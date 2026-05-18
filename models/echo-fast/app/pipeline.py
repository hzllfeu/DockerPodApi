"""Echo Fast backend: pyannote-community-1 (diar) + DiCoW_v3_3 (ASR).

~2x faster than Echo Light. Trade-off: lower quality on meetings.

Architecture problem: pyannote-community-1 requires pyannote.audio>=4.0 which
is INCOMPATIBLE with DiCoW (vendored pyannote.audio 3.1.1). We solve this by
running pyannote in a separate Python venv via subprocess, persisting the
RTTM to disk, then loading it back as an Annotation for DiCoW.

The dedicated pyannote venv lives at /opt/pyannote_venv (built at image
build time). DiCoW runs in the main /opt/python-runtime.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import torch

log = logging.getLogger(__name__)


TS_PAT = re.compile(r"<\|(\d+\.\d+)\|>")


class _BatchSizeTuner:
    """Adaptive batch_size finder for DiCoW inference.

    Behavior:
      - Starts at `initial` (env-overridable via ECHO_BATCH_SIZE_INIT, else
        derived from total VRAM at first call).
      - On OOM, halves `current` (down to 1) and retries on the SAME call.
      - On 3 consecutive successes at current value, tries to bump up one step
        (next power of 2, capped at `max_cap`).
    """

    def __init__(self, max_cap: int = 32):
        self._fixed = os.environ.get("ECHO_BATCH_SIZE_FIXED")
        if self._fixed:
            self._fixed = max(1, int(self._fixed))
            self.current = self._fixed
        else:
            init = os.environ.get("ECHO_BATCH_SIZE_INIT")
            if init:
                self.current = max(1, int(init))
            else:
                self.current = self._initial_from_vram()
        self.max_cap = max_cap
        self._consecutive_ok = 0

    @staticmethod
    def _initial_from_vram() -> int:
        try:
            if not torch.cuda.is_available():
                return 1
            _free, total = torch.cuda.mem_get_info()
            total_gb = total / 1e9
            if total_gb >= 70:
                return 16
            if total_gb >= 35:
                return 8
            if total_gb >= 22:
                return 4
            if total_gb >= 14:
                return 2
            return 1
        except Exception:
            return 1

    def run(self, fn) -> tuple:
        import logging as _log_mod
        _log = _log_mod.getLogger(__name__)
        attempts = 0
        bs = self.current
        while True:
            attempts += 1
            try:
                result = fn(bs)
                if self._fixed is None and bs == self.current:
                    self._consecutive_ok += 1
                    if self._consecutive_ok >= 3 and bs < self.max_cap:
                        new_bs = min(self.max_cap, bs * 2)
                        if new_bs != bs:
                            _log.info("batch_size tuner: bumping %d -> %d", bs, new_bs)
                            self.current = new_bs
                            self._consecutive_ok = 0
                return result, bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._consecutive_ok = 0
                if bs <= 1:
                    raise
                new_bs = max(1, bs // 2)
                _log.warning("batch_size tuner: OOM at bs=%d -> retry at %d", bs, new_bs)
                bs = new_bs
                self.current = bs
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    self._consecutive_ok = 0
                    if bs <= 1:
                        raise
                    new_bs = max(1, bs // 2)
                    _log.warning("batch_size tuner: OOM-like at bs=%d -> retry at %d", bs, new_bs)
                    bs = new_bs
                    self.current = bs
                else:
                    raise
            if attempts > 8:
                raise RuntimeError(f"batch_size tuner: too many attempts (bs={bs})")


PYANNOTE_VENV_PYTHON = os.environ.get(
    "ECHO_PYANNOTE_PYTHON", "/opt/pyannote_venv/bin/python"
)


def _parse_whisper_ts(text: str) -> list[tuple[float, float, str]]:
    pieces = TS_PAT.split(text)
    times: list[float] = []
    for i in range(1, len(pieces), 2):
        try:
            times.append(float(pieces[i]))
        except ValueError:
            pass
    texts = [pieces[i] for i in range(2, len(pieces), 2)]
    out: list[tuple[float, float, str]] = []
    for i, raw in enumerate(texts):
        if i + 1 >= len(times):
            break
        t0, t1 = times[i], times[i + 1]
        txt = raw.strip()
        if t1 > t0 and txt:
            out.append((t0, t1, txt))
    return out


def _patch_torch_load_weights_only() -> None:
    _orig = torch.load
    def _patched(*a, **kw):  # type: ignore[no-untyped-def]
        kw["weights_only"] = False
        return _orig(*a, **kw)
    torch.load = _patched  # type: ignore[assignment]


def _load_rttm_as_annotation(rttm_path: str, uri: str):
    from pyannote.core import Annotation, Segment
    a = Annotation()
    if not os.path.exists(rttm_path):
        return a
    with open(rttm_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 8 or p[0] != "SPEAKER":
                continue
            a[Segment(float(p[3]), float(p[3]) + float(p[4]))] = p[7]
    a.uri = uri
    return a


class _CachedDiarPipeline:
    """Wraps a precomputed RTTM as a DiCoW-compatible diarization callable."""

    def __init__(self, rttm_path: str, uri: str):
        self.rttm_path = rttm_path
        self.uri = uri

    def __call__(self, audio_path: str):
        return _load_rttm_as_annotation(self.rttm_path, self.uri)


def _run_pyannote_in_subprocess(audio_path: Path, rttm_out: Path) -> None:
    """Invoke the pyannote venv to produce an RTTM. Raises on failure."""
    if not os.path.exists(PYANNOTE_VENV_PYTHON):
        raise RuntimeError(
            f"pyannote venv python not found at {PYANNOTE_VENV_PYTHON}"
        )
    hf_token = os.environ.get("HF_TOKEN", "")
    script = """
import sys, torch
from pyannote.audio import Pipeline
import soundfile as sf

audio_path, rttm_out, hf_token = sys.argv[1], sys.argv[2], sys.argv[3] or None
pipe = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", token=hf_token,
)
pipe.to(torch.device("cuda:0"))
audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
audio = audio.T
wav_tensor = torch.from_numpy(audio.copy())
diar = pipe({"waveform": wav_tensor, "sample_rate": sr, "uri": "audio"})
ann = diar.speaker_diarization
with open(rttm_out, "w") as f:
    ann.write_rttm(f)
print("RTTM_OK")
"""
    res = subprocess.run(
        [PYANNOTE_VENV_PYTHON, "-c", script, str(audio_path), str(rttm_out), hf_token],
        capture_output=True, text=True, timeout=600,
    )
    if res.returncode != 0 or not rttm_out.exists():
        raise RuntimeError(
            f"pyannote subprocess failed (rc={res.returncode}): "
            f"stdout={res.stdout[-500:]} stderr={res.stderr[-500:]}"
        )


class EchoPyTorchBackend:
    """Echo Fast: pyannote-community-1 (subprocess) + DiCoW_v3_3."""

    def __init__(self):
        self._asr_model = None
        self._asr_fe = None
        self._asr_tok = None
        self._dicow_pipeline_cls = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        t0 = time.time()
        _patch_torch_load_weights_only()

        dicow_root = os.environ.get("ECHO_DICOW_ROOT", "/opt/DiCoW")
        sys.path.insert(0, dicow_root)

        log.info("loading DiCoW_v3_3 ASR...")
        from transformers import (
            AutoTokenizer,
            AutoFeatureExtractor,
            AutoModelForSpeechSeq2Seq,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "BUT-FIT/DiCoW_v3_3", trust_remote_code=True,
        ).to("cuda")
        fe = AutoFeatureExtractor.from_pretrained("BUT-FIT/DiCoW_v3_3")
        tok = AutoTokenizer.from_pretrained("BUT-FIT/DiCoW_v3_3")
        model.config.model_type = "whisper"
        model.tokenizer = tok
        if hasattr(model, "set_tokenizer"):
            model.set_tokenizer(tok)

        spec = importlib.util.spec_from_file_location(
            "dicow_pipeline_mod", os.path.join(dicow_root, "pipeline.py"),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not locate {dicow_root}/pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._dicow_pipeline_cls = mod.DiCoWPipeline
        self._asr_model = model
        self._asr_fe = fe
        self._asr_tok = tok
        self._loaded = True
        log.info(
            "Echo Fast ASR ready in %.1fs, VRAM=%.2f GB. Diarization runs "
            "on-demand via pyannote subprocess.",
            time.time() - t0, torch.cuda.memory_allocated() / 1e9,
        )

    def infer(self, audio_path: Path) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("backend not loaded")

        with tempfile.NamedTemporaryFile(
            suffix=".rttm", delete=False,
        ) as rttm_file:
            rttm_path = Path(rttm_file.name)
        try:
            t_diar0 = time.time()
            _run_pyannote_in_subprocess(audio_path, rttm_path)
            diar_elapsed = time.time() - t_diar0

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t_asr0 = time.time()

            diar = _CachedDiarPipeline(str(rttm_path), audio_path.stem)
            pipe = self._dicow_pipeline_cls(
                model=self._asr_model,
                diarization_pipeline=diar,
                feature_extractor=self._asr_fe,
                tokenizer=self._asr_tok,
                device=torch.device("cuda:0"),
            )
            out, used_bs = self._batch_tuner.run(

                lambda bs: pipe(str(audio_path), return_timestamps=True, batch_size=bs),

            )
            torch.cuda.synchronize()
            asr_elapsed = time.time() - t_asr0
        finally:
            rttm_path.unlink(missing_ok=True)

        per_spk = out.get("per_spk_outputs", []) or []
        speakers: list[dict[str, Any]] = []
        rttm_lines: list[str] = []
        for spk_idx, spk_text in enumerate(per_spk):
            spk_id = f"S{spk_idx}"
            spans = _parse_whisper_ts(spk_text)
            segments = [
                {"start": round(t0_, 3), "end": round(t1_, 3), "text": txt}
                for t0_, t1_, txt in spans
            ]
            speakers.append({"id": spk_id, "segments": segments})
            for t0_, t1_, _ in spans:
                rttm_lines.append(
                    f"SPEAKER {audio_path.stem} 1 {t0_:.3f} "
                    f"{t1_ - t0_:.3f} <NA> <NA> {spk_id} <NA> <NA>"
                )
        transcript = " ".join(
            seg["text"] for sp in speakers for seg in sp["segments"]
        )
        return {
            "transcript": transcript,
            "speakers": speakers,
            "rttm": "\n".join(rttm_lines),
            "meta": {
                "n_speakers": len(speakers),
                "diarization_seconds": round(diar_elapsed, 3),
                "asr_seconds": round(asr_elapsed, 3),
                "inference_seconds": round(diar_elapsed + asr_elapsed, 3),
                "backend": "echo-fast",
                "batch_size": used_bs,
            },
        }

    def infer_streaming(
        self, audio_path: Path,
    ) -> Iterator[dict[str, Any]]:
        result = self.infer(audio_path)
        events_by_time: list[tuple[float, dict[str, Any]]] = []
        for sp in result["speakers"]:
            for seg in sp["segments"]:
                events_by_time.append(
                    (seg["start"], {
                        "event": "segment",
                        "speaker_id": sp["id"],
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                    })
                )
        events_by_time.sort(key=lambda x: x[0])
        for _, ev in events_by_time:
            yield ev
        yield {"event": "done", "meta": result["meta"]}
