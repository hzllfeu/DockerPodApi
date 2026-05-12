"""Echo Dia + SE-DiCoW PyTorch backend.

Wraps the existing Hellfeu/echo recipe (DiariZen-based segmentation + SE-DiCoW
Whisper-style ASR with target-speaker self-enrollment) into a uniform infer()
that takes a WAV path and returns a normalised result dict.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch

log = logging.getLogger(__name__)


# Whisper-style timestamp tokens: <|12.34|>
TS_PAT = re.compile(r"<\|(\d+\.\d+)\|>")


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
    """Force weights_only=False; pyannote / DiariZen checkpoints contain
    Python objects that the new safe-load rejects. Our checkpoints come from
    trusted sources, baked into the image at build time."""
    _orig = torch.load
    def _patched(*a, **kw):  # type: ignore[no-untyped-def]
        kw["weights_only"] = False
        return _orig(*a, **kw)
    torch.load = _patched  # type: ignore[assignment]


class EchoPyTorchBackend:
    """Heavy PyTorch backend. Loaded once at boot."""

    def __init__(self):
        self._pipe = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        t0 = time.time()
        _patch_torch_load_weights_only()

        # Sys.path: DiCoW is vendored alongside the image; DiariZen is pip-installed
        dicow_root = os.environ.get("ECHO_DICOW_ROOT", "/opt/DiCoW")
        sys.path.insert(0, dicow_root)

        log.info("loading Hellfeu/echo-dia weights from local cache...")
        from huggingface_hub import hf_hub_download
        # Models are baked into the image at build time. At runtime the image
        # runs with HF_HUB_OFFLINE=1 and local_files_only=True; for local dev
        # on a pod that already has HF_TOKEN, ECHO_HF_OFFLINE=0 enables online
        # fallback.
        offline = os.environ.get("ECHO_HF_OFFLINE", "1") == "1"
        weights_path = hf_hub_download(
            repo_id="Hellfeu/echo-dia",
            filename="pytorch_model.bin",
            local_files_only=offline,
        )
        log.info("  echo-dia weights at %s", weights_path)

        log.info("loading DiariZen base pipeline...")
        from diarizen.pipelines.inference import DiariZenPipeline
        pipe_dia = DiariZenPipeline.from_pretrained(
            "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        ).to(torch.device("cuda:0"))
        sd = torch.load(weights_path, map_location="cuda:0")
        pipe_dia._segmentation.model.load_state_dict(sd, strict=False)

        log.info("loading SE-DiCoW ASR...")
        from transformers import (
            AutoTokenizer,
            AutoFeatureExtractor,
            AutoModelForSpeechSeq2Seq,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "BUT-FIT/SE-DiCoW", trust_remote_code=True,
        ).to("cuda")

        # SE-DiCoW remote code has a bug: it does an unconditional
        # `del self.enrollments` in _retrieve_init_tokens. After the first
        # call the attribute is gone and the second call raises. We patch
        # the bound method to be defensive.
        if hasattr(model, "_retrieve_init_tokens"):
            _orig = model._retrieve_init_tokens
            def _safe_retrieve(*args, **kwargs):
                try:
                    return _orig(*args, **kwargs)
                except AttributeError as e:
                    if "enrollments" in str(e):
                        # ensure attribute exists before the del, retry
                        model.enrollments = getattr(model, "enrollments", None)
                        return _orig(*args, **kwargs)
                    raise
            model._retrieve_init_tokens = _safe_retrieve
        fe = AutoFeatureExtractor.from_pretrained("BUT-FIT/SE-DiCoW")
        tok = AutoTokenizer.from_pretrained("BUT-FIT/SE-DiCoW")
        model.config.model_type = "whisper"
        model.tokenizer = tok
        if hasattr(model, "set_tokenizer"):
            model.set_tokenizer(tok)

        # Import DiCoWPipeline from the vendored repo
        spec = importlib.util.spec_from_file_location(
            "dicow_pipeline_mod", os.path.join(dicow_root, "pipeline.py"),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not locate {dicow_root}/pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        DiCoWPipeline = mod.DiCoWPipeline

        self._pipe = DiCoWPipeline(
            model=model,
            diarization_pipeline=pipe_dia,
            feature_extractor=fe,
            tokenizer=tok,
            device=torch.device("cuda:0"),
        )
        self._loaded = True
        log.info(
            "Echo PyTorch pipeline ready in %.1fs, VRAM=%.2f GB",
            time.time() - t0, torch.cuda.memory_allocated() / 1e9,
        )

    def infer(self, audio_path: Path) -> dict[str, Any]:
        """Run end-to-end pipeline. Returns the normalised result schema."""
        if not self._loaded or self._pipe is None:
            raise RuntimeError("backend not loaded")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.time()
        out = self._pipe(str(audio_path), return_timestamps=True)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
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
                "inference_seconds": round(elapsed, 3),
                "backend": "pytorch",
            },
        }

    # ---------- streaming variant ----------

    def infer_streaming(
        self, audio_path: Path,
    ) -> Iterator[dict[str, Any]]:
        """Stream segments as they are produced. Today's DiCoWPipeline runs
        non-streaming; we emit a synchronous batch then yield segments
        ordered by time. This is sufficient for the SSE contract (consumer
        sees one event per segment). A true streaming runtime is left for
        later (would require restructuring SE-DiCoW)."""
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
