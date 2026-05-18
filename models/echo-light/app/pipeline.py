"""Echo Light backend: DiariZen V4 (Echo Dia weights) + DiCoW_v3_3 ASR.

Best speed/quality tradeoff for meetings. No self-enrollment (unlike echo-l4).

Architecture:
  - Diarization: Echo Dia weights (Hellfeu/echo-dia) loaded into DiariZen V2 base
    (BUT-FIT/diarizen-wavlm-large-s80-md-v2)
  - ASR: BUT-FIT/DiCoW_v3_3 (vanilla, no SE-DiCoW)
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
    _orig = torch.load
    def _patched(*a, **kw):  # type: ignore[no-untyped-def]
        kw["weights_only"] = False
        return _orig(*a, **kw)
    torch.load = _patched  # type: ignore[assignment]


class EchoPyTorchBackend:
    """Echo Light: DiariZen V4 + DiCoW_v3_3. No self-enrollment."""

    def __init__(self):
        self._pipe = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        t0 = time.time()
        _patch_torch_load_weights_only()

        dicow_root = os.environ.get("ECHO_DICOW_ROOT", "/opt/DiCoW")
        sys.path.insert(0, dicow_root)

        log.info("loading Echo Dia (V4) weights from local cache...")
        from huggingface_hub import hf_hub_download
        offline = os.environ.get("ECHO_HF_OFFLINE", "1") == "1"
        weights_path = hf_hub_download(
            repo_id="Hellfeu/echo-dia",
            filename="pytorch_model.bin",
            local_files_only=offline,
        )

        log.info("loading DiariZen base pipeline...")
        from diarizen.pipelines.inference import DiariZenPipeline
        pipe_dia = DiariZenPipeline.from_pretrained(
            "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        ).to(torch.device("cuda:0"))
        sd = torch.load(weights_path, map_location="cuda:0")
        pipe_dia._segmentation.model.load_state_dict(sd, strict=False)

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
            "Echo Light pipeline ready in %.1fs, VRAM=%.2f GB",
            time.time() - t0, torch.cuda.memory_allocated() / 1e9,
        )

    def infer(self, audio_path: Path) -> dict[str, Any]:
        if not self._loaded or self._pipe is None:
            raise RuntimeError("backend not loaded")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.time()
        out = self._pipe(str(audio_path), return_timestamps=True, batch_size=1)
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
                "backend": "echo-light",
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
