"""Echo Dia backend: diarization-only (no ASR).

Pure diarization API: takes an audio file, returns RTTM-formatted speaker
turns + a structured list of segments. No transcription.

Architecture:
  - DiariZen V4 = DiariZen V2 base (BUT-FIT/diarizen-wavlm-large-s80-md-v2)
    + Echo Dia fine-tuned weights (Hellfeu/echo-dia).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator

import torch

log = logging.getLogger(__name__)


def _patch_torch_load_weights_only() -> None:
    _orig = torch.load
    def _patched(*a, **kw):  # type: ignore[no-untyped-def]
        kw["weights_only"] = False
        return _orig(*a, **kw)
    torch.load = _patched  # type: ignore[assignment]


class EchoPyTorchBackend:
    """Echo Dia: diarization-only backend."""

    def __init__(self):
        self._pipe = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        t0 = time.time()
        _patch_torch_load_weights_only()

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

        self._pipe = pipe_dia
        self._loaded = True
        log.info(
            "Echo Dia pipeline ready in %.1fs, VRAM=%.2f GB",
            time.time() - t0, torch.cuda.memory_allocated() / 1e9,
        )

    def infer(self, audio_path: Path) -> dict[str, Any]:
        if not self._loaded or self._pipe is None:
            raise RuntimeError("backend not loaded")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.time()
        # DiariZenPipeline returns a pyannote.core.Annotation
        annotation = self._pipe(str(audio_path))
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        speakers_map: dict[str, list[dict[str, float]]] = {}
        rttm_lines: list[str] = []
        for segment, _, spk_id in annotation.itertracks(yield_label=True):
            spk_key = str(spk_id)
            speakers_map.setdefault(spk_key, []).append({
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
            })
            rttm_lines.append(
                f"SPEAKER {audio_path.stem} 1 {segment.start:.3f} "
                f"{segment.duration:.3f} <NA> <NA> {spk_key} <NA> <NA>"
            )

        speakers = [
            {"id": spk_id, "segments": [
                {"start": s["start"], "end": s["end"], "text": ""}
                for s in segs
            ]}
            for spk_id, segs in sorted(speakers_map.items())
        ]
        return {
            "transcript": "",
            "speakers": speakers,
            "rttm": "\n".join(rttm_lines),
            "meta": {
                "n_speakers": len(speakers),
                "inference_seconds": round(elapsed, 3),
                "backend": "echo-dia",
                "note": "diarization-only, no ASR",
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
                        "text": "",
                    })
                )
        events_by_time.sort(key=lambda x: x[0])
        for _, ev in events_by_time:
            yield ev
        yield {"event": "done", "meta": result["meta"]}
