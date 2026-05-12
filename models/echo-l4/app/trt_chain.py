"""Backend wrapper.

Originally designed as a chain TRT → ONNX → PyTorch with fallbacks. For
this revision of the image we ship only the PyTorch backend (TRT/ONNX
removed at the user's request). The wrapper API is preserved so that we
can add other backends later without touching server.py.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.pipeline import EchoPyTorchBackend

log = logging.getLogger(__name__)


@dataclass
class BackendStatus:
    name: str
    ready: bool
    detail: str = ""
    compile_seconds: float | None = None


class BackendNotReady(Exception):
    pass


class PyTorchBackend:
    name = "pytorch"

    def __init__(self):
        self._backend: EchoPyTorchBackend | None = None
        self._ready = False
        self._reason = "not loaded"

    def try_load(self) -> BackendStatus:
        t0 = time.time()
        try:
            self._backend = EchoPyTorchBackend()
            self._backend.load()
            self._ready = True
            self._reason = "loaded"
            return BackendStatus(
                name=self.name, ready=True,
                detail="loaded", compile_seconds=time.time() - t0,
            )
        except Exception as e:
            log.exception("PyTorch backend load failed")
            self._reason = str(e)
            return BackendStatus(name=self.name, ready=False, detail=str(e))

    def run(self, audio_path: Path) -> dict[str, Any]:
        if not self._ready or self._backend is None:
            raise BackendNotReady(self._reason)
        return self._backend.infer(audio_path)


class Pipeline:
    """Single-backend wrapper today. Kept as a class so we can add more
    backends later without changing server.py."""

    def __init__(self):
        self._backend = PyTorchBackend()
        self._statuses: list[BackendStatus] = []

    def load(self) -> None:
        status = self._backend.try_load()
        self._statuses.append(status)
        log.info(
            "backend %s ready=%s detail=%s",
            self._backend.name, status.ready, status.detail,
        )

    @property
    def active_backend(self) -> str | None:
        return self._backend.name if self._backend._ready else None

    @property
    def statuses(self) -> list[BackendStatus]:
        return list(self._statuses)

    @property
    def _backends(self) -> dict[str, PyTorchBackend]:
        # Compatibility surface for server.py SSE path
        return {"pytorch": self._backend}

    def is_ready(self) -> bool:
        return self._backend._ready

    def run(self, audio_path: Path) -> dict[str, Any]:
        if not self._backend._ready:
            raise BackendNotReady("backend not ready")
        return self._backend.run(audio_path)
