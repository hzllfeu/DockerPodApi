"""Echo pod FastAPI server. Internal-only (Rust API calls in).

Endpoints:
  POST /infer           sync, JSON or SSE per Accept header
  POST /infer/async     async, returns 202 + job_id, webhook on completion
  GET  /jobs/{id}       status + result (polling fallback)
  GET  /healthz         liveness
  GET  /readyz          readiness (model loaded, backend selected)
  GET  /metrics         Prometheus format

Auth: every request must carry header `X-Internal-Secret: <ECHO_INTERNAL_SECRET>`.
Health endpoints (/healthz, /readyz) are exempt — Koyeb's load balancer
needs them.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, HttpUrl

from app.config import (
    ASYNC_MAX_SECONDS,
    INTERNAL_SECRET,
    SERVER_SYNC_TIMEOUT_S,
    SYNC_MAX_SECONDS,
    VRAM_REFUSE_RATIO,
    validate_boot_config,
)
from app.download import DownloadError, safe_download
from app.jobs import Job, JobManager, JobStatus
from app.trt_chain import Pipeline

log = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("ECHO_LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","name":"%(name)s","msg":%(message)r}',
)

PIPELINE: Pipeline | None = None
JOBS: JobManager | None = None
BOOT_TIME = time.time()

# Prometheus-flavored counters (plaintext, no client needed)
_METRICS = {
    "echo_requests_total": 0,
    "echo_requests_failed_total": 0,
    "echo_inferences_total": 0,
    "echo_inferences_failed_total": 0,
    "echo_audio_seconds_total": 0.0,
    "echo_inference_seconds_total": 0.0,
}


# ---------- helpers ----------

def _check_auth(x_internal_secret: str | None) -> None:
    if INTERNAL_SECRET is None:
        raise HTTPException(500, "server missing INTERNAL_SECRET")
    if not x_internal_secret or x_internal_secret != INTERNAL_SECRET:
        raise HTTPException(401, "missing or invalid X-Internal-Secret")


def _vram_ok() -> tuple[bool, float]:
    """Returns (ok, used_ratio)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return True, 0.0
        free, total = torch.cuda.mem_get_info()
        used = total - free
        ratio = used / total
        return ratio < VRAM_REFUSE_RATIO, ratio
    except Exception:
        return True, 0.0


# ---------- schemas ----------

class InferRequest(BaseModel):
    audio_url: HttpUrl
    # Optional caller-provided id for tracing (returned in /metrics + logs)
    request_id: str | None = None


class InferAsyncRequest(BaseModel):
    audio_url: HttpUrl
    callback_url: HttpUrl
    request_id: str | None = None


# ---------- lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE, JOBS
    validate_boot_config()
    log.info("loading pipeline...")
    PIPELINE = Pipeline()
    PIPELINE.load()
    if not PIPELINE.is_ready():
        log.error("no inference backend could be loaded; /readyz will 503")
    JOBS = JobManager()

    async def run_job(job: Job) -> dict:
        return await _run_inference(job.audio_url, ASYNC_MAX_SECONDS, job.id)

    await JOBS.start(run_job)
    log.info("server ready")
    yield
    log.info("shutting down...")
    if JOBS:
        await JOBS.stop()
    log.info("shutdown complete")


app = FastAPI(lifespan=lifespan, title="Echo (Hellfeu) inference pod")


# ---------- health ----------

@app.get("/healthz")
async def healthz():
    return {"status": "alive", "uptime_seconds": round(time.time() - BOOT_TIME, 1)}


@app.get("/readyz")
async def readyz():
    if PIPELINE is None or not PIPELINE.is_ready():
        statuses = []
        if PIPELINE is not None:
            statuses = [
                {"name": s.name, "ready": s.ready, "detail": s.detail}
                for s in PIPELINE.statuses
            ]
        return JSONResponse(
            {"status": "not_ready", "backends": statuses},
            status_code=503,
        )
    return {
        "status": "ready",
        "active_backend": PIPELINE.active_backend,
        "backends": [
            {"name": s.name, "ready": s.ready, "detail": s.detail}
            for s in PIPELINE.statuses
        ],
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    lines = []
    for k, v in _METRICS.items():
        lines.append(f"# TYPE {k} counter")
        lines.append(f"{k} {v}")
    if PIPELINE is not None and PIPELINE.is_ready():
        lines.append("# TYPE echo_backend_active gauge")
        for s in PIPELINE.statuses:
            v = 1 if (s.ready and s.name == PIPELINE.active_backend) else 0
            lines.append(f'echo_backend_active{{backend="{s.name}"}} {v}')
    return "\n".join(lines) + "\n"


# ---------- inference ----------

async def _run_inference(
    audio_url: str,
    max_seconds: int,
    request_id: str,
) -> dict:
    """Download + run pipeline. Pure async wrapper around the blocking PyTorch
    inference (run in a thread to keep the event loop responsive)."""
    if PIPELINE is None or not PIPELINE.is_ready():
        raise HTTPException(503, "pipeline not ready")
    vram_ok, vram_ratio = _vram_ok()
    if not vram_ok:
        raise HTTPException(
            503, f"VRAM busy (used ratio={vram_ratio:.2f})",
        )

    _METRICS["echo_requests_total"] += 1
    audio_path: Path | None = None
    try:
        audio_path, wav_info = safe_download(audio_url, max_seconds, request_id)
        _METRICS["echo_audio_seconds_total"] += wav_info.duration_s
        t0 = time.time()
        result = await asyncio.to_thread(PIPELINE.run, audio_path)
        elapsed = time.time() - t0
        _METRICS["echo_inferences_total"] += 1
        _METRICS["echo_inference_seconds_total"] += elapsed
        result.setdefault("meta", {}).update({
            "audio_duration_s": wav_info.duration_s,
            "inference_seconds": round(elapsed, 3),
            "rtf": round(wav_info.duration_s / elapsed, 2) if elapsed > 0 else None,
            "request_id": request_id,
        })
        return result
    except DownloadError as e:
        _METRICS["echo_requests_failed_total"] += 1
        raise HTTPException(e.status, e.msg)
    except HTTPException:
        _METRICS["echo_inferences_failed_total"] += 1
        raise
    except Exception as e:
        _METRICS["echo_inferences_failed_total"] += 1
        log.exception("inference failed")
        raise HTTPException(500, f"inference error: {e}")
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)


# ---------- endpoints ----------

@app.post("/infer")
async def infer(
    req: InferRequest,
    request: Request,
    x_internal_secret: str | None = Header(None),
    accept: str | None = Header(None),
):
    _check_auth(x_internal_secret)
    request_id = req.request_id or uuid.uuid4().hex

    if accept and "text/event-stream" in accept.lower():
        return StreamingResponse(
            _sse_stream(str(req.audio_url), request_id),
            media_type="text/event-stream",
            headers={"x-echo-request-id": request_id},
        )

    try:
        result = await asyncio.wait_for(
            _run_inference(str(req.audio_url), SYNC_MAX_SECONDS, request_id),
            timeout=SERVER_SYNC_TIMEOUT_S,
        )
        return JSONResponse(result, headers={"x-echo-request-id": request_id})
    except asyncio.TimeoutError:
        raise HTTPException(504, f"inference exceeded {SERVER_SYNC_TIMEOUT_S}s")


async def _sse_stream(audio_url: str, request_id: str):
    """SSE: server-sent events stream of segments + done.

    Today the pipeline runs as a blocking batch and we emit pre-sorted
    segments. As soon as we move to a streaming runtime, this function
    will yield segments while inference is still running with no API change.
    """
    if PIPELINE is None or not PIPELINE.is_ready():
        yield 'event: error\ndata: {"error":"pipeline not ready"}\n\n'
        return
    yield f'event: start\ndata: {json.dumps({"request_id": request_id})}\n\n'
    audio_path: Path | None = None
    try:
        audio_path, wav_info = safe_download(
            audio_url, SYNC_MAX_SECONDS, request_id,
        )
        # Run inference in a thread; collect ordered segments
        from app.pipeline import EchoPyTorchBackend
        backend = PIPELINE._backends.get("pytorch")
        if backend is None or not getattr(backend, "_ready", False):
            yield 'event: error\ndata: {"error":"streaming requires pytorch backend"}\n\n'
            return
        real = backend._backend  # type: ignore[attr-defined]
        if real is None:
            yield 'event: error\ndata: {"error":"backend internal not loaded"}\n\n'
            return
        events = await asyncio.to_thread(
            lambda: list(real.infer_streaming(audio_path)),
        )
        for ev in events:
            yield f"event: {ev.pop('event')}\ndata: {json.dumps(ev)}\n\n"
    except DownloadError as e:
        yield f'event: error\ndata: {json.dumps({"status": e.status, "error": e.msg})}\n\n'
    except Exception as e:
        log.exception("SSE stream failed")
        yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)


@app.post("/infer/async", status_code=202)
async def infer_async(
    req: InferAsyncRequest,
    x_internal_secret: str | None = Header(None),
):
    _check_auth(x_internal_secret)
    if JOBS is None:
        raise HTTPException(503, "job manager not ready")
    job = JOBS.submit(str(req.audio_url), str(req.callback_url))
    log.info("submitted job=%s", job.id)
    return {"job_id": job.id, "status": job.status}


@app.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    x_internal_secret: str | None = Header(None),
):
    _check_auth(x_internal_secret)
    if JOBS is None:
        raise HTTPException(503, "job manager not ready")
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.to_public()
