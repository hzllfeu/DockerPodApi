"""In-process async job manager with disk persistence.

Why not Redis? For a single-pod inference worker the queue is local. Koyeb
scales the pod horizontally; each pod has its own queue. We persist to disk
so that a pod restart keeps the queue (jobs visible via GET /jobs/{id}).

Job states: queued → running → done | failed
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from app.config import (
    JOBS_CACHE_DIR,
    WEBHOOK_BACKOFF_SECONDS,
    WEBHOOK_HMAC_SECRET,
    WEBHOOK_MAX_RETRIES,
    WEBHOOK_TIMEOUT,
)

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    audio_url: str
    callback_url: str | None
    status: str = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    def to_public(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result if self.status == JobStatus.DONE else None,
        }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._loop_runner = None  # set in start()
        self._running = False

    def _persist(self, job: Job) -> None:
        path = JOBS_CACHE_DIR / f"{job.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(job)))

    def _restore(self) -> None:
        """Load existing jobs from disk. Done jobs stay visible; queued jobs
        are re-enqueued in case the pod was restarted mid-run."""
        if not JOBS_CACHE_DIR.exists():
            return
        for f in JOBS_CACHE_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                job = Job(**data)
                self._jobs[job.id] = job
                if job.status == JobStatus.QUEUED:
                    self._queue.put_nowait(job.id)
                elif job.status == JobStatus.RUNNING:
                    # Pod crashed mid-job: mark failed.
                    job.status = JobStatus.FAILED
                    job.error = "interrupted by pod restart"
                    job.finished_at = time.time()
                    self._persist(job)
            except Exception:
                log.exception("failed to restore job from %s", f)

    async def start(self, runner) -> None:
        """`runner` is an async callable: runner(job) → result_dict. It must
        do download + inference and return the JSON-able result."""
        self._loop_runner = runner
        self._restore()
        self._running = True
        # One worker for now (single GPU). Add more if you want overlap with
        # multi-instance and have the VRAM.
        self._workers.append(asyncio.create_task(self._worker()))

    async def stop(self) -> None:
        self._running = False
        for w in self._workers:
            w.cancel()
        for w in self._workers:
            try:
                await w
            except asyncio.CancelledError:
                pass

    def submit(self, audio_url: str, callback_url: str | None) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, audio_url=audio_url, callback_url=callback_url)
        self._jobs[job_id] = job
        self._persist(job)
        self._queue.put_nowait(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def _worker(self) -> None:
        while self._running:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                return
            job = self._jobs.get(job_id)
            if not job:
                continue
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._persist(job)
            try:
                if self._loop_runner is None:
                    raise RuntimeError("job manager runner not set")
                result = await self._loop_runner(job)
                job.result = result
                job.status = JobStatus.DONE
            except Exception as e:
                log.exception("job %s failed", job_id)
                job.error = str(e)
                job.status = JobStatus.FAILED
            finally:
                job.finished_at = time.time()
                self._persist(job)
            if job.callback_url:
                asyncio.create_task(_send_webhook(job))


# ---------- webhook delivery ----------

def _sign(body: bytes) -> str:
    if not WEBHOOK_HMAC_SECRET:
        return ""
    return hmac.new(
        WEBHOOK_HMAC_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()


async def _send_webhook(job: Job) -> None:
    if not job.callback_url:
        return
    payload = job.to_public()
    body = json.dumps(payload, separators=(",", ":")).encode()
    sig = _sign(body)
    headers = {
        "content-type": "application/json",
        "x-echo-job-id": job.id,
        "x-echo-event": "job.completed" if job.status == JobStatus.DONE else "job.failed",
    }
    if sig:
        headers["x-echo-signature"] = f"sha256={sig}"

    timeouts = httpx.Timeout(WEBHOOK_TIMEOUT)
    last_error = None
    for attempt in range(WEBHOOK_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=timeouts) as client:
                r = await client.post(
                    job.callback_url, content=body, headers=headers,
                )
            if 200 <= r.status_code < 300:
                log.info(
                    "webhook delivered job=%s attempt=%d status=%d",
                    job.id, attempt + 1, r.status_code,
                )
                return
            last_error = f"http {r.status_code}"
        except Exception as e:
            last_error = str(e)
        if attempt < WEBHOOK_MAX_RETRIES - 1:
            await asyncio.sleep(WEBHOOK_BACKOFF_SECONDS[attempt])
    log.error(
        "webhook delivery FAILED for job=%s after %d attempts: %s",
        job.id, WEBHOOK_MAX_RETRIES, last_error,
    )
