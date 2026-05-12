# API contract — every model image

All model images in this repository expose this same HTTP contract.

## Auth

Every endpoint except `/healthz`, `/readyz`, `/metrics` requires:

    X-Internal-Secret: <shared secret, ≥ 24 chars>

The secret is set via the `ECHO_INTERNAL_SECRET` env var at container start.
A missing or wrong secret returns `401`.

## Endpoints

### POST /infer (synchronous)

For audio shorter than `ECHO_SYNC_MAX_SECONDS` (default 300 s = 5 min).

Request body:

    {
      "audio_url": "https://example.com/clip.wav",
      "request_id": "req_abc123"        // optional, echoed in X-Echo-Request-Id and logs
    }

The pod downloads the audio itself with full SSRF protection
(see [SECURITY.md](SECURITY.md)). The audio must be WAV 16 kHz mono
16-bit PCM. Anything else → 415.

If header `Accept: text/event-stream`, the response is SSE
(see [Streaming](#streaming)). Otherwise JSON inline:

    {
      "transcript": "yeah oh no sorry ok so",
      "speakers": [
        {
          "id": "S0",
          "total_speaking_time_s": 0.30,
          "segment_count": 1,
          "segments": [
            { "start": 9.780, "end": 10.080, "text": "yeah" }
          ]
        },
        {
          "id": "S1",
          "total_speaking_time_s": 2.04,
          "segment_count": 3,
          "segments": [
            { "start": 8.960,  "end": 10.760, "text": "oh no sorry" },
            { "start": 16.660, "end": 16.800, "text": "ok" },
            { "start": 28.900, "end": 29.500, "text": "so" }
          ]
        }
      ],
      "diarization": [
        { "start": 8.960,  "end": 10.760, "speaker_id": "S1", "text": "oh no sorry" },
        { "start": 9.780,  "end": 10.080, "speaker_id": "S0", "text": "yeah" },
        { "start": 16.660, "end": 16.800, "speaker_id": "S1", "text": "ok" },
        { "start": 28.900, "end": 29.500, "speaker_id": "S1", "text": "so" }
      ],
      "rttm": "SPEAKER req_abc123 1 9.780 0.300 <NA> <NA> S0 <NA> <NA>\nSPEAKER req_abc123 1 8.960 1.800 <NA> <NA> S1 <NA> <NA>\n...",
      "timing": {
        "download_seconds":     0.05,
        "inference_seconds":    6.34,
        "post_process_seconds": 0.002,
        "total_seconds":        6.39
      },
      "meta": {
        "n_speakers":       2,
        "audio_duration_s": 30.0,
        "rtf":              4.73,
        "backend":          "pytorch",
        "model_version":    "Hellfeu/echo@v1",
        "request_id":       "req_abc123"
      }
    }

Notes on the schema:

- `speakers[]` is the **per-speaker** view: each speaker's segments grouped
  under their id. Good for transcript display.
- `diarization[]` is the same data **flat and time-sorted**. Good for
  rendering a timeline.
- `rttm` is the standard NIST RTTM-string format (one line per segment)
  for direct consumption by `dscore`, `pyannote.metrics`, etc.
- All three views are derived from the same source. Pick what your client
  needs.

### Streaming

When `Accept: text/event-stream`:

    event: start
    data: {"request_id": "req_abc123"}

    event: segment
    data: {"speaker_id": "S1", "start": 8.96, "end": 10.76, "text": "oh no sorry"}

    event: segment
    data: {"speaker_id": "S0", "start": 9.78, "end": 10.08, "text": "yeah"}

    ...

    event: done
    data: {"timing": {...}, "meta": {...}}

Segments are emitted in time-sorted order. If inference fails an
`event: error` is emitted instead of `done`.

### POST /infer/async

For audio up to `ECHO_ASYNC_MAX_SECONDS` (default 3600 s = 60 min).

Request body:

    {
      "audio_url":    "https://example.com/long.wav",
      "callback_url": "https://my-rust-api.example.com/v1/echo/webhook",
      "request_id":   "req_xyz"   // optional
    }

Response `202 Accepted`:

    {"job_id": "abc123def456", "status": "queued"}

When inference finishes, the pod POSTs the **same** body as `/infer` to
`callback_url` with headers:

    Content-Type: application/json
    X-Echo-Job-Id: <job_id>
    X-Echo-Event: job.completed   (or job.failed)
    X-Echo-Signature: sha256=<hex>   (if ECHO_WEBHOOK_HMAC_SECRET is set)

Three delivery attempts, exponential backoff (2 s, 5 s, 15 s). After
that the job stays in `failed` state on the pod side; the caller can
poll `/jobs/{id}` to get the result.

To verify a webhook is from us, compute
`hmac.new(secret, body, sha256).hexdigest()` and compare with the
`X-Echo-Signature` value after the `sha256=` prefix.

### GET /jobs/{job_id}

Polling fallback for async. Same body as the webhook payload.

    {
      "job_id":      "abc123def456",
      "status":      "queued|running|done|failed",
      "created_at":  1736635123.45,
      "started_at":  1736635124.10,
      "finished_at": 1736635170.55,
      "error":       null,
      "result":      { ...same as /infer sync response... }
    }

### GET /healthz

Liveness only. Always 200 as long as the process is up.

    {"status": "alive", "uptime_seconds": 123.4}

### GET /readyz

Readiness: model loaded and at least one backend ready.

    {
      "status": "ready",
      "active_backend": "pytorch",
      "backends": [
        {"name": "pytorch", "ready": true, "detail": "loaded"}
      ]
    }

Returns `503` until the model finishes loading.

### GET /metrics

Prometheus plain-text format. Counters incremented on each request:

    echo_requests_total
    echo_requests_failed_total
    echo_inferences_total
    echo_inferences_failed_total
    echo_audio_seconds_total
    echo_inference_seconds_total
    echo_backend_active{backend="pytorch"} 1

## Error codes

| code | meaning |
|---|---|
| 400 | bad audio URL, redirect loop, malformed WAV, payload too small |
| 401 | missing or wrong `X-Internal-Secret` |
| 403 | URL resolves to private / loopback / link-local IP |
| 413 | audio too long (`Content-Length` or duration > limit) |
| 415 | wrong `Content-Type` or WAV not 16 kHz mono 16-bit PCM |
| 502 | upstream HEAD or GET on `audio_url` failed |
| 503 | model not loaded, VRAM full, or pod draining |
| 504 | sync inference exceeded `ECHO_SYNC_TIMEOUT_S` |

## Required env vars

| var | required | default | purpose |
|---|---|---|---|
| `ECHO_INTERNAL_SECRET` | yes | — | shared secret with caller (≥ 24 chars) |
| `ECHO_WEBHOOK_HMAC_SECRET` | no | — | enables HMAC signing of webhook bodies |
| `ECHO_JOBS_CACHE_DIR` | no | `/cache/jobs` | persistent volume for async job state |
| `ECHO_SYNC_MAX_SECONDS` | no | `300` | sync audio cap |
| `ECHO_ASYNC_MAX_SECONDS` | no | `3600` | async audio cap |
| `ECHO_SYNC_TIMEOUT_S` | no | `90` | server-side timeout for /infer |
| `ECHO_LOG_LEVEL` | no | `INFO` | python logging level |
| `ECHO_INSECURE_LOCAL_DOWNLOAD` | no | `0` | **dev only** — accept http:// and private IPs |
