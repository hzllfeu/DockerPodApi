# echo-l4 — Echo inference pod (Hellfeu/echo + SE-DiCoW)

Production-grade single-image pod for Echo audio diarization + ASR on
NVIDIA L4 / L40S. Designed to be called by a Rust API gateway over the
Koyeb internal network.

## Layout

    docker/echo-l4/
    ├── README.md
    ├── requirements.txt
    ├── app/
    │   ├── server.py          # FastAPI: /infer, /infer/async, /healthz, /readyz, /metrics
    │   ├── config.py          # env-driven runtime config
    │   ├── download.py        # SSRF-safe https-only WAV download + validation
    │   ├── trt_chain.py       # backend chain: TRT → ONNX → PyTorch
    │   ├── pipeline.py        # PyTorch backend = Hellfeu/echo (DiariZen V4 + SE-DiCoW)
    │   └── jobs.py            # in-process async job queue + HMAC-signed webhooks
    └── bazel/                 # build files for rules_oci (Bazel)

## API contract

All endpoints require `X-Internal-Secret: <ECHO_INTERNAL_SECRET>` except
`/healthz`, `/readyz`, `/metrics`.

### POST /infer (sync, ≤ 5 min audio)

Request:

    {
      "audio_url": "https://example.com/clip.wav",
      "request_id": "req_abc123"   // optional, echoed in logs and X-Echo-Request-Id
    }

If `Accept: text/event-stream`, the response is SSE: one `start` event,
one `segment` event per timestamped span, one `done` event with meta.
Otherwise JSON inline:

    {
      "transcript": "...",
      "speakers": [
        { "id": "S0", "segments": [{"start": 0.0, "end": 5.2, "text": "..."}, ...] },
        ...
      ],
      "rttm": "SPEAKER <stem> 1 0.000 5.200 <NA> <NA> S0 <NA> <NA>\n...",
      "meta": {
        "n_speakers": 4,
        "audio_duration_s": 280.5,
        "inference_seconds": 51.2,
        "rtf": 5.48,
        "backend": "pytorch",
        "request_id": "req_abc123"
      }
    }

### POST /infer/async (async, ≤ 60 min audio)

Request:

    {
      "audio_url": "https://example.com/long.wav",
      "callback_url": "https://my-rust-api.koyeb.app/v1/echo/webhook",
      "request_id": "..."  // optional
    }

Response 202:

    { "job_id": "abc123def", "status": "queued" }

On completion the pod POSTs the same JSON shape as /infer to
`callback_url`, with headers:

    Content-Type: application/json
    X-Echo-Job-Id: <job_id>
    X-Echo-Event: job.completed | job.failed
    X-Echo-Signature: sha256=<hex>   (if ECHO_WEBHOOK_HMAC_SECRET is set)

Three retry attempts, exponential backoff (2s, 5s, 15s). If all three
fail the job stays in `failed` state and callers must poll.

### GET /jobs/{id}

Polling fallback. Returns same payload as the webhook body.

### Errors

| code | meaning |
|---|---|
| 400 | bad audio URL, redirect loop, malformed WAV, payload too small |
| 401 | missing or wrong X-Internal-Secret |
| 403 | URL resolves to private / loopback / link-local IP |
| 413 | audio too long (Content-Length or duration > limit) |
| 415 | wrong Content-Type or WAV not 16 kHz mono 16-bit PCM |
| 502 | upstream HEAD/GET on audio_url failed |
| 503 | model not loaded, VRAM full, or pod draining |
| 504 | inference exceeded SERVER_SYNC_TIMEOUT_S |

## Required env vars

| var | required | default | purpose |
|---|---|---|---|
| `ECHO_INTERNAL_SECRET` | yes | — | shared secret with caller. Must be ≥ 24 chars |
| `ECHO_WEBHOOK_HMAC_SECRET` | no | — | enables HMAC signing of webhook bodies |
| `ECHO_TRT_CACHE_DIR` | no | `/cache/trt` | persistent volume for TRT engines |
| `ECHO_JOBS_CACHE_DIR` | no | `/cache/jobs` | persistent volume for async job state |
| `ECHO_SYNC_MAX_SECONDS` | no | `300` | sync audio cap |
| `ECHO_ASYNC_MAX_SECONDS` | no | `3600` | async audio cap |
| `ECHO_SYNC_TIMEOUT_S` | no | `90` | server-side timeout for /infer |
| `ECHO_LOG_LEVEL` | no | `INFO` | python logging level |

## Model assets

The image is built with all model weights baked in (Hellfeu/echo-dia,
BUT-FIT/diarizen-wavlm-large-s80-md-v2, BUT-FIT/SE-DiCoW). At runtime
the HF hub is offline (`HF_HUB_OFFLINE=1`). The image therefore needs
no HF_TOKEN at runtime and no outbound to huggingface.co.

## Backend chain

Boot tries TRT → ONNX → PyTorch in that order. Today TRT and ONNX are
stubs and the active backend is PyTorch. The chain is in place so that
exporting the components to TRT (Echo Dia segmentation, SE-DiCoW encoder
+ decoder) requires no API change.
