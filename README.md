# DockerPodApi

Production-grade Docker images that package speech AI models behind a uniform
HTTP API. Designed to be deployed on Koyeb GPU pods (L4 / L40S) and called
by a Rust API gateway over the internal Koyeb network.

## Layout

    DockerPodApi/
    ├── README.md
    ├── shared/                       # cross-model docs and conventions
    │   ├── API_CONTRACT.md           # exact request/response schema
    │   ├── SECURITY.md               # SSRF, auth, secrets
    │   └── DEPLOY.md                 # Koyeb deployment notes
    └── models/
        └── echo-l4/                  # first model: Echo (DiariZen V4 + SE-DiCoW)
            ├── README.md
            ├── requirements.txt
            ├── app/
            │   ├── server.py         # FastAPI: /infer, /infer/async, /healthz, /readyz, /metrics
            │   ├── config.py
            │   ├── download.py       # SSRF-safe https-only WAV downloader
            │   ├── pipeline.py       # model-specific inference wrapper
            │   ├── trt_chain.py      # backend selector (PyTorch today)
            │   └── jobs.py           # async job queue + signed webhooks
            └── bazel/                # rules_oci build (build image without docker daemon)

## Uniform API across models

Every model image exposes the same contract:

| endpoint | method | purpose |
|---|---|---|
| `/infer` | POST | sync inference (≤ 5 min audio). Returns JSON or SSE per `Accept` |
| `/infer/async` | POST | async inference (≤ 60 min audio). Returns 202 + `job_id`, webhook on completion |
| `/jobs/{id}` | GET | polling fallback for async |
| `/healthz` | GET | liveness probe |
| `/readyz` | GET | readiness probe (model loaded) |
| `/metrics` | GET | Prometheus counters |

Auth: `X-Internal-Secret` header on every endpoint except health/metrics.

Full request/response schema in [shared/API_CONTRACT.md](shared/API_CONTRACT.md).

## Switching / adding a new model

A model image is fully described by **one directory** under `models/`. To
add a new model, copy `models/echo-l4/` and adapt three files:

### 1. `models/<your-model>/app/pipeline.py`

Implement an `EchoPyTorchBackend`-shaped class with two methods:

    class MyBackend:
        def load(self) -> None:
            # Load weights, tokenizer, etc. Called once at boot.
            ...

        def infer(self, audio_path: Path) -> dict[str, Any]:
            # Take a local WAV path, return the canonical schema:
            return {
                "transcript": "...",
                "speakers": [
                    {"id": "S0", "segments": [{"start": 0.0, "end": 5.2, "text": "..."}]},
                    ...
                ],
                "rttm": "SPEAKER ... 1 0.000 5.200 <NA> <NA> S0 <NA> <NA>\n...",
                "meta": {
                    "n_speakers": <int>,
                    "inference_seconds": <float>,
                    "backend": "pytorch",
                },
            }

        def infer_streaming(self, audio_path: Path):
            # Yield events in time order:
            # {"event": "segment", "speaker_id": ..., "start": ..., "end": ..., "text": ...}
            # End with: {"event": "done", "meta": {...}}
            ...

The streaming variant can be a simple wrapper around `infer()` that yields
pre-sorted segments (this is what `echo-l4` does today since DiCoW does not
expose a true streaming runtime).

### 2. `models/<your-model>/app/trt_chain.py`

Wire your new backend class into `PyTorchBackend.try_load()`. If you want
to add ONNX or TensorRT variants, declare them as siblings of
`PyTorchBackend` and add them to the `Pipeline._backend` selection. Today
only PyTorch is implemented.

### 3. `models/<your-model>/requirements.txt`

List the pip dependencies needed at runtime. Pin versions. The build
process pre-installs these into a portable `site-packages` baked into the
image so the container does not run `pip install` at boot.

### 4. `models/<your-model>/bazel/BUILD.bazel`

Adjust the OCI image target if your model needs extra layers (e.g., a
vendored upstream repo like DiCoW). The default template covers
"FastAPI + Python deps + model weights + app".

### 5. Test natively before building the image

From inside `models/<your-model>/`:

    ECHO_INTERNAL_SECRET=dev-test-secret-at-least-24-chars-long \
    ECHO_INSECURE_LOCAL_DOWNLOAD=1 \
    PYTHONPATH=. \
      python3 -m uvicorn app.server:app --host 127.0.0.1 --port 8000

Curl smoke tests in `shared/SMOKE_TESTS.md`.

### 6. Build the image

On a Koyeb GPU pod (where Bazel rules_oci works without a docker
daemon), as a non-root user (rules_python refuses root):

    cd models/<your-model>/bazel
    bazel build //:image

Output: an OCI tarball under `bazel-bin/image/`.

### 7. Push to a registry

    bazel run //:push

Token / credentials configured in `shared/DEPLOY.md`.

## Why not just one Dockerfile per model

We considered classic Dockerfiles. On Koyeb pods the host has neither
`dockerd` nor user namespaces, so `docker build`, `buildah`, `kaniko`,
`buildctl` all fail. Bazel `rules_oci` is the only path that produces a
valid OCI image without those primitives, because it manipulates tar
files and manifests rather than chrooting.

## Conventions

- **Models are baked into the image** (no HF download at runtime). The
  image runs with `HF_HUB_OFFLINE=1`.
- **Audio is WAV 16 kHz mono 16-bit PCM, period.** The client is expected
  to do format conversion before posting. Pods refuse 415 otherwise.
- **The pod downloads the audio itself** from the URL the caller passed,
  with strict SSRF defenses (https only, IP allowlist).
- **No outbound HTTP at runtime except** the audio URL and the webhook
  callback. No HF, no model registry, no tracing endpoint.
- **TRT/ONNX are NOT used for echo-l4.** The chain is in place for future
  models that ship pre-exported engines, but echo-l4 stays on PyTorch.

## License

Each model directory inherits the license of its upstream weights
(typically CC BY-NC 4.0 for the BUT-FIT family). The wrapper code here is
MIT unless stated otherwise.
