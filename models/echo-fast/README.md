# echo-fast — Echo inference pod (pyannote-community-1 + DiCoW_v3_3)

~2x faster than `echo-light`. Trade-off: lower diarization quality on meetings.

## Architecture

- **Diarization**: `pyannote/speaker-diarization-community-1`, which requires
  `pyannote.audio>=4.0`.
- **ASR**: `BUT-FIT/DiCoW_v3_3`.

### Why two venvs

`pyannote.audio>=4.0` is incompatible with DiCoW (which vendors pyannote.audio
3.1.1). The image ships **two Python environments**:

- `/opt/conda/...` + `/opt/python-runtime/site-packages` — main venv with DiCoW,
  DiariZen, transformers, etc. Runs the FastAPI server + ASR.
- `/opt/pyannote_venv` — isolated venv with pyannote.audio 4.x. Invoked as a
  subprocess to produce an RTTM, then results are loaded back as a pyannote
  Annotation and passed to DiCoW.

Env var override: `ECHO_PYANNOTE_PYTHON` (default `/opt/pyannote_venv/bin/python`).

## Image

    docker.io/lexiapro/echo-fast:latest

Same FastAPI server, request/response contract and operational characteristics
as `echo-l4`. See `models/echo-l4/README.md` for the full API contract.

## Meta fields

The response `meta` object adds `diarization_seconds` and `asr_seconds`
separately (in addition to the combined `inference_seconds`).
