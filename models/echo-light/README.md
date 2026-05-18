# echo-light — Echo inference pod (DiariZen V4 + DiCoW_v3_3)

Best speed/quality tradeoff for meetings. No self-enrollment.

## Architecture

- **Diarization**: Echo Dia weights (`Hellfeu/echo-dia`) loaded into
  DiariZen V2 base (`BUT-FIT/diarizen-wavlm-large-s80-md-v2`).
- **ASR**: `BUT-FIT/DiCoW_v3_3` (vanilla, no self-enrollment).

Faster than `echo-l4` (which uses SE-DiCoW + self-enrollment) with comparable
quality on most meeting audio.

## Image

    docker.io/lexiapro/echo-light:latest

Same FastAPI server, request/response contract and operational characteristics
as `echo-l4`. See `models/echo-l4/README.md` for the full API contract.
