# echo-dia — Diarization-only pod (DiariZen V4)

Pure speaker diarization. No ASR — returns RTTM + structured segments only.

## Architecture

- **Diarization**: Echo Dia V4 = DiariZen V2 base
  (`BUT-FIT/diarizen-wavlm-large-s80-md-v2`) + Echo Dia fine-tuned weights
  (`Hellfeu/echo-dia`).

## Image

    docker.io/lexiapro/echo-dia:latest

Same FastAPI server as `echo-l4`. `/infer` and `/infer/async` return a payload
with `transcript=""` (empty) and `speakers[*].segments[*].text=""` — only the
`rttm` field and segment start/end timestamps are populated.

## Response shape

```json
{
  "transcript": "",
  "speakers": [
    {"id": "0", "segments": [{"start": 0.0, "end": 2.3, "text": ""}]}
  ],
  "rttm": "SPEAKER audio 1 0.000 2.300 <NA> <NA> 0 <NA> <NA>\n...",
  "meta": {
    "n_speakers": 2,
    "inference_seconds": 1.2,
    "backend": "echo-dia",
    "note": "diarization-only, no ASR"
  }
}
```
