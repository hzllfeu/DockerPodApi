# Smoke tests

These are the curls we run after every build to verify the pod works
end-to-end. Run them against `http://127.0.0.1:8000` when testing
locally, or against the internal Koyeb hostname in staging.

Set:

    export S=dev-test-secret-at-least-24-chars-long
    export URL=http://127.0.0.1:8000

## 1. Health endpoints (no auth)

    curl -s $URL/healthz
    # → {"status":"alive","uptime_seconds":42.1}

    curl -s $URL/readyz | jq
    # → {"status":"ready","active_backend":"pytorch","backends":[...]}

    curl -s $URL/metrics | head -5
    # → Prometheus plain text

## 2. Auth

    # no secret → 401
    curl -sw '\n[%{http_code}]\n' -X POST $URL/infer \
      -H 'Content-Type: application/json' \
      -d '{"audio_url":"https://x.example.com/a.wav"}'

    # wrong secret → 401
    curl -sw '\n[%{http_code}]\n' -X POST $URL/infer \
      -H 'X-Internal-Secret: bad' -H 'Content-Type: application/json' \
      -d '{"audio_url":"https://x.example.com/a.wav"}'

## 3. SSRF defenses

    # http:// → 400 (only https accepted)
    curl -sw '\n[%{http_code}]\n' -X POST $URL/infer \
      -H "X-Internal-Secret: $S" -H 'Content-Type: application/json' \
      -d '{"audio_url":"http://example.com/a.wav"}'

    # nip.io trick → AWS metadata IP → 403
    curl -sw '\n[%{http_code}]\n' -X POST $URL/infer \
      -H "X-Internal-Secret: $S" -H 'Content-Type: application/json' \
      -d '{"audio_url":"https://169-254-169-254.nip.io/a.wav"}'

    # nip.io → loopback → 403
    curl -sw '\n[%{http_code}]\n' -X POST $URL/infer \
      -H "X-Internal-Secret: $S" -H 'Content-Type: application/json' \
      -d '{"audio_url":"https://127-0-0-1.nip.io/a.wav"}'

## 4. Real inference (sync)

Needs a publicly reachable WAV. For local testing, serve a file:

    python3 -m http.server 8002   # in /data with a test_30s.wav

Set `ECHO_INSECURE_LOCAL_DOWNLOAD=1` on the server to allow http:// in dev.

    curl -s -X POST $URL/infer \
      -H "X-Internal-Secret: $S" -H 'Content-Type: application/json' \
      --max-time 120 \
      -d '{"audio_url":"http://127.0.0.1:8002/test_30s.wav","request_id":"smoke1"}' \
      | jq '.meta, .speakers | length'

Expected: meta block with timing, count of speakers > 0.

## 5. Streaming (SSE)

    curl -s --max-time 120 -X POST $URL/infer \
      -H "X-Internal-Secret: $S" \
      -H 'Accept: text/event-stream' \
      -H 'Content-Type: application/json' \
      -d '{"audio_url":"http://127.0.0.1:8002/test_30s.wav"}'

Expected output:

    event: start
    data: {...}

    event: segment
    data: {...}
    ...
    event: done
    data: {...}

## 6. Async + polling

    J=$(curl -s -X POST $URL/infer/async \
      -H "X-Internal-Secret: $S" -H 'Content-Type: application/json' \
      -d '{"audio_url":"http://127.0.0.1:8002/test_10min.wav",
           "callback_url":"http://127.0.0.1:8002/none"}')
    ID=$(echo $J | jq -r .job_id)
    echo "submitted $ID"

    # poll
    until curl -fsS -H "X-Internal-Secret: $S" $URL/jobs/$ID | jq -e '.status | IN("done","failed")' > /dev/null; do
      sleep 5
    done
    curl -s -H "X-Internal-Secret: $S" $URL/jobs/$ID | jq '.status, .result.meta'

## 7. Webhook (production)

If `callback_url` is reachable, the pod POSTs the same body to it on
completion. Verify the receiver gets:

    X-Echo-Job-Id: <id>
    X-Echo-Event: job.completed
    X-Echo-Signature: sha256=<hex>   (if HMAC enabled)
