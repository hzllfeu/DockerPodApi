# Security model

The pod is an inference worker. It runs in a Koyeb GPU container, called
by a Rust API over the internal Koyeb network. It is **not** intended
to be exposed directly to the public internet.

## Threat model

We defend against these attackers:

1. **Compromised client** — sends arbitrary `audio_url` to extract data
   from the pod's network position.
2. **Compromised sibling pod** — tries to call the inference endpoints
   directly to use compute without going through the Rust gateway.
3. **Webhook receiver impersonation** — third party tries to forge a
   webhook payload to the Rust API.

We do **not** defend against:

- Koyeb hypervisor compromise.
- A compromised Rust API (it has the internal secret).
- Quota exhaustion via legitimate calls (handled by Koyeb autoscaling
  + Rust API rate limiting).

## Layered defenses

### 1. Shared-secret auth

Every endpoint requires `X-Internal-Secret: <ECHO_INTERNAL_SECRET>`.
Boot validation enforces ≥ 24 chars (`config.py:validate_boot_config`).
The Rust API holds the same secret in its Koyeb secret store and adds
the header on every inference call.

If a secret leaks, rotate it on both sides simultaneously.

### 2. SSRF protection

The pod downloads the user-provided `audio_url` itself. Without
protection, an attacker could exfiltrate cloud metadata
(`http://169.254.169.254`), probe internal services, or reach loopback.

Defenses in `app/download.py`:

- **`https://` only.** http, file, ftp, data, gopher → 400.
- **DNS allowlist check.** All resolved IPs for the hostname must be
  public. Private (10.0.0.0/8, 172.16/12, 192.168/16, 169.254/16),
  loopback, link-local, multicast, reserved → 403.
- **All resolved IPs.** If `getaddrinfo()` returns multiple records and
  any one is private, refuse the whole request (DNS rebinding defense).
- **Manual redirect handling.** `follow_redirects=False`. We re-validate
  scheme + DNS on every `Location` hop (up to 5).
- **Strict `Content-Type`.** Only `audio/wav` and `audio/x-wav` accepted
  at the HEAD step (415 otherwise).
- **`Content-Length` cap.** Header check up-front + streaming hard cap
  during GET (413).
- **WAV magic + fmt parse.** First 64 KB of the file is parsed: must
  begin with `RIFF....WAVE`, must contain `fmt ` chunk with
  `format=1, channels=1, sample_rate=16000, bits_per_sample=16`.

### 3. Webhook signature

If `ECHO_WEBHOOK_HMAC_SECRET` is set, every webhook body is signed:

    X-Echo-Signature: sha256=<hmac_sha256_hex(secret, body)>

The Rust API verifies the signature before trusting the payload.
Without the env var the signature header is omitted; the receiver
should reject unsigned webhooks in production.

### 4. Resource limits

- `ECHO_SYNC_MAX_SECONDS` (300 s): hard cap on sync audio length.
- `ECHO_ASYNC_MAX_SECONDS` (3600 s): hard cap on async audio length.
- `ECHO_SYNC_TIMEOUT_S` (90 s): server-side timeout on /infer.
- VRAM check before each inference: refuse 503 if used > 90 % of total.

These prevent a malicious client from DoS-ing the pod by sending 24 h
of audio.

### 5. Output sanitization

The transcript is whatever the model produced. It is **not** HTML-
escaped or otherwise sanitized. The Rust API and downstream consumers
must treat the transcript text as untrusted user input.

The `request_id` field is echoed verbatim in headers and logs. The
Rust API should generate it server-side or validate that it matches
`^[A-Za-z0-9_-]{1,64}$`.

## Disabling SSRF for local dev

The env var `ECHO_INSECURE_LOCAL_DOWNLOAD=1` accepts `http://` URLs
and refuses to check resolved IPs. **This must never be set in
production.** It exists only so we can serve test audio from
`http://127.0.0.1:8002/` while developing.

## Secret rotation

| secret | storage | rotation |
|---|---|---|
| `ECHO_INTERNAL_SECRET` | Koyeb secret on both pod and Rust API | rotate both at once during a maintenance window. Hot rotation needs a brief overlap window where the pod accepts old + new. |
| `ECHO_WEBHOOK_HMAC_SECRET` | Koyeb secret on pod, application config on Rust API | hot rotate: deploy Rust API with new secret first, then deploy pod with new secret; the Rust API can verify both during overlap. |
| HF download tokens | only used at build time, never present at runtime | refresh on every release |

## Audit trail

Every request is logged as one JSON line on stdout with fields
`ts, lvl, name, msg`. Koyeb captures stdout/stderr by default. For
long-term retention, ship logs from the Rust API instead (which sees
both inbound and outbound).
