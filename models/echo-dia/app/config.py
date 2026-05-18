"""Runtime configuration. All values from env vars with safe defaults."""
from __future__ import annotations

import os
from pathlib import Path


# ----- Auth -----
INTERNAL_SECRET = os.environ.get("ECHO_INTERNAL_SECRET")  # required at boot
WEBHOOK_HMAC_SECRET = os.environ.get("ECHO_WEBHOOK_HMAC_SECRET")  # optional

# ----- Paths -----
JOBS_CACHE_DIR = Path(os.environ.get("ECHO_JOBS_CACHE_DIR", "/cache/jobs"))
DOWNLOAD_TMP_DIR = Path(os.environ.get("ECHO_DOWNLOAD_TMP_DIR", "/tmp/echo_dl"))

# ----- Audio limits -----
SR = 16000
SYNC_MAX_SECONDS = int(os.environ.get("ECHO_SYNC_MAX_SECONDS", "300"))     # 5 min
ASYNC_MAX_SECONDS = int(os.environ.get("ECHO_ASYNC_MAX_SECONDS", "3600"))  # 60 min
# Hard byte cap = ASYNC_MAX * SR * 2 bytes/sample + 200 KB header headroom
MAX_AUDIO_BYTES = ASYNC_MAX_SECONDS * SR * 2 + 200_000

# ----- HTTP -----
DOWNLOAD_HEAD_TIMEOUT = 5.0    # seconds for HEAD
DOWNLOAD_GET_TIMEOUT = 60.0    # total seconds for GET (depends on audio size)
DOWNLOAD_CONNECT_TIMEOUT = 5.0
SERVER_SYNC_TIMEOUT_S = int(os.environ.get("ECHO_SYNC_TIMEOUT_S", "90"))

# ----- Webhook -----
WEBHOOK_MAX_RETRIES = 3
WEBHOOK_TIMEOUT = 30.0
WEBHOOK_BACKOFF_SECONDS = [2.0, 5.0, 15.0]  # exponential-ish

# ----- VRAM safety -----
# Refuse new inference if used/total > this ratio. Set to >=1.0 to disable
# the check entirely. The default is intentionally loose (0.98) because
# PyTorch's mem_get_info reports the GPU free pool, which can stay low
# between inferences when activations are cached.
VRAM_REFUSE_RATIO = float(os.environ.get("ECHO_VRAM_REFUSE_RATIO", "0.98"))

# ----- Model + backend -----
MODEL_NAME = "Hellfeu/echo"
HF_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0") == "1"


def validate_boot_config() -> None:
    """Raise at boot if config is incoherent. Logged then process exits."""
    if not INTERNAL_SECRET:
        raise RuntimeError(
            "ECHO_INTERNAL_SECRET env var is required (shared secret with the "
            "calling Rust API)"
        )
    if len(INTERNAL_SECRET) < 24:
        raise RuntimeError("ECHO_INTERNAL_SECRET must be >= 24 chars")
    for d in (JOBS_CACHE_DIR, DOWNLOAD_TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
