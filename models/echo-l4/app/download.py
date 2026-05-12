"""SSRF-safe audio download.

Threat model: an attacker provides `audio_url`. We must:
  1. Refuse non-https schemes.
  2. Refuse URLs whose resolved IP is in private / link-local / loopback ranges.
     This blocks AWS metadata (169.254.169.254), localhost, internal services.
  3. Stop downloads that exceed MAX_AUDIO_BYTES.
  4. Validate the response is `audio/wav`.
  5. Validate magic bytes + WAV fmt chunk (16 kHz mono 16-bit PCM).
  6. Re-validate after every redirect (do not blindly trust follow_redirects).

Returns a Path to a local temp file containing the validated WAV.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
import struct
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.config import (
    DOWNLOAD_CONNECT_TIMEOUT,
    DOWNLOAD_GET_TIMEOUT,
    DOWNLOAD_HEAD_TIMEOUT,
    DOWNLOAD_TMP_DIR,
    MAX_AUDIO_BYTES,
    SR,
)

# Dev only: allow http:// and private IPs. NEVER set this in production.
_INSECURE_LOCAL = os.environ.get("ECHO_INSECURE_LOCAL_DOWNLOAD", "0") == "1"

log = logging.getLogger(__name__)


class DownloadError(Exception):
    """4xx-style: the request was bad (client's fault)."""
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status
        self.msg = msg


@dataclass
class WavInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    n_samples: int  # per channel
    duration_s: float


# ---------- SSRF guard ----------

def _is_unsafe_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_safe(hostname: str) -> str:
    """Resolve hostname and refuse if ANY resolved IP is unsafe. Returns the
    first safe IP. We refuse on any unsafe IP found (DNS rebinding defense)."""
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise DownloadError(400, f"DNS resolution failed for {hostname}: {e}")
    ips = {info[4][0] for info in infos}
    if not ips:
        raise DownloadError(400, f"no IP for {hostname}")
    for ip in ips:
        if _is_unsafe_ip(ip):
            raise DownloadError(
                403, f"refused: hostname {hostname} resolves to unsafe IP {ip}"
            )
    return next(iter(ips))


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http" if _INSECURE_LOCAL else "https"):
        raise DownloadError(400, f"only https URLs accepted, got {parsed.scheme}")
    if not parsed.hostname:
        raise DownloadError(400, "missing hostname")
    if not _INSECURE_LOCAL:
        _resolve_safe(parsed.hostname)
    return url


# ---------- WAV parsing ----------

def _parse_wav_header(data: bytes) -> WavInfo:
    """Minimal RIFF/WAVE parser. Validates 16 kHz mono 16-bit PCM."""
    if len(data) < 44:
        raise DownloadError(400, "file too small to be a WAV")
    if data[0:4] != b"RIFF":
        raise DownloadError(400, "not a WAV file (RIFF magic missing)")
    if data[8:12] != b"WAVE":
        raise DownloadError(400, "not a WAV file (WAVE magic missing)")
    # Walk chunks looking for fmt
    pos = 12
    fmt_chunk = None
    data_chunk_size = None
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if chunk_id == b"fmt ":
            fmt_chunk = data[pos + 8:pos + 8 + chunk_size]
        elif chunk_id == b"data":
            data_chunk_size = chunk_size
            break
        pos += 8 + chunk_size
        if chunk_size % 2 == 1:  # pad byte
            pos += 1
    if fmt_chunk is None:
        raise DownloadError(400, "WAV missing fmt chunk")
    if data_chunk_size is None:
        raise DownloadError(400, "WAV missing data chunk")
    if len(fmt_chunk) < 16:
        raise DownloadError(400, "WAV fmt chunk too short")
    audio_format, channels, sr, byte_rate, block_align, bps = struct.unpack(
        "<HHIIHH", fmt_chunk[:16]
    )
    if audio_format != 1:  # PCM
        raise DownloadError(415, f"WAV must be PCM (format=1), got format={audio_format}")
    if sr != SR:
        raise DownloadError(415, f"WAV must be {SR} Hz, got {sr}")
    if channels != 1:
        raise DownloadError(415, f"WAV must be mono, got {channels} channels")
    if bps != 16:
        raise DownloadError(415, f"WAV must be 16-bit, got {bps}-bit")
    n_samples = data_chunk_size // 2
    return WavInfo(
        sample_rate=sr,
        channels=channels,
        bits_per_sample=bps,
        n_samples=n_samples,
        duration_s=n_samples / sr,
    )


# ---------- Public API ----------

def safe_download(url: str, max_seconds: int, request_id: str) -> tuple[Path, WavInfo]:
    """Download `url` safely. Returns (local_path, wav_info)."""
    _validate_url(url)

    timeouts = httpx.Timeout(
        connect=DOWNLOAD_CONNECT_TIMEOUT,
        read=DOWNLOAD_HEAD_TIMEOUT,
        write=DOWNLOAD_CONNECT_TIMEOUT,
        pool=DOWNLOAD_CONNECT_TIMEOUT,
    )

    # HEAD first to fail fast on too-large or non-audio content
    with httpx.Client(timeout=timeouts, follow_redirects=False) as client:
        try:
            head = client.head(url)
        except httpx.HTTPError as e:
            raise DownloadError(502, f"HEAD failed: {e}")

        # Manually follow redirects with re-validation
        n_hops = 0
        while head.status_code in (301, 302, 303, 307, 308):
            n_hops += 1
            if n_hops > 5:
                raise DownloadError(400, "too many redirects")
            location = head.headers.get("location")
            if not location:
                raise DownloadError(502, "redirect without Location header")
            url = httpx.URL(url).join(location)
            _validate_url(str(url))
            try:
                head = client.head(str(url))
            except httpx.HTTPError as e:
                raise DownloadError(502, f"HEAD on redirect failed: {e}")

        if head.status_code != 200:
            raise DownloadError(
                502, f"HEAD returned {head.status_code} from {url}"
            )

        ct = head.headers.get("content-type", "").split(";")[0].strip().lower()
        if ct != "audio/wav" and ct != "audio/x-wav":
            raise DownloadError(415, f"Content-Type must be audio/wav, got {ct!r}")

        clen = head.headers.get("content-length")
        if clen is not None:
            clen_i = int(clen)
            max_bytes_strict = max_seconds * SR * 2 + 200_000
            if clen_i > max_bytes_strict:
                raise DownloadError(
                    413,
                    f"Content-Length {clen_i} > limit {max_bytes_strict} "
                    f"({max_seconds}s of 16kHz mono 16-bit WAV)",
                )

    # Stream GET with hard size cap
    timeouts_dl = httpx.Timeout(
        connect=DOWNLOAD_CONNECT_TIMEOUT,
        read=DOWNLOAD_GET_TIMEOUT,
        write=DOWNLOAD_CONNECT_TIMEOUT,
        pool=DOWNLOAD_CONNECT_TIMEOUT,
    )
    out_path = DOWNLOAD_TMP_DIR / f"{request_id}_{uuid.uuid4().hex}.wav"
    DOWNLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    max_bytes_strict = max_seconds * SR * 2 + 200_000
    head_buf = bytearray()
    try:
        with httpx.stream(
            "GET", str(url), timeout=timeouts_dl, follow_redirects=False
        ) as r:
            if r.status_code != 200:
                raise DownloadError(502, f"GET returned {r.status_code}")
            with open(out_path, "wb") as fout:
                for chunk in r.iter_bytes(chunk_size=64 * 1024):
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES or total > max_bytes_strict:
                        raise DownloadError(
                            413,
                            f"download exceeded {max_bytes_strict} bytes",
                        )
                    fout.write(chunk)
                    if len(head_buf) < 256:
                        head_buf.extend(chunk[: 256 - len(head_buf)])
    except httpx.HTTPError as e:
        out_path.unlink(missing_ok=True)
        raise DownloadError(502, f"GET failed: {e}")
    except DownloadError:
        out_path.unlink(missing_ok=True)
        raise

    if total < 44:
        out_path.unlink(missing_ok=True)
        raise DownloadError(400, "downloaded payload too small to be a WAV")

    # We need the full header AND the fmt chunk. The first 64 KB is plenty.
    with open(out_path, "rb") as f:
        first = f.read(64 * 1024)
    try:
        info = _parse_wav_header(first)
    except DownloadError:
        out_path.unlink(missing_ok=True)
        raise

    if info.duration_s > max_seconds + 0.5:
        out_path.unlink(missing_ok=True)
        raise DownloadError(
            413,
            f"audio duration {info.duration_s:.1f}s > limit {max_seconds}s",
        )

    log.info(
        "downloaded %s bytes (%.1fs audio) from %s -> %s",
        total, info.duration_s, url, out_path,
    )
    return out_path, info
