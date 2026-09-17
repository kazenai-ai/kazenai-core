"""FINAL_1 P3-5 — ingest URL normalization for Control FinOps/Lens endpoints."""

from __future__ import annotations

from typing import Tuple
from urllib.parse import urlparse, urlunparse


def normalize_ingest_base_url(url: str) -> str:
    """Return ingest *base* URL without a trailing ``/v1/events`` (or ``/events``).

    Producers and HttpSink always append ``/v1/events``. Passing an env value that
    already ends with that path must not produce ``.../v1/events/v1/events``.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    # Strip trailing slashes for comparison.
    trimmed = raw.rstrip("/")
    lower = trimmed.lower()
    for suffix in ("/v1/events", "/events"):
        if lower.endswith(suffix):
            trimmed = trimmed[: -len(suffix)].rstrip("/")
            break
    return trimmed


def ingest_post_url(base_or_full: str) -> str:
    """Canonical POST URL for event ingest."""
    base = normalize_ingest_base_url(base_or_full)
    if not base:
        return ""
    return f"{base}/v1/events"


def split_ingest_url(url: str) -> Tuple[str, str]:
    """Return ``(base, post_url)`` for HttpSinkConfig.ingest_url + post helper."""
    base = normalize_ingest_base_url(url)
    return base, ingest_post_url(base) if base else ("", "")


def is_loopback_or_private_host(url: str) -> bool:
    """Best-effort host classifier for docs/tests (not a substitute for SSRF guard)."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    if host.startswith("10.") or host.startswith("192.168.") or host.startswith("172."):
        return True
    return False
