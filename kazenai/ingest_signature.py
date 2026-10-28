"""Raw-body ingest HMAC. One implementation for Core, FinOps, and Lens.

The helper accepts already-serialized bytes. It does not parse or reserialize JSON.

Canonical string::

    METHOD \\n PATH \\n KEY_ID \\n TIMESTAMP \\n NONCE \\n CONTENT_TYPE \\n hex(sha256(raw_body))
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from typing import Mapping

INGEST_CONTENT_TYPE = "application/json"
_NONCE_RE = re.compile(r"^[\x21-\x7e]{1,128}$")


class IngestSignatureError(Exception):
    """Verification failed. The nonce must not be consumed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def body_sha256(raw_body: bytes) -> str:
    return hashlib.sha256(raw_body).hexdigest()


def tenant_key_id(api_key: str) -> str:
    """Non-secret id: ``tenant:`` plus the first 32 hex chars of SHA-256(api key)."""
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]
    return f"tenant:{digest}"


def canonical_message(
    *,
    method: str,
    path: str,
    key_id: str,
    timestamp: str,
    nonce: str,
    content_type: str,
    raw_body: bytes,
) -> bytes:
    if not isinstance(raw_body, (bytes, bytearray)):
        raise IngestSignatureError("raw_body must be bytes")
    line = "\n".join(
        (
            str(method or "").upper(),
            str(path or ""),
            str(key_id or ""),
            str(timestamp or ""),
            str(nonce or ""),
            str(content_type or ""),
            body_sha256(bytes(raw_body)),
        )
    )
    return line.encode("utf-8")


def sign_bytes(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def new_nonce() -> str:
    return secrets.token_hex(16)


def sign_request(
    *,
    secret: str,
    method: str,
    path: str,
    key_id: str,
    raw_body: bytes,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """Headers for a signed ingest. Content-Type is exactly application/json."""
    ts = str(int(time.time()) if timestamp is None else int(timestamp))
    nonce_value = nonce or new_nonce()
    if not _NONCE_RE.match(nonce_value):
        raise IngestSignatureError("invalid nonce")
    message = canonical_message(
        method=method,
        path=path,
        key_id=key_id,
        timestamp=ts,
        nonce=nonce_value,
        content_type=INGEST_CONTENT_TYPE,
        raw_body=raw_body,
    )
    return {
        "Content-Type": INGEST_CONTENT_TYPE,
        "X-Kazen-Key-Id": key_id,
        "X-Kazen-Timestamp": ts,
        "X-Kazen-Nonce": nonce_value,
        "X-Kazen-Signature": sign_bytes(secret, message),
    }


def content_type_values(headers: Mapping[str, str], values: list[str] | None = None) -> str:
    """Return the single supported content type, or raise.

    Exactly one ``Content-Type`` header whose value is exactly
    ``application/json``. Missing, duplicated, parameterised (``; charset=…``) or
    other values are refused, never normalised: the signer signs that exact value.
    """
    presented = list(values or [])
    if not presented:
        raw = headers.get("content-type") or headers.get("Content-Type") or ""
        presented = [raw] if raw else []
    if len(presented) != 1 or str(presented[0]).strip() != INGEST_CONTENT_TYPE:
        raise IngestSignatureError("unsupported content type")
    return INGEST_CONTENT_TYPE


def nonce_ttl_s(max_skew_s: int) -> int:
    """How long a claimed nonce must be remembered.

    A timestamp is accepted up to ``max_skew_s`` either side of now, so an
    envelope stays verifiable for ``2 * max_skew_s``. The nonce must outlive that.
    """
    return 2 * max(10, int(max_skew_s)) + 60


def signed_path(url: str) -> str:
    """The PATH component that is signed for a resolved ingest URL (no query)."""
    from urllib.parse import urlparse

    return urlparse(url).path or "/"


def verify_signature(
    *,
    secret: str,
    method: str,
    path: str,
    key_id: str,
    timestamp: str,
    nonce: str,
    content_type: str,
    raw_body: bytes,
    signature: str,
    now_s: int | None = None,
    max_skew_s: int = 300,
) -> None:
    """Check the envelope. Raises IngestSignatureError and does not touch a nonce store."""
    if not key_id or not timestamp or not nonce or not signature:
        raise IngestSignatureError("missing ingest signature headers")
    if not _NONCE_RE.match(nonce):
        raise IngestSignatureError("invalid ingest nonce")
    if content_type != INGEST_CONTENT_TYPE:
        raise IngestSignatureError("unsupported content type")
    try:
        ts_s = int(timestamp)
        if str(ts_s) != str(timestamp).strip():
            raise ValueError
    except ValueError as exc:
        raise IngestSignatureError("invalid ingest timestamp") from exc
    now = int(time.time()) if now_s is None else int(now_s)
    skew = max(10, int(max_skew_s))
    if abs(now - ts_s) > skew:
        raise IngestSignatureError("stale ingest signature timestamp")
    message = canonical_message(
        method=method,
        path=path,
        key_id=key_id,
        timestamp=str(ts_s),
        nonce=nonce,
        content_type=content_type,
        raw_body=raw_body,
    )
    expected = sign_bytes(secret, message)
    if not hmac.compare_digest(expected, str(signature).strip().lower()):
        raise IngestSignatureError("invalid ingest signature")
