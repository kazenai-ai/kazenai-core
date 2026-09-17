"""FINAL_1 P3-3 — private-by-default content capture policy.

Default is metadata-only: prompt/tool/result bodies are omitted from events,
sinks, logs, and offline queues. Explicit opt-in modes:

* ``metadata`` (default) — cost/identity metadata only; bodies ``kind=omitted``
* ``redacted`` — bodies stored after secret redaction
* ``full`` — requires ``KAZENAI_CAPTURE_CONSENT=1`` (or ``capture_consent=True``);
  still applies secret-pattern redaction (L55: FULL without redaction is illegal)

See ``docs/security/capture-and-retention.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Union

_log = logging.getLogger("kazenai.capture_policy")

# Canary-friendly + common provider secret shapes (aligned with AgentLens store).
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|token|secret|password|aws_access_key_id|aws_secret_access_key)"
)
_VALUE_SECRET_PATTERNS = (
    re.compile(r"sk-[a-zA-Z0-9]{8,}"),
    re.compile(r"kz_[a-zA-Z0-9]{8,}"),
    re.compile(r"kzn_(?:live|test)_[a-zA-Z0-9]{8,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
    re.compile(r"github_pat_[a-zA-Z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----"),
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    # Explicit test canary prefix used by P3-3 verification.
    re.compile(r"CANARY_SECRET_[A-Za-z0-9_\-]{8,}"),
)


class CaptureMode(str, Enum):
    METADATA = "metadata"
    REDACTED = "redacted"
    FULL = "full"


def resolve_capture_mode(
    *,
    explicit: Optional[str] = None,
    consent: Optional[bool] = None,
) -> CaptureMode:
    raw = (explicit if explicit is not None else os.getenv("KAZENAI_CAPTURE_MODE", "metadata")).strip().lower()
    aliases = {
        "metadata": CaptureMode.METADATA,
        "meta": CaptureMode.METADATA,
        "off": CaptureMode.METADATA,
        "disabled": CaptureMode.METADATA,
        "none": CaptureMode.METADATA,
        "redacted": CaptureMode.REDACTED,
        "redact": CaptureMode.REDACTED,
        "feedback_only": CaptureMode.REDACTED,
        "full": CaptureMode.FULL,
        "on": CaptureMode.FULL,
    }
    mode = aliases.get(raw, CaptureMode.METADATA)
    if mode is CaptureMode.FULL:
        consented = (
            bool(consent)
            if consent is not None
            else os.getenv("KAZENAI_CAPTURE_CONSENT", "").strip().lower() in ("1", "true", "yes")
        )
        if not consented:
            _log.warning(
                "capture_mode=full requested without consent; falling back to redacted "
                "(set KAZENAI_CAPTURE_CONSENT=1 or capture_consent=True)"
            )
            return CaptureMode.REDACTED
    return mode


def capture_disclosure(mode: CaptureMode) -> Dict[str, Any]:
    """Runtime disclosure for operators / demo honesty."""
    return {
        "capture_mode": mode.value,
        "bodies": "off" if mode is CaptureMode.METADATA else "on",
        "secret_redaction": "always",
        "retention_doc": "docs/security/capture-and-retention.md",
        "l55_note": "FULL/session-log aspirations require explicit consent + redaction; default is metadata-only",
    }


def _approx_bytes(value: Any) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, (bytes, bytearray)):
            return len(value)
        if isinstance(value, str):
            return len(value.encode("utf-8", errors="replace"))
        return len(json.dumps(value, default=str, ensure_ascii=False).encode("utf-8", errors="replace"))
    except Exception:
        return 0


def content_fingerprint(value: Any) -> Optional[str]:
    """Non-reversible fingerprint for omitted content (replay honesty / dedupe)."""
    try:
        raw = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
    except Exception:
        return None


def redact_string(text: str) -> str:
    redacted = text
    for pattern in _VALUE_SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    if _SECRET_KEY_RE.search(redacted):
        # Avoid leaking key-shaped substrings in free text when pattern is weak.
        redacted = _SECRET_KEY_RE.sub("[REDACTED]", redacted)
    return redacted


def redact_secrets(value: Any, *, _depth: int = 0) -> Any:
    """Recursively redact secret-shaped keys and values (nested dict/list/str)."""
    if _depth > 32:
        return "[REDACTED_DEPTH]"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            k = str(key)
            if _SECRET_KEY_RE.search(k):
                out[k] = "[REDACTED]"
            else:
                out[k] = redact_secrets(item, _depth=_depth + 1)
        return out
    if isinstance(value, list):
        return [redact_secrets(item, _depth=_depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item, _depth=_depth + 1) for item in value)
    if isinstance(value, str):
        return redact_string(value)
    return value


def build_content_ref(
    value: Any,
    *,
    mode: CaptureMode,
    role: str = "content",
) -> Dict[str, Any]:
    """Build inputs_ref / outputs_ref honoring capture mode."""
    if mode is CaptureMode.METADATA:
        return {
            "kind": "omitted",
            "present": False,
            "reason": "capture_mode=metadata",
            "role": role,
            "content_sha256": content_fingerprint(value),
            "approx_bytes": _approx_bytes(value),
        }
    # REDACTED and FULL both store bodies after secret redaction.
    return {
        "kind": "inline",
        "present": True,
        "redacted": True,
        "capture_mode": mode.value,
        "role": role,
        "value": redact_secrets(value),
    }


def inputs_absent(ref: Any) -> bool:
    """Replay honesty: True when bodies were never captured."""
    if not isinstance(ref, dict):
        return True
    if ref.get("kind") == "omitted" or ref.get("present") is False:
        return True
    return False


def sanitize_event_dict(data: Mapping[str, Any], *, mode: Optional[CaptureMode] = None) -> Dict[str, Any]:
    """Defense-in-depth scrub before durable/network/log boundaries."""
    out = dict(data)
    payload = out.get("payload")
    if isinstance(payload, dict):
        payload = dict(payload)
        for key in ("inputs_ref", "outputs_ref"):
            ref = payload.get(key)
            if not isinstance(ref, dict):
                continue
            if mode is CaptureMode.METADATA or ref.get("kind") == "omitted" or ref.get("present") is False:
                payload[key] = {
                    "kind": "omitted",
                    "present": False,
                    "reason": ref.get("reason") or "capture_mode=metadata",
                    "role": ref.get("role"),
                    "content_sha256": ref.get("content_sha256"),
                    "approx_bytes": ref.get("approx_bytes"),
                }
            elif "value" in ref:
                payload[key] = {
                    **ref,
                    "value": redact_secrets(ref.get("value")),
                    "redacted": True,
                }
        # Scrub any other nested payload fields.
        out["payload"] = redact_secrets(payload)
    else:
        out["payload"] = redact_secrets(payload)
    return out


def event_log_fields(event: Any) -> Dict[str, Any]:
    """Safe fields for INFO/debug logging — never include inline bodies."""
    try:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
    except Exception:
        return {"event": "<unserializable>"}
    safe = sanitize_event_dict(data, mode=CaptureMode.METADATA)
    payload = safe.get("payload") if isinstance(safe.get("payload"), dict) else {}
    return {
        "event_id": safe.get("event_id"),
        "event_type": safe.get("event_type"),
        "org_id": safe.get("org_id"),
        "workspace_id": safe.get("workspace_id"),
        "run_id": safe.get("run_id"),
        "step_id": safe.get("step_id"),
        "tokens_used": safe.get("tokens_used"),
        "cost_usd": safe.get("cost_usd"),
        "method": payload.get("method"),
        "model": payload.get("model"),
        "inputs_absent": inputs_absent(payload.get("inputs_ref")),
        "outputs_absent": inputs_absent(payload.get("outputs_ref")),
        "capture": capture_disclosure(CaptureMode.METADATA),
    }


CaptureModeLike = Union[CaptureMode, str, None]
