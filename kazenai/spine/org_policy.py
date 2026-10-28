"""What an organization chose for calls made while Agent FinOps cannot be reached.

An org admin sets it in Agent FinOps: ``fail_closed`` (the default) refuses the call,
``fail_open`` lets it run with only the local client caps. The SDK learns the choice
from the capabilities document it already fetches before each reserve, keeps the
last value it saw in memory and on disk (beside the settlement queue, mode 0600),
and reads it only when Agent FinOps does not answer. A process that has never seen
the setting fails closed.

The setting is kept per FinOps URL, declared org and API key (by fingerprint), so
a process only ever acts on the setting Agent FinOps sent for its own key, even
when two apps on one machine share the settlement queue folder.

Fail-open never overrides an answer: a budget denial, a pause, a tier breaker or a
rejected credential still refuses the call.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Mapping, Optional

FAIL_OPEN = "fail_open"
FAIL_CLOSED = "fail_closed"
_POLICIES = (FAIL_OPEN, FAIL_CLOSED)

_log = logging.getLogger("kazenai.spine")
_lock = threading.Lock()
# (FinOps URL, declared org, API key fingerprint) -> (policy, org id Agent FinOps reported)
_memory: dict[tuple[str, str, str], tuple[str, str]] = {}
_warned: dict[str, tuple[float, int]] = {}
_WARN_EVERY_S = 60.0


def _key(base: str, org_id: str, api_key: str) -> tuple[str, str, str]:
    secret = str(api_key or "").strip()
    fingerprint = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:32] if secret else ""
    return (str(base or "").rstrip("/"), str(org_id or "").strip(), fingerprint)


def _usable(key: tuple[str, str, str]) -> bool:
    return bool(key[0] and key[1])


def _path(key: tuple[str, str, str]) -> Optional[str]:
    try:
        from .settle_queue import _default_root

        root = _default_root()
    except Exception:
        return None
    digest = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:40]
    return os.path.join(root, "org-policy", f"{digest}.json")


def policy_from_capabilities(capabilities: Mapping[str, Any]) -> Optional[str]:
    block = capabilities.get("org_policy") if isinstance(capabilities, Mapping) else None
    value = block.get("when_unreachable") if isinstance(block, Mapping) else None
    return value if value in _POLICIES else None


def _reported_org(capabilities: Mapping[str, Any]) -> str:
    block = capabilities.get("org_policy") if isinstance(capabilities, Mapping) else None
    value = block.get("org_id") if isinstance(block, Mapping) else None
    return str(value or "").strip()[:200]


def remember(base: str, org_id: str, capabilities: Mapping[str, Any], *, api_key: str) -> None:
    """Keep the org's setting from a capabilities document. Older servers send none."""
    policy = policy_from_capabilities(capabilities)
    key = _key(base, org_id, api_key)
    if policy is None or not _usable(key):
        return
    entry = (policy, _reported_org(capabilities) or key[1])
    with _lock:
        if _memory.get(key) == entry:
            return
        _memory[key] = entry
    path = _path(key)
    if path is None:
        return
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"when_unreachable": policy, "org_id": entry[1], "seen_at": time.time()}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        _log.debug("could not save the organization's unreachable policy", exc_info=True)
        with contextlib.suppress(Exception):
            os.remove(tmp)


def _lookup(key: tuple[str, str, str]) -> Optional[tuple[str, str]]:
    if not _usable(key):
        return None
    with _lock:
        entry = _memory.get(key)
    if entry is not None:
        return entry
    path = _path(key)
    try:
        with open(path or "", encoding="utf-8") as fh:
            stored = json.load(fh)
        policy = stored.get("when_unreachable")
        reported = str(stored.get("org_id") or "").strip()[:200] or key[1]
    except Exception:
        return None
    if policy not in _POLICIES:
        return None
    with _lock:
        return _memory.setdefault(key, (policy, reported))


def allows_fail_open(base: str, org_id: str, *, api_key: str) -> bool:
    """True only when the last setting seen for this FinOps URL, org and API key was fail_open."""
    entry = _lookup(_key(base, org_id, api_key))
    return entry is not None and entry[0] == FAIL_OPEN


def reported_org_id(base: str, org_id: str, *, api_key: str) -> str:
    """The org id Agent FinOps reported with the setting, for log lines."""
    entry = _lookup(_key(base, org_id, api_key))
    return entry[1] if entry is not None else str(org_id or "").strip()


def note_fail_open_call(org_id: str, reason: str) -> None:
    """One WARNING per org per minute, with the number of calls let through since the last one."""
    now = time.time()
    with _lock:
        last, count = _warned.get(org_id, (0.0, 0))
        count += 1
        if now - last < _WARN_EVERY_S and last:
            _warned[org_id] = (last, count)
            return
        _warned[org_id] = (now, 0)
    _log.warning(
        "kazenai: Agent FinOps unreachable (%s); %d call(s) for org %s ran under the organization's "
        "fail-open setting with local caps only and are not charged to its Agent FinOps limits",
        reason,
        count,
        org_id,
    )


def reset_for_tests() -> None:
    with _lock:
        _memory.clear()
        _warned.clear()
