"""Single enforcement-owner context (FINAL_1_b P3-2).

Prevents double authorize/settle when both a gateway and the SDK could intercept.
Ownership is declared via ``KAZENAI_ENFORCEMENT_OWNER`` and/or a ContextVar claim.
Ambiguous ownership raises :class:`EnforcementOwnershipError` (configuration error).

Propagates ``decision_id`` / ``reservation_id`` through trusted context so nested
same-owner scopes reuse one reservation instead of double-charging.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from dataclasses import dataclass, replace
from typing import Iterator, Optional

_OWNER: contextvars.ContextVar[Optional["EnforcementOwnerState"]] = contextvars.ContextVar(
    "kazenai_enforcement_owner",
    default=None,
)


class EnforcementOwnershipError(RuntimeError):
    """Ambiguous or conflicting economic enforcement ownership."""

    def __init__(self, message: str, *, code: str = "ambiguous_enforcement_owner") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EnforcementOwnerState:
    owner: str
    decision_id: Optional[str] = None
    reservation_id: Optional[str] = None
    call_id: Optional[str] = None
    attempt: Optional[int] = None


def configured_enforcement_owner() -> Optional[str]:
    raw = (os.getenv("KAZENAI_ENFORCEMENT_OWNER") or "").strip().lower()
    return raw or None


def get_enforcement_owner() -> Optional[EnforcementOwnerState]:
    return _OWNER.get()


def _normalize_owner(owner: str) -> str:
    text = (owner or "").strip().lower()
    if not text:
        raise EnforcementOwnershipError("enforcement owner must be non-empty", code="invalid_enforcement_owner")
    return text


@contextlib.contextmanager
def claim_enforcement_owner(
    owner: str = "sdk",
    *,
    decision_id: Optional[str] = None,
    reservation_id: Optional[str] = None,
    call_id: Optional[str] = None,
    attempt: Optional[int] = None,
) -> Iterator[EnforcementOwnerState]:
    """Claim (or nest under) a single enforcement owner for this ContextVar scope."""
    want = _normalize_owner(owner)
    env = configured_enforcement_owner()
    if env and env != want:
        raise EnforcementOwnershipError(
            f"ambiguous enforcement ownership: KAZENAI_ENFORCEMENT_OWNER={env!r} "
            f"conflicts with claim {want!r}",
            code="ambiguous_enforcement_owner",
        )
    current = _OWNER.get()
    if current is not None and current.owner != want:
        raise EnforcementOwnershipError(
            f"ambiguous enforcement ownership: active={current.owner!r} conflicts with {want!r}",
            code="ambiguous_enforcement_owner",
        )
    if current is not None and current.owner == want:
        merged = EnforcementOwnerState(
            owner=want,
            decision_id=decision_id or current.decision_id,
            reservation_id=reservation_id or current.reservation_id,
            call_id=call_id or current.call_id,
            attempt=attempt if attempt is not None else current.attempt,
        )
        token = _OWNER.set(merged)
        try:
            yield merged
        finally:
            _OWNER.reset(token)
        return

    state = EnforcementOwnerState(
        owner=want,
        decision_id=decision_id,
        reservation_id=reservation_id,
        call_id=call_id,
        attempt=attempt,
    )
    token = _OWNER.set(state)
    try:
        yield state
    finally:
        _OWNER.reset(token)


def bind_decision_refs(
    *,
    decision_id: Optional[str] = None,
    reservation_id: Optional[str] = None,
    call_id: Optional[str] = None,
    attempt: Optional[int] = None,
) -> Optional[EnforcementOwnerState]:
    """Propagate authorize outcome into trusted owner context (mutable ContextVar set)."""
    current = _OWNER.get()
    if current is None:
        return None
    updated = replace(
        current,
        decision_id=decision_id if decision_id is not None else current.decision_id,
        reservation_id=reservation_id if reservation_id is not None else current.reservation_id,
        call_id=call_id if call_id is not None else current.call_id,
        attempt=attempt if attempt is not None else current.attempt,
    )
    _OWNER.set(updated)
    return updated


def should_skip_sdk_reserve() -> bool:
    """True when an outer owner already authorized this attempt (avoid double-charge)."""
    current = _OWNER.get()
    return bool(current and current.reservation_id)


def assert_sdk_may_reserve() -> None:
    """Raise when SDK reserve would create ambiguous double enforcement."""
    if should_skip_sdk_reserve():
        return
    env = configured_enforcement_owner()
    current = _OWNER.get()
    if env and env != "sdk":
        raise EnforcementOwnershipError(
            f"KAZENAI_ENFORCEMENT_OWNER={env!r}: SDK must not independently reserve "
            "(propagate reservation_id through trusted context or unset the env)",
            code="ambiguous_enforcement_owner",
        )
    if current is not None and current.owner != "sdk":
        raise EnforcementOwnershipError(
            f"active enforcement owner {current.owner!r} without reservation_id; "
            "SDK reserve would double-charge or leave ownership ambiguous",
            code="ambiguous_enforcement_owner",
        )
