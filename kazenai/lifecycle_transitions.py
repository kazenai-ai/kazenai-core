"""Pure economic-lifecycle helpers shared by sync and async execution paths.

Transport and iteration stay in spine/streaming; this module owns identity
ordering, settle payload shape, and settlement-status classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .schema import new_id

LIFECYCLE_VERSION = "reservation.lifecycle.v1"

# Reserved → in_flight → settled | pending_reconciliation
# Reserved → cancelled_before_call
KNOWN_RESERVATION_STATES = frozenset(
    {
        "reserved",
        "in_flight",
        "settled",
        "cancelled_before_call",
        "pending_reconciliation",
    }
)

SETTLE_EVENT_SUFFIX = {
    "provider_started": "started",
    "usage_known": "usage",
    "outcome_unknown": "unknown",
    "cancel_before_call": "cancelled",
    "replay_pending": "replay",
}


@dataclass(frozen=True)
class AttemptIdentities:
    """Stable identities for one potentially billable provider execution."""

    operation_id: str
    call_id: str
    attempt: int
    attempt_id: str
    idempotency_key: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "call_id": self.call_id,
            "attempt": self.attempt,
            "attempt_id": self.attempt_id,
            "idempotency_key": self.idempotency_key,
        }


def create_attempt_identities(
    *,
    operation_id: Optional[str] = None,
    call_id: Optional[str] = None,
    attempt: int = 1,
    attempt_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> AttemptIdentities:
    """Create identities before awaiting admission (crash-safe ordering)."""
    op = (operation_id or "").strip() or f"op_{new_id()}"
    cid = (call_id or "").strip() or f"call_{new_id()}"
    att = max(1, int(attempt or 1))
    aid = (attempt_id or "").strip() or f"{cid}:attempt:{att}"
    idem = (idempotency_key or "").strip() or f"reserve:{cid}:{att}:{new_id()}"
    return AttemptIdentities(
        operation_id=op,
        call_id=cid,
        attempt=att,
        attempt_id=aid,
        idempotency_key=idem,
    )


def settle_idempotency_key(reservation_id: str, event: str) -> str:
    suffix = SETTLE_EVENT_SUFFIX.get(event)
    if not suffix:
        raise ValueError(f"unsupported settle event: {event}")
    return f"settle:{reservation_id}:{suffix}"


def build_settle_payload(
    *,
    reservation_id: str,
    call_id: str,
    attempt: int,
    org_id: str,
    event: str,
    actual_usd_micros: Optional[int] = None,
    lifecycle_version: str = LIFECYCLE_VERSION,
    idempotency_key: Optional[str] = None,
    settlement_capability: Optional[str] = None,
    usage_evidence: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Canonical settle request body shared by sync/async clients."""
    payload: dict[str, Any] = {
        "lifecycle_version": lifecycle_version,
        "reservation_id": reservation_id,
        "call_id": call_id,
        "attempt": max(1, int(attempt)),
        "idempotency_key": idempotency_key or settle_idempotency_key(reservation_id, event),
        "event": event,
        "org_id": org_id,
    }
    if settlement_capability:
        payload["settlement_capability"] = settlement_capability
    if event == "usage_known":
        if actual_usd_micros is None:
            raise ValueError("actual_usd_micros is required for usage_known")
        payload["actual_usd_micros"] = int(actual_usd_micros)
        if usage_evidence:
            payload["usage_evidence"] = dict(usage_evidence)
    return payload


def classify_settlement_delivery(
    *,
    local_finalized: bool,
    financial_ack: bool,
    authoritative_usage: bool,
) -> str:
    """Separate local stream finalization from durable financial settlement."""
    if not local_finalized:
        return "open"
    if financial_ack and authoritative_usage:
        return "settled"
    if financial_ack and not authoritative_usage:
        return "pending_usage"
    if authoritative_usage and not financial_ack:
        return "exact_usage_pending_settlement"
    return "pending_reconciliation"


def settlement_evidence_matches(
    *,
    reservation: Mapping[str, Any],
    call_id: str,
    attempt: int,
    reservation_id: str,
    actual_usd_micros: Optional[int],
) -> bool:
    """True when a settled reservation is an identical replay of this work item.

    A message containing the word "settled" is never sufficient on its own —
    bound identities and monetary evidence must match.
    """
    if str(reservation.get("reservation_id") or "") != str(reservation_id):
        return False
    if str(reservation.get("call_id") or "") != str(call_id):
        return False
    try:
        if int(reservation.get("attempt") or 0) != int(attempt):
            return False
    except (TypeError, ValueError):
        return False
    if str(reservation.get("state") or "") != "settled":
        return False
    settle_result = reservation.get("settle_result")
    if settle_result not in (None, "applied", "duplicate"):
        return False
    if actual_usd_micros is None:
        return True
    stored = reservation.get("last_settle_json") or {}
    stored_actual = None
    if isinstance(stored, Mapping):
        stored_actual = stored.get("actual_usd_micros")
    if stored_actual is None:
        stored_actual = reservation.get("actual_usd_micros")
    if stored_actual is None:
        return False
    try:
        return int(stored_actual) == int(actual_usd_micros)
    except (TypeError, ValueError):
        return False
