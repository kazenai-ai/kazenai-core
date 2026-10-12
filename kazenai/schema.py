"""KazenEvent surface for the core SDK.

``kazen-event-schema`` is the canonical Python event model and normalizer.
This module re-exports it while retaining stable helpers (``now_ms``, ``new_id``,
``validate_payload``) used throughout the SDK.
"""

from __future__ import annotations

from typing import Any, Dict  # noqa: F401 — re-exported for callers

from pydantic import ValidationError  # noqa: F401 — re-exported

from kazen_event_schema import (
    FinOpsPayload,
    GateDecisionPayload,
    KazenEvent,
    ModelCallPayload,
    RunLifecyclePayload,
    ToolCallPayload,
    canonicalize_event_dict,
    new_id,
    now_ms,
    validate_kazen_event,
    validate_payload,
)

__all__ = [
    "KazenEvent",
    "ModelCallPayload",
    "ToolCallPayload",
    "GateDecisionPayload",
    "RunLifecyclePayload",
    "FinOpsPayload",
    "ValidationError",
    "canonicalize_event_dict",
    "validate_kazen_event",
    "validate_payload",
    "now_ms",
    "new_id",
]
