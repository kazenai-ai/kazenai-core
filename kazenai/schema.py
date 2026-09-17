from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator  # noqa: F401 – re-exported


# ---------------------------------------------------------------------------
# Typed payload classes (mirrored from kazen-event-schema).
# extra="allow" so new payload fields don't break older consumers.
# ---------------------------------------------------------------------------

class ModelCallPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = ""
    prompt_hash: Optional[str] = None
    response_hash: Optional[str] = None


class ToolCallPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    tool_name: str = ""
    args_hash: Optional[str] = None
    result_hash: Optional[str] = None


class GateDecisionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    gate_type: Optional[str] = None
    decision: Optional[str] = None
    reason: Optional[str] = None
    brain_trace_node_id: Optional[str] = None


class RunLifecyclePayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_summary: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class FinOpsPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    threshold_usd: Optional[float] = None
    current_usd: Optional[float] = None
    reason: Optional[str] = None


_PAYLOAD_VALIDATORS: Dict[str, type[BaseModel]] = {
    "model.call": ModelCallPayload,
    "llm.call": ModelCallPayload,
    "tool.call": ToolCallPayload,
    "mcp.call": ToolCallPayload,
    "gate.decision": GateDecisionPayload,
    "gate.override.requested": GateDecisionPayload,
    "run.started": RunLifecyclePayload,
    "run.finished": RunLifecyclePayload,
    "run.failed": RunLifecyclePayload,
    "finops.circuit_breaker.opened": FinOpsPayload,
    "finops.pause": FinOpsPayload,
    "finops.trajectory": FinOpsPayload,
}


def validate_payload(event_type: str, payload: Dict[str, Any]) -> None:
    """Validate payload contents for the given event_type. Unknown types pass through."""
    validator_cls = _PAYLOAD_VALIDATORS.get(event_type)
    if validator_cls is not None:
        validator_cls.model_validate(payload)


class KazenEvent(BaseModel):
    """
    Canonical event schema (single source of truth for SDK + backend).

    Strictly forbids extra fields to ensure wire compatibility.
    """

    model_config = ConfigDict(extra="forbid")

    # Contract / versioning
    schema_version: str = "1.2"

    # Timing + identity
    ts_ms: int
    event_id: str

    # Multi-tenant + surface
    org_id: str
    workspace_id: str = "default"
    project_id: str
    surface: str

    # Agent identity
    agent_id: str
    agent_role: str

    # Trace identity
    run_id: str
    parent_run_id: Optional[str] = None
    root_run_id: Optional[str] = None
    trace_id: Optional[str] = None
    client_id: Optional[str] = None
    step_id: Optional[str] = None
    parent_step_id: Optional[str] = None

    # Type + common metrics
    event_type: str
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None

    # Governance / gates (set only when applicable)
    gate_decision: Optional[str] = None
    gate_latency_s: Optional[float] = None
    difficulty_score: Optional[float] = None
    artifact_refs: Dict[str, Any] = Field(default_factory=dict)

    # Score-5 reliability spine fields (additive in v1.2).
    stage_budget_usd: Optional[float] = None
    remaining_budget_usd: Optional[float] = None
    projected_total_cost_usd: Optional[float] = None
    avoided_cost_usd: Optional[float] = None
    expected_success_probability: Optional[float] = None
    cost_quality_score: Optional[float] = None
    replay_group_id: Optional[str] = None
    frozen_trace_ref: Optional[str] = None
    drift_baseline_id: Optional[str] = None
    sandbox_backend: Optional[str] = None

    # Event-specific details live here (free-form, redacted by producers as needed).
    payload: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_root_run_id(self) -> "KazenEvent":
        if self.root_run_id is None and self.run_id:
            self.root_run_id = self.run_id
        return self


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id() -> str:
    # UUID4 hex is stable, portable, and dependency-free. We can switch to ULID later.
    return uuid.uuid4().hex
