"""KazenEvent helpers and payload validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kazenai.schema import (
    FinOpsPayload,
    GateDecisionPayload,
    KazenEvent,
    ModelCallPayload,
    ToolCallPayload,
    new_id,
    now_ms,
    validate_payload,
)


def test_now_ms_is_positive_int():
    assert isinstance(now_ms(), int)
    assert now_ms() > 0


def test_new_id_hex_length():
    nid = new_id()
    assert len(nid) == 32
    assert all(c in "0123456789abcdef" for c in nid)


def test_validate_payload_model_call():
    validate_payload("model.call", {"model": "gpt-4o-mini"})
    validate_payload("llm.call", {"model": "claude-3"})


def test_validate_payload_tool_call():
    validate_payload("tool.call", {"tool_name": "search"})
    validate_payload("mcp.call", {"tool_name": "filesystem"})


def test_validate_payload_finops():
    validate_payload("finops.circuit_breaker.opened", {"reason": "budget", "threshold_usd": 1.0})


def test_validate_payload_run_lifecycle():
    validate_payload("run.started", {"task_summary": "bootstrap"})
    validate_payload("run.failed", {"error_type": "TimeoutError"})


def test_validate_payload_unknown_type_passes():
    validate_payload("custom.event", {"anything": True})


def test_model_call_payload_extra_allowed():
    p = ModelCallPayload.model_validate({"model": "x", "extra_field": 1})
    assert p.model == "x"


def test_gate_decision_payload():
    p = GateDecisionPayload(gate_type="budget", decision="allow")
    assert p.decision == "allow"


def test_finops_payload_optional_fields():
    p = FinOpsPayload()
    assert p.threshold_usd is None


def test_kazenevent_defaults_schema_version():
    ev = KazenEvent(
        ts_ms=now_ms(),
        event_id=new_id(),
        org_id="o",
        project_id="p",
        surface="test",
        agent_id="a",
        agent_role="agent",
        run_id="r",
        step_id="s",
        event_type="run.started",
    )
    assert ev.schema_version == "1.2"
    assert ev.workspace_id == "default"


def test_kazenevent_rejects_unknown_top_level_field():
    with pytest.raises(ValidationError):
        KazenEvent(
            ts_ms=1,
            event_id="e",
            org_id="o",
            project_id="p",
            surface="s",
            agent_id="a",
            agent_role="agent",
            run_id="r",
            step_id="s",
            event_type="model.call",
            payload={},
            surprise=True,
        )
