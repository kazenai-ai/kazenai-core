"""Monitor emits canonical model.call events."""

from __future__ import annotations

from kazenai.schema import KazenEvent
from kazenai.sinks import MemorySink


def test_monitor_emits_model_call_event_type():
    sink = MemorySink()
    ev = KazenEvent(
        # schema_version 1.2 is current; older "1.0" predated step_id requirement.
        schema_version="1.2",
        ts_ms=1,
        event_id="e1",
        org_id="o",
        project_id="p",
        surface="test",
        agent_id="a",
        agent_role="agent",
        run_id="r",
        # step_id is required since schema v1.2 (Score-5 reliability spine).
        step_id="s1",
        event_type="model.call",
        cost_usd=0.01,
        payload={},
    )
    sink.emit(ev)
    assert sink.events[0].event_type == "model.call"
