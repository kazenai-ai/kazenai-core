"""CrewAI listener ownership routes real LLM/task/tool sources, never across crews."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from kazenai.integrations import crewai as crew_mod
from kazenai.integrations.adapter_contract import new_operation
from kazenai.integrations.crewai import CrewAdapter, KazenCrewListener


def _op(name: str) -> Any:
    return new_operation(
        framework="crewai",
        org_id=f"org-{name}",
        workspace_id="ws",
        project_id="p",
        agent_id=f"agent-{name}",
        capability="observe",
    )


def _phases(events: list[Any], phase: str) -> int:
    return sum(1 for e in events if (getattr(e, "payload", None) or {}).get("phase") == phase)


# ---------------------------------------------------------------------------
# Unit layer (no crewai install required)
# ---------------------------------------------------------------------------


class _FakeCrew:
    def __init__(self, tag: str) -> None:
        self.llm = SimpleNamespace(model=f"m-{tag}")
        self.tool = SimpleNamespace(name=f"tool-{tag}")
        self.agent = SimpleNamespace(id=f"agent-uuid-{tag}", llm=self.llm, tools=[self.tool])
        self.task = SimpleNamespace(id=f"task-uuid-{tag}", agent=self.agent, tools=[])
        self.agents = [self.agent]
        self.tasks = [self.task]

    def kickoff(self, *a: Any, **k: Any) -> Any:
        return {"ok": True}


def _listener_for(crew: Any) -> KazenCrewListener:
    ltn = KazenCrewListener()
    ltn.bind_crew(crew)
    return ltn


def test_unit_registry_accepts_llm_task_tool_sources_of_own_crew():
    a, b = _FakeCrew("a"), _FakeCrew("b")
    la, lb = _listener_for(a), _listener_for(b)
    for source in (a, a.llm, a.task, a.agent, a.tool):
        assert la.owns_event(source, None)
        assert not lb.owns_event(source, None)


def test_unit_event_agent_task_references_resolve_owner():
    a, b = _FakeCrew("a"), _FakeCrew("b")
    la, lb = _listener_for(a), _listener_for(b)
    tool_usage_helper = object()  # unregistered emitter, like CrewAI ToolUsage
    ev = SimpleNamespace(from_agent=a.agent, from_task=None, agent_id=None, task_id=None)
    assert la.owns_event(tool_usage_helper, ev)
    assert not lb.owns_event(tool_usage_helper, ev)
    ev_ids = SimpleNamespace(agent_id=None, task_id="task-uuid-b")
    assert lb.owns_event(tool_usage_helper, ev_ids)
    assert not la.owns_event(tool_usage_helper, ev_ids)


def test_unit_unrelated_source_rejected_outside_scope():
    a = _FakeCrew("a")
    la = _listener_for(a)
    assert not la.owns_event(object(), SimpleNamespace())
    assert not la.owns_event(None, None)


def test_unit_kickoff_scope_accepts_own_and_rejects_foreign_crew():
    a, b = _FakeCrew("a"), _FakeCrew("b")
    ad_a = CrewAdapter(a, capability="observe")
    ad_b = CrewAdapter(b, capability="observe")
    try:
        tok = ad_a._bind(_op("a"))
        try:
            # Unregistered emitter inside A's kickoff context → A only.
            assert ad_a._listener.owns_event(object(), None)
            assert not ad_b._listener.owns_event(object(), None)
            # Even B's registered LLM is rejected by B while A's scope is active.
            assert not ad_b._listener.owns_event(b.llm, None)
        finally:
            ad_a._unbind(tok)
    finally:
        ad_a.detach()
        ad_b.detach()


def test_unit_detach_clears_registry():
    a = _FakeCrew("a")
    adapter = CrewAdapter(a, capability="observe")
    assert adapter._listener.owns_event(a.llm, None)
    adapter.detach()
    assert adapter._listener._owned_ids == frozenset()
    assert adapter._listener._owned_objs == []


def test_unit_refresh_on_bind_picks_up_late_agents():
    a = _FakeCrew("a")
    adapter = CrewAdapter(a, capability="observe")
    try:
        late_llm = SimpleNamespace(model="late")
        a.agents.append(SimpleNamespace(id="late-agent", llm=late_llm, tools=[]))
        assert not adapter._listener.owns_event(late_llm, None)
        tok = adapter._bind(_op("a"))
        adapter._unbind(tok)
        assert adapter._listener.owns_event(late_llm, None)
    finally:
        adapter.detach()


# ---------------------------------------------------------------------------
# Real CrewAI 1.15.x event bus
# ---------------------------------------------------------------------------


@pytest.fixture
def real_crews(monkeypatch):
    if os.getenv("KAZENAI_REQUIRE_CREWAI") == "1":
        import crewai  # CI gate: a missing install must fail, not skip
    else:
        crewai = pytest.importorskip("crewai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    from crewai import LLM, Agent, Crew, Task

    def build(tag: str):
        llm = LLM(model="gpt-4o-mini", api_key="sk-test-not-used")
        agent = Agent(role=f"r-{tag}", goal="g", backstory="b", llm=llm)
        task = Task(description=f"d-{tag}", expected_output="o", agent=agent)
        return Crew(agents=[agent], tasks=[task]), agent, task, llm

    return crewai, build


def _emit(source: Any, event: Any) -> None:
    from crewai.events import crewai_event_bus

    fut = crewai_event_bus.emit(source, event)
    if fut is not None:
        fut.result(timeout=5)


def test_real_bus_llm_task_tool_sources_route_to_owner_only(real_crews):
    _, build = real_crews
    from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallStartedEvent
    from crewai.events.types.task_events import TaskStartedEvent
    from crewai.events.types.tool_usage_events import ToolUsageStartedEvent

    crew_a, agent_a, task_a, llm_a = build("a")
    crew_b, _, _, _ = build("b")
    ev_a: list[Any] = []
    ev_b: list[Any] = []
    ad_a = CrewAdapter(crew_a, capability="observe", on_event=ev_a.append)
    ad_b = CrewAdapter(crew_b, capability="observe", on_event=ev_b.append)
    tok_a = ad_a._bind(_op("a"))
    tok_b = None
    try:
        crew_mod._TASK_SCOPE.reset(tok_a)  # emit outside any task scope: registry only
        tok_a = None
        _emit(llm_a, LLMCallStartedEvent(call_id="c1", model="gpt-4o-mini", from_agent=agent_a))
        _emit(
            llm_a,
            LLMCallCompletedEvent(
                call_id="c1", model="gpt-4o-mini", response="hi", call_type="llm_call", from_agent=agent_a
            ),
        )
        _emit(task_a, TaskStartedEvent(context="", task=task_a))
        _emit(
            object(),
            ToolUsageStartedEvent(
                tool_name="search", tool_args={}, agent_key=agent_a.key, from_agent=agent_a, from_task=task_a
            ),
        )
        assert _phases(ev_a, "llm_start") == 1
        assert _phases(ev_a, "llm_end") == 1
        assert _phases(ev_a, "task_start") == 1
        assert _phases(ev_a, "tool_start") == 1
        assert ev_b == []
    finally:
        ad_a._unbind(tok_a)
        ad_a.detach()
        ad_b.detach()


def test_real_bus_kickoff_scope_isolates_concurrent_crews(real_crews):
    _, build = real_crews
    from crewai.events.types.llm_events import LLMCallStartedEvent

    crew_a, _, _, _ = build("a")
    crew_b, _, _, llm_b = build("b")
    ev_a: list[Any] = []
    ev_b: list[Any] = []
    ad_a = CrewAdapter(crew_a, capability="observe", on_event=ev_a.append)
    ad_b = CrewAdapter(crew_b, capability="observe", on_event=ev_b.append)
    try:
        tok = ad_b._bind(_op("b"))
        try:
            # Emitter not in any registry, inside B's kickoff context (bus copies contextvars).
            _emit(object(), LLMCallStartedEvent(call_id="c2", model="m"))
            _emit(llm_b, LLMCallStartedEvent(call_id="c3", model="m"))
        finally:
            ad_b._unbind(tok)
        assert _phases(ev_b, "llm_start") == 2
        assert _phases(ev_a, "llm_start") == 0
    finally:
        ad_a.detach()
        ad_b.detach()


def test_real_bus_detach_stops_delivery(real_crews):
    _, build = real_crews
    from crewai.events.types.llm_events import LLMCallStartedEvent

    crew_a, agent_a, _, llm_a = build("a")
    ev_a: list[Any] = []
    ad_a = CrewAdapter(crew_a, capability="observe", on_event=ev_a.append)
    tok = ad_a._bind(_op("a"))
    ad_a._unbind(tok)
    ad_a.detach()
    tok = crew_mod._TASK_SCOPE.set(None)
    try:
        _emit(llm_a, LLMCallStartedEvent(call_id="c4", model="m", from_agent=agent_a))
    finally:
        crew_mod._TASK_SCOPE.reset(tok)
    assert ev_a == []
