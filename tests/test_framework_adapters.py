"""Shared adapter contract + framework wrappers (hermetic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kazenai.integrations.adapter_contract import (
    CapabilityLevel,
    UnsupportedFrameworkPath,
    correlation_envelope,
    enter_operation,
    extract_chat_usage,
    mark_attached,
    new_operation,
    reject_openai_responses_api,
    require_control_client,
    restore_operation_from_envelope,
)
from kazenai.integrations.crewai import CrewAdapter, wrap_crew_kickoff
from kazenai.integrations.langchain import KazenCallbackHandler, wrap_langchain_runnable
from kazenai.integrations.langgraph import LangGraphAdapter, wrap_graph, wrap_graph_invoke

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
RESULTS: list[dict] = []


def _record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"scenario": name, "status": status, "detail": detail})


@pytest.fixture(scope="module", autouse=True)
def _write_artifact():
    yield
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"kazenai-adapters-{ts}.json"
    path.write_text(
        json.dumps(
            {
                "recorded_at": ts,
                "gate": "FRAMEWORK_ADAPTER_CONTRACT_VERIFIED",
                "results": RESULTS,
                "pass_count": sum(1 for r in RESULTS if r["status"] == "pass"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class _Runnable:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, x):
        self.calls += 1
        return {"echo": x}

    async def ainvoke(self, x):
        self.calls += 1
        return {"echo": x}

    def stream(self, x):
        self.calls += 1
        yield {"chunk": 1}
        yield {"chunk": 2}

    async def astream(self, x):
        self.calls += 1
        yield {"chunk": 1}


class _Graph:
    def __init__(self) -> None:
        self.configs: list[Any] = []

    def invoke(self, state, config=None, **kwargs):
        self.configs.append(config)
        return {"ok": True, "state": state}

    async def ainvoke(self, state, config=None, **kwargs):
        self.configs.append(config)
        return {"ok": True, "state": state}

    def stream(self, state, config=None, **kwargs):
        self.configs.append(config)
        yield {"n": 1}

    async def astream(self, state, config=None, **kwargs):
        self.configs.append(config)
        yield {"n": 1}


class _Crew:
    def kickoff(self, inputs=None):
        return {"ok": True, "inputs": inputs}

    async def akickoff(self, inputs=None):
        return {"ok": True, "inputs": inputs, "async": True}


def test_callback_uses_wire_1_2():
    events = []
    h = KazenCallbackHandler(on_event=lambda e: events.append(e), org_id="org-a")
    h.on_llm_end({"llm_output": {"token_usage": {"total_tokens": 42}}})
    assert events[0].schema_version == "1.2"
    assert events[0].payload.get("wire_version") == "1.2"
    assert "kwargs" not in events[0].payload  # no arbitrary kwargs dump
    _record("langchain_callback_wire_1_2", "pass")


def test_missing_usage_is_unknown_not_zero():
    tokens, meta = extract_chat_usage(object())
    assert tokens is None
    assert meta["usage_provenance"] == "unknown"
    _record("missing_usage_unknown", "pass")


def test_wrap_langchain_not_monitor_delegation():
    r = _Runnable()
    adapter = wrap_langchain_runnable(r, org_id="o", capability="observe")
    assert adapter.invoke({"q": 1}) == {"echo": {"q": 1}}
    assert r.calls == 1
    _record("langchain_runnable_modes_observe", "pass")


@pytest.mark.asyncio
async def test_langchain_ainvoke_stream():
    r = _Runnable()
    adapter = wrap_langchain_runnable(r, org_id="o")
    assert await adapter.ainvoke(2) == {"echo": 2}
    chunks = list(adapter.stream(3))
    assert len(chunks) == 2
    async_chunks = [c async for c in adapter.astream(4)]
    assert async_chunks == [{"chunk": 1}]
    _record("langchain_ainvoke_stream_astream", "pass")


def test_attachment_idempotent():
    r = _Runnable()
    a = wrap_langchain_runnable(r)
    assert mark_attached(a) is False  # already attached in ctor
    _record("attachment_idempotent", "pass")


def test_control_requires_guarded_client():
    with pytest.raises(UnsupportedFrameworkPath):
        require_control_client(object())
    _record("control_requires_guarded_client", "pass")


def test_langchain_control_rejects_unrelated_guarded_client():
    class _Echo:
        def invoke(self, x):
            return x

    class FakeGuarded:
        _kazenai_monitored = True

    with pytest.raises(UnsupportedFrameworkPath) as ei:
        wrap_langchain_runnable(
            _Echo(),
            org_id="o",
            capability="control",
            guarded_client=FakeGuarded(),
        )
    assert ei.value.code == "unguarded_control_boundary"
    _record("langchain_control_decoy_guarded_client", "pass")


def test_reject_responses_api_flag():
    class Fake:
        use_responses_api = True

    with pytest.raises(UnsupportedFrameworkPath):
        reject_openai_responses_api(Fake())
    _record("reject_responses_api", "pass")


def test_langgraph_modes_and_envelope():
    g = _Graph()
    events = []
    adapter = wrap_graph(g, org_id="org-g", workspace_id="ws-g", on_event=lambda e: events.append(e))
    out = adapter.invoke({"x": 1}, config={"configurable": {"thread_id": "t1"}})
    assert out["ok"] is True
    env = g.configs[0]["configurable"]["kazenai_correlation"]
    assert env["v"].startswith("kazenai.framework.correlation")
    assert env["org_id"] == "org-g"
    op_a = env["operation_id"]
    # Independent invocations without an envelope must not reuse operation identity.
    adapter.invoke({"x": 2}, config={"configurable": {"thread_id": "t2"}})
    op_b = g.configs[1]["configurable"]["kazenai_correlation"]["operation_id"]
    assert op_a != op_b
    # Checkpoint restore rebinds to constructor tenancy
    restored = restore_operation_from_envelope({**env, "org_id": "forged"})
    assert restored is not None
    # Adapter uses trusted constructor org on resume path
    adapter2 = LangGraphAdapter(g, org_id="trusted", workspace_id="ws")
    adapter2.invoke({}, config={"configurable": {"kazenai_correlation": {**env, "org_id": "forged"}}})
    assert any(e.event_type == "workflow.start" for e in events)
    # compat shim preserves positional config
    fn = wrap_graph_invoke(g, org_id="o")
    assert fn({"a": 1})["ok"] is True
    assert fn({"a": 2}, {"configurable": {"thread_id": "pos"}})["ok"] is True
    _record("langgraph_modes_envelope", "pass")


@pytest.mark.asyncio
async def test_langgraph_async_stream():
    g = _Graph()
    adapter = wrap_graph(g, org_id="o")
    assert (await adapter.ainvoke({"a": 1}))["ok"] is True
    assert [x async for x in adapter.astream({"a": 2})] == [{"n": 1}]
    _record("langgraph_ainvoke_astream", "pass")


def test_crew_kickoff_and_control_guard():
    crew = _Crew()
    events = []
    fn = wrap_crew_kickoff(crew, org_id="o", on_event=lambda e: events.append(e))
    assert fn(inputs={"q": 1})["ok"] is True
    assert any(e.event_type == "workflow.start" for e in events)
    with pytest.raises(UnsupportedFrameworkPath):
        CrewAdapter(crew, capability="control")  # no guarded client

    class FakeGuarded:
        _kazenai_monitored = True

    with pytest.raises(UnsupportedFrameworkPath) as ei:
        CrewAdapter(crew, capability="control", guarded_client=FakeGuarded())
    assert ei.value.code == "unguarded_control_boundary"
    _record("crew_kickoff_control_guard", "pass")


@pytest.mark.asyncio
async def test_crew_akickoff():
    crew = _Crew()
    adapter = CrewAdapter(crew, org_id="o", capability="observe")
    result = await adapter.akickoff(inputs={})
    assert result["async"] is True
    _record("crew_akickoff", "pass")


def test_operation_context_restored_on_error():
    from kazenai.integrations.adapter_contract import get_framework_operation

    op = new_operation(framework="langchain", org_id="o")
    with pytest.raises(RuntimeError):
        with enter_operation(op):
            assert get_framework_operation() is not None
            raise RuntimeError("boom")
    assert get_framework_operation() is None
    _record("operation_context_cleanup", "pass")


def test_correlation_envelope_has_no_secrets():
    op = new_operation(framework="langgraph", org_id="o", metadata={"api_key": "secret"})
    env = correlation_envelope(op)
    assert "api_key" not in env
    assert env["capability"] == CapabilityLevel.OBSERVE.value
    _record("envelope_no_secrets", "pass")


def test_emit_framework_event_spine_and_callback(monkeypatch):
    """Default spine delivery + optional on_event + light control_span."""
    from kazenai.integrations.adapter_contract import emit_framework_event

    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("KAZENAI_OTEL_ENABLED", "1")
    seen: list[str] = []

    def _on_event(ev):
        seen.append(ev.event_type)

    op = new_operation(framework="langchain", org_id="org-u12", project_id="p")
    ev = emit_framework_event(
        op=op,
        event_type="workflow.step",
        payload={"workflow_id": "wf-1", "feature_id": "feat-1", "api_key": "drop-me"},
        on_event=_on_event,
    )
    assert ev.event_type == "workflow.step"
    assert "api_key" not in (ev.payload or {})
    assert seen == ["workflow.step"]

    emitted: list[object] = []
    monkeypatch.setattr(
        "kazenai.spine.guard._emit_event",
        lambda event, event_sink=None: emitted.append(event),
    )
    ev2 = emit_framework_event(
        op=op,
        event_type="workflow.end",
        payload={"workflow_id": "wf-1"},
    )
    assert ev2.event_type == "workflow.end"
    assert len(emitted) == 1
    _record("emit_framework_event_delivery", "pass")
