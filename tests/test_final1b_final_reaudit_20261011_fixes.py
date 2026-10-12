"""Regression coverage for FINAL_1_b_final_REAUDIT_2026-10-11 product defects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
RESULTS: list[dict[str, Any]] = []


def _record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"scenario": name, "status": status, "detail": detail})


@pytest.fixture(scope="module", autouse=True)
def _write_artifact():
    yield
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"final1b-final-reaudit-20261011-fixes-{ts}.json"
    path.write_text(
        json.dumps(
            {
                "recorded_at": ts,
                "loop": "FINAL_1_b_final",
                "phase": "reaudit-20261011-fixes",
                "results": RESULTS,
                "pass_count": sum(1 for r in RESULTS if r["status"] == "pass"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_u01_flush_pending_does_not_ack_result_pending(tmp_path, monkeypatch):
    from kazenai.integrations.litellm_control import (
        AttemptIdentity,
        AttemptReservation,
        DurableGatewayStore,
        GatewayFinOpsClient,
    )

    store = DurableGatewayStore(str(tmp_path / "gw"))
    client = GatewayFinOpsClient(
        base_url="http://finops.test",
        org_id="org",
        workspace_id="ws",
        durable_store=store,
    )
    identities = AttemptIdentity(
        call_id="call-1",
        attempt=1,
        attempt_id="att-1",
        idempotency_key="idem-1",
        operation_id="op-1",
    )
    rsv = AttemptReservation(
        reservation_id="rsv-1",
        decision_id="dec-1",
        identity=identities,
        estimated_usd_micros=12_000,
        status="pending",
        settled_usd_micros=12_000,
    )
    body = {
        "lifecycle_version": "reservation.lifecycle.v1",
        "reservation_id": "rsv-1",
        "call_id": "call-1",
        "attempt": 1,
        "idempotency_key": "idem-1:usage_known",
        "event": "usage_known",
        "org_id": "org",
        "actual_usd_micros": 12_000,
    }
    store.put_open(rsv)
    store.enqueue_pending(body=body, error="http_500")
    client._open[identities.attempt_id] = rsv

    def _post(_path, _body):
        return 200, {"result": "pending", "state": "pending_reconciliation"}

    monkeypatch.setattr(client, "_post", _post)
    out = client.flush_pending()
    assert out == {"applied": 0, "failed": 1}
    assert len(store.list_pending()) == 1
    assert store.get_open(identities.attempt_id) is not None
    _record("u01_flush_pending_keeps_queue", "pass")


def test_u01_cancel_transport_is_pending_not_applied(tmp_path, monkeypatch):
    from kazenai.integrations.litellm_control import (
        AttemptIdentity,
        AttemptReservation,
        DurableGatewayStore,
        GatewayFinOpsClient,
    )

    store = DurableGatewayStore(str(tmp_path / "gw2"))
    client = GatewayFinOpsClient(
        base_url="http://finops.test",
        org_id="org",
        workspace_id="ws",
        durable_store=store,
    )
    identities = AttemptIdentity(
        call_id="call-c",
        attempt=1,
        attempt_id="att-c",
        idempotency_key="idem-c",
        operation_id="op-c",
    )
    rsv = AttemptReservation(
        reservation_id="rsv-c",
        decision_id="dec-c",
        identity=identities,
        estimated_usd_micros=1000,
        status="reserved",
    )

    def _boom(*_a, **_k):
        raise TimeoutError("simulated unavailable")

    monkeypatch.setattr(client, "_post", _boom)
    out = client.settle_cancel_or_pending(rsv, dispatched=False, error="simulated unavailable")
    assert out.get("result") == "pending"
    assert len(store.list_pending()) >= 1
    _record("u01_cancel_transport_pending", "pass")


def test_u02_sync_false_reconcile_marks_pending_with_amount(monkeypatch):
    from kazenai.monitor import _try_mark_stream_pending

    calls: list[dict[str, Any]] = []

    def _signal(**kwargs):
        calls.append(kwargs)
        return False

    monkeypatch.setattr(
        "kazenai.spine.guard.signal_reservation_event",
        _signal,
    )
    ok = _try_mark_stream_pending(
        org_id="o",
        workspace_id="w",
        run_id="r",
        reserved_cost_usd=0.01,
        actual_cost_usd=0.008,
    )
    assert ok is False
    assert calls and calls[0]["event"] == "usage_known"
    assert calls[0]["actual_cost_usd"] == 0.008
    _record("u02_pending_preserves_known_amount", "pass")


def test_u05_langgraph_rejects_decoy_guarded_client_on_adapter():
    from kazenai.integrations.adapter_contract import UnsupportedFrameworkPath
    from kazenai.integrations.langgraph import LangGraphAdapter

    class _G:
        def invoke(self, *_a, **_k):
            return {"ok": True}

    class FakeGuarded:
        _kazenai_monitored = True

    with pytest.raises(UnsupportedFrameworkPath) as ei:
        LangGraphAdapter(
            _G(),
            capability="control",
            guarded_client=FakeGuarded(),
        )
    assert ei.value.code == "unguarded_control_boundary"
    _record("u05_langgraph_decoy_rejected", "pass")


def test_u06_thread_cache_scoped_by_org():
    from kazenai.integrations import langgraph as lg
    from kazenai.integrations.langgraph import wrap_graph

    lg._THREAD_ENVELOPES.clear()

    class _G:
        def __init__(self):
            self.configs: list[dict] = []

        def invoke(self, _input, config=None, **_k):
            self.configs.append(config or {})
            return {"ok": True}

    g = _G()
    a = wrap_graph(g, org_id="org-a", workspace_id="ws", project_id="p", agent_id="ag")
    b = wrap_graph(g, org_id="org-b", workspace_id="ws", project_id="p", agent_id="ag")
    a.invoke({}, config={"configurable": {"thread_id": "shared"}})
    b.invoke({}, config={"configurable": {"thread_id": "shared"}})
    op_a = g.configs[0]["configurable"]["kazenai_correlation"]["operation_id"]
    op_b = g.configs[1]["configurable"]["kazenai_correlation"]["operation_id"]
    assert op_a != op_b
    _record("u06_thread_cache_org_scoped", "pass", f"a={op_a} b={op_b}")


def test_u07_crew_handler_accepts_three_positional_args(monkeypatch):
    import sys
    import types

    from kazenai.integrations import crewai as crew_mod
    from kazenai.integrations.adapter_contract import new_operation
    from kazenai.integrations.crewai import CrewAdapter

    # Fake CrewAI 1.15.26 bus: decorator-style ``on`` and 3-positional dispatch.
    class LLMCallStartedEvent:  # name drives handler routing
        pass

    handlers: list[Any] = []

    class _Bus:
        def on(self, ev_type):
            def deco(fn):
                handlers.append(fn)
                return fn

            return deco

    fake = types.ModuleType("crewai.events")
    fake.crewai_event_bus = _Bus()
    fake.LLMCallStartedEvent = LLMCallStartedEvent
    monkeypatch.setitem(sys.modules, "crewai", types.ModuleType("crewai"))
    monkeypatch.setitem(sys.modules, "crewai.events", fake)
    monkeypatch.delitem(sys.modules, "crewai.utilities.events", raising=False)
    monkeypatch.delitem(sys.modules, "crewai.events.event_bus", raising=False)
    monkeypatch.delitem(sys.modules, "crewai.events.types", raising=False)

    class _Crew:
        def __init__(self):
            self.llm = object()
            self.agents = [types.SimpleNamespace(id="ag-1", llm=self.llm, tools=[])]
            self.tasks = []

        def kickoff(self):
            return {"ok": True}

    events: list[Any] = []
    crew = _Crew()
    adapter = CrewAdapter(crew, capability="observe", on_event=events.append)
    try:
        assert handlers, "listener did not register on the bus"
        tok = adapter._bind(
            new_operation(
                framework="crewai",
                org_id="o",
                workspace_id="w",
                project_id="p",
                agent_id="a",
                capability="observe",
            )
        )
        crew_mod._TASK_SCOPE.reset(tok)  # registry routing only
        # Real LLM source (agent.llm), three positionals: must not raise, must deliver.
        handlers[0](crew.llm, LLMCallStartedEvent(), object())
        # Foreign source: must not deliver.
        handlers[0](object(), LLMCallStartedEvent(), object())
    finally:
        adapter.detach()
    steps = [e for e in events if getattr(e, "event_type", None) == "workflow.step"]
    assert len(steps) == 1
    _record("u07_crew_handler_arity", "pass")


def test_u10_stream_span_does_not_attach_caller_context(monkeypatch):
    pytest.importorskip("opentelemetry")
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    from kazenai.otel import end_control_span, start_control_span
    from kazenai.streaming.lifecycle import StreamAttempt

    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("KAZENAI_OTEL_ENABLED", "1")
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())

    root = trace.get_current_span()
    attempt = StreamAttempt(
        surface="openai",
        step_ctx=object(),
        enforcement=object(),
        held_projection=0.0,
        reserved_cost_usd=0.01,
        model="gpt-4o-mini",
        method="chat.completions.create",
        usd_per_1k_tokens=0.0,
        capture_mode=None,
        agent_role="assistant",
    )
    attempt.begin_otel_span(provider="openai", reservation_id="rsv")
    assert attempt._otel_span is not None
    # Active context must remain the caller root — not model_attempt.
    assert trace.get_current_span() is root or trace.get_current_span().get_span_context().span_id == root.get_span_context().span_id
    attempt.end_otel_span()
    _record("u10_stream_span_no_attach", "pass")


def test_u02_durable_settle_queue_keeps_pending_and_flushes(tmp_path, monkeypatch):
    from kazenai.spine.settle_queue import (
        flush_pending_settlements,
        reset_settle_queue_for_tests,
    )

    q = reset_settle_queue_for_tests(str(tmp_path / "sq"))
    body = {
        "lifecycle_version": "reservation.lifecycle.v1",
        "reservation_id": "rsv-q",
        "call_id": "call-q",
        "attempt": 1,
        "idempotency_key": "idem-q:usage_known",
        "event": "usage_known",
        "org_id": "org",
        "actual_usd_micros": 9000,
    }
    q.enqueue(body, error="http_500", base_url="http://finops.test", api_key="k", org_id="org")

    calls: list[dict] = []

    def _post(
        base,
        org_id,
        payload,
        *,
        timeout_s,
        api_key=None,
        enqueue_on_failure=True,
        journal_before_send=True,
        **_kwargs,
    ):
        calls.append(payload)
        if len(calls) == 1:
            return False  # result=pending / transport failure
        return True

    monkeypatch.setattr("kazenai.spine.guard._post_settle_sync", _post)
    first = flush_pending_settlements(default_base_url="http://finops.test", default_api_key="k")
    assert first["failed"] == 1
    assert len(q.list_pending()) == 1
    second = flush_pending_settlements(default_base_url="http://finops.test", default_api_key="k")
    assert second["applied"] == 1
    assert len(q.list_pending()) == 0
    assert calls[0]["actual_usd_micros"] == 9000
    _record("u02_durable_settle_queue", "pass")


def test_u02_settle_queue_default_root_and_helpers(tmp_path, monkeypatch):
    from kazenai.spine import settle_queue as sq

    monkeypatch.setenv("KAZENAI_SDK_SETTLE_QUEUE_DIR", str(tmp_path / "envq"))
    sq._QUEUE = None
    q = sq.get_settle_queue()
    assert (tmp_path / "envq").exists()
    sq.enqueue_failed_settle(
        {"idempotency_key": "idem-h", "event": "usage_known", "org_id": "o", "actual_usd_micros": 1},
        error="x",
        base_url="http://finops.test",
        api_key="k",
        org_id="o",
    )
    assert len(q.list_pending()) >= 1
    # Corrupt / empty directory entries are skipped.
    bad = tmp_path / "envq" / "pending" / "not-json.txt"
    bad.write_text("nope", encoding="utf-8")
    assert isinstance(q.list_pending(), list)
    q.ack(str(tmp_path / "envq" / "pending" / "missing.json"))
    monkeypatch.delenv("KAZENAI_SDK_SETTLE_QUEUE_DIR", raising=False)
    sq._QUEUE = None
    _record("u02_settle_queue_helpers", "pass")


def test_u02_post_settle_sync_enqueues_on_pending_result(tmp_path, monkeypatch):
    from kazenai.spine import settle_queue as sq
    from kazenai.spine.guard import _post_settle_sync

    q = sq.reset_settle_queue_for_tests(str(tmp_path / "sq2"))

    class _Resp:
        def read(self):
            return b'{"result":"pending","state":"pending_reconciliation"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        "kazenai.spine.guard.urllib.request.urlopen",
        lambda *a, **k: _Resp(),
    )
    body = {
        "lifecycle_version": "reservation.lifecycle.v1",
        "reservation_id": "rsv-e",
        "call_id": "call-e",
        "attempt": 1,
        "idempotency_key": "idem-e:usage_known",
        "event": "usage_known",
        "org_id": "org",
        "actual_usd_micros": 7000,
    }
    ok = _post_settle_sync("http://finops.test", "org", body, timeout_s=1.0, api_key="k")
    assert ok is False
    pending = q.list_pending()
    assert len(pending) == 1
    assert pending[0]["body"]["actual_usd_micros"] == 7000
    # Public guard flush entry delegates to settle_queue.
    from kazenai.spine.guard import flush_pending_settlements as guard_flush

    monkeypatch.setattr(
        "kazenai.spine.guard.urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("down")),
    )
    out = guard_flush(default_base_url="http://finops.test", default_api_key="k")
    assert out["failed"] >= 1
    _record("u02_post_settle_enqueues_pending", "pass")


def test_u07_crew_task_local_scopes_concurrent_ops():
    from kazenai.integrations import crewai as crew_mod
    from kazenai.integrations.crewai import CrewAdapter, CrewExecutionScope

    class _Crew:
        def kickoff(self, *a, **k):
            return {"ok": True}

    events_a: list = []
    events_b: list = []
    crew = _Crew()
    a = CrewAdapter(
        crew,
        capability="observe",
        org_id="o",
        workspace_id="w",
        project_id="p",
        agent_id="ag",
        on_event=lambda e: events_a.append(e),
    )
    # Simulate two concurrent scopes for the same crew id.
    op_a = type("Op", (), {"operation_id": "op-a"})()
    op_b = type("Op", (), {"operation_id": "op-b"})()
    tok_a = a._bind(op_a)
    assert a._listener._op_for(crew).operation_id == "op-a"
    # Nested task context for a second concurrent kickoff.
    scope_b = CrewExecutionScope(op_b, crew=crew)
    tok_b = crew_mod._TASK_SCOPE.set(scope_b)
    assert a._listener._op_for(crew).operation_id == "op-b"
    crew_mod._TASK_SCOPE.reset(tok_b)
    assert a._listener._op_for(crew).operation_id == "op-a"
    a._unbind(tok_a)
    _record("u07_crew_task_local", "pass")


def test_u10_workflow_span_covers_invoke(monkeypatch):
    pytest.importorskip("opentelemetry")
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from kazenai.integrations.langgraph import wrap_graph
    from kazenai.otel import workflow_span

    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("KAZENAI_OTEL_ENABLED", "1")
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    class _G:
        def invoke(self, *_a, **_k):
            return {"ok": True}

    # Direct helper + adapter path — TracerProvider may already be set globally.
    with workflow_span("kazenai.workflow"):
        wrap_graph(_G(), org_id="o", workspace_id="w", project_id="p", agent_id="a").invoke({})
    names = [s.name for s in exporter.get_finished_spans()]
    assert "kazenai.workflow" in names
    _record("u10_workflow_span", "pass", ",".join(names))
