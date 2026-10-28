"""CrewAI 1.15.26 Control certification cells (real CrewAI objects, simulated providers).

Every cell drives a real ``crewai.Crew`` whose native OpenAI / Anthropic LLM is
bound to a monitored client over ``httpx.MockTransport``, and checks the Control
contract: exact recorded cost, denial before any provider request, events routed
to the owning crew, and refusal of every path KazenAI cannot guard.

Live counterparts: ``tests/test_live_framework_smoke.py`` (``crewai.*`` cells).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

if os.getenv("KAZENAI_REQUIRE_CREWAI") == "1":
    import crewai  # noqa: F401  # fail, never skip, in the CrewAI CI job
crewai = pytest.importorskip("crewai")
openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

from kazenai import BudgetExceeded, SharedBudget, UnsupportedModeError, monitor  # noqa: E402
from kazenai.integrations import crewai as crew_mod  # noqa: E402
from kazenai.integrations.adapter_contract import UnsupportedFrameworkPath  # noqa: E402
from kazenai.integrations.crewai import bind_crew_llm_client, wrap_crew  # noqa: E402
from kazenai.model_pricing import model_cost_usd_per_million  # noqa: E402

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"
FINAL = "Thought: I now can give a great answer\nFinal Answer: ok"

OPENAI_BODY = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": OPENAI_MODEL,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": FINAL}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
}
ANTHROPIC_BODY = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": ANTHROPIC_MODEL,
    "content": [{"type": "text", "text": FINAL}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 1000, "output_tokens": 100},
}
OPENAI_COST = model_cost_usd_per_million(OPENAI_MODEL, input_tokens=1000, output_tokens=100)
ANTHROPIC_COST = model_cost_usd_per_million(ANTHROPIC_MODEL, input_tokens=1000, output_tokens=100)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL", "KAZENAI_ENFORCEMENT_OWNER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")


# -- fixtures ------------------------------------------------------------------------


class Hits:
    def __init__(self) -> None:
        self.paths: list[str] = []
        self.lock = threading.Lock()

    def add(self, path: str) -> None:
        with self.lock:
            self.paths.append(path)


def _handler(hits: Hits, body: dict[str, Any], delay: float = 0.0):
    def handle(request: httpx.Request) -> httpx.Response:
        hits.add(request.url.path)
        if delay:
            time.sleep(delay)
        return httpx.Response(200, json=body, request=request)

    return handle


class _AsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, hits: Hits, body: dict[str, Any], delay: float = 0.0) -> None:
        self.hits, self.body, self.delay = hits, body, delay

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.hits.add(request.url.path)
        if self.delay:
            await asyncio.sleep(self.delay)
        return httpx.Response(200, json=self.body, request=request)


def _openai_llm(hits: Hits, *, body: dict[str, Any] = OPENAI_BODY, delay: float = 0.0, **llm_kwargs: Any):
    llm = crewai.LLM(model=OPENAI_MODEL, api_key="sk-fake", base_url="https://o.test/v1", **llm_kwargs)
    llm._async_client = openai.AsyncOpenAI(
        api_key="sk-fake", base_url="https://o.test/v1", max_retries=0,
        http_client=httpx.AsyncClient(transport=_AsyncTransport(hits, body, delay)),
    )
    raw = openai.OpenAI(
        api_key="sk-fake", base_url="https://o.test/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_handler(hits, body, delay))),
    )
    return llm, raw


def _anthropic_llm(hits: Hits, *, body: dict[str, Any] = ANTHROPIC_BODY):
    llm = crewai.LLM(model=f"anthropic/{ANTHROPIC_MODEL}", api_key="sk-ant-fake", base_url="https://a.test")
    llm._async_client = anthropic.AsyncAnthropic(
        api_key="sk-ant-fake", base_url="https://a.test", max_retries=0,
        http_client=httpx.AsyncClient(transport=_AsyncTransport(hits, body)),
    )
    raw = anthropic.Anthropic(
        api_key="sk-ant-fake", base_url="https://a.test", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_handler(hits, body))),
    )
    return llm, raw


def _agent(llm: Any, role: str = "writer", **kwargs: Any):
    kwargs.setdefault("max_iter", 1)
    return crewai.Agent(role=role, goal="g", backstory="b", llm=llm, allow_delegation=False, **kwargs)


def _crew(*agents: Any, **kwargs: Any):
    tasks = [crewai.Task(description=f"say ok ({a.role})", expected_output="ok", agent=a) for a in agents]
    return crewai.Crew(agents=list(agents), tasks=tasks, **kwargs)


def _spent(client: Any) -> float:
    return float(client._kazenai_enforcement.snapshot()["cumulative_cost_usd"])


def _events() -> tuple[list[Any], Any]:
    seen: list[Any] = []
    return seen, seen.append


def _types(events: list[Any]) -> list[str]:
    return [e.event_type for e in events]


def _payloads(events: list[Any], event_type: str) -> list[dict[str, Any]]:
    return [dict(e.payload or {}) for e in events if e.event_type == event_type]


# -- kickoff / akickoff × provider ----------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_kickoff_exact_cost_and_events(provider):
    hits = Hits()
    llm, raw = _openai_llm(hits) if provider == "openai" else _anthropic_llm(hits)
    agent = _agent(llm)
    crew = _crew(agent)
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    events, sink = _events()
    adapter = wrap_crew(crew, capability="control", guarded_client=guarded, on_event=sink)
    result = adapter.kickoff()
    assert "ok" in str(result)
    expected = OPENAI_COST if provider == "openai" else ANTHROPIC_COST
    assert _spent(guarded) == pytest.approx(expected)
    assert len(hits.paths) == 1
    types = _types(events)
    assert types[0] == "workflow.start" and types[-1] == "workflow.end"
    assert "model.call" in types
    assert crew_mod._ACTIVE_OPS.get(id(crew)) is None


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_native_akickoff_exact_cost_on_one_budget(provider):
    hits = Hits()
    llm, raw = _openai_llm(hits) if provider == "openai" else _anthropic_llm(hits)
    crew = _crew(_agent(llm))
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    # Sync and async calls of the LLM draw from one budget.
    assert llm._async_client._kazenai_enforcement is guarded._kazenai_enforcement
    events, sink = _events()
    adapter = wrap_crew(crew, capability="control", guarded_client=guarded, on_event=sink)
    asyncio.run(adapter.akickoff())
    expected = OPENAI_COST if provider == "openai" else ANTHROPIC_COST
    assert _spent(guarded) == pytest.approx(expected)
    assert len(hits.paths) == 1
    assert _types(events)[-1] == "workflow.end"


def test_kickoff_async_runs_on_a_thread_with_scope():
    hits = Hits()
    llm, raw = _openai_llm(hits)
    crew = _crew(_agent(llm))
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    events, sink = _events()
    adapter = wrap_crew(crew, capability="control", guarded_client=guarded, on_event=sink)
    asyncio.run(adapter.kickoff_async())
    assert _spent(guarded) == pytest.approx(OPENAI_COST)
    assert "model.call" in _types(events)


def test_kickoff_for_each_keeps_the_guard_on_copied_agents():
    hits = Hits()
    llm, raw = _openai_llm(hits)
    agent = _agent(llm)
    crew = crewai.Crew(agents=[agent], tasks=[crewai.Task(description="summarise ticket {ticket}", expected_output="ok", agent=agent)])
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    # Distinct inputs per run; identical prompts back to back would trip the loop guard.
    crew.kickoff_for_each(inputs=[{"ticket": "billing question about invoice 1001"}, {"ticket": "password reset for the admin account"}])
    assert len(hits.paths) == 2
    assert _spent(guarded) == pytest.approx(2 * OPENAI_COST)


# -- denial ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_async", [False, True], ids=["kickoff", "akickoff"])
def test_denial_is_not_retried_and_reaches_no_provider(use_async):
    hits = Hits()
    llm, raw = _openai_llm(hits)
    agent = _agent(llm)  # default max_retry_limit=2
    assert agent.max_retry_limit == 2
    crew = _crew(agent)
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1e-9)
    attempts: list[str] = []
    cls = type(llm)
    inner_call, inner_acall = cls.call.__wrapped__, cls.acall.__wrapped__

    def counting_call(self, *a, **k):
        attempts.append("call")
        return inner_call(self, *a, **k)

    async def counting_acall(self, *a, **k):
        attempts.append("acall")
        return await inner_acall(self, *a, **k)

    crew_mod._STOP_PATCHED.discard(cls)
    cls.call, cls.acall = counting_call, counting_acall
    crew_mod._install_crewai_stop_patch(cls)
    try:
        events, sink = _events()
        adapter = wrap_crew(crew, capability="control", guarded_client=guarded, on_event=sink)
        with pytest.raises(BudgetExceeded) as info:
            asyncio.run(adapter.akickoff()) if use_async else adapter.kickoff()
    finally:
        crew_mod._STOP_PATCHED.discard(cls)
        cls.call, cls.acall = inner_call, inner_acall
        crew_mod._install_crewai_stop_patch(cls)
    from crewai.hooks.dispatch import HookAborted

    assert isinstance(info.value, HookAborted)
    assert len(attempts) == 1, f"a budget denial must not be retried by max_retry_limit: {attempts}"
    assert hits.paths == []
    assert _payloads(events, "workflow.suspend")
    assert "workflow.end" not in _types(events)


# -- structured output and streaming ---------------------------------------------------


class Verdict(BaseModel):
    ok: bool


def test_openai_output_pydantic_is_guarded_and_priced():
    hits = Hits()
    body = {**OPENAI_BODY, "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps({"ok": True})}, "finish_reason": "stop"}]}
    llm, raw = _openai_llm(hits, body=body)
    agent = _agent(llm)
    crew = crewai.Crew(agents=[agent], tasks=[crewai.Task(description="judge", expected_output="verdict", agent=agent, output_pydantic=Verdict)])
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    result = wrap_crew(crew, capability="control", guarded_client=guarded).kickoff()
    assert result.pydantic is not None and result.pydantic.ok is True
    assert len(hits.paths) >= 1
    assert _spent(guarded) == pytest.approx(len(hits.paths) * OPENAI_COST)


def test_anthropic_native_structured_output_is_refused_before_dispatch():
    hits = Hits()
    llm, raw = _anthropic_llm(hits)
    agent = _agent(llm)
    crew = crewai.Crew(agents=[agent], tasks=[crewai.Task(description="judge", expected_output="verdict", agent=agent, output_pydantic=Verdict)])
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    with pytest.raises(UnsupportedModeError):
        wrap_crew(crew, capability="control", guarded_client=guarded).kickoff()
    assert hits.paths == []


OPENAI_SSE = "".join(
    f"data: {json.dumps(chunk)}\n\n"
    for chunk in [
        {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": OPENAI_MODEL,
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": FINAL}, "finish_reason": None}]},
        {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": OPENAI_MODEL,
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": OPENAI_MODEL, "choices": [],
         "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}},
    ]
) + "data: [DONE]\n\n"


def test_streaming_llm_settles_exact():
    hits = Hits()

    def handle(request: httpx.Request) -> httpx.Response:
        hits.add(request.url.path)
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=OPENAI_SSE.encode(), request=request)

    llm = crewai.LLM(model=OPENAI_MODEL, api_key="sk-fake", base_url="https://o.test/v1", stream=True)
    llm._async_client = openai.AsyncOpenAI(api_key="sk-fake", base_url="https://o.test/v1")
    raw = openai.OpenAI(api_key="sk-fake", base_url="https://o.test/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    crew = _crew(_agent(llm))
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    wrap_crew(crew, capability="control", guarded_client=guarded).kickoff()
    assert hits.paths == ["/v1/chat/completions"]
    assert _spent(guarded) == pytest.approx(OPENAI_COST)


# -- two providers, one budget ---------------------------------------------------------


def test_two_provider_crew_on_one_shared_budget():
    hits = Hits()
    o_llm, o_raw = _openai_llm(hits)
    a_llm, a_raw = _anthropic_llm(hits)
    researcher, writer = _agent(o_llm, "researcher"), _agent(a_llm, "writer")
    crew = _crew(researcher, writer)
    budget = SharedBudget(1.0)
    g_openai = bind_crew_llm_client(researcher, o_raw, budget=budget)
    g_anthropic = bind_crew_llm_client(writer, a_raw, budget=budget)
    with pytest.raises(UnsupportedFrameworkPath):
        wrap_crew(crew, capability="control", guarded_client=g_openai)  # writer is not listed
    adapter = wrap_crew(crew, capability="control", guarded_clients=[g_openai, g_anthropic])
    adapter.kickoff()
    assert budget.spent_usd == pytest.approx(OPENAI_COST + ANTHROPIC_COST)
    assert len(hits.paths) == 2


# -- cancellation ------------------------------------------------------------------------


def test_akickoff_cancelled_mid_call_emits_no_end_and_cleans_up():
    """Cancelling akickoff stops the crew; a call already sent finishes and is settled exactly.

    CrewAI 1.15 runs agent LLM calls on worker threads even under ``akickoff``,
    so an in-flight provider call cannot be interrupted.
    """
    hits = Hits()
    llm, raw = _openai_llm(hits, delay=1.0)
    crew = _crew(_agent(llm))
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    events, sink = _events()
    adapter = wrap_crew(crew, capability="control", guarded_client=guarded, on_event=sink)

    async def _run():
        task = asyncio.create_task(adapter.akickoff())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if hits.paths:
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    assert hits.paths == ["/v1/chat/completions"]
    types = _types(events)
    assert "workflow.end" not in types
    assert any(p.get("terminal") == "cancelled" for p in _payloads(events, "workflow.error"))
    assert crew_mod._ACTIVE_OPS.get(id(crew)) is None
    deadline = time.time() + 5
    while _spent(guarded) == 0.0 and time.time() < deadline:
        time.sleep(0.05)
    snap = guarded._kazenai_enforcement.snapshot()
    assert snap["cumulative_cost_usd"] == pytest.approx(OPENAI_COST)  # charged exactly once
    assert snap["pending_projected_usd"] == 0.0  # nothing left held


def test_kickoff_async_cancel_does_not_stop_the_crew():
    """``kickoff_async`` is ``kickoff`` on a worker thread. Cancelling the awaiter
    does not stop that thread: the crew finishes, its spend is checked and settled,
    and it emits ``workflow.end`` (not ``workflow.error``). Use ``akickoff`` when
    cancellation must stop the crew."""
    hits = Hits()
    llm, raw = _openai_llm(hits, delay=1.0)
    crew = _crew(_agent(llm))
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    events, sink = _events()
    adapter = wrap_crew(crew, capability="control", guarded_client=guarded, on_event=sink)

    async def _run():
        task = asyncio.create_task(adapter.kickoff_async())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if hits.paths:
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    deadline = time.time() + 5
    while "workflow.end" not in _types(events) and time.time() < deadline:
        time.sleep(0.05)
    types = _types(events)
    assert "workflow.end" in types
    assert not any(p.get("terminal") == "cancelled" for p in _payloads(events, "workflow.error"))
    assert hits.paths == ["/v1/chat/completions"]
    assert _spent(guarded) == pytest.approx(OPENAI_COST)


def test_max_execution_time_timeout_still_settles_the_orphaned_call():
    hits = Hits()
    llm, raw = _openai_llm(hits, delay=1.5)
    crew = _crew(_agent(llm, max_execution_time=1))
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    with pytest.raises(Exception):
        wrap_crew(crew, capability="control", guarded_client=guarded).kickoff()
    # CrewAI cannot interrupt the running call; it finishes and is settled exactly.
    deadline = time.time() + 5
    while _spent(guarded) == 0.0 and time.time() < deadline:
        time.sleep(0.05)
    assert _spent(guarded) == pytest.approx(len(hits.paths) * OPENAI_COST)


# -- nesting, sharing, routing -----------------------------------------------------------


def test_nested_wrapped_crew_links_to_parent_and_both_spend():
    hits = Hits()
    inner_llm, inner_raw = _openai_llm(hits)
    inner = _crew(_agent(inner_llm, "inner"))
    g_inner = bind_crew_llm_client(inner, inner_raw, max_budget_usd=1.0)
    inner_events, inner_sink = _events()
    inner_adapter = wrap_crew(inner, capability="control", guarded_client=g_inner, on_event=inner_sink)

    outer_llm, outer_raw = _openai_llm(hits)
    outer_agent = _agent(outer_llm, "outer")
    outer = crewai.Crew(
        agents=[outer_agent],
        tasks=[crewai.Task(description="say ok", expected_output="ok", agent=outer_agent,
                           callback=lambda _out: inner_adapter.kickoff())],
    )
    g_outer = bind_crew_llm_client(outer, outer_raw, max_budget_usd=1.0)
    outer_events, outer_sink = _events()
    wrap_crew(outer, capability="control", guarded_client=g_outer, on_event=outer_sink).kickoff()

    assert _spent(g_outer) == pytest.approx(OPENAI_COST)
    assert _spent(g_inner) == pytest.approx(OPENAI_COST)
    outer_op = outer_events[0].payload["operation_id"]
    inner_op = inner_events[0].payload["operation_id"]
    assert inner_op != outer_op
    assert all(e.payload["operation_id"] == inner_op for e in inner_events)
    assert all(e.payload["operation_id"] == outer_op for e in outer_events)
    assert inner_adapter._crew is inner
    started = [e for e in inner_events if e.event_type == "workflow.start"]
    assert started and started[0].payload.get("operation_id") == inner_op


def test_nested_operation_records_parent_operation_id():
    hits = Hits()
    llm, raw = _openai_llm(hits)
    crew = _crew(_agent(llm))
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    adapter = wrap_crew(crew, capability="control", guarded_client=guarded)
    outer = crew_mod.CrewExecutionScope(crew_mod.new_operation(framework="crewai", operation_id="op_outer"), crew=object())
    token = crew_mod._TASK_SCOPE.set(outer)
    try:
        op = adapter._new_op()
    finally:
        crew_mod._TASK_SCOPE.reset(token)
    assert op.metadata["parent_operation_id"] == "op_outer"


def test_concurrent_kickoffs_of_two_crews_sharing_one_llm_route_by_scope():
    hits = Hits()
    llm, raw = _openai_llm(hits, delay=0.05)
    agent_a, agent_b = _agent(llm, "billing analyst"), _agent(llm, "travel planner")
    crew_a = crewai.Crew(agents=[agent_a], tasks=[crewai.Task(description="reconcile the March invoices for account 1001", expected_output="ok", agent=agent_a)])
    crew_b = crewai.Crew(agents=[agent_b], tasks=[crewai.Task(description="plan a three-day trip to Lisbon in spring", expected_output="ok", agent=agent_b)])
    guarded = bind_crew_llm_client(crew_a, raw, max_budget_usd=1.0)
    bind_crew_llm_client(crew_b, guarded)
    ev_a, sink_a = _events()
    ev_b, sink_b = _events()
    ad_a = wrap_crew(crew_a, capability="control", guarded_client=guarded, on_event=sink_a)
    ad_b = wrap_crew(crew_b, capability="control", guarded_client=guarded, on_event=sink_b)

    async def _run():
        await asyncio.gather(ad_a.akickoff(), ad_b.akickoff())

    asyncio.run(_run())
    assert _spent(guarded) == pytest.approx(2 * OPENAI_COST)
    op_a = {e.payload["operation_id"] for e in ev_a}
    op_b = {e.payload["operation_id"] for e in ev_b}
    assert len(op_a) == 1 and len(op_b) == 1 and op_a != op_b
    assert _types(ev_a).count("model.call") == 1
    assert _types(ev_b).count("model.call") == 1

    # Outside any kickoff, an event from the shared LLM belongs to neither crew.
    assert not ad_a._listener.owns_event(llm, None)
    assert not ad_b._listener.owns_event(llm, None)


def test_unbound_listener_owns_nothing():
    listener = crew_mod.KazenCrewListener()
    assert listener.owns_event(object(), None) is False


def test_two_adapters_on_one_crew_both_receive_and_detach_independently():
    hits = Hits()
    llm, raw = _openai_llm(hits)
    agent = _agent(llm)
    crew = crewai.Crew(agents=[agent], tasks=[crewai.Task(description="answer the question: {q}", expected_output="ok", agent=agent)])
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    ev_1, sink_1 = _events()
    ev_2, sink_2 = _events()
    ad_1 = wrap_crew(crew, capability="control", guarded_client=guarded, on_event=sink_1)
    ad_2 = wrap_crew(crew, capability="control", guarded_client=guarded, on_event=sink_2)
    ad_1.kickoff(inputs={"q": "what is the capital of France"})
    assert "model.call" in _types(ev_1)
    assert "model.call" in _types(ev_2)  # both adapters on one crew receive events
    ad_1.detach()
    ad_2.kickoff(inputs={"q": "how many moons does Jupiter have"})
    assert "model.call" in _types(ev_2)
    assert _types(ev_1).count("model.call") == 1  # detached adapter got nothing more
    assert _types(ev_2).count("model.call") == 2


# -- refusals: every path KazenAI cannot guard --------------------------------------------


def _bound_crew(hits: Hits, **crew_kwargs: Any):
    llm, raw = _openai_llm(hits)
    agent = _agent(llm)
    crew = _crew(agent, **crew_kwargs)
    guarded = bind_crew_llm_client(crew, raw, max_budget_usd=1.0)
    return crew, agent, guarded


def test_unbound_second_agent_is_refused():
    hits = Hits()
    llm, raw = _openai_llm(hits)
    bound_agent = _agent(llm, "bound")
    loose_agent = _agent(crewai.LLM(model=OPENAI_MODEL, api_key="sk-fake"), "loose")
    crew = _crew(bound_agent, loose_agent)
    guarded = bind_crew_llm_client(bound_agent, raw, max_budget_usd=1.0)
    with pytest.raises(UnsupportedFrameworkPath, match=r"agent\[1\]\.llm"):
        wrap_crew(crew, capability="control", guarded_client=guarded)


@pytest.mark.parametrize(
    "crew_kwargs,gap",
    [
        ({"memory": True}, "crew.memory"),
        ({"planning": True}, "crew.planning_llm"),
        ({"process": "hierarchical", "manager_llm": "gpt-4o-mini"}, "crew.manager_llm"),
    ],
    ids=["memory", "planning", "hierarchical-unbound-manager"],
)
def test_unguarded_crew_features_are_refused(crew_kwargs, gap):
    hits = Hits()
    try:
        crew, _agent_, guarded = _bound_crew(hits, **crew_kwargs)
    except Exception as exc:  # CrewAI rejected the configuration itself
        pytest.skip(f"CrewAI refused {crew_kwargs}: {exc}")
    with pytest.raises(UnsupportedFrameworkPath, match=gap.replace(".", r"\.")):
        wrap_crew(crew, capability="control", guarded_client=guarded)


def test_hierarchical_crew_with_bound_manager_is_accepted():
    hits = Hits()
    llm, raw = _openai_llm(hits)
    manager_llm, manager_raw = _openai_llm(hits)
    worker = _agent(llm, "worker")
    crew = crewai.Crew(agents=[worker], tasks=[crewai.Task(description="say ok", expected_output="ok")],
                       process="hierarchical", manager_llm=manager_llm)
    budget = SharedBudget(1.0)
    g_worker = bind_crew_llm_client(worker, raw, budget=budget)
    g_manager = bind_crew_llm_client(crewai.Agent(role="m", goal="g", backstory="b", llm=manager_llm), manager_raw, budget=budget)
    assert crew_mod.unguarded_crew_paths(crew, [g_worker, g_manager]) == []


def test_non_certified_crewai_llm_class_is_refused_untouched():
    from crewai.llms.providers.openai.completion import OpenAICompletion

    class CustomCompletion(OpenAICompletion):
        pass

    llm = CustomCompletion(model=OPENAI_MODEL, api_key="sk-fake")
    agent = _agent(llm)
    before = llm._client
    with pytest.raises(UnsupportedFrameworkPath, match="not a certified CrewAI LLM"):
        bind_crew_llm_client(agent, openai.OpenAI(api_key="sk-fake"), max_budget_usd=1.0)
    assert llm._client is before
    assert agent.llm is llm


def test_wrong_provider_client_is_refused():
    hits = Hits()
    llm, _raw = _anthropic_llm(hits)
    with pytest.raises(UnsupportedFrameworkPath, match="Anthropic"):
        bind_crew_llm_client(_agent(llm), openai.OpenAI(api_key="sk-fake"), max_budget_usd=1.0)


def test_streaming_crew_is_refused_at_kickoff():
    hits = Hits()
    crew, _agent_, guarded = _bound_crew(hits, stream=True)
    adapter = wrap_crew(crew, capability="control", guarded_client=guarded)
    with pytest.raises(UnsupportedFrameworkPath, match="stream=True"):
        adapter.kickoff()
    assert hits.paths == []


def test_llm_overlay_is_refused_at_kickoff():
    from crewai.llm_overlay import llm_overlay

    hits = Hits()
    crew, _agent_, guarded = _bound_crew(hits)
    adapter = wrap_crew(crew, capability="control", guarded_client=guarded)
    with llm_overlay({"writer": "gpt-4o"}):
        with pytest.raises(UnsupportedFrameworkPath, match="llm_overlay"):
            adapter.kickoff()
    assert hits.paths == []
