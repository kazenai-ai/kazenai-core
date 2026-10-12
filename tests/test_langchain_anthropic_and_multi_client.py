"""ChatAnthropic Control bind, multi-client boundaries, raw-response streams, beta block."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from kazenai import BudgetExceeded, SharedBudget, UnsupportedModeError, monitor
from kazenai.integrations.adapter_contract import UnsupportedFrameworkPath
from kazenai.model_pricing import model_cost_usd_per_million
from test_final1b_final_eight_cell_cancel_matrix import ANTHROPIC_JSON, ANTHROPIC_SSE, OPENAI_JSON, OPENAI_SSE

pytest.importorskip("langchain_anthropic")
pytest.importorskip("langgraph")


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL"):
        monkeypatch.delenv(key, raising=False)


def _anthropic_handler(hits: list):
    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        import json

        body = json.loads(request.content or b"{}")
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=ANTHROPIC_SSE.encode())
        return httpx.Response(200, json=ANTHROPIC_JSON)

    return handler


def _bound_chat_anthropic(hits: list, **monitor_kwargs: Any):
    import anthropic
    from langchain_anthropic import ChatAnthropic

    from kazenai.integrations.langchain import bind_monitored_chat_anthropic

    handler = _anthropic_handler(hits)
    chat = ChatAnthropic(model="claude-haiku-4-5", api_key="t", max_retries=0, max_tokens=64)
    bind_monitored_chat_anthropic(
        chat,
        anthropic.Anthropic(api_key="t", base_url="https://a.test", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(handler))),
        async_client=anthropic.AsyncAnthropic(api_key="t", base_url="https://a.test", max_retries=0,
                                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))),
        **monitor_kwargs,
    )
    return chat


EXPECTED = model_cost_usd_per_million("claude-haiku-4-5", input_tokens=5, output_tokens=2)


@pytest.mark.parametrize("mode", ["invoke", "stream", "ainvoke", "astream"])
def test_chat_anthropic_bound_records_exact_cost(mode):
    hits: list = []
    budget = SharedBudget(max_usd=1.0)
    chat = _bound_chat_anthropic(hits, budget=budget)
    prompt = [("user", "hi")]
    if mode == "invoke":
        msg = chat.invoke(prompt)
    elif mode == "ainvoke":
        msg = asyncio.run(chat.ainvoke(prompt))
    elif mode == "stream":
        msg = None
        for chunk in chat.stream(prompt):
            msg = chunk if msg is None else msg + chunk
    else:
        async def _go():
            acc = None
            async for chunk in chat.astream(prompt):
                acc = chunk if acc is None else acc + chunk
            return acc

        msg = asyncio.run(_go())
    assert "Hi" in str(msg.content)
    assert hits, "provider was called through the monitored client"
    assert budget.spent_usd == pytest.approx(EXPECTED), mode


def test_chat_anthropic_denied_before_dispatch():
    hits: list = []
    chat = _bound_chat_anthropic(hits, budget=SharedBudget(max_usd=1e-9))
    with pytest.raises(BudgetExceeded):
        chat.invoke([("user", "hi")])
    assert hits == []


def test_chat_anthropic_betas_rejected_at_bind():
    from langchain_anthropic import ChatAnthropic

    from kazenai.integrations.langchain import bind_monitored_chat_anthropic

    chat = ChatAnthropic(model="claude-haiku-4-5", api_key="t", betas=["token-efficient-tools-2025-02-19"])
    with pytest.raises(UnsupportedFrameworkPath):
        bind_monitored_chat_anthropic(chat, max_budget_usd=1.0)


def test_monitored_anthropic_beta_messages_is_blocked_not_bypassed():
    import anthropic

    hits: list = []
    client = monitor(
        anthropic.Anthropic(api_key="t", base_url="https://a.test", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(_anthropic_handler(hits)))),
        max_budget_usd=1.0,
    )
    with pytest.raises(UnsupportedModeError):
        client.beta.messages.create(model="claude-haiku-4-5", max_tokens=8, messages=[{"role": "user", "content": "x"}])
    with pytest.raises(UnsupportedModeError):
        client.beta.messages.with_raw_response.create(model="claude-haiku-4-5", max_tokens=8, messages=[])
    assert hits == []


def test_wrap_langchain_runnable_accepts_bound_chat_anthropic_and_rejects_decoys():
    import anthropic

    from kazenai.integrations.langchain import wrap_langchain_runnable

    chat = _bound_chat_anthropic([], max_budget_usd=1.0)
    adapter = wrap_langchain_runnable(chat, capability="control", guarded_client=chat._kazenai_bound_guarded_client)
    assert "Hi" in str(adapter.invoke([("user", "hi")]).content)
    decoy = monitor(anthropic.Anthropic(api_key="t"), max_budget_usd=1.0)
    with pytest.raises(UnsupportedFrameworkPath):
        wrap_langchain_runnable(chat, capability="control", guarded_client=decoy)


def _openai_client(budget):
    import openai

    def handler(request):
        import json

        if json.loads(request.content or b"{}").get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=OPENAI_SSE.encode())
        return httpx.Response(200, json=OPENAI_JSON)

    return monitor(openai.OpenAI(api_key="t", base_url="https://o.test/v1", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler))), budget=budget)


def _anthropic_client(budget):
    import anthropic

    return monitor(anthropic.Anthropic(api_key="t", base_url="https://a.test", max_retries=0,
                                       http_client=httpx.Client(transport=httpx.MockTransport(_anthropic_handler([])))),
                   budget=budget)


def test_langgraph_declares_and_wraps_multiple_provider_clients():
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    from kazenai.integrations.langgraph import declare_langgraph_control_boundary, wrap_graph

    budget = SharedBudget(max_usd=1.0)
    oa, an = _openai_client(budget), _anthropic_client(budget)

    class S(TypedDict, total=False):
        a: str
        b: str

    def plan(state):
        r = oa.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "plan"}])
        return {"a": r.choices[0].message.content}

    def draft(state):
        r = an.messages.create(model="claude-haiku-4-5", max_tokens=64, messages=[{"role": "user", "content": "draft"}])
        return {"b": r.content[0].text}

    g = StateGraph(S)
    g.add_node("plan", plan)
    g.add_node("draft", draft)
    g.set_entry_point("plan")
    g.add_edge("plan", "draft")
    g.add_edge("draft", END)
    compiled = g.compile()
    declare_langgraph_control_boundary(compiled, oa, an)
    out = wrap_graph(compiled, capability="control", guarded_clients=[oa, an]).invoke({})
    assert out["a"] and out["b"]
    assert budget.spent_usd == pytest.approx(
        model_cost_usd_per_million("gpt-4o-mini", input_tokens=5, output_tokens=2) + EXPECTED
    )
    decoy = _anthropic_client(SharedBudget(1.0))
    with pytest.raises(UnsupportedFrameworkPath):
        wrap_graph(compiled, capability="control", guarded_clients=[oa, decoy])


def test_langchain_runnable_declares_multiple_clients():
    from kazenai.integrations.langchain import declare_langchain_control_boundary, wrap_langchain_runnable

    budget = SharedBudget(max_usd=1.0)
    oa, an = _openai_client(budget), _anthropic_client(budget)

    class TwoModelChain:
        def invoke(self, x):
            oa.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": x}])
            return an.messages.create(model="claude-haiku-4-5", max_tokens=64, messages=[{"role": "user", "content": x}])

    chain = declare_langchain_control_boundary(TwoModelChain(), oa, an)
    adapter = wrap_langchain_runnable(chain, capability="control", guarded_clients=[oa, an])
    adapter.invoke("hi")
    assert budget.spent_usd > 0
    decoy = _openai_client(SharedBudget(1.0))
    with pytest.raises(UnsupportedFrameworkPath):
        wrap_langchain_runnable(chain, capability="control", guarded_clients=[oa, decoy])


def test_openai_with_raw_response_stream_is_metered():
    budget = SharedBudget(max_usd=1.0)
    client = _openai_client(budget)
    raw = client.chat.completions.with_raw_response.create(
        model="gpt-4o-mini", stream=True, messages=[{"role": "user", "content": "x"}]
    )
    text = "".join((c.choices[0].delta.content or "") if c.choices else "" for c in raw.parse())
    assert text == "Hi!"
    assert budget.spent_usd == pytest.approx(model_cost_usd_per_million("gpt-4o-mini", input_tokens=5, output_tokens=2))


def test_wrap_langchain_runnable_uses_the_bound_client_by_default():
    from kazenai.integrations.langchain import wrap_langchain_runnable

    chat = _bound_chat_anthropic([], max_budget_usd=1.0)
    adapter = wrap_langchain_runnable(chat, capability="control")
    assert adapter._guarded_client is chat._kazenai_bound_guarded_client
    assert "Hi" in str(adapter.invoke([("user", "hi")]).content)


def test_bound_guarded_client_accessor():
    from langchain_anthropic import ChatAnthropic

    from kazenai.integrations.langchain import bound_guarded_client

    chat = _bound_chat_anthropic([], max_budget_usd=1.0)
    assert bound_guarded_client(chat) is chat._kazenai_bound_guarded_client
    with pytest.raises(UnsupportedFrameworkPath):
        bound_guarded_client(ChatAnthropic(model="claude-haiku-4-5", api_key="t"))
