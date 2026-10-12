"""SharedBudget: one local cap across several monitor() clients (multi-model workflows)."""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from kazenai import BudgetExceeded, SharedBudget, monitor
from kazenai.model_pricing import model_cost_usd_per_million

_OPENAI = {
    "id": "x", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
}
_ANTHROPIC = {
    "id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
    "usage": {"input_tokens": 1000, "output_tokens": 100},
}


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL"):
        monkeypatch.delenv(key, raising=False)


def _openai(budget, hits, *, delay: float = 0.0):
    openai = pytest.importorskip("openai")

    def handler(request):
        hits.append("openai")
        if delay:
            time.sleep(delay)
        return httpx.Response(200, json=_OPENAI)

    return monitor(
        openai.OpenAI(api_key="t", base_url="https://o.test/v1", max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(handler))),
        budget=budget, loop_anomaly_threshold=1.0,
    )


def _anthropic(budget, hits):
    anthropic = pytest.importorskip("anthropic")

    def handler(request):
        hits.append("anthropic")
        return httpx.Response(200, json=_ANTHROPIC)

    return monitor(
        anthropic.Anthropic(api_key="t", base_url="https://a.test", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(handler))),
        budget=budget, loop_anomaly_threshold=1.0,
    )


def test_two_providers_charge_one_budget():
    budget = SharedBudget(max_usd=1.0, name="workflow")
    hits: list[str] = []
    oa, an = _openai(budget, hits), _anthropic(budget, hits)
    oa.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "plan"}])
    an.messages.create(model="claude-haiku-4-5", max_tokens=64, messages=[{"role": "user", "content": "draft"}])
    expected = model_cost_usd_per_million("gpt-4o-mini", input_tokens=1000, output_tokens=100) + \
        model_cost_usd_per_million("claude-haiku-4-5", input_tokens=1000, output_tokens=100)
    assert budget.spent_usd == pytest.approx(expected)
    assert oa._kazenai_enforcement is an._kazenai_enforcement is budget.enforcement


def test_shared_cap_denies_the_other_provider_before_dispatch():
    # Enough headroom for one OpenAI call's pre-call estimate, not for the next step.
    budget = SharedBudget(max_usd=0.012)
    hits: list[str] = []
    oa, an = _openai(budget, hits), _anthropic(budget, hits)
    oa.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "plan"}])
    with pytest.raises(BudgetExceeded):
        an.messages.create(model="claude-haiku-4-5", max_tokens=4096, messages=[{"role": "user", "content": "draft"}])
    assert hits == ["openai"], "the denied step must never reach the provider"


def test_without_shared_budget_each_client_has_its_own_cap():
    hits: list[str] = []
    openai = pytest.importorskip("openai")
    a = monitor(openai.OpenAI(api_key="t", base_url="https://o.test/v1", max_retries=0,
                              http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_OPENAI)))),
                max_budget_usd=1.0)
    b = _anthropic(None, hits)
    assert a._kazenai_enforcement is not b._kazenai_enforcement


def test_concurrent_calls_across_clients_cannot_oversubscribe():
    budget = SharedBudget(max_usd=0.05)
    hits: list[str] = []
    clients = [_openai(budget, hits, delay=0.2) for _ in range(3)]
    allowed, denied = [], []

    def call(client, i):
        try:
            client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"q{i}"}])
            allowed.append(i)
        except BudgetExceeded:
            denied.append(i)

    threads = [threading.Thread(target=call, args=(clients[i % 3], i)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    per_call_estimate = 0.05 / 4.9  # ~0.0102 USD pre-call hold for gpt-4o-mini here
    assert len(allowed) <= int(0.05 // per_call_estimate) + 1
    assert denied, "some concurrent calls must be denied by the shared cap"
    assert len(hits) == len(allowed)
    assert budget.spent_usd <= 0.05


def test_budget_and_max_budget_usd_are_exclusive():
    openai = pytest.importorskip("openai")
    with pytest.raises(ValueError):
        monitor(openai.OpenAI(api_key="t"), budget=SharedBudget(1.0), max_budget_usd=1.0)
    with pytest.raises(TypeError):
        monitor(openai.OpenAI(api_key="t"), budget=1.0)
    with pytest.raises(ValueError):
        SharedBudget(-1)


def test_langchain_bind_shares_one_budget_across_sync_and_async():
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI
    from openai import AsyncOpenAI, OpenAI

    from kazenai.integrations.langchain import bind_monitored_chat_openai

    budget = SharedBudget(max_usd=1.0)
    chat = ChatOpenAI(model="gpt-4o-mini", api_key="t", max_retries=0)
    bind_monitored_chat_openai(chat, OpenAI(api_key="t"), async_client=AsyncOpenAI(api_key="t"), budget=budget)
    assert chat._kazenai_bound_guarded_client._kazenai_enforcement is budget.enforcement
    assert chat._kazenai_bound_guarded_async_client._kazenai_enforcement is budget.enforcement
