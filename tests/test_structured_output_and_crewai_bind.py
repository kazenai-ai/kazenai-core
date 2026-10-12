"""Structured-output parse() is guarded like create(); CrewAI native LLMs bind correctly.

Found 2026-10-12 while building framework examples:
- chat.completions.parse / beta.chat.completions.parse posted to the provider
  directly, skipping the budget check and recording no cost.
- bind_crew_llm_client replaced CrewAI 1.15's native LLM with a raw client
  ("LLM must be resolved before creating agent executor").
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import BaseModel

from kazenai import BudgetExceeded, SharedBudget, UnsupportedModeError, monitor
from kazenai.model_pricing import model_cost_usd_per_million

openai = pytest.importorskip("openai")

_BODY = {
    "id": "x", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"a": 1}'}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
}
_CREW_BODY = {
    **_BODY,
    "choices": [{"index": 0, "message": {"role": "assistant",
                                         "content": "Thought: I now can give a great answer\nFinal Answer: ok"},
                 "finish_reason": "stop"}],
}
EXPECTED = model_cost_usd_per_million("gpt-4o-mini", input_tokens=1000, output_tokens=100)


class Answer(BaseModel):
    a: int


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL"):
        monkeypatch.delenv(key, raising=False)


def _client(budget, hits, *, body=_BODY, aio=False):
    def handler(request):
        hits.append(request.url.path)
        return httpx.Response(200, json=body)

    if aio:
        raw = openai.AsyncOpenAI(api_key="t", base_url="https://o.test/v1", max_retries=0,
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    else:
        raw = openai.OpenAI(api_key="t", base_url="https://o.test/v1", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return monitor(raw, budget=budget)


def _resolve(client, path):
    obj = client
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


_PARSE_PATHS = ["chat.completions.parse", "beta.chat.completions.parse", "chat.completions.with_raw_response.parse"]


@pytest.mark.parametrize("path", _PARSE_PATHS)
def test_parse_is_denied_before_dispatch_over_budget(path):
    hits: list = []
    client = _client(SharedBudget(1e-9), hits)
    with pytest.raises(BudgetExceeded):
        _resolve(client, path)(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}], response_format=Answer)
    assert hits == []


@pytest.mark.parametrize("path", _PARSE_PATHS)
def test_parse_records_exact_cost(path):
    budget = SharedBudget(1.0)
    client = _client(budget, [])
    _resolve(client, path)(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}], response_format=Answer)
    assert budget.spent_usd == pytest.approx(EXPECTED)


def test_async_parse_records_exact_cost():
    budget = SharedBudget(1.0)
    client = _client(budget, [], aio=True)
    asyncio.run(client.chat.completions.parse(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}], response_format=Answer
    ))
    assert budget.spent_usd == pytest.approx(EXPECTED)


def test_beta_structured_stream_is_blocked():
    client = _client(SharedBudget(1.0), [])
    with pytest.raises(UnsupportedModeError):
        client.beta.chat.completions.stream(model="gpt-4o-mini", messages=[], response_format=Answer)


def _crew(llm_budget, hits, monkeypatch, *, body=_CREW_BODY):
    crewai = pytest.importorskip("crewai")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    from kazenai.integrations.crewai import bind_crew_llm_client

    def handler(request):
        hits.append(request.url.path)
        return httpx.Response(200, json=body)

    raw = openai.OpenAI(api_key="sk-fake", base_url="https://o.test/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    llm = crewai.LLM(model="gpt-4o-mini", api_key="sk-fake", base_url="https://o.test/v1")
    agent = crewai.Agent(role="r", goal="g", backstory="b", llm=llm, allow_delegation=False,
                         max_iter=1, max_retry_limit=0)
    crew = crewai.Crew(agents=[agent], tasks=[crewai.Task(description="say ok", expected_output="ok", agent=agent)])
    guarded = bind_crew_llm_client(crew, raw, budget=llm_budget)
    return crew, llm, guarded


def test_crewai_native_llm_keeps_its_llm_and_is_guarded(monkeypatch):
    hits: list = []
    budget = SharedBudget(1.0)
    crew, llm, guarded = _crew(budget, hits, monkeypatch)
    assert llm._client is guarded
    assert crew.agents[0].llm is llm  # the LLM object is kept, not replaced
    crew.kickoff()
    assert hits == ["/v1/chat/completions"]
    assert budget.spent_usd == pytest.approx(EXPECTED)


def test_crewai_budget_denial_reaches_no_provider(monkeypatch):
    hits: list = []
    crew, _llm, _g = _crew(SharedBudget(1e-9), hits, monkeypatch)
    with pytest.raises(BudgetExceeded):
        crew.kickoff()
    assert hits == []
