"""Regressions for bugs the live framework smoke found on 2026-10-12."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from kazenai import monitor
from kazenai.context import RunContext, use_context
from kazenai.model_pricing import model_cost_usd_per_million

_OPENAI_BODY = {
    "id": "x",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
}
_ANTHROPIC_BODY = {
    "id": "m",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1000, "output_tokens": 100},
}


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL"):
        monkeypatch.delenv(key, raising=False)


def _recorded(client) -> float:
    return float(client._kazenai_enforcement.snapshot()["cumulative_cost_usd"])


def test_openai_with_raw_response_records_cost_sync_and_async():
    """langchain-openai calls chat.completions.with_raw_response.create for every non-stream call."""
    openai = pytest.importorskip("openai")
    expected = model_cost_usd_per_million("gpt-4o-mini", input_tokens=1000, output_tokens=100)

    sync = monitor(
        openai.OpenAI(
            api_key="t", base_url="https://o.test/v1", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_OPENAI_BODY))),
        ),
        max_budget_usd=1.0,
    )
    raw = sync.chat.completions.with_raw_response.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
    assert raw.parse().choices[0].message.content == "hi"  # caller's parse() still works
    assert _recorded(sync) == pytest.approx(expected)

    aclient = monitor(
        openai.AsyncOpenAI(
            api_key="t", base_url="https://o.test/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_OPENAI_BODY))),
        ),
        max_budget_usd=1.0,
    )

    async def _go():
        return await aclient.chat.completions.with_raw_response.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "y"}]
        )

    asyncio.run(_go())
    assert _recorded(aclient) == pytest.approx(expected)


def test_anthropic_with_raw_response_records_cost():
    anthropic = pytest.importorskip("anthropic")
    client = monitor(
        anthropic.Anthropic(
            api_key="t", base_url="https://a.test", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_ANTHROPIC_BODY))),
        ),
        max_budget_usd=1.0,
    )
    client.messages.with_raw_response.create(
        model="claude-haiku-4-5", max_tokens=64, messages=[{"role": "user", "content": "x"}]
    )
    expected = model_cost_usd_per_million("claude-haiku-4-5", input_tokens=1000, output_tokens=100)
    assert _recorded(client) == pytest.approx(expected)


def test_gateway_marker_never_adds_a_top_level_litellm_params_key():
    """Deployment-hook kwargs become the provider request; providers reject unknown fields."""
    from kazenai.integrations.litellm_control import (
        AttemptReservation,
        GatewayFinOpsClient,
        new_attempt_identity,
    )

    identity = new_attempt_identity(operation_id="op-1", call_id="c-1", attempt=1)
    reservation = AttemptReservation(identity=identity, reservation_id="rsv_1", decision_id="dec_1")
    kwargs: dict = {"model": "gpt-4o-mini", "messages": []}
    GatewayFinOpsClient(base_url="http://finops.test").bind_kwargs_metadata(kwargs, reservation)
    assert "litellm_params" not in kwargs
    assert kwargs["metadata"]["kazenai_reservation_id"] == "rsv_1"

    existing: dict = {"model": "gpt-4o-mini", "litellm_params": {"metadata": {}}}
    GatewayFinOpsClient(base_url="http://finops.test").bind_kwargs_metadata(existing, reservation)
    assert existing["litellm_params"]["metadata"]["kazenai_reservation_id"] == "rsv_1"


def test_framework_default_tenant_does_not_override_client_tenant():
    from kazenai.monitor import _call_context

    client_ctx = RunContext.new(org_id="org-acme", project_id="p", workspace_id="ws-acme", agent_id="a")
    adapter_default = RunContext.new(org_id="local", project_id="p", workspace_id="default", agent_id="langgraph")
    with use_context(adapter_default):
        ctx = _call_context(client_ctx)
    assert (ctx.org_id, ctx.workspace_id) == ("org-acme", "ws-acme")
    assert ctx.run_id == adapter_default.run_id  # lineage still comes from the framework run

    explicit = RunContext.new(org_id="org-other", project_id="p", workspace_id="ws-other", agent_id="langgraph")
    with use_context(explicit):
        assert _call_context(client_ctx).org_id == "org-other"  # an explicit ambient tenant still wins
