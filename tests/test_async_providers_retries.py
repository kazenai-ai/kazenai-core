"""Async providers: retries contract, LiteLLM false-zero, cancel helpers."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from kazenai import UnsupportedModeError, monitor
from kazenai.monitor import configure_direct_provider_client
from kazenai.spine.guard import _embedding_cost, _extract_usage_cost
from kazenai.streaming.lifecycle import StreamOutcome

openai = pytest.importorskip("openai")


OPENAI_SSE = """data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}

data: [DONE]

"""


@pytest.fixture(autouse=True)
def _standalone(monkeypatch):
    for key in (
        "KAZENAI_FINOPS_URL",
        "KAZENAI_REQUIRE_ZERO_PROVIDER_RETRIES",
        "KAZENAI_ALLOW_PROVIDER_RETRIES",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")


def test_require_zero_retries_rejects_nonzero(monkeypatch):
    monkeypatch.setenv("KAZENAI_REQUIRE_ZERO_PROVIDER_RETRIES", "1")
    client = openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
    )
    with pytest.raises(UnsupportedModeError, match="max_retries"):
        monitor(client, max_budget_usd=1.0)


def test_configure_direct_provider_client_zeros_retries():
    client = openai.OpenAI(api_key="test", base_url="https://openai.test/v1", max_retries=2)
    safe = configure_direct_provider_client(client, max_retries=0)
    assert getattr(safe, "max_retries", None) == 0
    # Original left unchanged.
    assert client.max_retries == 2


def test_litellm_zero_or_missing_falls_through_to_cost_engine():
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000)
    resp = SimpleNamespace(usage=usage)

    # Absent LiteLLM — engine prices gpt-4o-mini.
    tokens, cost = _extract_usage_cost("gpt-4o-mini", resp)
    assert tokens == 2000
    assert cost > 0

    # Present LiteLLM returning 0 must not authorize false zero settlement.
    fake = MagicMock()
    fake.completion_cost.return_value = 0.0
    with patch.dict("sys.modules", {"litellm": fake}):
        tokens2, cost2 = _extract_usage_cost("gpt-4o-mini", resp)
    assert tokens2 == 2000
    assert cost2 > 0
    assert cost2 == cost


def test_embedding_cost_ignores_litellm_false_zero():
    fake = MagicMock()
    fake.embedding_cost.return_value = 0.0
    usage = SimpleNamespace(prompt_tokens=500, completion_tokens=0, total_tokens=500)
    resp = SimpleNamespace(usage=usage)
    with patch.dict("sys.modules", {"litellm": fake}):
        cost = _embedding_cost("gpt-4o-mini", resp, ["hello world" * 50])
    assert cost > 0
    fake.embedding_cost.assert_called()


def test_async_stream_repeated_aclose_is_once_only():
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=OPENAI_SSE.encode(),
            request=request,
        )

    async def _run():
        client = monitor(
            openai.AsyncOpenAI(
                api_key="test",
                base_url="https://openai.test/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ),
            ),
            max_budget_usd=5.0,
        )
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        await stream.__anext__()
        await stream.aclose()
        await stream.aclose()
        assert stream._attempt.finalized is True
        assert stream._attempt.terminal_outcome == StreamOutcome.CLIENT_CANCELLED
        await client.close()

    asyncio.run(_run())


def test_async_stream_manager_enter_exit_settles_or_pending():
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=OPENAI_SSE.encode(),
            request=request,
        )

    async def _run():
        client = monitor(
            openai.AsyncOpenAI(
                api_key="test",
                base_url="https://openai.test/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ),
            ),
            max_budget_usd=5.0,
        )
        if not hasattr(client.chat.completions, "stream"):
            pytest.skip("no stream helper")
        async with client.chat.completions.stream(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            async for _ in stream:
                break
        assert stream._attempt.finalized is True
        assert stream._attempt.terminal_outcome in {
            StreamOutcome.CLIENT_CANCELLED,
            StreamOutcome.COMPLETE,
        }
        await client.close()

    asyncio.run(_run())
