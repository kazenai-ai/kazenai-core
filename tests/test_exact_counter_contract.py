"""The exact input counters are called the way the official SDKs accept.

``responses.input_tokens.count`` and ``messages.count_tokens`` take fewer fields than
``create``. Earlier tests faked the counters with ``**kwargs``, which hid that every
long-context Responses call with ``max_output_tokens`` (and every Anthropic call that
needed a count) was refused with ``input_counter_failed``. These use the real SDK
resources over a mocked transport, so a signature mismatch fails here.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from kazenai import monitor
from kazenai.request_pricing import _counter_kwargs

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

_RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-6-luna",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 12,
        "output_tokens": 2,
        "total_tokens": 14,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}
_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 2},
}


@pytest.fixture(autouse=True)
def _local_caps_only(monkeypatch):
    # Local client caps only, as in the live provider smoke; FinOps is not part of this contract.
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL"):
        monkeypatch.delenv(key, raising=False)


class _Upstream:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append((request.url.path, body))
        if request.url.path.endswith("/responses/input_tokens"):
            return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 12})
        if request.url.path.endswith("/messages/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 12})
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json=_RESPONSE)
        return httpx.Response(200, json=_MESSAGE)

    def body(self, suffix: str) -> dict:
        return next(b for path, b in self.requests if path.endswith(suffix))


def _require_openai_counter() -> None:
    probe = openai.OpenAI(api_key="test", base_url="https://openai.test/v1")
    if not callable(getattr(getattr(probe.responses, "input_tokens", None), "count", None)):
        pytest.skip("this openai version has no responses.input_tokens.count (openai 2.x has it)")


def test_long_context_responses_call_with_create_only_options_is_counted_and_sent():
    _require_openai_counter()
    up = _Upstream()
    raw = openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(up)),
    )
    client = monitor(raw, max_budget_usd=1.0, agent_id="counter-contract")
    out = client.responses.create(
        model="gpt-6-luna",
        input="Reply with ok",
        max_output_tokens=64,
        temperature=0.2,
        store=False,
        metadata={"case": "counter"},
    )
    assert out.output_text == "ok"
    assert [path for path, _ in up.requests] == ["/v1/responses/input_tokens", "/v1/responses"]
    counted = up.body("/responses/input_tokens")
    assert counted["model"] == "gpt-6-luna" and counted["input"] == "Reply with ok"
    assert not {"max_output_tokens", "temperature", "store", "metadata"} & set(counted)


def test_async_long_context_responses_call_is_counted_and_sent():
    _require_openai_counter()
    up = _Upstream()

    async def run() -> str:
        raw = openai.AsyncOpenAI(
            api_key="test",
            base_url="https://openai.test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(up)),
        )
        client = monitor(raw, max_budget_usd=1.0, agent_id="counter-contract-async")
        out = await client.responses.create(model="gpt-6-luna", input="Reply with ok, async", max_output_tokens=64)
        return out.output_text

    assert asyncio.run(run()) == "ok"
    assert [path for path, _ in up.requests] == ["/v1/responses/input_tokens", "/v1/responses"]
    assert "max_output_tokens" not in up.body("/responses/input_tokens")


def test_long_context_anthropic_call_is_counted_without_create_only_fields():
    up = _Upstream()
    raw = anthropic.Anthropic(
        api_key="test",
        base_url="https://anthropic.test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(up)),
    )
    client = monitor(raw, max_budget_usd=1.0, agent_id="counter-contract-anthropic")
    out = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=16,
        temperature=0.2,
        metadata={"user_id": "u-1"},
        messages=[{"role": "user", "content": "Reply with ok"}],
    )
    assert out.content[0].text == "ok"
    assert [path for path, _ in up.requests] == ["/v1/messages/count_tokens", "/v1/messages"]
    counted = up.body("/messages/count_tokens")
    assert counted["model"] == "claude-haiku-5-5"
    assert not {"max_tokens", "temperature", "metadata"} & set(counted)


def test_anthropic_image_call_is_counted_and_sent():
    up = _Upstream()
    raw = anthropic.Anthropic(
        api_key="test",
        base_url="https://anthropic.test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(up)),
    )
    client = monitor(raw, max_budget_usd=1.0, agent_id="counter-contract-image")
    # An image by URL: counted by the provider, no Pillow needed (CI installs only .[dev]).
    image = {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}
    client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=16,
        messages=[{"role": "user", "content": [image, {"type": "text", "text": "What is this?"}]}],
    )
    assert [path for path, _ in up.requests] == ["/v1/messages/count_tokens", "/v1/messages"]
    assert "max_tokens" not in up.body("/messages/count_tokens")


def test_counter_kwargs_keeps_input_fields_for_a_permissive_counter():
    def permissive(**_kwargs):
        return None

    kept = _counter_kwargs(
        permissive,
        {"model": "m", "input": "x", "tools": [], "max_output_tokens": 5, "stream": True, "store": False},
        provider="openai",
    )
    assert kept == {"model": "m", "input": "x", "tools": []}


def test_parse_text_format_is_counted_as_the_schema_the_sdk_sends():
    class Answer(BaseModel):
        answer: str

    def counter(*, model, input, text=None):
        return None

    kept = _counter_kwargs(counter, {"model": "m", "input": "x", "text_format": Answer}, provider="openai")
    assert kept["text"]["format"]["type"] == "json_schema"
    assert "answer" in json.dumps(kept["text"]["format"])
