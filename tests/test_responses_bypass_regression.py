"""No OpenAI spend path on a monitored client may reach the provider unchecked.

Regression for the 1.1.3 bypass: ``responses.parse``, ``responses.with_raw_response.parse``
and a ``with_raw_response`` wrapper cached before ``monitor()`` dispatched with no
budget check. Every case below uses an exhausted budget and asserts that zero HTTP
requests left the process.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx
import pytest

from kazenai import monitor
from kazenai.enforcement import KazenBlock

openai = pytest.importorskip("openai")

_EXHAUSTED = 1e-9

_CHAT_BODY = {
    "id": "chatcmpl-x",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


@pytest.fixture(autouse=True)
def _standalone(monkeypatch):
    for key in (
        "KAZENAI_FINOPS_URL",
        "KAZENAI_FINOPS_INGEST_URL",
        "KAZENAI_INGEST_URL",
        "KAZENAI_UNKNOWN_MODEL_POLICY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")
    # Local budget only, so a block below can only come from the budget or a stub.
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")


def _sync_client(requests: list[httpx.Request]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_CHAT_BODY, request=request)

    return openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _async_client(requests: list[httpx.Request]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_CHAT_BODY, request=request)

    return openai.AsyncOpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


_RESP_KW = {"model": "gpt-4o-mini", "input": "hello"}
_CHAT_KW = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}


def _has(obj: Any, dotted: str) -> bool:
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


SYNC_CASES: list[tuple[str, str, Callable[[Any], Any]]] = [
    ("responses.create", "responses.create", lambda c: c.responses.create(**_RESP_KW)),
    ("responses.parse", "responses.parse", lambda c: c.responses.parse(**_RESP_KW)),
    (
        "responses.with_raw_response.parse",
        "responses.parse",
        lambda c: c.responses.with_raw_response.parse(**_RESP_KW),
    ),
    (
        "responses.with_streaming_response.create",
        "responses.create",
        lambda c: c.responses.with_streaming_response.create(**_RESP_KW).__enter__(),
    ),
    ("responses.compact", "responses.compact", lambda c: c.responses.compact(model="gpt-4o-mini", input="x")),
    ("completions.create (legacy)", "completions.create", lambda c: c.completions.create(model="gpt-3.5-turbo-instruct", prompt="x")),
    ("realtime.connect", "realtime.connect", lambda c: c.realtime.connect(model="gpt-realtime").__enter__()),
    ("beta.realtime.connect", "beta.realtime.connect", lambda c: c.beta.realtime.connect(model="gpt-realtime").__enter__()),
]


@pytest.mark.parametrize("label,needs,call", SYNC_CASES, ids=[c[0] for c in SYNC_CASES])
def test_sync_spend_surface_never_dispatches_unchecked(label: str, needs: str, call: Callable[[Any], Any]) -> None:
    requests: list[httpx.Request] = []
    raw = _sync_client(requests)
    if not _has(raw, needs):
        pytest.skip(f"{needs} not present in openai {openai.__version__}")
    client = monitor(raw, max_budget_usd=_EXHAUSTED)
    with pytest.raises(KazenBlock):
        call(client)
    assert requests == [], f"{label} reached the provider without a budget check"


ASYNC_CASES: list[tuple[str, str, Callable[[Any], Any]]] = [
    ("responses.create", "responses.create", lambda c: c.responses.create(**_RESP_KW)),
    ("responses.parse", "responses.parse", lambda c: c.responses.parse(**_RESP_KW)),
    (
        "responses.with_raw_response.parse",
        "responses.parse",
        lambda c: c.responses.with_raw_response.parse(**_RESP_KW),
    ),
    ("completions.create (legacy)", "completions.create", lambda c: c.completions.create(model="gpt-3.5-turbo-instruct", prompt="x")),
]


@pytest.mark.parametrize("label,needs,call", ASYNC_CASES, ids=[c[0] for c in ASYNC_CASES])
def test_async_spend_surface_never_dispatches_unchecked(label: str, needs: str, call: Callable[[Any], Any]) -> None:
    requests: list[httpx.Request] = []
    raw = _async_client(requests)
    if not _has(raw, needs):
        pytest.skip(f"{needs} not present in openai {openai.__version__}")
    client = monitor(raw, max_budget_usd=_EXHAUSTED)

    async def _go() -> None:
        await call(client)

    with pytest.raises(KazenBlock):
        asyncio.run(_go())
    assert requests == [], f"async {label} reached the provider without a budget check"


@pytest.mark.parametrize("resource", ["chat.completions", "responses"])
def test_raw_response_wrapper_cached_before_monitor_is_still_guarded(resource: str) -> None:
    """``with_raw_response`` is a cached_property that captures ``create`` at first access."""
    requests: list[httpx.Request] = []
    raw = _sync_client(requests)
    target = raw
    for part in resource.split("."):
        target = getattr(target, part)
    _ = target.with_raw_response  # populate the SDK cache with the unpatched create
    _ = target.with_streaming_response
    client = monitor(raw, max_budget_usd=_EXHAUSTED)
    kwargs = _CHAT_KW if resource == "chat.completions" else _RESP_KW
    with pytest.raises(KazenBlock):
        target.with_raw_response.create(**kwargs)
    with pytest.raises(KazenBlock):
        target.with_streaming_response.create(**kwargs).__enter__()
    assert requests == []
    assert client is raw


def test_async_raw_response_wrapper_cached_before_monitor_is_still_guarded() -> None:
    requests: list[httpx.Request] = []
    raw = _async_client(requests)
    _ = raw.chat.completions.with_raw_response
    monitor(raw, max_budget_usd=_EXHAUSTED)

    async def _go() -> None:
        await raw.chat.completions.with_raw_response.create(**_CHAT_KW)

    with pytest.raises(KazenBlock):
        asyncio.run(_go())
    assert requests == []


def test_raw_response_after_monitor_still_meters_and_parses() -> None:
    """The cache reset must not break the supported langchain-openai raw path."""
    requests: list[httpx.Request] = []
    raw = _sync_client(requests)
    _ = raw.chat.completions.with_raw_response
    client = monitor(raw, max_budget_usd=1.0)
    resp = client.chat.completions.with_raw_response.create(**_CHAT_KW)
    assert resp.parse().choices[0].message.content == "hi"
    assert len(requests) == 1
    assert float(client._kazenai_enforcement.snapshot()["cumulative_cost_usd"]) > 0
