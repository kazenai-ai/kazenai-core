"""References taken before ``monitor()`` cannot spend without a budget check.

``monitor()`` replaces SDK methods on the client's resources, so a bound method
or ``with_raw_response`` wrapper saved earlier still points at the original. The
request gate on the client's ``request()`` refuses those calls before dispatch.
Every refusal case below asserts that zero HTTP requests left the process.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx
import pytest

from kazenai import monitor
from kazenai.enforcement import KazenBlock, UnsupportedModeError

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
_EMBED_BODY = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
}
_MODERATION_BODY = {"id": "modr-x", "model": "omni-moderation-latest", "results": []}
_MODELS_BODY = {"object": "list", "data": []}
_ANTHROPIC_BODY = {
    "id": "msg_x",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 2},
}

_CHAT_KW = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}
_RESP_KW = {"model": "gpt-4o-mini", "input": "hello"}
_ANTHROPIC_KW = {
    "model": "claude-haiku-4-5",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hello"}],
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
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")


def _body_for(path: str) -> dict:
    if path.endswith("/embeddings"):
        return _EMBED_BODY
    if path.endswith("/moderations"):
        return _MODERATION_BODY
    if path.endswith("/models"):
        return _MODELS_BODY
    return _CHAT_BODY


def _handler(requests: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_body_for(request.url.path), request=request)

    return handler


def _openai(requests: list[httpx.Request], http_client: Any = None) -> Any:
    return openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=http_client or httpx.Client(transport=httpx.MockTransport(_handler(requests))),
    )


def _async_openai(requests: list[httpx.Request]) -> Any:
    return openai.AsyncOpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler(requests))),
    )


# (label, take the reference before monitor(), call it after)
_HELD_SYNC: list[tuple[str, Callable[[Any], Any], Callable[[Any], Any]]] = [
    ("chat.create", lambda c: c.chat.completions.create, lambda h: h(**_CHAT_KW)),
    (
        "chat.with_raw_response",
        lambda c: c.chat.completions.with_raw_response,
        lambda h: h.create(**_CHAT_KW),
    ),
    (
        "chat.with_streaming_response",
        lambda c: c.chat.completions.with_streaming_response,
        lambda h: h.create(**_CHAT_KW).__enter__(),
    ),
    ("chat.stream", lambda c: c.chat.completions.stream, lambda h: h(**_CHAT_KW).__enter__()),
    ("responses.create", lambda c: c.responses.create, lambda h: h(**_RESP_KW)),
    ("responses.parse", lambda c: c.responses.parse, lambda h: h(**_RESP_KW)),
    (
        "responses.with_raw_response",
        lambda c: c.responses.with_raw_response,
        lambda h: h.create(**_RESP_KW),
    ),
    ("client.post", lambda c: c.post, lambda h: h("/chat/completions", body=_CHAT_KW, cast_to=dict)),
]


@pytest.mark.parametrize("label,take,call", _HELD_SYNC, ids=[c[0] for c in _HELD_SYNC])
def test_reference_saved_before_monitor_cannot_dispatch(label: str, take: Any, call: Any) -> None:
    requests: list[httpx.Request] = []
    raw = _openai(requests)
    held = take(raw)
    monitor(raw, max_budget_usd=_EXHAUSTED)
    with pytest.raises(KazenBlock):
        call(held)
    assert requests == [], f"{label} saved before monitor() reached the provider"


@pytest.mark.parametrize(
    "label,take,call",
    [
        ("chat.create", lambda c: c.chat.completions.create, lambda h: h(**_CHAT_KW)),
        (
            "chat.with_raw_response",
            lambda c: c.chat.completions.with_raw_response,
            lambda h: h.create(**_CHAT_KW),
        ),
        ("responses.parse", lambda c: c.responses.parse, lambda h: h(**_RESP_KW)),
    ],
    ids=["chat.create", "chat.with_raw_response", "responses.parse"],
)
def test_async_reference_saved_before_monitor_cannot_dispatch(label: str, take: Any, call: Any) -> None:
    requests: list[httpx.Request] = []
    raw = _async_openai(requests)
    held = take(raw)
    monitor(raw, max_budget_usd=_EXHAUSTED)

    async def _go() -> None:
        await call(held)

    with pytest.raises(KazenBlock):
        asyncio.run(_go())
    assert requests == [], f"async {label} saved before monitor() reached the provider"


def test_held_reference_is_refused_even_with_budget_left() -> None:
    """The refusal comes from the gate, not from an exhausted budget."""
    requests: list[httpx.Request] = []
    raw = _openai(requests)
    held = raw.chat.completions.create
    client = monitor(raw, max_budget_usd=10.0)
    with pytest.raises(UnsupportedModeError) as exc:
        held(**_CHAT_KW)
    assert exc.value.error_code == "unguarded_dispatch"
    assert requests == []
    # The method read after monitor() is guarded and works.
    assert client.chat.completions.create(**_CHAT_KW).choices[0].message.content == "hi"
    assert len(requests) == 1


def test_unmetered_endpoints_are_refused_by_default() -> None:
    requests: list[httpx.Request] = []
    client = monitor(_openai(requests), max_budget_usd=10.0)
    with pytest.raises(UnsupportedModeError, match="allow_unmetered"):
        client.embeddings.create(model="text-embedding-3-small", input="hello")
    with pytest.raises(UnsupportedModeError):
        client.images.generate(model="gpt-image-1", prompt="a cat")
    with pytest.raises(UnsupportedModeError):
        client.batches.create(input_file_id="file-x", endpoint="/v1/chat/completions", completion_window="24h")
    assert requests == []


def test_allow_unmetered_sends_named_endpoint_without_a_budget_check() -> None:
    requests: list[httpx.Request] = []
    client = monitor(_openai(requests), max_budget_usd=_EXHAUSTED, allow_unmetered=("/embeddings",))
    out = client.embeddings.create(model="text-embedding-3-small", input="hello")
    assert out.data[0].embedding == [0.1, 0.2]
    assert [r.url.path for r in requests] == ["/v1/embeddings"]
    # Only the named endpoint is opened.
    with pytest.raises(UnsupportedModeError):
        client.images.generate(model="gpt-image-1", prompt="a cat")
    assert len(requests) == 1


def test_free_and_read_only_requests_pass() -> None:
    requests: list[httpx.Request] = []
    client = monitor(_openai(requests), max_budget_usd=_EXHAUSTED)
    client.moderations.create(input="hello")
    client.models.list()
    assert [r.url.path for r in requests] == ["/v1/moderations", "/v1/models"]


def test_clone_keeps_the_gate() -> None:
    requests: list[httpx.Request] = []
    client = monitor(_openai(requests), max_budget_usd=10.0)
    clone = client.with_options(timeout=5)
    with pytest.raises(UnsupportedModeError):
        clone.embeddings.create(model="text-embedding-3-small", input="hello")
    assert requests == []


def test_shared_http_client_does_not_gate_an_unmonitored_client() -> None:
    requests: list[httpx.Request] = []
    shared = httpx.Client(transport=httpx.MockTransport(_handler(requests)))
    monitored = monitor(_openai(requests, http_client=shared), max_budget_usd=10.0)
    plain = _openai(requests, http_client=shared)
    with pytest.raises(UnsupportedModeError):
        monitored.embeddings.create(model="text-embedding-3-small", input="hello")
    plain.embeddings.create(model="text-embedding-3-small", input="hello")
    assert [r.url.path for r in requests] == ["/v1/embeddings"]


def test_undo_removes_the_gate() -> None:
    from kazenai.monitor import patch_openai

    requests: list[httpx.Request] = []
    raw = _openai(requests)
    undo = patch_openai(raw, org_id="o", project_id="p", agent_id="a")
    with pytest.raises(UnsupportedModeError):
        raw.embeddings.create(model="text-embedding-3-small", input="hello")
    undo()
    raw.embeddings.create(model="text-embedding-3-small", input="hello")
    assert [r.url.path for r in requests] == ["/v1/embeddings"]


def test_anthropic_reference_saved_before_monitor_cannot_dispatch() -> None:
    anthropic = pytest.importorskip("anthropic")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_ANTHROPIC_BODY, request=request)

    raw = anthropic.Anthropic(
        api_key="test",
        base_url="https://anthropic.test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    held_create = raw.messages.create
    held_raw = raw.messages.with_raw_response
    client = monitor(raw, max_budget_usd=10.0)
    with pytest.raises(UnsupportedModeError):
        held_create(**_ANTHROPIC_KW)
    with pytest.raises(UnsupportedModeError):
        held_raw.create(**_ANTHROPIC_KW)
    assert requests == []
    assert client.messages.create(**_ANTHROPIC_KW).content[0].text == "hi"
    assert [r.url.path for r in requests] == ["/v1/messages"]


@pytest.mark.parametrize(
    "method,url,allowed",
    [
        ("post", "/v1/messages/count_tokens", True),
        ("post", "/v1/responses/input_tokens", True),
        ("post", "/v1/responses/resp_123/cancel", True),
        ("post", "/v1/files", True),
        ("get", "/v1/responses/resp_123", True),
        ("delete", "/v1/files/file-1", True),
        ("post", "/v1/messages", False),
        ("post", "/v1/responses/resp_123/other", False),
        ("post", "/v1/audio/speech", False),
        ("post", "/v1/embeddings", False),
        # A base URL with its own path prefix (a proxy) is stripped first.
        ("post", "/openai/v1/moderations", True),
        ("post", "/openai/v1/embeddings", False),
    ],
)
def test_gate_path_policy(method: str, url: str, allowed: bool) -> None:
    from kazenai._dispatch_gate import DispatchGate

    gate = DispatchGate(base_path="/openai/v1" if url.startswith("/openai") else "/v1")
    if allowed:
        gate.check(method, url)
    else:
        with pytest.raises(UnsupportedModeError):
            gate.check(method, url)


def test_allow_unmetered_matches_behind_a_proxy_base_path() -> None:
    from kazenai._dispatch_gate import DispatchGate

    gate = DispatchGate(base_path="/openai/v1", allow_unmetered=("/embeddings",))
    gate.check("post", "/openai/v1/embeddings")


# -- authorization is per client --------------------------------------------------------


def test_a_guarded_call_on_one_client_does_not_authorize_another() -> None:
    """A callback that runs while client A dispatches cannot send through client B."""
    requests: list[httpx.Request] = []
    refused: list[str] = []
    b = monitor(_openai(requests), max_budget_usd=_EXHAUSTED)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        try:
            b.embeddings.create(model="text-embedding-3-small", input="hello")
        except UnsupportedModeError as exc:
            refused.append(exc.error_code)
        return httpx.Response(200, json=_CHAT_BODY, request=request)

    a = monitor(
        openai.OpenAI(
            api_key="test",
            base_url="https://openai.test/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        max_budget_usd=10.0,
    )
    a.chat.completions.create(**_CHAT_KW)
    assert [r.url.path for r in requests] == ["/v1/chat/completions"]
    assert refused == ["unguarded_dispatch"]


def test_client_request_saved_before_monitor_cannot_dispatch() -> None:
    from openai._models import FinalRequestOptions

    requests: list[httpx.Request] = []
    raw = _openai(requests)
    held = raw.request
    monitor(raw, max_budget_usd=10.0)
    options = FinalRequestOptions.construct(method="post", url="/embeddings", json_data={"model": "m", "input": "x"})
    # Refused at the transport, below the saved method; the SDK reports it as a
    # connection error whose cause is the refusal.
    with pytest.raises(openai.APIConnectionError) as exc:
        held(dict, options)
    assert isinstance(exc.value.__cause__, UnsupportedModeError)
    assert requests == []


def test_async_client_request_saved_before_monitor_cannot_dispatch() -> None:
    from openai._models import FinalRequestOptions

    requests: list[httpx.Request] = []
    raw = _async_openai(requests)
    held = raw.request
    monitor(raw, max_budget_usd=10.0)
    options = FinalRequestOptions.construct(method="post", url="/embeddings", json_data={"model": "m", "input": "x"})

    async def _go() -> None:
        await held(dict, options)

    with pytest.raises(openai.APIConnectionError) as exc:
        asyncio.run(_go())
    assert isinstance(exc.value.__cause__, UnsupportedModeError)
    assert requests == []


def test_monitoring_twice_keeps_one_working_gate() -> None:
    requests: list[httpx.Request] = []
    client = monitor(monitor(_openai(requests), max_budget_usd=10.0), max_budget_usd=10.0)
    assert client.chat.completions.create(**_CHAT_KW).choices[0].message.content == "hi"
    with pytest.raises(UnsupportedModeError):
        client.embeddings.create(model="text-embedding-3-small", input="hello")
    assert [r.url.path for r in requests] == ["/v1/chat/completions"]


def test_the_gated_view_does_not_close_the_shared_connection_pool() -> None:
    import gc

    from kazenai.monitor import patch_openai

    requests: list[httpx.Request] = []
    raw = _openai(requests)
    undo = patch_openai(raw, org_id="o", project_id="p", agent_id="a")
    undo()  # drops the gated view
    gc.collect()
    assert raw._client.is_closed is False
    raw.models.list()
    assert [r.url.path for r in requests] == ["/v1/models"]
