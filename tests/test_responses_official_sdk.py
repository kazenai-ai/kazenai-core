"""OpenAI Responses API on the Control path, using the official SDK over MockTransport."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import httpx
import pytest

from kazenai import BudgetExceeded, monitor
from kazenai.enforcement import UnsupportedModeError
from kazenai.model_pricing import default_max_output_tokens, model_cost_usd_per_million
from kazenai.streaming.lifecycle import StreamOutcome

openai = pytest.importorskip("openai")
if not hasattr(openai.OpenAI(api_key="x"), "responses"):  # pragma: no cover - openai < 1.66
    pytest.skip("openai SDK without the Responses API", allow_module_level=True)

MODEL = "gpt-4o-mini"
USAGE = {
    "input_tokens": 1000,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 100,
    "output_tokens_details": {"reasoning_tokens": 40},
    "total_tokens": 1100,
}
EXPECTED_USD = model_cost_usd_per_million(MODEL, input_tokens=1000, output_tokens=100)

_MESSAGE = {
    "type": "message",
    "id": "msg_1",
    "status": "completed",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "hi", "annotations": []}],
}


def _response(status: str = "completed", usage: Any = USAGE, output: Any = None) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": MODEL,
        "status": status,
        "output": [_MESSAGE] if output is None else output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    }


def _sse(terminal: str = "response.completed", usage: Any = USAGE) -> str:
    in_progress = _response(status="in_progress", usage=None, output=[])
    item = dict(_MESSAGE, status="in_progress", content=[])
    part = {"type": "output_text", "text": "", "annotations": []}
    final_status = {"response.completed": "completed", "response.incomplete": "incomplete", "response.failed": "failed"}[terminal]
    events = [
        {"type": "response.created", "sequence_number": 0, "response": in_progress},
        {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": item},
        {"type": "response.content_part.added", "sequence_number": 2, "item_id": "msg_1", "output_index": 0, "content_index": 0, "part": part},
        {"type": "response.output_text.delta", "sequence_number": 3, "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "hi", "logprobs": []},
        {"type": "response.output_text.done", "sequence_number": 4, "item_id": "msg_1", "output_index": 0, "content_index": 0, "text": "hi", "logprobs": []},
        {"type": "response.content_part.done", "sequence_number": 5, "item_id": "msg_1", "output_index": 0, "content_index": 0, "part": dict(part, text="hi")},
        {"type": "response.output_item.done", "sequence_number": 6, "output_index": 0, "item": _MESSAGE},
        {"type": terminal, "sequence_number": 7, "response": _response(status=final_status, usage=usage)},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


@pytest.fixture(autouse=True)
def _standalone(monkeypatch):
    for key in (
        "KAZENAI_FINOPS_URL",
        "KAZENAI_FINOPS_INGEST_URL",
        "KAZENAI_INGEST_URL",
        "KAZENAI_DENY_STREAMING",
        "KAZENAI_UNKNOWN_MODEL_POLICY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")


class _Recorder:
    def __init__(self, *, sse: str | None = None, body: dict[str, Any] | None = None) -> None:
        self.sse = sse
        self.body = body or _response()
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content.decode() or "{}")
        self.requests.append(sent)
        if sent.get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=(self.sse or _sse()).encode(), request=request
            )
        return httpx.Response(200, json=copy.deepcopy(self.body), request=request)


def _client(rec: _Recorder, **kw: Any) -> Any:
    raw = openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(rec)),
    )
    return monitor(raw, max_budget_usd=kw.pop("max_budget_usd", 5.0), **kw)


def _aclient(rec: _Recorder, **kw: Any) -> Any:
    raw = openai.AsyncOpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(rec)),
    )
    return monitor(raw, max_budget_usd=kw.pop("max_budget_usd", 5.0), **kw)


def _recorded(client: Any) -> float:
    return float(client._kazenai_enforcement.snapshot()["cumulative_cost_usd"])


# -- non-streaming ---------------------------------------------------------------


def test_create_settles_exact_cost_and_bounds_output() -> None:
    rec = _Recorder()
    client = _client(rec)
    resp = client.responses.create(model=MODEL, input="hello")
    assert resp.output_text == "hi"
    assert _recorded(client) == pytest.approx(EXPECTED_USD)
    assert rec.requests[0]["max_output_tokens"] == default_max_output_tokens(MODEL)
    assert "max_tokens" not in rec.requests[0] and "stream_options" not in rec.requests[0]


def test_caller_output_limit_above_bound_is_clamped_below_is_kept() -> None:
    bound = int(default_max_output_tokens(MODEL))
    rec = _Recorder()
    client = _client(rec)
    client.responses.create(model=MODEL, input="a", max_output_tokens=bound * 10)
    client.responses.create(model=MODEL, input="b", max_output_tokens=7)
    assert [r["max_output_tokens"] for r in rec.requests] == [bound, 7]


def test_parse_is_metered() -> None:
    rec = _Recorder()
    client = _client(rec)
    parsed = client.responses.parse(model=MODEL, input="hello")
    assert parsed.output_text == "hi"
    assert len(rec.requests) == 1
    assert _recorded(client) == pytest.approx(EXPECTED_USD)


def test_raw_response_create_is_metered() -> None:
    rec = _Recorder()
    client = _client(rec)
    raw = client.responses.with_raw_response.create(model=MODEL, input="hello")
    assert raw.parse().output_text == "hi"
    assert _recorded(client) == pytest.approx(EXPECTED_USD)


def test_function_tools_are_allowed() -> None:
    rec = _Recorder()
    client = _client(rec)
    tool = {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}
    client.responses.create(model=MODEL, input="hello", tools=[tool])
    assert rec.requests[0]["tools"][0]["name"] == "lookup"


def test_exhausted_budget_blocks_before_dispatch() -> None:
    rec = _Recorder()
    client = _client(rec, max_budget_usd=1e-9)
    with pytest.raises(BudgetExceeded):
        client.responses.create(model=MODEL, input="hello")
    assert rec.requests == []


# -- requests whose cost cannot be bounded ------------------------------------------

_DENIED = [
    ({"background": True}, "responses_background_unsupported"),
    ({"previous_response_id": "resp_0"}, "responses_server_state_unsupported"),
    ({"conversation": "conv_1"}, "responses_server_state_unsupported"),
    ({"prompt": {"id": "pmpt_1"}}, "responses_server_state_unsupported"),
    ({"tools": [{"type": "web_search"}]}, "responses_builtin_tool_unsupported"),
    ({"tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}]}, "responses_builtin_tool_unsupported"),
    ({"tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]}, "responses_builtin_tool_unsupported"),
    ({"tools": [{"type": "mcp", "server_label": "x", "server_url": "https://mcp.test"}]}, "responses_builtin_tool_unsupported"),
]


@pytest.mark.parametrize("extra,code", _DENIED, ids=[c for _, c in _DENIED])
@pytest.mark.parametrize("method", ["create", "parse", "stream"])
def test_unboundable_requests_fail_before_dispatch(method: str, extra: dict[str, Any], code: str) -> None:
    rec = _Recorder()
    client = _client(rec)
    with pytest.raises(UnsupportedModeError) as info:
        result = getattr(client.responses, method)(model=MODEL, input="hello", **extra)
        if method == "stream":
            result.__enter__()
    assert info.value.error_code == code
    assert rec.requests == []


def test_stream_resume_by_response_id_is_refused() -> None:
    rec = _Recorder()
    client = _client(rec)
    with pytest.raises(UnsupportedModeError):
        client.responses.stream(response_id="resp_0").__enter__()
    assert rec.requests == []


# -- streaming ----------------------------------------------------------------------


def test_create_stream_settles_exact_from_terminal_event() -> None:
    rec = _Recorder()
    client = _client(rec)
    stream = client.responses.create(model=MODEL, input="hello", stream=True)
    deltas = [e.delta for e in stream if e.type == "response.output_text.delta"]
    assert deltas == ["hi"]
    assert "stream_options" not in rec.requests[0]
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
    assert stream._attempt.cost_confidence == "exact"
    assert stream._attempt.observed_usage.total_tokens == 1100
    assert _recorded(client) == pytest.approx(EXPECTED_USD)


def test_stream_helper_reserves_once_and_settles_exact() -> None:
    rec = _Recorder()
    client = _client(rec)
    with client.responses.stream(model=MODEL, input="hello") as stream:
        final = stream.get_final_response()
    assert final.output_text == "hi"
    assert len(rec.requests) == 1
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
    assert stream._attempt.cost_confidence == "exact"
    assert _recorded(client) == pytest.approx(EXPECTED_USD)


def test_incomplete_stream_with_usage_settles_exact() -> None:
    rec = _Recorder(sse=_sse("response.incomplete"))
    client = _client(rec)
    stream = client.responses.create(model=MODEL, input="hello", stream=True)
    list(stream)
    assert stream._attempt.cost_confidence == "exact"
    assert _recorded(client) == pytest.approx(EXPECTED_USD)


def test_failed_stream_without_usage_stays_unresolved() -> None:
    rec = _Recorder(sse=_sse("response.failed", usage=None))
    client = _client(rec)
    stream = client.responses.create(model=MODEL, input="hello", stream=True)
    list(stream)
    assert stream._attempt.cost_confidence != "exact"
    assert not stream._attempt.observed_usage.authoritative


def test_early_break_is_client_cancelled() -> None:
    rec = _Recorder()
    client = _client(rec)
    stream = client.responses.create(model=MODEL, input="hello", stream=True)
    for _event in stream:
        break
    assert stream._attempt.terminal_outcome == StreamOutcome.CLIENT_CANCELLED


# -- async --------------------------------------------------------------------------


def test_async_create_parse_and_stream() -> None:
    rec = _Recorder()
    client = _aclient(rec)

    async def _go() -> Any:
        await client.responses.create(model=MODEL, input="a")
        await client.responses.parse(model=MODEL, input="b")
        stream = await client.responses.create(model=MODEL, input="c", stream=True)
        async for _event in stream:
            pass
        async with client.responses.stream(model=MODEL, input="d") as helper:
            final = await helper.get_final_response()
        return stream, final

    stream, final = asyncio.run(_go())
    assert final.output_text == "hi"
    assert len(rec.requests) == 4
    assert stream._attempt.cost_confidence == "exact"
    assert _recorded(client) == pytest.approx(4 * EXPECTED_USD)


@pytest.mark.parametrize("extra,code", _DENIED[:3], ids=[c for _, c in _DENIED[:3]])
def test_async_unboundable_requests_fail_before_dispatch(extra: dict[str, Any], code: str) -> None:
    rec = _Recorder()
    client = _aclient(rec)

    async def _go() -> None:
        await client.responses.create(model=MODEL, input="hello", **extra)

    with pytest.raises(UnsupportedModeError) as info:
        asyncio.run(_go())
    assert info.value.error_code == code
    assert rec.requests == []


@pytest.mark.parametrize("resource", ["responses", "chat.completions"])
def test_async_stream_manager_denied_before_dispatch_leaves_no_pending_coroutine(resource: str) -> None:
    import gc
    import warnings

    rec = _Recorder()
    client = _aclient(rec, max_budget_usd=1e-9)
    target = client.responses if resource == "responses" else client.chat.completions
    kwargs = {"model": MODEL, "input": "x"} if resource == "responses" else {"model": MODEL, "messages": [{"role": "user", "content": "x"}]}

    async def _go() -> None:
        with pytest.raises(BudgetExceeded):
            async with target.stream(**kwargs):
                pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(_go())
        gc.collect()
    assert rec.requests == []
    assert not [w for w in caught if "was never awaited" in str(w.message)]
