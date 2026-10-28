"""Chat Completions on a model with a long-context price band.

Chat Completions has no exact input counter. Such a request is priced at the standard
band only when its UTF-8 byte size (an upper bound on its tokens) plus framing stays
below the threshold. Anything that cannot be bounded that way is refused before dispatch.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from kazenai import monitor
from kazenai.enforcement import UnsupportedModeError
from kazenai.request_pricing import (
    _chat_input_upper_bound,
    _effective_request,
    _pin_service_tier,
    prepare_provider_call,
)

openai = pytest.importorskip("openai")

THRESHOLD = 272_000  # GPT-5.4 and later, GPT-6


class Colour(BaseModel):
    name: str
    hex_code: str


def _chat_body(content: str = "ok") -> dict:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-6-luna",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def _client(requests: list, *, content: str = "ok", is_async: bool = False):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_chat_body(content), request=request)

    if is_async:
        raw = openai.AsyncOpenAI(
            api_key="test", base_url="https://openai.test/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
    else:
        raw = openai.OpenAI(
            api_key="test", base_url="https://openai.test/v1", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    return monitor(raw, org_id="org-1", project_id="p", agent_id="bound", max_budget_usd=5.0)


@pytest.fixture(autouse=True)
def _local_caps_only(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    "text",
    ["plain ascii text", "日本語のテキストです", "emoji 🚀🔥 and accents éèê", "{\"json\": [1, 2, 3]}"],
)
def test_bound_is_never_below_the_utf8_bytes_of_the_text(text):
    view = {"model": "gpt-6-luna", "messages": [{"role": "system", "content": text}, {"role": "user", "content": text}]}
    bound = _chat_input_upper_bound(view)
    assert bound is not None
    assert bound >= 2 * len(text.encode("utf-8"))


def test_bound_counts_tools_and_the_response_format_schema():
    tool = {
        "type": "function",
        "function": {"name": "lookup", "description": "d" * 5000, "parameters": {"type": "object", "properties": {}}},
    }
    base = {"model": "gpt-6-luna", "messages": [{"role": "user", "content": "hi"}]}
    # Tool definitions and schemas reach the model rewritten as text, up to a few times their JSON size.
    assert _chat_input_upper_bound({**base, "tools": [tool]}) - _chat_input_upper_bound(base) >= 4 * 5000
    with_schema = _chat_input_upper_bound({**base, "response_format": Colour})
    assert with_schema is not None
    assert with_schema - _chat_input_upper_bound(base) >= 4 * len(json.dumps(Colour.model_json_schema()))


def test_tool_text_rewritten_with_comment_lines_stays_within_the_bound():
    # Each description line becomes a "// " comment line; every rendered byte can be a token.
    description = "a\n" * 2000
    tool = {"type": "function", "function": {"name": "f", "description": description, "parameters": {}}}
    rendered = "namespace functions {\n" + "".join(f"// {line}\n" for line in description.splitlines()) + "type f = () => any;\n}"
    base = {"model": "gpt-6-luna", "messages": [{"role": "user", "content": "hi"}]}
    assert _chat_input_upper_bound({**base, "tools": [tool]}) - _chat_input_upper_bound(base) >= len(rendered.encode())


def test_a_reference_to_earlier_audio_output_has_no_bound():
    view = {
        "model": "gpt-6-luna",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "audio": {"id": "audio_1"}},
            {"role": "user", "content": "again"},
        ],
    }
    assert _chat_input_upper_bound(view) is None


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        {"type": "file", "file": {"file_id": "file-1"}},
    ],
)
def test_parts_whose_tokens_bytes_cannot_bound_have_no_bound(part):
    view = {"model": "gpt-6-luna", "messages": [{"role": "user", "content": [part, {"type": "text", "text": "hi"}]}]}
    assert _chat_input_upper_bound(view) is None


def test_an_unmeasurable_object_has_no_bound():
    class Opaque:
        pass

    view = {"model": "gpt-6-luna", "messages": [{"role": "user", "content": "hi"}], "metadata": Opaque()}
    assert _chat_input_upper_bound(view) is None


def test_short_request_is_priced_at_the_standard_band():
    pricing = prepare_provider_call(
        {"model": "gpt-6-luna", "max_completion_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
        provider="openai",
        surface="chat",
        org_id="org-1",
    )
    assert pricing.decision.context_band != "long"
    assert pricing.decision.threshold_decision == "short"


def _effective_bound(request: dict) -> int:
    # Pricing measures the request the provider receives (service tier pinned, extra_body merged).
    body = dict(request)
    _pin_service_tier(body, provider="openai")
    return _chat_input_upper_bound(_effective_request(body))


def test_threshold_edge():
    base = {"model": "gpt-6-luna", "max_completion_tokens": 16}

    def request_with_bound(target: int) -> dict:
        probe = _effective_bound({**base, "messages": [{"role": "user", "content": ""}]})
        return {**base, "messages": [{"role": "user", "content": "x" * (target - probe)}]}

    under = request_with_bound(THRESHOLD)
    assert _effective_bound(under) == THRESHOLD
    prepare_provider_call(dict(under), provider="openai", surface="chat", org_id="org-1")
    over = request_with_bound(THRESHOLD + 1)
    with pytest.raises(UnsupportedModeError) as exc:
        prepare_provider_call(dict(over), provider="openai", surface="chat", org_id="org-1")
    assert exc.value.error_code == "long_context_chat_unsupported"


def test_sync_create_and_stream_on_gpt_6_luna_are_dispatched():
    requests: list = []
    client = _client(requests)
    out = client.chat.completions.create(
        model="gpt-6-luna", max_completion_tokens=64, messages=[{"role": "user", "content": "one"}]
    )
    assert out.choices[0].message.content == "ok"
    assert len(requests) == 1


def test_async_create_on_gpt_6_luna_is_dispatched():
    requests: list = []

    async def run():
        client = _client(requests, is_async=True)
        out = await client.chat.completions.create(
            model="gpt-6-luna", max_completion_tokens=64, messages=[{"role": "user", "content": "two"}]
        )
        return out.choices[0].message.content

    assert asyncio.run(run()) == "ok"
    assert len(requests) == 1


def test_parse_with_a_response_format_model_is_dispatched():
    parse = getattr(getattr(openai.OpenAI(api_key="t").chat, "completions"), "parse", None)
    if parse is None:
        pytest.skip("this openai version has no chat.completions.parse")
    requests: list = []
    client = _client(requests, content=json.dumps({"name": "teal", "hex_code": "#008080"}))
    out = client.chat.completions.parse(
        model="gpt-6-luna", max_completion_tokens=64, response_format=Colour,
        messages=[{"role": "user", "content": "three"}],
    )
    assert out.choices[0].message.parsed.name == "teal"
    assert len(requests) == 1


def test_long_input_and_images_are_refused_before_dispatch():
    requests: list = []
    client = _client(requests)
    with pytest.raises(UnsupportedModeError) as long_exc:
        client.chat.completions.create(model="gpt-6-luna", messages=[{"role": "user", "content": "y" * THRESHOLD}])
    assert long_exc.value.error_code == "long_context_chat_unsupported"
    image = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
    with pytest.raises(UnsupportedModeError) as image_exc:
        client.chat.completions.create(
            model="gpt-6-luna", messages=[{"role": "user", "content": [image, {"type": "text", "text": "four"}]}]
        )
    assert image_exc.value.error_code == "long_context_chat_unsupported"
    assert requests == []
