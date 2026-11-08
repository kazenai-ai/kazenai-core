"""Pre-call input estimate: plain text is bounded by its bytes; anything else keeps the floor.

A short text prompt used to be priced as at least 2,000 input tokens, so a cheap call
reserved several times its cost. One UTF-8 byte is at most one token, so the byte count
already bounds a plain-text request. Images, audio, files, tools and provider-stored
context can cost tokens the request's bytes don't show, so they keep the floor.
"""

from __future__ import annotations

import importlib

import pytest

m = importlib.import_module("kazenai.monitor")
FLOOR = m._PRECALL_MIN_TOKENS
PROMPT = "Summarise the plot of a short novel in two sentences. " * 2


def _est(**kw) -> int:
    return m._estimate_input_tokens({"model": "gpt-4o-mini", "max_tokens": 100, **kw})


def test_plain_text_chat_is_priced_from_its_bytes():
    n = _est(messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": PROMPT}])
    assert len(PROMPT.encode()) < n < FLOOR


def test_plain_text_never_estimates_fewer_tokens_than_bytes():
    text = "naïve café 東京 " * 40  # multi-byte text: still at least one token per byte
    n = _est(messages=[{"role": "user", "content": text}])
    assert n >= len(text.encode("utf-8"))


def test_long_plain_text_goes_above_the_old_floor():
    text = "x" * 5000
    assert _est(messages=[{"role": "user", "content": text}]) > 5000


def test_text_parts_in_chat_responses_and_anthropic_count_as_text():
    assert _est(messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]) < FLOOR
    assert _est(input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": PROMPT}]}]) < FLOOR
    assert _est(input=PROMPT, instructions="Be brief.") < FLOOR
    assert _est(system="Be brief.", messages=[{"role": "user", "content": PROMPT}]) < FLOOR


@pytest.mark.parametrize(
    "kw",
    [
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]}]},
        {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://x.test/a.png"}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://x.test/a.png"}}]}]},
        {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]}]},
        {"messages": [{"role": "user", "content": [{"type": "file", "file": {"file_id": "file-1"}}]}]},
        {"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "f"}}]},
        {"messages": [{"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function"}]}]},
        {"input": "hi", "previous_response_id": "resp_123"},
        {"input": "hi", "conversation": "conv_123"},
        {"input": "hi", "prompt": {"id": "pmpt_123"}},
        {"messages": [{"role": "user", "content": "hi"}], "extra_body": {"anything": 1}},
    ],
    ids=["chat-image", "responses-image", "anthropic-image", "audio", "file", "tools", "tool-calls",
         "stored-response", "stored-conversation", "prompt-template", "extra-body"],
)
def test_requests_the_bytes_dont_bound_keep_the_floor(kw):
    assert _est(**kw) >= FLOOR


def test_empty_tools_list_is_still_plain_text():
    assert _est(messages=[{"role": "user", "content": PROMPT}], tools=[]) < FLOOR
