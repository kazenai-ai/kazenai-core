"""Streamed calls settle at input/output split pricing, like non-streamed calls.

Regression: the stream attempt carried the model's *output* rate (used for the
in-flight cutoff estimate) and settlement treated it as a flat caller rate, so
every streamed token — input included — was charged at the output price.
Found by the live-provider smoke on 2026-10-12 (2.6–2.9x overstatement).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kazenai import monitor
from kazenai.model_pricing import model_cost_usd_per_million
from test_streaming_contract import _FakeAnthropic, _FakeAnthropicEventStream, _FakeOpenAI, _standalone

IN_TOKENS = 1000
OUT_TOKENS = 100


def _recorded(client) -> float:
    return float(client._kazenai_enforcement.snapshot()["cumulative_cost_usd"])


def _openai_stream_chunks():
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))], usage=None),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=IN_TOKENS,
                completion_tokens=OUT_TOKENS,
                total_tokens=IN_TOKENS + OUT_TOKENS,
            ),
        ),
    ]


def _anthropic_stream_events():
    return [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=IN_TOKENS, output_tokens=1)),
        ),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="ok")),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=OUT_TOKENS)),
        SimpleNamespace(type="message_stop"),
    ]


def test_openai_stream_settles_at_split_pricing(monkeypatch):
    _standalone(monkeypatch)
    client = _FakeOpenAI()
    client.chat.completions.chunks = _openai_stream_chunks()
    monitor(client, max_budget_usd=5.0)
    list(client.chat.completions.create(model="gpt-4o-mini", stream=True, messages=[{"role": "user", "content": "hi"}]))
    expected = model_cost_usd_per_million("gpt-4o-mini", input_tokens=IN_TOKENS, output_tokens=OUT_TOKENS)
    assert _recorded(client) == pytest.approx(expected, rel=1e-9)
    # The old bug: every token at the output rate.
    assert _recorded(client) < (IN_TOKENS + OUT_TOKENS) * 0.60 / 1_000_000


def test_anthropic_stream_settles_at_split_pricing(monkeypatch):
    _standalone(monkeypatch)
    # The fake's final Message reports fixed usage; align it with the events.
    monkeypatch.setattr(
        _FakeAnthropicEventStream,
        "get_final_message",
        lambda self: SimpleNamespace(usage=SimpleNamespace(input_tokens=IN_TOKENS, output_tokens=OUT_TOKENS)),
    )
    client = _FakeAnthropic()
    client.messages.events = _anthropic_stream_events()
    monitor(client, max_budget_usd=5.0)
    list(
        client.messages.create(
            model="claude-haiku-4-5", stream=True, max_tokens=256, messages=[{"role": "user", "content": "hi"}]
        )
    )
    expected = model_cost_usd_per_million("claude-haiku-4-5", input_tokens=IN_TOKENS, output_tokens=OUT_TOKENS)
    assert _recorded(client) == pytest.approx(expected, rel=1e-9)
    assert _recorded(client) < (IN_TOKENS + OUT_TOKENS) * 5.00 / 1_000_000


def test_explicit_caller_rate_still_applies_to_streams(monkeypatch):
    """monitor(..., usd_per_1k_tokens=) is a caller price; settlement honours it."""
    _standalone(monkeypatch)
    import sys

    m = sys.modules["kazenai.monitor"]

    captured = {}
    real = m._postcall_cost_usd

    def _spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(m, "_postcall_cost_usd", _spy)
    client = _FakeOpenAI()
    client.chat.completions.chunks = _openai_stream_chunks()
    monitor(client, max_budget_usd=5.0)
    list(client.chat.completions.create(model="gpt-4o-mini", stream=True, messages=[{"role": "user", "content": "hi"}]))
    # No caller rate: settlement must not receive the cutoff (output) rate.
    assert captured["usd_per_1k_tokens"] is None
