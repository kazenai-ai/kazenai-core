"""Regression tests for streaming wrappers (Train B local cutoff; no stream-tick)."""

from __future__ import annotations

import importlib

import pytest

monitor_mod = importlib.import_module("kazenai.monitor")
from kazenai.enforcement import StreamCutoffError


class _Ctx:
    org_id = "org-test"
    workspace_id = "ws-test"
    run_id = "run-test"


def _chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def _wrap(chunks, *, usd_per_1k_tokens: float = 0.002, stream_cutoff_usd=None, stream_enforcement=True):
    return monitor_mod._wrap_stream_iterator(
        iter(chunks),
        step_ctx=_Ctx(),
        model="test-model",
        stream_enforcement=stream_enforcement,
        usd_per_1k_tokens=usd_per_1k_tokens,
        stream_cutoff_usd=stream_cutoff_usd,
    )


def test_guarded_stream_iterates_end_to_end():
    chunks = [_chunk("hello world, this is a streamed chunk " * 4) for _ in range(3)]
    seen = list(_wrap(chunks, stream_cutoff_usd=None, stream_enforcement=False))
    assert seen == chunks


def test_guarded_stream_local_cutoff():
    chunks = [_chunk("x" * 800), _chunk("never reached")]
    stream = _wrap(chunks, usd_per_1k_tokens=1.0, stream_cutoff_usd=0.01)
    with pytest.raises(StreamCutoffError):
        list(stream)


def test_unguarded_stream_passthrough():
    chunks = [_chunk("hello")]
    raw = iter(chunks)
    out = monitor_mod._wrap_stream_iterator(
        raw,
        step_ctx=_Ctx(),
        model="test-model",
        stream_enforcement=False,
        usd_per_1k_tokens=0.002,
    )
    assert out is raw
