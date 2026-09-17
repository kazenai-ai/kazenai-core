"""Regression tests for the mid-flight stream budget guard (P0-1).

_GuardedStream.__iter__ previously did augmented assignment on the closure
variables pending_tokens / pending_usd without a ``nonlocal`` declaration,
raising UnboundLocalError on the first chunk containing text — the FinOps
mid-stream cutoff never functioned.
"""

from __future__ import annotations

import importlib

import pytest

# kazenai/__init__.py re-exports the monitor *function*, shadowing the
# submodule for plain attribute access — import the module explicitly.
monitor_mod = importlib.import_module("kazenai.monitor")
from kazenai.enforcement import StreamCutoffError


class _Ctx:
    org_id = "org-test"
    workspace_id = "ws-test"
    run_id = "run-test"


def _chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def _wrap(chunks, *, usd_per_1k_tokens: float = 0.002):
    return monitor_mod._wrap_stream_iterator(
        iter(chunks),
        step_ctx=_Ctx(),
        model="test-model",
        stream_enforcement=True,
        usd_per_1k_tokens=usd_per_1k_tokens,
    )


def test_guarded_stream_iterates_end_to_end(monkeypatch):
    """A guarded stream with real text chunks must yield every chunk (no UnboundLocalError)."""
    ticks = []

    def fake_tick(**kwargs):
        ticks.append(kwargs)
        return True

    monkeypatch.setattr(monitor_mod, "_try_stream_tick", fake_tick)

    chunks = [_chunk("hello world, this is a streamed chunk " * 4) for _ in range(5)]
    seen = list(_wrap(chunks))

    assert seen == chunks
    # The final-flush path (pending_usd > 0 after the loop) must also tick.
    assert len(ticks) >= 1
    assert all(t["incremental_cost_usd"] > 0 for t in ticks)


def test_guarded_stream_cutoff_when_tick_denied(monkeypatch):
    """When FinOps denies the tick, StreamCutoffError must fire mid-stream."""
    monkeypatch.setattr(monitor_mod, "_try_stream_tick", lambda **kw: False)

    # >=16 estimated tokens (len//4) in one chunk forces an in-loop flush.
    chunks = [_chunk("x" * 200), _chunk("never reached")]
    stream = iter(_wrap(chunks))

    with pytest.raises(StreamCutoffError):
        for _ in stream:
            pass


def test_guarded_stream_final_flush_cutoff(monkeypatch):
    """A small tail below the 16-token flush threshold still ticks (and can cut off) at stream end."""
    monkeypatch.setattr(monitor_mod, "_try_stream_tick", lambda **kw: False)

    chunks = [_chunk("tiny")]  # ~1 token: only the post-loop flush runs
    with pytest.raises(StreamCutoffError):
        list(_wrap(chunks))


def test_unguarded_stream_passthrough():
    """stream_enforcement=False must return the original stream untouched."""
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
