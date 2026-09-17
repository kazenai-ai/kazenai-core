"""Additional monitor.py edge-case coverage."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kazenai.monitor import (
    _extract_input_text,
    _extract_openai_input,
    _extract_usage,
    _safe_output,
    patch_openai,
)
from kazenai.sinks import MemorySink
from test_monitor_helpers import _FakeClient


def test_extract_usage_empty():
    assert _extract_usage(object()) == (None, None)


def test_extract_input_text_fallback_str():
    assert "{" in _extract_input_text({"unexpected": [1, 2, 3]})


def test_extract_openai_input_empty():
    payload = _extract_openai_input((), {})
    assert payload == {"model": None}


def test_safe_output_uses_str_for_plain_objects():
    class _Weird:
        def __str__(self):
            return "weird"

    assert _safe_output(_Weird()) == "weird"


def test_patch_openai_sink_emit_failure_still_returns(monkeypatch):
    # This verifies sink-emit-failure resilience, not budget enforcement. The pre-call
    # FinOps reserve fails closed by default (no FinOps URL configured) and would raise
    # before the sink is ever touched — run in explicit dev/fail-open so the reserve is
    # skipped and the sink path is actually exercised.
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")

    client = _FakeClient()
    sink = MagicMock()
    sink.emit.side_effect = RuntimeError("sink down")

    patch_openai(client, org_id="o", project_id="p", agent_id="a", event_sink=sink)
    resp = client.chat.completions.create(messages=[])
    assert resp is not None
