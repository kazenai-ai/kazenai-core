"""monitor.py helper and patch_openai integration tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kazenai.enforcement import BudgetExceeded
from kazenai.finops import FinOpsConfig, FinOpsController
from kazenai.monitor import (
    _detect_client_kind,
    _extract_anthropic_usage,
    _extract_input_text,
    _extract_openai_input,
    _extract_tool_names,
    _extract_usage,
    _safe_output,
    monitor,
    patch_anthropic,
    patch_openai,
)
from kazenai.schema import KazenEvent
from kazenai.sinks import MemorySink


@pytest.fixture(autouse=True)
def _dev_fail_open_reserve(monkeypatch):
    """Monitor patch tests use fake clients — allow reserve skip without FinOps."""
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_FINOPS_INGEST_URL", raising=False)


def test_extract_openai_input_messages():
    payload = _extract_openai_input((), {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o"})
    assert "messages" in payload
    assert payload["model"] == "gpt-4o"


def test_extract_openai_input_responses_api():
    payload = _extract_openai_input((), {"input": "hello", "model": "gpt-4o"})
    assert payload["input"] == "hello"


def test_extract_openai_input_from_args():
    payload = _extract_openai_input(("arg",), {"model": "m"})
    assert payload["args"] == ("arg",)


def test_extract_tool_names_from_dict_tools():
    names = _extract_tool_names({"tools": [{"function": {"name": "search"}}, {"function": {"name": "calc"}}]})
    assert names == ["search", "calc"]


def test_extract_tool_names_none_when_missing():
    assert _extract_tool_names({}) is None


def test_extract_input_text_from_messages():
    text = _extract_input_text({"messages": [{"content": "a"}, {"content": "b"}]})
    assert "a" in text and "b" in text


def test_extract_usage_from_dict():
    total, usage = _extract_usage({"usage": {"total_tokens": 99}})
    assert total == 99
    assert usage == {"total_tokens": 99}


def test_extract_usage_from_object():
    resp = SimpleNamespace(usage=SimpleNamespace(total_tokens=12))
    total, _ = _extract_usage(resp)
    assert total == 12


def test_safe_output_model_dump():
    class _M:
        def model_dump(self):
            return {"ok": True}

    assert _safe_output(_M()) == {"ok": True}


def test_safe_output_primitives():
    assert _safe_output({"x": 1}) == {"x": 1}
    assert _safe_output(None) is None


class _FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(total_tokens=50),
            model_dump=lambda: {"usage": {"total_tokens": 50}},
        )


class _FakeResponses:
    def create(self, **kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(total_tokens=25),
            model_dump=lambda: {"usage": {"total_tokens": 25}},
        )


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()
        self.responses = _FakeResponses()


class _FakeAnthropicMessages:
    def create(self, **kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            model_dump=lambda: {"usage": {"input_tokens": 10, "output_tokens": 20}},
        )


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeAnthropicMessages()


def test_extract_tool_names_non_dict_tools():
    names = _extract_tool_names({"tools": ["raw-tool", 42]})
    assert "raw-tool" in names


def test_extract_tool_names_exception_returns_none():
    class _Bad:
        def __iter__(self):
            raise RuntimeError("bad tools")

    assert _extract_tool_names({"tools": _Bad()}) is None


def test_extract_anthropic_usage_from_object():
    resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=5, output_tokens=15))
    total, usage = _extract_anthropic_usage(resp)
    assert total == 20
    assert usage is not None


def test_patch_anthropic_emits_model_call():
    sink = MemorySink()
    client = _FakeAnthropicClient()
    undo = patch_anthropic(
        client,
        org_id="org",
        project_id="proj",
        agent_id="agent",
        event_sink=sink,
        usd_per_1k_tokens=0.01,
    )
    client.messages.create(model="claude-3-haiku", messages=[{"role": "user", "content": "hi"}])
    undo()
    assert any(e.event_type == "model.call" for e in sink.events)


def test_patch_anthropic_unsupported_client():
    with pytest.raises(TypeError, match="Unsupported client"):
        patch_anthropic(object(), org_id="o", project_id="p", agent_id="a")


def test_patch_openai_emits_model_call():
    sink = MemorySink()
    client = _FakeClient()
    finops = FinOpsController(cfg=FinOpsConfig(budget_usd=100.0))
    undo = patch_openai(
        client,
        org_id="org",
        project_id="proj",
        agent_id="agent",
        event_sink=sink,
        finops=finops,
        usd_per_1k_tokens=0.01,
    )
    client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    undo()
    assert len(sink.events) >= 2
    types = {e.event_type for e in sink.events}
    assert "model.call" in types
    assert "finops.trajectory" in types


def test_patch_openai_patches_responses_api():
    client = _FakeClient()
    sink = MemorySink()
    patch_openai(client, org_id="o", project_id="p", agent_id="a", event_sink=sink)
    client.responses.create(input="hi")
    assert any(e.event_type == "model.call" for e in sink.events)


def test_patch_openai_undo_restores():
    client = _FakeClient()
    sink = MemorySink()
    undo = patch_openai(client, org_id="o", project_id="p", agent_id="a", event_sink=sink)
    client.chat.completions.create(messages=[])
    assert len(sink.events) >= 1
    undo()
    sink2 = MemorySink()
    client.chat.completions.create(messages=[])
    assert len(sink2.events) == 0


def test_patch_openai_unsupported_client():
    with pytest.raises(TypeError, match="Unsupported client"):
        patch_openai(object(), org_id="o", project_id="p", agent_id="a")


def test_patch_openai_with_debug_and_endpoint_url(monkeypatch):
    client = _FakeClient()
    sink = MemorySink()
    patch_openai(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        event_sink=sink,
        debug=True,
        endpoint_url="http://127.0.0.1:9999/ingest",
    )
    client.chat.completions.create(messages=[{"role": "user", "content": "x"}])
    assert sink.events


def test_detect_client_kind_openai():
    assert _detect_client_kind(_FakeClient()) == "openai"


def test_detect_client_kind_anthropic():
    assert _detect_client_kind(_FakeAnthropicClient()) == "anthropic"


def test_detect_client_kind_unsupported():
    with pytest.raises(TypeError, match="Unsupported client"):
        _detect_client_kind(object())


def test_monitor_entrypoint_uses_finops_ingest_env(monkeypatch):
    monkeypatch.delenv("KAZENAI_INGEST_URL", raising=False)
    monkeypatch.delenv("KAZENAI_API_KEY", raising=False)
    monkeypatch.setenv("KAZENAI_FINOPS_INGEST_URL", "http://127.0.0.1:8090")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "finops-key")
    client = _FakeClient()

    monitor(client, org_id="o", project_id="p", agent_id="a", max_budget_usd=100.0)
    client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])


def test_monitor_entrypoint_wraps_client(monkeypatch):
    monkeypatch.delenv("KAZENAI_INGEST_URL", raising=False)
    monkeypatch.delenv("KAZENAI_API_KEY", raising=False)
    monkeypatch.delenv("KAZENAI_FINOPS_INGEST_URL", raising=False)
    monkeypatch.delenv("KAZENAI_FINOPS_API_KEY", raising=False)
    client = _FakeClient()

    monitor(client, org_id="o", project_id="p", agent_id="a", max_budget_usd=100.0)
    client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])


def test_monitor_entrypoint_wraps_anthropic_client(monkeypatch, tmp_path):
    monkeypatch.delenv("KAZENAI_FINOPS_INGEST_URL", raising=False)
    monkeypatch.delenv("KAZENAI_FINOPS_API_KEY", raising=False)
    timeline = tmp_path / "timeline.jsonl"
    client = _FakeAnthropicClient()

    monitor(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        max_budget_usd=100.0,
        timeline_path=str(timeline),
    )
    client.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert timeline.exists()
    assert "model.call" in timeline.read_text()


def test_patch_openai_loop_detection_raises():
    from kazenai.enforcement import LoopDetected
    from kazenai.loop_detector import LoopDetector

    client = _FakeClient()
    detector = LoopDetector(h1_threshold=0.0)
    undo = patch_openai(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        loop_detector=detector,
        loop_block_threshold=0.0,
    )
    try:
        client.chat.completions.create(messages=[{"role": "user", "content": "repeat same text"}])
        with pytest.raises(LoopDetected):
            client.chat.completions.create(messages=[{"role": "user", "content": "repeat same text"}])
    finally:
        undo()


def test_patch_openai_budget_enforcement_blocks():
    from kazenai.enforcement import Enforcement

    sink = MemorySink()
    client = _FakeClient()
    enforcement = Enforcement(max_cost_usd=0.0)
    enforcement.record_call(cost_usd=0.01)
    patch_openai(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        enforcement=enforcement,
        event_sink=sink,
    )
    with pytest.raises(BudgetExceeded):
        client.chat.completions.create(messages=[])
