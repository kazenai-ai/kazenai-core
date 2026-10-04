"""model.call timeline payloads must carry commercial attribution by default."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from kazenai import monitor


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, *args, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )


class _FakeOpenAI:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def _hermetic(monkeypatch) -> None:
    for key in (
        "KAZENAI_FINOPS_URL",
        "KAZENAI_FINOPS_INGEST_URL",
        "KAZENAI_INGEST_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")


def test_model_call_payload_includes_attribution(monkeypatch, tmp_path: Path) -> None:
    _hermetic(monkeypatch)
    timeline = tmp_path / "timeline.jsonl"
    client = _FakeOpenAI()
    monitored = monitor(
        client,
        org_id="local",
        project_id="default",
        workspace_id="default",
        agent_id="attr-agent",
        max_budget_usd=1.0,
        soft_pause_pct=100.0,
        loop_anomaly_threshold=1.0,
        timeline_path=str(timeline),
        business_subject_ref="cust:acme-42",
        feature_id="support_assistant",
        workflow_id="customer_support_reply.v1",
    )
    monitored.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "attribution payload check"}],
    )
    assert client.chat.completions.calls == 1
    rows = [json.loads(line) for line in timeline.read_text().splitlines() if line.strip()]
    model_calls = [r for r in rows if r.get("event_type") == "model.call"]
    assert model_calls, "expected model.call in timeline"
    payload = model_calls[0].get("payload") or {}
    assert payload.get("business_subject_ref") == "cust:acme-42"
    assert payload.get("feature_id") == "support_assistant"
    assert payload.get("workflow_id") == "customer_support_reply.v1"
