"""Optional paid OpenAI/Anthropic smokes — never run in default CI.

Enable explicitly:

```bash
export KAZENAI_LIVE_PROVIDER_SMOKE=1
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
pytest -m live tests/test_live_provider_smoke.py -q
```

Spend is capped with tiny max_tokens and max_budget_usd. Skip when keys/gate absent.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


def _live_enabled() -> bool:
    return os.getenv("KAZENAI_LIVE_PROVIDER_SMOKE", "").strip() in {"1", "true", "yes", "on"}


def _hermetic_local_env(monkeypatch) -> None:
    # Keep shared FinOps optional; local hard cap still applies.
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _require_live_gate(monkeypatch):
    if not _live_enabled():
        pytest.skip("Set KAZENAI_LIVE_PROVIDER_SMOKE=1 to run paid provider smokes")
    _hermetic_local_env(monkeypatch)


def test_live_openai_chat_completions_smoke(monkeypatch):
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    openai = pytest.importorskip("openai")
    from kazenai import monitor

    client = monitor(
        openai.OpenAI(api_key=key),
        agent_id="live-smoke-openai",
        max_budget_usd=0.05,
        soft_pause_pct=100.0,
        loop_anomaly_threshold=1.0,
        business_subject_ref="smoke:openai",
        feature_id="live_provider_smoke",
        workflow_id="paid_smoke.v1",
    )
    resp = client.chat.completions.create(
        model=os.getenv("KAZENAI_LIVE_OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=8,
    )
    assert resp.choices and resp.choices[0].message.content


def test_live_anthropic_messages_smoke(monkeypatch):
    key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    anthropic = pytest.importorskip("anthropic")
    from kazenai import monitor

    client = monitor(
        anthropic.Anthropic(api_key=key),
        agent_id="live-smoke-anthropic",
        max_budget_usd=0.05,
        soft_pause_pct=100.0,
        loop_anomaly_threshold=1.0,
        business_subject_ref="smoke:anthropic",
        feature_id="live_provider_smoke",
        workflow_id="paid_smoke.v1",
    )
    resp = client.messages.create(
        model=os.getenv("KAZENAI_LIVE_ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        max_tokens=8,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    assert resp.content
