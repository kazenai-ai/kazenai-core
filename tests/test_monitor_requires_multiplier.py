"""monitor() refuses to start under Control without the tier multiplier.

Importing never raises (tests/test_import_never_raises_under_control.py). The
setup call is where a missing KAZENAI_TIER_MISMATCH_MULTIPLIER surfaces, with
the variable and the docs page named, before any provider traffic.
"""

from __future__ import annotations

import importlib

import openai
import pytest

from kazenai import monitor
from kazenai.price_catalog import PricingConfigError


def _client() -> openai.OpenAI:
    # Control refuses hidden provider retries; customers construct clients the same way.
    return openai.OpenAI(api_key="sk-test", base_url="https://api.openai.com/v1", max_retries=0)


@pytest.fixture
def control_env(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "https://finops.example.test")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "sdk-key")
    monkeypatch.delenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", raising=False)


def test_control_without_multiplier_warns_once_then_fails_with_guidance(control_env, monkeypatch):
    messages: list[str] = []

    class CapturingLog:
        def warning(self, message, *args, **kwargs):
            del args, kwargs
            messages.append(str(message))

        def info(self, *args, **kwargs):
            del args, kwargs

    monitor_module = importlib.import_module("kazenai.monitor")
    monkeypatch.setattr(monitor_module, "_monitor_log", lambda: CapturingLog())
    with pytest.raises(PricingConfigError) as excinfo:
        monitor(_client())
    message = str(excinfo.value)
    assert "KAZENAI_TIER_MISMATCH_MULTIPLIER" in message
    assert "docs.kazenai.com/reference/configuration" in message
    assert len(messages) == 1
    assert "KAZENAI_TIER_MISMATCH_MULTIPLIER" in messages[0]
    assert "docs.kazenai.com/reference/configuration" in messages[0]


def test_control_with_multiplier_starts(control_env, monkeypatch):
    monkeypatch.setenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", "1.5")
    assert monitor(_client()) is not None


def test_without_finops_the_multiplier_is_not_required(monkeypatch):
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_TIER_MISMATCH_MULTIPLIER"):
        monkeypatch.delenv(key, raising=False)
    assert monitor(_client()) is not None
