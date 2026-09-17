"""Loop 25 P0-1 — secret hygiene startup guard."""

from __future__ import annotations

import os

import pytest

from kazenai.secret_hygiene import validate_secret_hygiene_startup

_CRITICAL = (
    "KAZEN_MASTER_KEY",
    "KAZENAI_FINOPS_MASTER_KEY",
    "KAZENAI_FINOPS_API_KEY",
    "KAZENAI_PLATFORM_SERVICE_TOKEN",
)


@pytest.fixture(autouse=True)
def _clear_secret_env(monkeypatch):
    for key in (*_CRITICAL, "STRIPE_SECRET_KEY", "KAZENAI_DEPLOYMENT_MODE", "KAZENAI_ENV"):
        monkeypatch.delenv(key, raising=False)


def _strong(prefix: str) -> str:
    return f"{prefix}-loop25-strong-secret-value"


def test_passes_with_distinct_strong_secrets(monkeypatch):
    monkeypatch.setenv("KAZEN_MASTER_KEY", _strong("master"))
    monkeypatch.setenv("KAZENAI_FINOPS_MASTER_KEY", _strong("finops-master"))
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", _strong("finops-api"))
    monkeypatch.setenv("KAZENAI_PLATFORM_SERVICE_TOKEN", _strong("platform"))
    validate_secret_hygiene_startup(service_name="test")


def test_rejects_reused_critical_secrets(monkeypatch):
    shared = _strong("shared-reuse")
    monkeypatch.setenv("KAZEN_MASTER_KEY", shared)
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", shared)
    with pytest.raises(SystemExit):
        validate_secret_hygiene_startup(service_name="test")


def test_rejects_placeholder_secret(monkeypatch):
    monkeypatch.setenv("KAZEN_MASTER_KEY", "test-signing-key")
    with pytest.raises(SystemExit):
        validate_secret_hygiene_startup(service_name="test")


def test_rejects_live_stripe_outside_production(monkeypatch):
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_live_loop25_fake_key_for_test_only")
    with pytest.raises(SystemExit):
        validate_secret_hygiene_startup(service_name="test")


def test_allows_live_stripe_in_production(monkeypatch):
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_live_loop25_fake_key_for_test_only")
    validate_secret_hygiene_startup(service_name="test")


def test_rejects_short_critical_secret(monkeypatch):
    monkeypatch.setenv("KAZEN_MASTER_KEY", "too-short-key")
    with pytest.raises(SystemExit):
        validate_secret_hygiene_startup(service_name="test")
