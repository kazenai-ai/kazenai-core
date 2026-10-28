"""Secret hygiene startup guard."""

from __future__ import annotations

import os

import pytest

from kazenai.secret_hygiene import validate_secret_hygiene_startup

_CRITICAL = (
    "KAZEN_MASTER_KEY",
    "KAZENAI_FINOPS_MASTER_KEY",
    "KAZENAI_FINOPS_API_KEY",
    "KAZENAI_PLATFORM_SERVICE_TOKEN",
    "KAZENAI_PLATFORM_FINOPS_CLIENT_TOKEN",
    "KAZENAI_PLATFORM_LENS_CLIENT_TOKEN",
    "KAZENAI_LENS_FINOPS_OUTBOX_KEY",
)
_PLATFORM_ONLY = (
    "SUPABASE_JWT_SECRET",
    "KAZENAI_SUPABASE_JWT_SECRET",
    "KAZENAI_JWT_SECRET",
    "KAZENAI_PLATFORM_JWT_SECRET",
    "JWT_SECRET",
    "KAZENAI_SESSION_JWT_PRIVATE_KEY",
)


@pytest.fixture(autouse=True)
def _clear_secret_env(monkeypatch):
    for key in (*_CRITICAL, *_PLATFORM_ONLY, "STRIPE_SECRET_KEY", "KAZENAI_DEPLOYMENT_MODE", "KAZENAI_ENV", "KAZENAI_FINOPS_ENV"):
        monkeypatch.delenv(key, raising=False)


def _strong(prefix: str) -> str:
    return f"{prefix}-test-strong-secret-value"


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
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_live_example_fake_key_for_test_only")
    with pytest.raises(SystemExit):
        validate_secret_hygiene_startup(service_name="test")


def test_allows_live_stripe_in_production(monkeypatch):
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_live_example_fake_key_for_test_only")
    validate_secret_hygiene_startup(service_name="test")


def test_rejects_short_critical_secret(monkeypatch):
    monkeypatch.setenv("KAZEN_MASTER_KEY", "too-short-key")
    with pytest.raises(SystemExit):
        validate_secret_hygiene_startup(service_name="test")


@pytest.mark.parametrize("name", _PLATFORM_ONLY)
def test_downstream_services_refuse_platform_signing_secrets_in_production(monkeypatch, name):
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv(name, "a-perfectly-strong-but-misplaced-secret")
    with pytest.raises(SystemExit):
        validate_secret_hygiene_startup(service_name="finops")
    # Platform holds them by design.
    validate_secret_hygiene_startup(service_name="platform", holds_signing_secrets=True)


def test_signing_secrets_allowed_outside_production(monkeypatch):
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "a-perfectly-strong-local-dev-secret")
    validate_secret_hygiene_startup(service_name="finops")
