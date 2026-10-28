"""0.11 — one deployment mode resolver for services and the SDK."""

from __future__ import annotations

import pytest

from kazenai.deployment import (
    DeploymentModeConflict,
    is_production_like,
    resolve_deployment_mode,
    validate_deployment_mode_startup,
)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, "development"),
        ({"KAZENAI_DEPLOYMENT_MODE": "production"}, "production"),
        ({"KAZENAI_ENV": "saas"}, "production"),
        ({"KAZENAI_ENV": "prod"}, "production"),
        ({"KAZENAI_FINOPS_ENV": "staging"}, "staging"),
        ({"KAZENAI_ENV": "dev"}, "development"),
        ({"KAZENAI_DEPLOYMENT_MODE": "test"}, "test"),
        ({"KAZENAI_DEPLOYMENT_MODE": "production", "KAZENAI_ENV": "saas", "KAZENAI_FINOPS_ENV": "prod"}, "production"),
    ],
)
def test_aliases_resolve_to_canonical_modes(env, expected):
    assert resolve_deployment_mode(env) == expected
    assert resolve_deployment_mode(env, strict=True) == expected


def test_lenient_resolution_picks_the_most_restrictive_mode():
    env = {"KAZENAI_DEPLOYMENT_MODE": "development", "KAZENAI_FINOPS_ENV": "production"}
    assert resolve_deployment_mode(env) == "production"
    assert is_production_like(env)
    assert resolve_deployment_mode({"KAZENAI_ENV": "sandbox"}) == "production"


@pytest.mark.parametrize(
    "env",
    [
        {"KAZENAI_DEPLOYMENT_MODE": "development", "KAZENAI_FINOPS_ENV": "production"},
        {"KAZENAI_DEPLOYMENT_MODE": "staging", "KAZENAI_ENV": "production"},
        {"KAZENAI_ENV": "sandbox"},
    ],
)
def test_strict_resolution_refuses_conflicts_and_unknown_values(env):
    with pytest.raises(DeploymentModeConflict):
        resolve_deployment_mode(env, strict=True)


def test_services_refuse_to_start_on_conflict(monkeypatch):
    for key in ("KAZENAI_DEPLOYMENT_MODE", "KAZENAI_ENV", "KAZENAI_FINOPS_ENV"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("KAZENAI_FINOPS_ENV", "development")
    with pytest.raises(SystemExit):
        validate_deployment_mode_startup("finops")
    monkeypatch.setenv("KAZENAI_FINOPS_ENV", "production")
    assert validate_deployment_mode_startup("finops") == "production"


def test_sdk_import_never_raises_on_conflict(monkeypatch):
    import importlib

    import kazenai

    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("KAZENAI_FINOPS_ENV", "development")
    importlib.reload(kazenai)
