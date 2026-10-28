"""Effective lifecycle resolver."""

from __future__ import annotations

import pytest

from kazenai.lifecycle_config import (
    LifecycleConfigError,
    control_profile_active,
    lifecycle_budget_enabled,
    resolve_effective_lifecycle,
)


def test_unset_legacy_reports_what_runtime_enforces() -> None:
    # Legacy fail_open applies only under KAZENAI_ENV=dev; elsewhere the
    # runtime fails closed and the snapshot says so.
    cfg = resolve_effective_lifecycle({})
    assert cfg.profile == "legacy"
    assert cfg.accounting_mode == "legacy_check_reconcile"
    assert cfg.failure_policy == "fail_closed"
    assert any(d.startswith("fail_open_ignored") for d in cfg.diagnostics)
    assert cfg.source == "unset"
    assert not cfg.uses_reservation_settlement
    dev = resolve_effective_lifecycle({"KAZENAI_ENV": "dev"})
    assert dev.failure_policy == "fail_open"


def test_runtime_policy_matches_the_snapshot(monkeypatch) -> None:
    from kazenai.deployment import enforcement_fail_closed, finops_reservation_fail_closed

    for key in ("KAZENAI_ENV", "KAZENAI_DEPLOYMENT_MODE", "KAZENAI_FINOPS_ENV", "KAZENAI_ENFORCEMENT_MODE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    assert resolve_effective_lifecycle().failure_policy == "fail_closed"
    assert enforcement_fail_closed() is True and finops_reservation_fail_closed() is True
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    assert resolve_effective_lifecycle().failure_policy == "fail_open"
    assert enforcement_fail_closed() is False


def test_conflicting_deployment_modes_raise_at_resolution() -> None:
    with pytest.raises(LifecycleConfigError) as excinfo:
        resolve_effective_lifecycle({"KAZENAI_DEPLOYMENT_MODE": "production", "KAZENAI_FINOPS_ENV": "development"})
    assert "deployment_mode_conflict" in str(excinfo.value)
    # Aliases of the same mode are not a conflict.
    resolve_effective_lifecycle({"KAZENAI_DEPLOYMENT_MODE": "production", "KAZENAI_ENV": "saas"})


def test_control_via_finops_control_profile() -> None:
    cfg = resolve_effective_lifecycle({"KAZENAI_FINOPS_CONTROL_PROFILE": "control"})
    assert cfg.profile == "control"
    assert cfg.accounting_mode == "reservation_settlement"
    assert cfg.failure_policy == "fail_closed"
    assert cfg.lifecycle_version == "reservation.lifecycle.v1"
    assert cfg.config_version == "control-config.v1"
    assert cfg.accounting_owner == "sdk"
    assert lifecycle_budget_enabled({"KAZENAI_FINOPS_CONTROL_PROFILE": "1"})


def test_finops_url_defaults_to_control_without_profile_variable() -> None:
    cfg = resolve_effective_lifecycle(
        {
            "KAZENAI_FINOPS_URL": "https://finops.example.test",
            "KAZENAI_FINOPS_API_KEY": "sdk-key",
        }
    )
    assert cfg.profile == "control"
    assert cfg.uses_reservation_settlement
    assert cfg.failure_policy == "fail_closed"
    assert cfg.source == "default_finops_configured"
    assert cfg.sources["KAZENAI_FINOPS_URL"] == "configured"


def test_ingest_url_alone_also_defaults_to_control() -> None:
    # The guard reserves against KAZENAI_FINOPS_INGEST_URL when it is the only
    # FinOps URL, so lifecycle resolution must select Control for it too.
    cfg = resolve_effective_lifecycle(
        {
            "KAZENAI_FINOPS_INGEST_URL": "https://finops.example.test",
            "KAZENAI_FINOPS_API_KEY": "sdk-key",
        }
    )
    assert cfg.profile == "control"
    assert cfg.source == "default_finops_configured"
    assert cfg.sources["KAZENAI_FINOPS_INGEST_URL"] == "configured"


def test_no_finops_url_keeps_the_unset_default() -> None:
    cfg = resolve_effective_lifecycle({"KAZENAI_FINOPS_API_KEY": "sdk-key"})
    assert cfg.profile == "legacy"
    assert cfg.source == "unset"


def test_explicit_legacy_overrides_finops_default_and_is_deprecated() -> None:
    cfg = resolve_effective_lifecycle(
        {
            "KAZENAI_FINOPS_URL": "https://finops.example.test",
            "KAZENAI_FINOPS_API_KEY": "sdk-key",
            "KAZENAI_PROFILE": "legacy",
        }
    )
    assert cfg.profile == "legacy"
    assert cfg.source == "KAZENAI_PROFILE"
    assert any("explicit legacy lifecycle is deprecated" in item for item in cfg.deprecations)


def test_alias_precedence_and_conflicts() -> None:
    # Disagreeing aliases always conflict (including FINOPS vs PROFILE).
    with pytest.raises(LifecycleConfigError):
        resolve_effective_lifecycle(
            {
                "KAZENAI_FINOPS_CONTROL_PROFILE": "control",
                "KAZENAI_PROFILE": "legacy",
            }
        )
    with pytest.raises(LifecycleConfigError):
        resolve_effective_lifecycle(
            {
                "KAZENAI_CONTROL_PROFILE": "control",
                "KAZENAI_PROFILE": "legacy",
            }
        )
    # Same enabling value via aliases is fine; deprecate non-primary keys.
    cfg = resolve_effective_lifecycle(
        {
            "KAZENAI_FINOPS_CONTROL_PROFILE": "1",
            "KAZENAI_CONTROL_PROFILE": "true",
        }
    )
    assert cfg.is_control
    assert cfg.source == "KAZENAI_FINOPS_CONTROL_PROFILE"
    assert any("KAZENAI_CONTROL_PROFILE" in d for d in cfg.deprecations)


def test_strict_budget_alone_is_not_control_lifecycle() -> None:
    cfg = resolve_effective_lifecycle({"KAZENAI_FINOPS_STRICT_BUDGET": "strict"})
    assert cfg.profile == "legacy"
    assert cfg.failure_policy == "fail_closed"
    assert not cfg.uses_reservation_settlement
    assert not control_profile_active({"KAZENAI_FINOPS_STRICT_BUDGET": "1"})
    assert any("failure_policy only" in d for d in cfg.deprecations)


def test_control_plus_strict_disable_conflicts() -> None:
    with pytest.raises(LifecycleConfigError) as excinfo:
        resolve_effective_lifecycle(
            {
                "KAZENAI_FINOPS_CONTROL_PROFILE": "control",
                "KAZENAI_FINOPS_STRICT_BUDGET": "0",
            }
        )
    assert excinfo.value.category == "conflict"


def test_ambiguous_migration_escape_hatch() -> None:
    cfg = resolve_effective_lifecycle(
        {
            "KAZENAI_CONTROL_PROFILE": "control",
            "KAZENAI_PROFILE": "legacy",
            "KAZENAI_LIFECYCLE_MIGRATION_ALLOW_AMBIGUOUS": "1",
        }
    )
    # Precedence still picks CONTROL_PROFILE when FINOPS key unset
    assert cfg.profile == "control"


def test_production_legacy_defaults_fail_closed() -> None:
    cfg = resolve_effective_lifecycle(
        {"KAZENAI_ENV": "production"},
        deployment_mode="production",
    )
    assert cfg.profile == "legacy"
    assert cfg.failure_policy == "fail_closed"


def test_invalid_profile_value() -> None:
    with pytest.raises(LifecycleConfigError) as excinfo:
        resolve_effective_lifecycle({"KAZENAI_PROFILE": "purple"})
    assert excinfo.value.category == "invalid"


def test_public_dict_and_receipt_block_sanitized() -> None:
    cfg = resolve_effective_lifecycle({"KAZENAI_FINOPS_CONTROL_PROFILE": "control"})
    pub = cfg.to_public_dict()
    assert "profile" in pub and "org_id" not in pub
    block = cfg.to_receipt_block()
    assert block["lifecycle_version"] == "reservation.lifecycle.v1"
