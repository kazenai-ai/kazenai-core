"""FINAL_1_b_final P3 — effective lifecycle resolver."""

from __future__ import annotations

import pytest

from kazenai.lifecycle_config import (
    LifecycleConfigError,
    control_profile_active,
    lifecycle_budget_enabled,
    resolve_effective_lifecycle,
)


def test_unset_defaults_to_legacy_fail_open() -> None:
    cfg = resolve_effective_lifecycle({})
    assert cfg.profile == "legacy"
    assert cfg.accounting_mode == "legacy_check_reconcile"
    assert cfg.failure_policy == "fail_open"
    assert cfg.source == "unset"
    assert not cfg.uses_reservation_settlement


def test_control_via_finops_control_profile() -> None:
    cfg = resolve_effective_lifecycle({"KAZENAI_FINOPS_CONTROL_PROFILE": "control"})
    assert cfg.profile == "control"
    assert cfg.accounting_mode == "reservation_settlement"
    assert cfg.failure_policy == "fail_closed"
    assert cfg.lifecycle_version == "reservation.lifecycle.v1"
    assert cfg.config_version == "control-config.v1"
    assert cfg.accounting_owner == "sdk"
    assert lifecycle_budget_enabled({"KAZENAI_FINOPS_CONTROL_PROFILE": "1"})


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
