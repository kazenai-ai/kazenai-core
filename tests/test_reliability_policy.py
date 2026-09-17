"""Tests for autonomy-tier reliability policy."""

from __future__ import annotations

import os

import pytest

from kazenai.reliability_policy import (
    AutonomyTier,
    FailureKind,
    ReliabilityAction,
    ReliabilityContext,
    ReliabilitySurface,
    resolve,
    should_escalate,
)


@pytest.fixture(autouse=True)
def _dev_env(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENV", "development")
    monkeypatch.delenv("KAZENAI_DEPLOYMENT_MODE", raising=False)


def test_critic_error_escalates_under_execute_with_approval():
    d = resolve(
        ReliabilityContext(
            autonomy_tier=AutonomyTier.EXECUTE_WITH_APPROVAL.value,
            failure_kind=FailureKind.CRITIC_ERROR.value,
        )
    )
    assert d.action == ReliabilityAction.ESCALATE_HUMAN
    assert should_escalate(d)


def test_critic_error_never_approves_in_prod_continuous_without_trusted(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENV", "production")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "production")
    d = resolve(
        ReliabilityContext(
            autonomy_tier=AutonomyTier.EXECUTE_CONTINUOUSLY.value,
            failure_kind=FailureKind.CRITIC_ERROR.value,
            difficulty=8,
            confidence=0.5,
        )
    )
    assert d.action == ReliabilityAction.ESCALATE_HUMAN


def test_entitlement_missing_blocks_in_prod(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENV", "production")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "saas")
    d = resolve(
        ReliabilityContext(failure_kind=FailureKind.ENTITLEMENT_CLIENT_MISSING.value)
    )
    assert d.action == ReliabilityAction.BLOCK


def test_optional_context_degrades_visible():
    d = resolve(
        ReliabilityContext(
            autonomy_tier=AutonomyTier.EXECUTE_WITH_APPROVAL.value,
            failure_kind=FailureKind.OPTIONAL_CONTEXT.value,
            surface=ReliabilitySurface.FOUNDER_OS.value,
        )
    )
    assert d.action == ReliabilityAction.DEGRADE_VISIBLE
    assert d.emit_degradation is True


def test_trusted_bypass_allows_degrade_on_critic_in_continuous():
    d = resolve(
        ReliabilityContext(
            autonomy_tier=AutonomyTier.EXECUTE_CONTINUOUSLY.value,
            failure_kind=FailureKind.CRITIC_ERROR.value,
            run_profile="trusted",
            difficulty=5,
            confidence=0.95,
        )
    )
    assert d.action == ReliabilityAction.DEGRADE_VISIBLE
