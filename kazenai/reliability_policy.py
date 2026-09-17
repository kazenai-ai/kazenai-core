"""Autonomy-tier reliability policy — uniform fail-open / fail-closed decisions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from kazenai.deployment import is_saas_deployment


class AutonomyTier(str, Enum):
    SUGGEST = "suggest"
    DRAFT = "draft"
    EXECUTE_WITH_APPROVAL = "execute_with_approval"
    EXECUTE_WITHIN_BUDGET = "execute_within_budget"
    EXECUTE_CONTINUOUSLY = "execute_continuously"


class ReliabilitySurface(str, Enum):
    ORCHESTRATOR = "orchestrator"
    BRAIN = "brain"
    GATEWAY = "gateway"
    FOUNDER_OS = "founder_os"
    GROWTHOPS = "growthops"
    LENS = "lens"
    PLATFORM = "platform"


class FailureKind(str, Enum):
    CRITIC_ERROR = "critic_error"
    VERIFIER_ERROR = "verifier_error"
    BRAIN_UNREACHABLE = "brain_unreachable"
    ENTITLEMENT_CLIENT_MISSING = "entitlement_client_missing"
    ENTITLEMENT_DENIED = "entitlement_denied"
    FINOPS_UNREACHABLE = "finops_unreachable"
    BUDGET_EXCEEDED = "budget_exceeded"
    OPTIONAL_CONTEXT = "optional_context"
    UPSTREAM_ERROR = "upstream_error"


class ReliabilityAction(str, Enum):
    ESCALATE_HUMAN = "escalate_human"
    BLOCK = "block"
    DEGRADE_VISIBLE = "degrade_visible"
    FAIL_OPEN_LOG = "fail_open_log"


@dataclass(frozen=True)
class ReliabilityContext:
    autonomy_tier: str = AutonomyTier.EXECUTE_WITH_APPROVAL.value
    surface: str = ReliabilitySurface.ORCHESTRATOR.value
    failure_kind: str = FailureKind.OPTIONAL_CONTEXT.value
    difficulty: int = 10
    confidence: float = 0.0
    run_profile: str = ""
    reliability_profile: str = "standard"  # lenient | standard | strict


@dataclass(frozen=True)
class ReliabilityDecision:
    action: ReliabilityAction
    reason: str
    emit_degradation: bool = False


def _normalize_tier(raw: str) -> AutonomyTier:
    try:
        return AutonomyTier(str(raw or "").strip().lower())
    except ValueError:
        return AutonomyTier.EXECUTE_WITH_APPROVAL


def _is_development() -> bool:
    return not is_saas_deployment() and os.getenv("KAZENAI_ENV", "development").strip().lower() in {
        "development",
        "dev",
        "local",
        "",
    }


def _trusted_bypass(ctx: ReliabilityContext) -> bool:
    if ctx.run_profile.lower() != "trusted":
        return False
    return ctx.difficulty <= 7 and ctx.confidence >= 0.9


def resolve(ctx: ReliabilityContext) -> ReliabilityDecision:
    """Return the reliability action for a failure under the given context."""
    tier = _normalize_tier(ctx.autonomy_tier)
    kind = FailureKind(ctx.failure_kind) if ctx.failure_kind in FailureKind._value2member_map_ else FailureKind.OPTIONAL_CONTEXT
    profile = (ctx.reliability_profile or "standard").strip().lower()

    if profile == "lenient" and _is_development():
        return ReliabilityDecision(
            action=ReliabilityAction.FAIL_OPEN_LOG,
            reason=f"lenient_dev:{kind.value}",
        )

    # Budget / entitlement — always block in prod regardless of tier
    if kind in {FailureKind.BUDGET_EXCEEDED, FailureKind.ENTITLEMENT_DENIED}:
        return ReliabilityDecision(action=ReliabilityAction.BLOCK, reason=kind.value)

    if kind == FailureKind.ENTITLEMENT_CLIENT_MISSING:
        if _is_development():
            return ReliabilityDecision(
                action=ReliabilityAction.FAIL_OPEN_LOG,
                reason="entitlement_client_missing_dev",
            )
        return ReliabilityDecision(action=ReliabilityAction.BLOCK, reason="entitlement_client_missing_prod")

    # Safety-critical: critic / verifier failures
    if kind in {FailureKind.CRITIC_ERROR, FailureKind.VERIFIER_ERROR}:
        if tier == AutonomyTier.EXECUTE_CONTINUOUSLY and _trusted_bypass(ctx):
            return ReliabilityDecision(
                action=ReliabilityAction.DEGRADE_VISIBLE,
                reason="trusted_bypass_safety_check",
                emit_degradation=True,
            )
        if tier in {AutonomyTier.SUGGEST, AutonomyTier.DRAFT, AutonomyTier.EXECUTE_WITH_APPROVAL}:
            return ReliabilityDecision(action=ReliabilityAction.ESCALATE_HUMAN, reason=kind.value)
        if tier == AutonomyTier.EXECUTE_WITHIN_BUDGET:
            return ReliabilityDecision(action=ReliabilityAction.ESCALATE_HUMAN, reason=kind.value)
        # execute_continuously without trusted bypass
        return ReliabilityDecision(action=ReliabilityAction.ESCALATE_HUMAN, reason=kind.value)

    # Brain / FinOps unreachable
    if kind == FailureKind.BRAIN_UNREACHABLE:
        if tier in {AutonomyTier.SUGGEST, AutonomyTier.DRAFT, AutonomyTier.EXECUTE_WITH_APPROVAL}:
            if profile == "strict":
                return ReliabilityDecision(action=ReliabilityAction.ESCALATE_HUMAN, reason=kind.value)
            return ReliabilityDecision(
                action=ReliabilityAction.DEGRADE_VISIBLE,
                reason=kind.value,
                emit_degradation=True,
            )
        if tier == AutonomyTier.EXECUTE_WITHIN_BUDGET:
            return ReliabilityDecision(
                action=ReliabilityAction.DEGRADE_VISIBLE,
                reason=kind.value,
                emit_degradation=True,
            )
        return ReliabilityDecision(
            action=ReliabilityAction.DEGRADE_VISIBLE,
            reason=kind.value,
            emit_degradation=True,
        )

    if kind == FailureKind.FINOPS_UNREACHABLE:
        if _is_development():
            return ReliabilityDecision(action=ReliabilityAction.FAIL_OPEN_LOG, reason=kind.value)
        return ReliabilityDecision(action=ReliabilityAction.BLOCK, reason=kind.value)

    if kind == FailureKind.OPTIONAL_CONTEXT:
        return ReliabilityDecision(
            action=ReliabilityAction.DEGRADE_VISIBLE,
            reason=kind.value,
            emit_degradation=True,
        )

    if kind == FailureKind.UPSTREAM_ERROR:
        if _is_development():
            return ReliabilityDecision(action=ReliabilityAction.FAIL_OPEN_LOG, reason=kind.value)
        return ReliabilityDecision(action=ReliabilityAction.BLOCK, reason=kind.value)

    return ReliabilityDecision(
        action=ReliabilityAction.DEGRADE_VISIBLE,
        reason="default",
        emit_degradation=True,
    )


def parse_autonomy_tier_from_headers(headers: dict) -> str:
    """Extract autonomy tier from request headers (case-insensitive)."""
    lowered = {str(k).lower(): v for k, v in headers.items()}
    for key in ("x-kazen-autonomy-tier", "x-kazenai-autonomy-tier"):
        raw = lowered.get(key)
        if raw:
            return str(raw).strip().lower()
    return os.getenv("KAZENAI_AUTONOMY_TIER", AutonomyTier.EXECUTE_WITH_APPROVAL.value)


def should_block(decision: ReliabilityDecision) -> bool:
    return decision.action == ReliabilityAction.BLOCK


def should_escalate(decision: ReliabilityDecision) -> bool:
    return decision.action == ReliabilityAction.ESCALATE_HUMAN
