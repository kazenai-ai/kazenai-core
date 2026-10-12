"""Effective budget lifecycle configuration — single validated resolver.

Replaces scattered ``CONTROL_PROFILE`` / ``STRICT_BUDGET`` truthy checks so
Control (reservation/settlement) is never silently confused with a fail-closed
legacy check/reconcile path.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

CONFIG_VERSION = "control-config.v1"
LIFECYCLE_VERSION_CONTROL = "reservation.lifecycle.v1"
LIFECYCLE_VERSION_LEGACY = "legacy.check_reconcile.v1"

# Precedence for profile selection (most specific first).
_PROFILE_ENV_KEYS: tuple[str, ...] = (
    "KAZENAI_FINOPS_CONTROL_PROFILE",
    "KAZENAI_CONTROL_PROFILE",
    "KAZENAI_PROFILE",
)
_STRICT_ENV_KEY = "KAZENAI_FINOPS_STRICT_BUDGET"
_MIGRATION_ALLOW_AMBIGUOUS = "KAZENAI_LIFECYCLE_MIGRATION_ALLOW_AMBIGUOUS"

_CONTROL_ENABLE = frozenset({"1", "true", "yes", "control"})
_CONTROL_DISABLE = frozenset({"0", "false", "no", "off", "legacy", "disabled"})
_STRICT_ENABLE = frozenset({"1", "true", "yes", "strict"})
_STRICT_DISABLE = frozenset({"0", "false", "no", "off", "disabled"})


class LifecycleConfigError(ValueError):
    """Raised when lifecycle env aliases conflict or are invalid."""

    def __init__(self, message: str, *, category: str = "conflict") -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class EffectiveLifecycleConfig:
    profile: str  # control | legacy
    accounting_mode: str  # reservation_settlement | legacy_check_reconcile
    failure_policy: str  # fail_closed | fail_open
    accounting_owner: str  # sdk | server
    lifecycle_version: str
    config_version: str = CONFIG_VERSION
    source: str = "unset"
    sources: Mapping[str, str] = field(default_factory=dict)
    deprecations: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @property
    def uses_reservation_settlement(self) -> bool:
        return self.profile == "control" and self.accounting_mode == "reservation_settlement"

    @property
    def is_control(self) -> bool:
        return self.profile == "control"

    def to_public_dict(self) -> dict[str, Any]:
        """Sanitized snapshot for logs/diagnostics (no secrets / tenant IDs)."""
        return {
            "profile": self.profile,
            "accounting_mode": self.accounting_mode,
            "failure_policy": self.failure_policy,
            "accounting_owner": self.accounting_owner,
            "enforcement_owner": map_accounting_owner_to_enforcement(self.accounting_owner),
            "lifecycle_version": self.lifecycle_version,
            "config_version": self.config_version,
            "source": self.source,
            "sources": dict(self.sources),
            "deprecations": list(self.deprecations),
            "diagnostics": list(self.diagnostics),
        }

    def to_receipt_block(self) -> dict[str, Any]:
        """Compact block for decision-receipt ``accounting.effective_lifecycle``."""
        return {
            "profile": self.profile,
            "accounting_mode": self.accounting_mode,
            "failure_policy": self.failure_policy,
            # Resolver vocabulary: sdk|server. Enforcement env uses sdk|gateway.
            "accounting_owner": self.accounting_owner,
            "enforcement_owner": map_accounting_owner_to_enforcement(self.accounting_owner),
            "lifecycle_version": self.lifecycle_version,
            "config_version": self.config_version,
            "source": self.source,
        }


def map_accounting_owner_to_enforcement(owner: str) -> str:
    """Map lifecycle ``sdk|server`` to enforcement ``sdk|gateway``."""
    value = (owner or "").strip().lower()
    if value in {"server", "gateway"}:
        return "gateway"
    return "sdk"


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _classify_profile_token(raw: str) -> Optional[str]:
    """Return 'control', 'legacy', or None if unset/unknown."""
    if not raw:
        return None
    if raw in _CONTROL_ENABLE:
        return "control"
    if raw in _CONTROL_DISABLE:
        return "legacy"
    # Unknown non-empty values are treated as explicit requests to fail closed.
    raise LifecycleConfigError(
        f"invalid_lifecycle_profile_value: {raw!r}",
        category="invalid",
    )


def _classify_strict_token(raw: str) -> Optional[bool]:
    if not raw:
        return None
    if raw in _STRICT_ENABLE:
        return True
    if raw in _STRICT_DISABLE:
        return False
    raise LifecycleConfigError(
        f"invalid_strict_budget_value: {raw!r}",
        category="invalid",
    )


def _env_map(environ: Optional[Mapping[str, str]] = None) -> Mapping[str, str]:
    if environ is None:
        return os.environ
    return environ


def resolve_effective_lifecycle(
    environ: Optional[Mapping[str, str]] = None,
    *,
    allow_ambiguous: Optional[bool] = None,
    accounting_owner: str = "sdk",
    deployment_mode: Optional[str] = None,
) -> EffectiveLifecycleConfig:
    """Resolve one effective lifecycle configuration from env aliases.

    Precedence for Control profile selection:
      1. ``KAZENAI_FINOPS_CONTROL_PROFILE``
      2. ``KAZENAI_CONTROL_PROFILE``
      3. ``KAZENAI_PROFILE``

    ``KAZENAI_FINOPS_STRICT_BUDGET`` only selects failure policy for the legacy
    path; it does **not** enable reservation/settlement by itself.

    Conflicting aliases raise ``LifecycleConfigError`` unless
    ``KAZENAI_LIFECYCLE_MIGRATION_ALLOW_AMBIGUOUS`` is truthy (or
    ``allow_ambiguous=True``).
    """
    env = _env_map(environ)
    if allow_ambiguous is None:
        allow_ambiguous = _norm(env.get(_MIGRATION_ALLOW_AMBIGUOUS)) in {
            "1",
            "true",
            "yes",
        }

    sources: dict[str, str] = {}
    classified: list[tuple[str, str]] = []
    deprecations: list[str] = []
    diagnostics: list[str] = []

    for key in _PROFILE_ENV_KEYS:
        raw = _norm(env.get(key))
        if not raw:
            continue
        sources[key] = raw
        label = _classify_profile_token(raw)
        if label is None:
            continue
        classified.append((key, label))
        if key != "KAZENAI_FINOPS_CONTROL_PROFILE":
            deprecations.append(
                f"{key} is a compatible alias; prefer KAZENAI_FINOPS_CONTROL_PROFILE"
            )

    strict_raw = _norm(env.get(_STRICT_ENV_KEY))
    strict_enabled: Optional[bool] = None
    if strict_raw:
        sources[_STRICT_ENV_KEY] = strict_raw
        strict_enabled = _classify_strict_token(strict_raw)
        deprecations.append(
            "KAZENAI_FINOPS_STRICT_BUDGET selects failure_policy only; "
            "it does not enable reservation.lifecycle.v1"
        )

    # Conflict: two profile aliases disagree.
    labels = {label for _, label in classified}
    if len(labels) > 1 and not allow_ambiguous:
        detail = ", ".join(f"{k}={v}" for k, v in classified)
        raise LifecycleConfigError(
            "lifecycle_profile_conflict: aliases disagree "
            f"({detail}). Set a single Control/legacy value or "
            f"{_MIGRATION_ALLOW_AMBIGUOUS}=1 during migration.",
            category="conflict",
        )

    # Winning profile: first in precedence that is set.
    profile = "legacy"
    source = "unset"
    for key in _PROFILE_ENV_KEYS:
        raw = sources.get(key)
        if not raw:
            continue
        profile = _classify_profile_token(raw) or "legacy"
        source = key
        break

    if profile == "control":
        accounting_mode = "reservation_settlement"
        lifecycle_version = LIFECYCLE_VERSION_CONTROL
        # Control defaults fail-closed; STRICT_BUDGET cannot downgrade it.
        failure_policy = "fail_closed"
        if strict_enabled is False and not allow_ambiguous:
            raise LifecycleConfigError(
                "lifecycle_failure_policy_conflict: Control profile requires "
                "fail_closed but KAZENAI_FINOPS_STRICT_BUDGET disables strict mode. "
                f"Remove the conflict or set {_MIGRATION_ALLOW_AMBIGUOUS}=1.",
                category="conflict",
            )
        if strict_enabled is False and allow_ambiguous:
            diagnostics.append(
                "ambiguous_migration: Control profile retained fail_closed despite "
                "STRICT_BUDGET disable"
            )
        # Fold explicit legacy fail_open into the *same* effective policy
        # that runtime reads — never leave diagnostics disagreeing with execution.
        enf = _norm(env.get("KAZENAI_ENFORCEMENT_MODE"))
        res = _norm(env.get("KAZENAI_FINOPS_RESERVATION_MODE"))
        if enf == "fail_open" or res == "fail_open":
            failure_policy = "fail_open"
            diagnostics.append(
                "effective_failure_policy=fail_open: ENFORCEMENT_MODE/"
                "RESERVATION_MODE=fail_open applied under Control profile"
            )
    else:
        accounting_mode = "legacy_check_reconcile"
        lifecycle_version = LIFECYCLE_VERSION_LEGACY
        if strict_enabled is True:
            failure_policy = "fail_closed"
            if source == "unset":
                source = _STRICT_ENV_KEY
        else:
            # Unset / explicit false → fail_open for legacy unless production forces closed.
            mode = _norm(deployment_mode or env.get("KAZENAI_DEPLOYMENT_MODE") or env.get("KAZENAI_ENV"))
            if mode in {"saas", "production", "prod", "staging"} and strict_enabled is not False:
                failure_policy = "fail_closed"
                diagnostics.append(
                    "production_default: legacy profile uses fail_closed in production-like modes"
                )
            else:
                failure_policy = "fail_open"
        # Explicit mode env is part of the effective snapshot.
        enf = _norm(env.get("KAZENAI_ENFORCEMENT_MODE"))
        res = _norm(env.get("KAZENAI_FINOPS_RESERVATION_MODE"))
        if enf in {"fail_open", "fail_closed"}:
            failure_policy = enf
        elif res in {"fail_open", "fail_closed"}:
            failure_policy = res

    owner = (accounting_owner or "sdk").strip().lower() or "sdk"
    # Accept gateway as an alias for server (gateway-owned Control).
    if owner == "gateway":
        owner = "server"
    if owner not in {"sdk", "server"}:
        raise LifecycleConfigError(
            f"invalid_accounting_owner: {owner!r}",
            category="invalid",
        )

    return EffectiveLifecycleConfig(
        profile=profile,
        accounting_mode=accounting_mode,
        failure_policy=failure_policy,
        accounting_owner=owner,
        lifecycle_version=lifecycle_version,
        config_version=CONFIG_VERSION,
        source=source,
        sources=sources,
        deprecations=tuple(dict.fromkeys(deprecations)),
        diagnostics=tuple(diagnostics),
    )


def control_profile_active(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True when the effective profile is Control (reservation/settlement)."""
    return resolve_effective_lifecycle(environ).is_control


def lifecycle_budget_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True when callers must use reserve/settle (not legacy check/reconcile)."""
    return resolve_effective_lifecycle(environ).uses_reservation_settlement


class ClientServerLifecycleMismatch(LifecycleConfigError):
    """Client effective lifecycle is incompatible with server capabilities."""

    def __init__(self, message: str, *, client: Mapping[str, Any], server: Mapping[str, Any]) -> None:
        super().__init__(message, category="capability_mismatch")
        self.client = dict(client)
        self.server = dict(server)


def sdk_lifecycle_diagnostics(
    environ: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Sanitized SDK-side lifecycle diagnostics for logs and client tooling."""
    cfg = resolve_effective_lifecycle(environ)
    return {
        "component": "kazenai-sdk",
        "effective_lifecycle": cfg.to_public_dict(),
        "capabilities": {
            "reservation_settlement": cfg.uses_reservation_settlement,
            "legacy_check_reconcile": not cfg.uses_reservation_settlement,
        },
    }


def assert_compatible_with_server(
    server_capabilities: Mapping[str, Any],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> EffectiveLifecycleConfig:
    """Fail closed when the client Control lifecycle cannot be served.

    ``server_capabilities`` is the authenticated FinOps
    ``GET /v1/budget/lifecycle`` (or equivalent) payload / ``capabilities`` block.
    """
    client = resolve_effective_lifecycle(environ)
    caps = server_capabilities.get("capabilities")
    if not isinstance(caps, Mapping):
        caps = server_capabilities
    server_eff = server_capabilities.get("effective_lifecycle")
    if not isinstance(server_eff, Mapping):
        server_eff = {}

    server_supports_control = bool(caps.get("reservation_settlement"))
    server_profile = str(server_eff.get("profile") or ("control" if server_supports_control else "legacy"))
    server_lifecycle = str(
        server_eff.get("lifecycle_version")
        or (
            LIFECYCLE_VERSION_CONTROL
            if server_supports_control
            else LIFECYCLE_VERSION_LEGACY
        )
    )

    if client.uses_reservation_settlement and not server_supports_control:
        raise ClientServerLifecycleMismatch(
            "client_server_lifecycle_mismatch: client requires reservation_settlement "
            "but server capabilities do not advertise it",
            client=client.to_receipt_block(),
            server={
                "profile": server_profile,
                "lifecycle_version": server_lifecycle,
                "reservation_settlement": False,
            },
        )

    if client.uses_reservation_settlement and server_lifecycle not in {
        "",
        LIFECYCLE_VERSION_CONTROL,
    }:
        raise ClientServerLifecycleMismatch(
            "client_server_lifecycle_mismatch: unsupported server lifecycle_version "
            f"{server_lifecycle!r}",
            client=client.to_receipt_block(),
            server={
                "profile": server_profile,
                "lifecycle_version": server_lifecycle,
                "reservation_settlement": server_supports_control,
            },
        )
    return client
