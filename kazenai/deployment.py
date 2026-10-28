"""Shared deployment-mode helpers for fail-safe production defaults."""

from __future__ import annotations

import os


from typing import Mapping, Optional

# M10/M13 (0.11): one deployment mode for every service and the SDK.
#
#   KAZENAI_DEPLOYMENT_MODE  canonical: development | test | staging | production
#   KAZENAI_ENV, KAZENAI_FINOPS_ENV, KAZENAI_AGENTLENS_ENV  legacy signals, same aliases
#
# Normal calls never raise: on conflicting signals they use the most restrictive
# mode. Services call validate_deployment_mode_startup() and refuse to start on a
# conflict or an unknown value; the SDK raises LifecycleConfigError at monitor().
CANONICAL_MODES = ("development", "test", "staging", "production")
MODE_ENV_VARS = ("KAZENAI_DEPLOYMENT_MODE", "KAZENAI_ENV", "KAZENAI_FINOPS_ENV", "KAZENAI_AGENTLENS_ENV")
_MODE_ALIASES = {
    "development": "development",
    "dev": "development",
    "local": "development",
    "test": "test",
    "testing": "test",
    "ci": "test",
    "staging": "staging",
    "stage": "staging",
    "production": "production",
    "prod": "production",
    "saas": "production",
}
_RESTRICTIVENESS = {"development": 0, "test": 1, "staging": 2, "production": 3}
PRODUCTION_LIKE_MODES = frozenset({"staging", "production"})


class DeploymentModeConflict(ValueError):
    """Deployment-mode signals disagree, or one is not a known mode."""


def deployment_mode_signals(env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Raw (lower-cased) value of each mode variable that is set."""
    source = os.environ if env is None else env
    out: dict[str, str] = {}
    for name in MODE_ENV_VARS:
        raw = str(source.get(name, "") or "").strip().lower()
        if raw:
            out[name] = raw
    return out


def resolve_deployment_mode(env: Optional[Mapping[str, str]] = None, *, strict: bool = False) -> str:
    """Canonical mode. ``strict`` raises on unknown values or disagreeing signals."""
    canonical: dict[str, str] = {}
    for name, raw in deployment_mode_signals(env).items():
        mode = _MODE_ALIASES.get(raw)
        if mode is None:
            if strict:
                raise DeploymentModeConflict(
                    f"{name}={raw!r} is not a deployment mode (use one of {', '.join(CANONICAL_MODES)})"
                )
            mode = "production"  # unknown: fail safe
        canonical[name] = mode
    if not canonical:
        return "development"
    distinct = set(canonical.values())
    if len(distinct) > 1:
        if strict:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(canonical.items()))
            raise DeploymentModeConflict(f"deployment mode signals disagree: {detail}")
        return max(distinct, key=_RESTRICTIVENESS.__getitem__)
    return distinct.pop()


def deployment_mode() -> str:
    return resolve_deployment_mode()


def is_production_like(env: Optional[Mapping[str, str]] = None) -> bool:
    return resolve_deployment_mode(env) in PRODUCTION_LIKE_MODES


def is_saas_deployment() -> bool:
    return is_production_like()


def validate_deployment_mode_startup(service_name: str) -> str:
    """Services refuse to start when mode signals conflict or are unknown."""
    try:
        return resolve_deployment_mode(strict=True)
    except DeploymentModeConflict as exc:
        import logging

        logging.getLogger("kazenai.deployment").critical("%s: %s", service_name, exc)
        raise SystemExit(f"{service_name}: {exc}") from exc


def _env_truthy(name: str) -> bool | None:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def is_dev_env() -> bool:
    """Explicit local dev only — ``KAZENAI_ENV=dev`` (not ``development``)."""
    return os.getenv("KAZENAI_ENV", "").strip().lower() == "dev"


def _enforcement_mode_value(name: str) -> str:
    raw = os.getenv(name, "").strip().lower()
    return raw if raw in {"fail_closed", "fail_open"} else "fail_closed"


def enforcement_fail_closed() -> bool:
    """Consumer-side FinOps unavailability policy.

    Always execute the lifecycle resolver's effective ``failure_policy``
    so diagnostics and runtime cannot disagree.
    """
    try:
        from kazenai.lifecycle_config import resolve_effective_lifecycle

        cfg = resolve_effective_lifecycle()
        # The resolver already applies the KAZENAI_ENV=dev rule.
        return cfg.failure_policy != "fail_open"
    except Exception:
        pass
    raw = os.getenv("KAZENAI_ENFORCEMENT_MODE", "").strip().lower()
    if raw in {"fail_closed", "fail_open"}:
        if raw == "fail_closed":
            return True
        return not is_dev_env()
    return True


def finops_reservation_fail_closed() -> bool:
    """Pre-call reservation HTTP error policy (``KAZENAI_FINOPS_RESERVATION_MODE``)."""
    try:
        from kazenai.lifecycle_config import resolve_effective_lifecycle

        cfg = resolve_effective_lifecycle()
        # The resolver already applies the KAZENAI_ENV=dev rule.
        return cfg.failure_policy != "fail_open"
    except Exception:
        pass
    raw = os.getenv("KAZENAI_FINOPS_RESERVATION_MODE", "").strip().lower()
    if raw in {"fail_closed", "fail_open"}:
        if raw == "fail_closed":
            return True
        return not is_dev_env()
    return True


_ENFORCEMENT_VARS = (
    "KAZENAI_ENFORCEMENT_MODE",
    "KAZENAI_FINOPS_ENFORCEMENT_MODE",
    "KAZENAI_FINOPS_RESERVATION_MODE",
)


def validate_prod_fail_closed_posture(service_name: str) -> None:
    """Refuse startup when prod-like deployment enables fail-open enforcement."""
    if not is_saas_deployment():
        return
    import logging

    log = logging.getLogger("kazenai.deployment")
    mode = deployment_mode()
    for var in _ENFORCEMENT_VARS:
        raw = os.getenv(var, "").strip().lower()
        if raw == "fail_open":
            log.critical(
                "%s: %s=fail_open refused when KAZENAI_DEPLOYMENT_MODE=%s",
                service_name,
                var,
                mode,
            )
            raise SystemExit(1)


def canonical_default_org(dev_default: str = "kazen-local") -> str:
    """Resolve the canonical default org id.

    A single ``KAZENAI_DEFAULT_ORG_ID`` overrides every service's local literal so
    multi-tenant defaults don't silently diverge. Safe to call on the request path:
    it never raises (use :func:`require_canonical_default_org` at startup to enforce
    that production actually sets it). In dev, ``dev_default`` is used.
    """
    explicit = os.getenv("KAZENAI_DEFAULT_ORG_ID", "").strip()
    if explicit:
        return explicit
    return dev_default


def require_canonical_default_org() -> None:
    """Startup guard: in saas/prod, ``KAZENAI_DEFAULT_ORG_ID`` must be set explicitly."""
    if is_saas_deployment() and not os.getenv("KAZENAI_DEFAULT_ORG_ID", "").strip():
        raise SystemExit(
            "KAZENAI_DEFAULT_ORG_ID must be set explicitly in saas/production "
            "deployments — relying on a hardcoded per-service default org is unsafe."
        )
