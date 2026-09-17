"""Shared deployment-mode helpers for fail-safe production defaults."""

from __future__ import annotations

import os


def deployment_mode() -> str:
    return os.getenv(
        "KAZENAI_DEPLOYMENT_MODE",
        os.getenv("KAZENAI_ENV", os.getenv("KAZENAI_FINOPS_ENV", "development")),
    ).strip().lower()


def is_saas_deployment() -> bool:
    return deployment_mode() in {"saas", "production", "prod", "staging"}


def _env_truthy(name: str) -> bool | None:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def brain_strict_enabled() -> bool:
    """Strict Brain when explicitly enabled or in saas/staging/prod deploys."""
    explicit = _env_truthy("KAZENAI_BRAIN_STRICT")
    if explicit is not None:
        return explicit
    return is_saas_deployment()


def graph_strict_enabled() -> bool:
    explicit = _env_truthy("KAZENAI_GRAPH_STRICT")
    if explicit is not None:
        return explicit
    return is_saas_deployment()


def is_dev_env() -> bool:
    """Explicit local dev only — ``KAZENAI_ENV=dev`` (not ``development``)."""
    return os.getenv("KAZENAI_ENV", "").strip().lower() == "dev"


def _enforcement_mode_value(name: str) -> str:
    raw = os.getenv(name, "").strip().lower()
    return raw if raw in {"fail_closed", "fail_open"} else "fail_closed"


def enforcement_fail_closed() -> bool:
    """Consumer-side FinOps unavailability policy (``KAZENAI_ENFORCEMENT_MODE``)."""
    mode = _enforcement_mode_value("KAZENAI_ENFORCEMENT_MODE")
    if mode == "fail_closed":
        return True
    if mode == "fail_open" and is_dev_env():
        return False
    return True


def finops_reservation_fail_closed() -> bool:
    """Pre-call reservation HTTP error policy (``KAZENAI_FINOPS_RESERVATION_MODE``)."""
    mode = _enforcement_mode_value("KAZENAI_FINOPS_RESERVATION_MODE")
    if mode == "fail_closed":
        return True
    if mode == "fail_open" and is_dev_env():
        return False
    return True


_ENFORCEMENT_VARS = (
    "KAZENAI_ENFORCEMENT_MODE",
    "KAZENAI_FINOPS_ENFORCEMENT_MODE",
    "KAZENAI_FINOPS_RESERVATION_MODE",
)


def validate_prod_fail_closed_posture(service_name: str) -> None:
    """Refuse startup when prod-like deployment enables fail-open enforcement (P0-2)."""
    if not is_saas_deployment():
        return
    import logging
    import sys

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
    if service_name in {"growthops", "kazenai-agent-growthops"}:
        if _env_truthy("GROWTHOPS_PRODUCT_TRUTH_FAIL_OPEN") is True:
            log.critical(
                "%s: GROWTHOPS_PRODUCT_TRUTH_FAIL_OPEN refused when KAZENAI_DEPLOYMENT_MODE=%s",
                service_name,
                mode,
            )
            raise SystemExit(1)
    if service_name == "brain" and _env_truthy("KAZENAI_BRAIN_STRICT") is False:
        log.critical(
            "brain: KAZENAI_BRAIN_STRICT=0 refused when KAZENAI_DEPLOYMENT_MODE=%s",
            mode,
        )
        raise SystemExit(1)
    if service_name == "builder":
        if _env_truthy("KAZENAI_WEBHOOK_FF") is True:
            log.critical(
                "builder: KAZENAI_WEBHOOK_FF refused when KAZENAI_DEPLOYMENT_MODE=%s",
                mode,
            )
            raise SystemExit(1)
        if _env_truthy("KAZENAI_WEBHOOK_INSECURE_DEV") is True and not is_dev_env():
            log.critical(
                "builder: KAZENAI_WEBHOOK_INSECURE_DEV refused outside KAZENAI_ENV=dev "
                "(KAZENAI_DEPLOYMENT_MODE=%s)",
                mode,
            )
            raise SystemExit(1)
        if not os.getenv("GITHUB_WEBHOOK_SECRET", "").strip():
            log.critical(
                "builder: GITHUB_WEBHOOK_SECRET required when KAZENAI_DEPLOYMENT_MODE=%s",
                mode,
            )
            raise SystemExit(1)
        sandbox_mode = os.getenv("KAZENAI_SECURITY_SANDBOX", "off").strip().lower()
        if sandbox_mode not in {"docker", "gvisor"}:
            log.critical(
                "builder: KAZENAI_SECURITY_SANDBOX must be docker or gvisor when "
                "KAZENAI_DEPLOYMENT_MODE=%s (got %r)",
                mode,
                sandbox_mode,
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
