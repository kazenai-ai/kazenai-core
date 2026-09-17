"""Fail-closed startup checks for critical secret hygiene (Loop 25 P0-1).

Compares secrets via digests only — never logs or echoes secret values.
"""

from __future__ import annotations

import hashlib
import logging
import os

from kazenai.deployment import is_saas_deployment

_log = logging.getLogger("kazenai.secret_hygiene")

_CRITICAL_SECRET_ENV_VARS = (
    "KAZEN_MASTER_KEY",
    "KAZENAI_FINOPS_MASTER_KEY",
    "KAZENAI_FINOPS_API_KEY",
    "KAZENAI_PLATFORM_SERVICE_TOKEN",
)

_STRIPE_ENV = "STRIPE_SECRET_KEY"
_MIN_SECRET_LEN = 16

# Weak / placeholder literals — stored as SHA-256 digests (never embed live values).
_PLACEHOLDER_DIGESTS: frozenset[str] = frozenset(
    hashlib.sha256(p.encode("utf-8")).hexdigest()
    for p in (
        "changeme",
        "change-me",
        "secret",
        "test",
        "test-key",
        "test-signing-key",
        "your-key-here",
        "placeholder",
        "kazenai-dev-insecure-secret",
        "sk_test_placeholder",
        "sk_live_placeholder",
    )
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _env_value(name: str) -> str:
    return os.getenv(name, "").strip()


def validate_secret_hygiene_startup(*, service_name: str = "kazenai") -> None:
    """Refuse startup when critical secrets are reused, weak, or live Stripe in non-prod."""
    errors: list[str] = []

    stripe = _env_value(_STRIPE_ENV)
    if stripe.startswith("sk_live_") and not is_saas_deployment():
        errors.append(
            f"{_STRIPE_ENV} appears to be a live Stripe key but deployment mode "
            "is not production/staging/saas"
        )

    present: dict[str, str] = {}
    for name in _CRITICAL_SECRET_ENV_VARS:
        val = _env_value(name)
        if not val:
            continue
        if len(val) < _MIN_SECRET_LEN:
            errors.append(f"{name} is set but shorter than {_MIN_SECRET_LEN} characters")
        elif _digest(val) in _PLACEHOLDER_DIGESTS:
            errors.append(f"{name} matches a known placeholder or weak value")
        present[name] = val

    by_value: dict[str, list[str]] = {}
    for name, val in present.items():
        by_value.setdefault(val, []).append(name)
    for names in by_value.values():
        if len(names) > 1:
            errors.append(
                "Critical secrets must be unique; these variables share the same value: "
                + ", ".join(sorted(names))
            )

    if errors:
        _log.critical("%s: secret hygiene validation failed:", service_name)
        for err in errors:
            _log.critical("  %s", err)
        raise SystemExit(1)


__all__ = ["validate_secret_hygiene_startup"]
