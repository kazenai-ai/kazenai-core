"""Hermetic test defaults for standalone CI (no host .env leakage)."""

from __future__ import annotations

import pytest

# Host/dev shells often export KAZENAI_* from a workspace .env; clear the ones
# that change default pricing behavior so unit tests match GitHub Actions.
_CLEAR_ENV = (
    "KAZENAI_UNKNOWN_MODEL_POLICY",
    "KAZENAI_DEPLOYMENT_MODE",
    "KAZENAI_ENV",
    "KAZENAI_PRICING_JSON",
    "KAZENAI_MODEL_ALIASES_JSON",
    "KAZENAI_TIER_MISMATCH_MULTIPLIER",
    # Optional-framework imports must not inherit host fallback ingest aliases.
    "KAZENAI_FINOPS_INGEST_URL",
    "KAZENAI_INGEST_URL",
    "KAZENAI_AGENTLENS_INGEST_URL",
    "KAZENAI_LENS_INGEST_URL",
    "KAZENAI_FINOPS_URL",
    "KAZENAI_ENFORCEMENT_OWNER",
    "KAZENAI_ENFORCEMENT_MODE",
    "KAZENAI_FINOPS_RESERVATION_MODE",
)


@pytest.fixture(autouse=True)
def _hermetic_kazenai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _CLEAR_ENV:
        monkeypatch.delenv(key, raising=False)
    # Control startup requires a finite multiplier >= 1. Tests that prove the
    # missing-multiplier failure delete this.
    monkeypatch.setenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", "1")
    # The flusher daemon starts after the first spooled record. Tests flush explicitly.
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_AUTO_FLUSH", "0")
    # Mock provider hosts bill at list price. A non-first-party host is
    # otherwise priced with the 10% regional/third-party uplift; the uplift has
    # its own tests (tests/test_pricing_modifiers.py), which unset this.
    monkeypatch.setenv(
        "KAZENAI_ENDPOINT_PRICE_MULTIPLIERS",
        '{"openai.test": "1", "o.test": "1", "anthropic.test": "1", "a.test": "1", "127.0.0.1": "1", "localhost": "1"}',
    )


@pytest.fixture(autouse=True)
def _fresh_pricing_scope():
    """Each test starts outside any armed pricing scope (no state across tests)."""
    from kazenai import pricing_context as pc

    saved = [(var, var.set(_default_for(var))) for var in pc.ALL_VARS]
    try:
        yield
    finally:
        for var, token in reversed(saved):
            var.reset(token)


def _default_for(var):
    from kazenai import pricing_context as pc

    return {
        pc.CURRENT_PRICING: None,
        pc.CALL_PROVIDER: "",
        pc.CALL_SURFACE: "",
        pc.CALL_CLIENT: None,
        pc.CALL_ORG: "",
        pc.OBSERVE_UNBOUNDED: (),
        pc.CALL_KWARGS_ID: 0,
    }[var]
