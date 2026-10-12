"""Loop 25 P0-3 / H2 — shared model pricing resolution."""

from __future__ import annotations

import pytest

from kazenai.model_pricing import (
    FINOPS_MODEL_PRICING,
    model_cost_usd_per_million,
    resolve_model_pricing_key,
)


@pytest.mark.parametrize(
    ("model_id", "expected_key"),
    [
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("claude-sonnet-4-6-20250514", "claude-sonnet-4-6"),
        ("bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
        ("anthropic.claude-opus-5-5", "claude-opus-5-5"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("gpt-5.4-mini", "gpt-5.4-mini"),
        ("openai/gpt-5-mini-2025-08-07", "gpt-5-mini"),
        ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite"),
    ],
)
def test_resolve_versioned_and_provider_prefixed_models(model_id, expected_key):
    assert resolve_model_pricing_key(model_id) == expected_key


def test_wrapper_spoof_not_resolved():
    assert resolve_model_pricing_key("wrapper-gpt-4o-mini-evil") is None


def test_dated_model_has_nonzero_price():
    cost = model_cost_usd_per_million(
        "claude-sonnet-4-6-20250514",
        input_tokens=1_000_000,
        output_tokens=0,
    )
    assert cost == pytest.approx(3.0)


def test_finops_table_is_non_empty():
    assert FINOPS_MODEL_PRICING


@pytest.mark.parametrize("model_id", ["gpt-5.3", "gpt-5.7-sol", "claude-opus-5-6", "us.anthropic.claude-opus-5-6-v1"])
def test_unpriced_version_bump_not_resolved(model_id):
    assert resolve_model_pricing_key(model_id) is None
