"""One verified price table: model_pricing is canonical, cost_engine derives from it."""

from __future__ import annotations

from datetime import date

import pytest

from kazenai.cost_engine import _DEFAULT_PRICING, TokenCostEngine
from kazenai.model_pricing import (
    FINOPS_MODEL_PRICING,
    MODEL_PRICING,
    PRICING_LAST_VERIFIED,
    PRICING_SOURCES,
    default_max_output_tokens,
    output_limit_param,
)
from kazenai.monitor import _apply_output_token_bound


def test_cost_engine_matches_canonical_table_for_every_model():
    engine = TokenCostEngine()
    for model, (inp, out) in MODEL_PRICING.items():
        p = engine.price(model)
        assert p.input_per_1k == pytest.approx(inp / 1000.0), model
        assert p.output_per_1k == pytest.approx(out / 1000.0), model


def test_no_priced_model_outside_canonical_table():
    # Only genuinely free local runtimes may be added on the cost-engine side.
    extra = set(_DEFAULT_PRICING) - set(MODEL_PRICING)
    assert extra == {"ollama"}


def test_every_price_is_positive():
    for model, (inp, out) in MODEL_PRICING.items():
        assert inp > 0 and out > 0, model


def test_finops_name_is_the_same_table():
    assert FINOPS_MODEL_PRICING is MODEL_PRICING


def test_last_verified_date_and_sources():
    assert date.fromisoformat(PRICING_LAST_VERIFIED) <= date.today()
    assert {"openai", "anthropic"} <= set(PRICING_SOURCES)
    assert all(url.startswith("https://") for url in PRICING_SOURCES.values())


def test_every_priced_model_has_an_output_bound():
    for model in MODEL_PRICING:
        assert default_max_output_tokens(model), model


@pytest.mark.parametrize(
    ("model", "param"),
    [
        ("gpt-5-mini", "max_completion_tokens"),
        ("gpt-5.4", "max_completion_tokens"),
        ("gpt-6-luna", "max_completion_tokens"),
        ("o3", "max_completion_tokens"),
        ("o4-mini-2025-04-16", "max_completion_tokens"),
        ("gpt-4o", "max_tokens"),
        ("gpt-4.1-mini", "max_tokens"),
        ("claude-haiku-4-5", "max_tokens"),
        ("openai/o1", "max_completion_tokens"),
        ("o1x", "max_tokens"),
    ],
)
def test_output_limit_param(model, param):
    assert output_limit_param(model) == param


def test_reasoning_model_gets_max_completion_tokens_injected():
    kwargs = {"model": "gpt-5-mini", "messages": []}
    _apply_output_token_bound(kwargs)
    assert "max_tokens" not in kwargs
    assert kwargs["max_completion_tokens"] == default_max_output_tokens("gpt-5-mini")


def test_reasoning_model_explicit_limit_is_clamped_not_duplicated():
    kwargs = {"model": "o3", "max_completion_tokens": 10_000_000}
    _apply_output_token_bound(kwargs)
    assert kwargs["max_completion_tokens"] == default_max_output_tokens("o3")
    assert "max_tokens" not in kwargs


def test_settlement_micros_do_not_ceil_float_noise():
    from kazenai.spine.guard import _usd_to_micros_ceil

    assert _usd_to_micros_ceil(0.00010200000000000001) == 102
    assert _usd_to_micros_ceil(18 * 2.0 / 1e6 + 4 * 10.0 / 1e6) == 76  # sonnet 5.5, 18 in / 4 out
    assert _usd_to_micros_ceil(3.3e-06) == 4  # genuine fraction still rounds up
    assert _usd_to_micros_ceil(0.0) == 0
