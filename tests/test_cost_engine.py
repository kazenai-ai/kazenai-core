"""TokenCostEngine (spend pricing) tests."""

from __future__ import annotations

import json
import logging

import pytest

from kazenai.cost_engine import ModelPricing, TokenCostEngine


def test_known_model_pricing():
    engine = TokenCostEngine()
    p = engine.price("gpt-4o-mini")
    assert p.input_per_1k == pytest.approx(0.00015)
    assert p.output_per_1k == pytest.approx(0.00060)


def test_alias_resolves_to_canonical():
    engine = TokenCostEngine()
    p = engine.price("claude-sonnet-4")
    assert p.input_per_1k == pytest.approx(0.00300)


def test_deepseek_dot_version_is_priced():
    engine = TokenCostEngine()
    p = engine.price("deepseek.v3.2")
    assert p.input_per_1k == pytest.approx(0.000270)
    assert p.output_per_1k == pytest.approx(0.001100)


def test_cost_usd_computation():
    engine = TokenCostEngine()
    cost = engine.cost_usd(model="gpt-4o-mini", input_tokens=1000, output_tokens=500)
    expected = (1000 * 0.00015 + 500 * 0.00060) / 1000.0
    assert cost == pytest.approx(expected)


def test_from_usage_dict():
    engine = TokenCostEngine()
    inp, out, cost = engine.from_usage(
        model="gpt-4o",
        usage={"prompt_tokens": 200, "completion_tokens": 100},
    )
    assert inp == 200
    assert out == 100
    assert cost > 0


def test_from_usage_object_attributes():
    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    engine = TokenCostEngine()
    inp, out, cost = engine.from_usage(model="gpt-4o-mini", usage=_Usage())
    assert inp == 10
    assert out == 5
    assert cost >= 0


def test_env_pricing_override(monkeypatch):
    monkeypatch.setenv(
        "KAZENAI_PRICING_JSON",
        json.dumps({"custom-model": {"input_per_1k": 0.01, "output_per_1k": 0.02}}),
    )
    engine = TokenCostEngine()
    p = engine.price("custom-model")
    assert p.input_per_1k == 0.01
    assert engine.cost_usd(model="custom-model", input_tokens=1000, output_tokens=0) == pytest.approx(0.01)


def test_constructor_pricing_override():
    engine = TokenCostEngine(pricing={"inline": ModelPricing(1.0, 2.0)})
    assert engine.price("inline").input_per_1k == 1.0


def test_unknown_model_warns_and_returns_zero(caplog):
    engine = TokenCostEngine()
    with caplog.at_level(logging.WARNING, logger="kazenai.cost_engine"):
        p = engine.price("totally-unknown-model-xyz")
    assert p.input_per_1k == 0.0
    assert "unknown model" in caplog.text.lower()


def test_prefix_match_versioned_model():
    engine = TokenCostEngine()
    p = engine.price("gpt-4o-2024-11-20-snapshot")
    assert p.input_per_1k == pytest.approx(0.00250)


def test_versioned_id_resolves_to_correct_base_not_pricier_family():
    """Regression for P0-1: a real dated model id must resolve to its own base
    family (longest-prefix), not the insertion-first pricier row."""
    engine = TokenCostEngine()
    # gpt-4o-mini-2024-07-18 must price as gpt-4o-mini, NOT gpt-4o (17x).
    assert engine.price("gpt-4o-mini-2024-07-18") == engine.price("gpt-4o-mini")
    assert engine.price("gpt-4o-mini-2024-07-18") != engine.price("gpt-4o")
    # gpt-4-turbo-2024-04-09 must resolve to gpt-4-turbo, not gpt-4.
    assert engine.price("gpt-4-turbo-2024-04-09") == engine.price("gpt-4-turbo")
    assert engine.price("gpt-4-turbo-2024-04-09") != engine.price("gpt-4")
    # o1-preview-2024-... must resolve to o1, not the unknown path.
    assert engine.price("o1-preview-2024-09-12") == engine.price("o1")


def test_truncated_stems_fall_through_to_unknown(caplog):
    """Regression for P0-1: short/truncated stems must NOT resolve to an
    arbitrary specific model (the dropped k.startswith(key) branch)."""
    engine = TokenCostEngine()
    for stem in ("gpt", "claude", "gemini"):
        with caplog.at_level(logging.WARNING, logger="kazenai.cost_engine"):
            p = engine.price(stem)
        assert p == ModelPricing(0.0, 0.0), f"{stem!r} should be unknown, got {p}"


def test_prefix_match_requires_token_boundary():
    """A prefix that isn't at a token boundary must not match."""
    engine = TokenCostEngine()
    # "gpt-4omega" startswith "gpt-4o" but is not a versioned suffix.
    assert engine.price("gpt-4omega") == ModelPricing(0.0, 0.0)


def test_known_models_sorted():
    engine = TokenCostEngine()
    models = engine.known_models()
    assert models == sorted(models)
    assert "gpt-4o-mini" in models


def test_negative_tokens_clamped_to_zero():
    engine = TokenCostEngine()
    assert engine.cost_usd(model="gpt-4o-mini", input_tokens=-5, output_tokens=-1) == 0.0
