"""TokenCostEngine (spend pricing) tests."""

from __future__ import annotations

import json
import logging

import pytest

from kazenai.cost_engine import ModelPricing, TokenCostEngine
from kazenai.pricing_policy import UnknownModelError


def test_known_model_pricing():
    engine = TokenCostEngine()
    p = engine.price("gpt-4o-mini")
    assert p.input_per_1k == pytest.approx(0.00015)
    assert p.output_per_1k == pytest.approx(0.00060)


def test_dated_snapshot_resolves_to_base():
    engine = TokenCostEngine()
    p = engine.price("claude-haiku-4-5-20251001")
    assert p.input_per_1k == pytest.approx(0.00100)
    assert p.output_per_1k == pytest.approx(0.00500)


def test_dotted_version_is_priced():
    engine = TokenCostEngine()
    p = engine.price("gpt-5.4-mini")
    assert p.input_per_1k == pytest.approx(0.00075)
    assert p.output_per_1k == pytest.approx(0.00450)


def test_unpriced_minor_version_is_unknown_not_base_family(monkeypatch):
    """gpt-5.3 / claude-opus-5-6 are different models; they must not borrow a sibling's price."""
    monkeypatch.setenv("KAZENAI_UNKNOWN_MODEL_POLICY", "warn")
    engine = TokenCostEngine()
    for model in ("gpt-5.3", "gpt-5.3-mini", "claude-opus-5-6", "claude-sonnet-4-7-20270101"):
        assert engine.price(model) == ModelPricing(0.0, 0.0), model


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


def test_unknown_model_blocks_by_default():
    engine = TokenCostEngine()
    with pytest.raises(UnknownModelError):
        engine.price("totally-unknown-model-xyz")


def test_unknown_model_warn_opt_in_returns_zero_and_logs(monkeypatch, caplog):
    monkeypatch.setenv("KAZENAI_UNKNOWN_MODEL_POLICY", "warn")
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
    # gpt-4.1-mini-2025-04-14 must resolve to gpt-4.1-mini, not gpt-4.1.
    assert engine.price("gpt-4.1-mini-2025-04-14") == engine.price("gpt-4.1-mini")
    assert engine.price("gpt-4.1-mini-2025-04-14") != engine.price("gpt-4.1")
    # o1-preview-2024-... must resolve to o1, not the unknown path.
    assert engine.price("o1-preview-2024-09-12") == engine.price("o1")


def test_truncated_stems_fall_through_to_unknown(caplog, monkeypatch):
    """Regression for P0-1: short/truncated stems must NOT resolve to an
    arbitrary specific model (the dropped k.startswith(key) branch)."""
    monkeypatch.setenv("KAZENAI_UNKNOWN_MODEL_POLICY", "warn")
    engine = TokenCostEngine()
    for stem in ("gpt", "claude", "gemini"):
        with caplog.at_level(logging.WARNING, logger="kazenai.cost_engine"):
            p = engine.price(stem)
        assert p == ModelPricing(0.0, 0.0), f"{stem!r} should be unknown, got {p}"


def test_prefix_match_requires_token_boundary(monkeypatch):
    """A prefix that isn't at a token boundary must not match."""
    monkeypatch.setenv("KAZENAI_UNKNOWN_MODEL_POLICY", "warn")
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
