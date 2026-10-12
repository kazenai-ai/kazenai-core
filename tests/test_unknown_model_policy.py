"""Unknown-model pricing policy tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from kazenai.cost_engine import ModelPricing, TokenCostEngine, UnknownModelError, _ALIASES
from kazenai.pricing_policy import (
    UnknownModelError as PricingUnknownModelError,
    enforce_unknown_model,
    estimate_unknown_model_pricing,
    is_known_model,
    unknown_model_policy,
)


class TestUnknownModelPolicy(unittest.TestCase):
    def test_block_policy_raises(self):
        with patch.dict(os.environ, {"KAZENAI_UNKNOWN_MODEL_POLICY": "block"}):
            engine = TokenCostEngine()
            with self.assertRaises(UnknownModelError):
                engine.cost_usd(model="totally-unknown-model-xyz", input_tokens=1000, output_tokens=500)

    def test_estimate_policy_nonzero(self):
        with patch.dict(os.environ, {"KAZENAI_UNKNOWN_MODEL_POLICY": "estimate"}):
            cost = TokenCostEngine().cost_usd(model="unknown-model-abc", input_tokens=1000, output_tokens=1000)
            self.assertGreater(cost, 0.0)

    def test_block_is_the_default_in_every_environment(self):
        self.assertEqual(unknown_model_policy(), "block")
        with patch.dict(os.environ, {"KAZENAI_DEPLOYMENT_MODE": "development"}, clear=False):
            self.assertEqual(unknown_model_policy(), "block")

    def test_warn_opt_in_honoured_only_outside_production(self):
        with patch.dict(os.environ, {"KAZENAI_UNKNOWN_MODEL_POLICY": "warn"}, clear=False):
            self.assertEqual(unknown_model_policy(), "warn")
        with patch.dict(
            os.environ,
            {"KAZENAI_UNKNOWN_MODEL_POLICY": "warn", "KAZENAI_DEPLOYMENT_MODE": "production"},
            clear=False,
        ):
            self.assertEqual(unknown_model_policy(), "block")

    def test_unrecognised_policy_value_fails_closed(self):
        with patch.dict(os.environ, {"KAZENAI_UNKNOWN_MODEL_POLICY": "allow"}, clear=False):
            self.assertEqual(unknown_model_policy(), "block")

    def test_production_defaults_to_block(self):
        with patch.dict(os.environ, {"KAZENAI_DEPLOYMENT_MODE": "production"}, clear=False):
            self.assertEqual(unknown_model_policy(), "block")

    def test_is_known_model_prefix_and_empty(self):
        known = {"gpt-4o", "claude-3"}
        self.assertFalse(is_known_model("", known))
        self.assertTrue(is_known_model("gpt-4o", known))
        self.assertTrue(is_known_model("gpt-4o-mini", known))

    def test_enforce_unknown_model_allows_known(self):
        enforce_unknown_model("gpt-4o", known_keys={"gpt-4o"})

    def test_enforce_unknown_model_blocks_in_production(self):
        with patch.dict(os.environ, {"KAZENAI_UNKNOWN_MODEL_POLICY": "block"}, clear=False):
            with self.assertRaises(PricingUnknownModelError):
                enforce_unknown_model("mystery-model", known_keys={"gpt-4o"})

    def test_estimate_unknown_model_pricing_returns_positive_rates(self):
        inp, out = estimate_unknown_model_pricing()
        self.assertGreater(inp, 0.0)
        self.assertGreater(out, 0.0)


class TestKnownPricedInvariant(unittest.TestCase):
    """Is_known_model and price must share one resolver.

    Invariant: for every model string, is_known_model(m) is True iff
    price(m) returns a real (non-fallback) pricing row.
    """

    def test_proxy_prefixed_names_are_unknown_and_not_billed_zero_silently(self):
        engine = TokenCostEngine()
        known = set(engine.known_models())
        # Common deployment pattern: gateway/proxy-prefixed model names. These
        # must NOT be treated as known (substring match was the bug) — they are
        # unknown, which the policy layer can then block/estimate.
        for m in ("my-gpt-4o-proxy", "internal-claude-haiku-4-5-router", "team-gpt-4o-gateway"):
            self.assertFalse(is_known_model(m, known, _ALIASES), f"{m!r} should be unknown")

    def test_known_iff_priced_nonzero(self):
        os.environ["KAZENAI_UNKNOWN_MODEL_POLICY"] = "warn"
        self.addCleanup(os.environ.pop, "KAZENAI_UNKNOWN_MODEL_POLICY", None)
        engine = TokenCostEngine()
        known = set(engine.known_models())
        models = [
            # proxy/gateway-prefixed (unknown)
            "my-gpt-4o-proxy", "internal-claude-haiku-4-5-router", "team-gpt-4o-gateway",
            # real dated ids (known via longest-prefix)
            "gpt-4o-mini-2024-07-18", "gpt-4.1-nano-2025-04-14", "o1-preview-2024-09-12",
            # exact keys
            "gpt-4o", "claude-opus-4-8", "gemini-2.5-pro",
            # unpriced version bumps (unknown)
            "gpt-5.3", "claude-opus-5-6",
            # truncated stems (unknown)
            "gpt", "claude", "gemini",
        ]
        for m in models:
            known_flag = is_known_model(m, known, _ALIASES)
            priced_nonzero = engine.price(m) != ModelPricing(0.0, 0.0)
            self.assertEqual(
                known_flag, priced_nonzero,
                f"invariant violated for {m!r}: known={known_flag} priced_nonzero={priced_nonzero}",
            )

    def test_block_policy_stops_proxy_prefixed_unknown(self):
        with patch.dict(os.environ, {"KAZENAI_UNKNOWN_MODEL_POLICY": "block"}, clear=False):
            engine = TokenCostEngine()
            with self.assertRaises(UnknownModelError):
                engine.cost_usd(model="my-gpt-4o-proxy", input_tokens=1_000_000, output_tokens=1_000_000)


if __name__ == "__main__":
    unittest.main()
