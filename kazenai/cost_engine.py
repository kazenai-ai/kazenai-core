from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .model_pricing import MODEL_PRICING
from .pricing_policy import (
    UnknownModelError,  # noqa: F401  (re-exported for callers)
    enforce_unknown_model,
    estimate_unknown_model_pricing,
    resolve_model_key,
    unknown_model_policy,
)

_log = logging.getLogger("kazenai.cost_engine")


@dataclass(frozen=True)
class ModelPricing:
    input_per_1k: float
    output_per_1k: float


# ---------------------------------------------------------------------------
# Default pricing table — USD per 1,000 tokens, derived from the canonical
# per-1M table in ``model_pricing`` (single source of truth, see
# ``PRICING_LAST_VERIFIED``). Override individual models via the
# KAZENAI_PRICING_JSON env var without code changes; the env var takes
# priority over these defaults.
# ---------------------------------------------------------------------------
_DEFAULT_PRICING: Dict[str, ModelPricing] = {
    key: ModelPricing(input_per_1k=inp / 1000.0, output_per_1k=out / 1000.0)
    for key, (inp, out) in MODEL_PRICING.items()
}
# Local runtimes are genuinely free; listed so they are known, not blocked.
_DEFAULT_PRICING["ollama"] = ModelPricing(input_per_1k=0.0, output_per_1k=0.0)

# Aliases: map API model IDs that are not prefix-resolvable to canonical keys.
# Dated snapshots (``claude-haiku-4-5-20251001``) resolve by prefix and need no
# alias. Only add an alias the provider documents as the same model and price.
_ALIASES: Dict[str, str] = {}


class TokenCostEngine:
    """
    Provider-agnostic token->USD cost engine.

    Does not require network access. Supports:
    - built-in baseline pricing table (all major 2026 models)
    - env override JSON (KAZENAI_PRICING_JSON)
    - caller-supplied override dict
    - warns (never silently returns $0) when model is unknown
    """

    def __init__(self, *, pricing: Optional[Dict[str, Any]] = None) -> None:
        merged: Dict[str, ModelPricing] = dict(_DEFAULT_PRICING)

        # Env override: KAZENAI_PRICING_JSON='{"gpt-4o":{"input_per_1k":0.0025,"output_per_1k":0.01}}'
        raw = os.getenv("KAZENAI_PRICING_JSON", "").strip()
        if raw:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, dict):
                            merged[str(k)] = ModelPricing(
                                input_per_1k=float(v.get("input_per_1k", 0.0) or 0.0),
                                output_per_1k=float(v.get("output_per_1k", 0.0) or 0.0),
                            )
            except Exception:
                pass

        if isinstance(pricing, dict):
            for k, v in pricing.items():
                if isinstance(v, ModelPricing):
                    merged[str(k)] = v
                elif isinstance(v, dict):
                    merged[str(k)] = ModelPricing(
                        input_per_1k=float(v.get("input_per_1k", 0.0) or 0.0),
                        output_per_1k=float(v.get("output_per_1k", 0.0) or 0.0),
                    )

        self._pricing = merged

    def price(self, model: str) -> ModelPricing:
        # Resolve via the single shared resolver (exact → alias → longest-prefix
        # at a token boundary) so "known" and "priced" can never disagree.
        # See pricing_policy.resolve_model_key / P0-2.
        resolved = resolve_model_key(model, self._pricing.keys(), _ALIASES)
        if resolved is not None:
            return self._pricing[resolved]
        enforce_unknown_model(model, known_keys=set(self._pricing.keys()), aliases=_ALIASES)
        # Reaching here means the policy is an explicit, non-production opt-in
        # (``estimate`` or ``warn``); ``block`` already raised above.
        if unknown_model_policy() == "estimate":
            inp, out = estimate_unknown_model_pricing()
            return ModelPricing(inp, out)
        _log.warning(
            "unknown model %r — cost will be reported as $0.00. "
            "Set KAZENAI_PRICING_JSON to add pricing or pin to a known model name.",
            model,
        )
        return ModelPricing(0.0, 0.0)

    def cost_usd(self, *, model: str, input_tokens: int, output_tokens: int) -> float:
        p = self.price(model)
        return ((max(0, int(input_tokens)) * p.input_per_1k) + (max(0, int(output_tokens)) * p.output_per_1k)) / 1000.0

    def from_usage(self, *, model: str, usage: Any) -> Tuple[int, int, float]:
        input_tokens = 0
        output_tokens = 0
        try:
            if isinstance(usage, dict):
                input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            else:
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        except Exception:
            input_tokens, output_tokens = 0, 0
        return input_tokens, output_tokens, float(self.cost_usd(model=model, input_tokens=input_tokens, output_tokens=output_tokens))

    def known_models(self) -> list[str]:
        """Return all model names with pricing in this engine."""
        return sorted(self._pricing.keys())
