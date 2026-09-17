from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .pricing_policy import (
    UnknownModelError,
    enforce_unknown_model,
    estimate_unknown_model_pricing,
    resolve_model_key,
)

_log = logging.getLogger("kazenai.cost_engine")


@dataclass(frozen=True)
class ModelPricing:
    input_per_1k: float
    output_per_1k: float


# ---------------------------------------------------------------------------
# Default pricing table — USD per 1,000 tokens (as of June 2026).
# Override individual models via KAZENAI_PRICING_JSON env var without code
# changes. The env var takes priority over these defaults.
# ---------------------------------------------------------------------------
_DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # Claude 4.x (Anthropic, 2026)
    "claude-opus-4-8":              ModelPricing(input_per_1k=0.01500,  output_per_1k=0.07500),
    "claude-opus-4-7":              ModelPricing(input_per_1k=0.01500,  output_per_1k=0.07500),
    "claude-opus-4-6":              ModelPricing(input_per_1k=0.01500,  output_per_1k=0.07500),
    "claude-sonnet-4-6":            ModelPricing(input_per_1k=0.00300,  output_per_1k=0.01500),
    "claude-sonnet-4-5":            ModelPricing(input_per_1k=0.00300,  output_per_1k=0.01500),
    "claude-haiku-4-5":             ModelPricing(input_per_1k=0.00080,  output_per_1k=0.00400),
    "claude-haiku-4-5-20251001":    ModelPricing(input_per_1k=0.00080,  output_per_1k=0.00400),
    # Claude 3.x (Anthropic, legacy)
    "claude-3-opus-20240229":       ModelPricing(input_per_1k=0.01500,  output_per_1k=0.07500),
    "claude-3-5-sonnet-20241022":   ModelPricing(input_per_1k=0.00300,  output_per_1k=0.01500),
    "claude-3-5-haiku-20241022":    ModelPricing(input_per_1k=0.00080,  output_per_1k=0.00400),
    "claude-3-haiku-20240307":      ModelPricing(input_per_1k=0.00025,  output_per_1k=0.00125),
    # GPT-4o (OpenAI, 2025)
    "gpt-4o":                       ModelPricing(input_per_1k=0.00250,  output_per_1k=0.01000),
    "gpt-4o-mini":                  ModelPricing(input_per_1k=0.00015,  output_per_1k=0.00060),
    "gpt-4o-2024-11-20":            ModelPricing(input_per_1k=0.00250,  output_per_1k=0.01000),
    "gpt-4-turbo":                  ModelPricing(input_per_1k=0.01000,  output_per_1k=0.03000),
    "gpt-4":                        ModelPricing(input_per_1k=0.03000,  output_per_1k=0.06000),
    "gpt-3.5-turbo":                ModelPricing(input_per_1k=0.00050,  output_per_1k=0.00150),
    # o1/o3 reasoning models (OpenAI)
    "o1":                           ModelPricing(input_per_1k=0.01500,  output_per_1k=0.06000),
    "o1-mini":                      ModelPricing(input_per_1k=0.00300,  output_per_1k=0.01200),
    "o3-mini":                      ModelPricing(input_per_1k=0.00110,  output_per_1k=0.00440),
    # Gemini (Google, 2025-2026)
    "gemini-1.5-pro":               ModelPricing(input_per_1k=0.00125,  output_per_1k=0.00500),
    "gemini-1.5-flash":             ModelPricing(input_per_1k=0.000075, output_per_1k=0.000300),
    "gemini-2.0-flash":             ModelPricing(input_per_1k=0.000100, output_per_1k=0.000400),
    "gemini-2.5-pro":               ModelPricing(input_per_1k=0.001250, output_per_1k=0.010000),
    # Mistral (2025-2026)
    "mistral-large-2":              ModelPricing(input_per_1k=0.00200,  output_per_1k=0.00600),
    "mistral-medium-3":             ModelPricing(input_per_1k=0.00040,  output_per_1k=0.00200),
    "mistral-small-3.1":            ModelPricing(input_per_1k=0.00010,  output_per_1k=0.00030),
    "codestral-2501":               ModelPricing(input_per_1k=0.00030,  output_per_1k=0.00090),
    # DeepSeek (2025)
    "deepseek-r1":                  ModelPricing(input_per_1k=0.000550, output_per_1k=0.002190),
    "deepseek-v3":                  ModelPricing(input_per_1k=0.000270, output_per_1k=0.001100),
    "deepseek-v3.2":                ModelPricing(input_per_1k=0.000270, output_per_1k=0.001100),
    "deepseek.v3.2":                ModelPricing(input_per_1k=0.000270, output_per_1k=0.001100),
    "deepseek-chat":                ModelPricing(input_per_1k=0.000270, output_per_1k=0.001100),
    # Llama (Meta via Groq/Together, indicative rates)
    "llama-3.3-70b":                ModelPricing(input_per_1k=0.00059,  output_per_1k=0.00079),
    "llama-3.1-8b":                 ModelPricing(input_per_1k=0.00005,  output_per_1k=0.00008),
    # Qwen (Alibaba, indicative rates)
    "qwen-2.5-72b":                 ModelPricing(input_per_1k=0.00040,  output_per_1k=0.00120),
    # Local / free models
    "ollama":                       ModelPricing(input_per_1k=0.0,      output_per_1k=0.0),
    "ollama/llama3":                ModelPricing(input_per_1k=0.0,      output_per_1k=0.0),
}

# Aliases: map common API model IDs to canonical pricing keys
_ALIASES: Dict[str, str] = {
    "claude-3-5-sonnet-latest":     "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-latest":      "claude-3-5-haiku-20241022",
    "claude-sonnet-4":              "claude-sonnet-4-6",
    "claude-opus-4":                "claude-opus-4-7",
    "gpt-4o-latest":                "gpt-4o",
    "gemini-flash":                 "gemini-1.5-flash",
    "gemini-pro":                   "gemini-1.5-pro",
}


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
        policy = os.getenv("KAZENAI_UNKNOWN_MODEL_POLICY", "warn").strip().lower()
        if policy == "estimate":
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
