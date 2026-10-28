"""Canonical LLM model pricing — a compatibility view of ``price_catalog``.

``PRICE_CATALOG`` is the single ceiling. ``MODEL_PRICING`` is the pinned
standard short-context (input, output) USD-per-1M slice so existing FinOps
imports keep working. Do not add a second price table.

Price lookup is exact id, then the explicit alias table. Family matching
remains only inside ``default_max_output_tokens``.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping, Optional, Tuple

from .price_catalog import (
    BUILTIN_ALIASES,
    PRICING_SOURCES,  # noqa: F401  re-exported for FinOps imports
    VERIFIED_ON,
    standard_short_rates,
)
from .price_catalog import PRICE_CATALOG as PRICE_CATALOG  # noqa: F401  re-export

PRICING_LAST_VERIFIED = VERIFIED_ON

MODEL_PRICING: Dict[str, Tuple[float, float]] = standard_short_rates()

# Backward-compatible name used by Agent FinOps and older imports.
FINOPS_MODEL_PRICING = MODEL_PRICING

# KazenAI's default pre-dispatch output bound per model. These are policy
# defaults at or below each provider's maximum, not the provider maximum; a
# caller's smaller explicit limit is kept, a larger one is clamped.
_DEFAULT_MAX_OUTPUT: Dict[str, int] = {
    "gpt-6": 16384,
    "gpt-5": 16384,
    "gpt-4.1": 16384,
    "gpt-4o": 4096,
    "gpt-4o-mini": 16384,
    "o1": 16384,
    "o3": 16384,
    "o4-mini": 16384,
    "claude-fable-5-1": 16384,
    "claude-fable-5": 16384,
    "claude-opus-5-5": 16384,
    "claude-opus-5": 16384,
    "claude-sonnet-5-5": 16384,
    "claude-sonnet-5": 16384,
    "claude-haiku-5-5": 8192,
    "claude-opus-4-8": 8192,
    "claude-opus-4-7": 8192,
    "claude-opus-4-6": 8192,
    "claude-opus-4-5": 8192,
    "claude-opus-4-1": 8192,
    "claude-sonnet-4-6": 8192,
    "claude-sonnet-4-5": 8192,
    "claude-haiku-4-5": 8192,
    "gemini-2.5-pro": 8192,
    "gemini-2.5-flash": 8192,
    "gemini-2.5-flash-lite": 8192,
    "gemini-3": 8192,
    "deepseek-flash": 8192,
    "deepseek-v4-pro": 8192,
}

# OpenAI reasoning families reject ``max_tokens`` on Chat Completions and
# require ``max_completion_tokens``.
_COMPLETION_TOKENS_PREFIXES: Tuple[str, ...] = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def normalize_model_id(model: str) -> str:
    raw = str(model or "").strip().lower()
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    # HF Inference Router provider suffix (e.g. meta-llama/...:novita).
    if ":" in raw:
        raw = raw.split(":", 1)[0]
    return raw


_VERSION_BUMP_SUFFIX = re.compile(r"^[-_.]\d{1,2}(?:$|[-_.])")


def _is_version_bump(suffix: str) -> bool:
    """True when *suffix* names a different model version, not a snapshot.

    ``gpt-5`` + ``.3`` or ``claude-opus-5`` + ``-6`` is another (unpriced)
    model; ``-2024-05-13`` / ``-20251001`` / ``-v2`` are snapshots of the same one.
    """
    return bool(_VERSION_BUMP_SUFFIX.match(suffix))


def resolve_model_pricing_key(
    model: str,
    pricing: Mapping[str, Tuple[float, float]] | None = None,
    *,
    allow_version_suffix: bool = False,
) -> Optional[str]:
    """Exact catalog id, then an explicit alias. No prefix inheritance.

    ``allow_version_suffix`` is ignored for prices. Output bounds use
    ``_resolve_output_family`` instead.
    """
    del allow_version_suffix
    table = pricing if pricing is not None else MODEL_PRICING
    norm = str(model or "").strip().lower()
    keys = {k.lower(): k for k in table}
    if norm in keys:
        return keys[norm]
    canonical = BUILTIN_ALIASES.get(norm)
    if canonical and canonical.lower() in keys:
        return keys[canonical.lower()]
    return None


def _resolve_output_family(model: str) -> Optional[str]:
    """Family match for default output bounds only. Not used for prices."""
    norm = normalize_model_id(model)
    keys = {k.lower(): k for k in _DEFAULT_MAX_OUTPUT}
    if norm in keys:
        return keys[norm]
    best = ""
    best_canonical: Optional[str] = None
    for k_lower, k_orig in keys.items():
        if not norm.startswith(k_lower) or len(k_lower) <= len(best):
            continue
        if len(norm) == len(k_lower) or norm[len(k_lower)] in "-_.":
            best = k_lower
            best_canonical = k_orig
    return best_canonical


def model_cost_usd_per_million(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    pricing: Mapping[str, Tuple[float, float]] | None = None,
) -> Optional[float]:
    """USD from the shared micro-USD ceiling; None when the model is unknown."""
    if not model or (input_tokens <= 0 and output_tokens <= 0):
        return 0.0
    if pricing is not None:
        key = resolve_model_pricing_key(model, pricing)
        if key is None:
            return None
        input_rate, output_rate = pricing[key]
        from decimal import Decimal

        from .price_catalog import decimal_to_micro_usd, micro_usd_to_decimal

        amount = (
            Decimal(max(0, int(input_tokens))) * Decimal(str(input_rate))
            + Decimal(max(0, int(output_tokens))) * Decimal(str(output_rate))
        ) / Decimal(1_000_000)
        return float(micro_usd_to_decimal(decimal_to_micro_usd(amount)))
    from .cost_engine import TokenCostEngine

    try:
        return TokenCostEngine().cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception:
        return None


def default_max_output_tokens(model: str) -> Optional[int]:
    # Family matching is allowed for the output bound and forbidden for prices.
    key = _resolve_output_family(model)
    if key and key in _DEFAULT_MAX_OUTPUT:
        return _DEFAULT_MAX_OUTPUT[key]
    return None


def output_limit_param(model: str) -> str:
    """Chat Completions parameter that carries the output limit for *model*."""
    norm = normalize_model_id(model)
    for prefix in _COMPLETION_TOKENS_PREFIXES:
        if norm == prefix or (norm.startswith(prefix) and norm[len(prefix)] in "-_."):
            return "max_completion_tokens"
    return "max_tokens"
