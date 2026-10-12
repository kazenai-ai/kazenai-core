"""Canonical LLM model pricing — the single source of truth for Core and FinOps.

Rates are USD per 1M tokens (input, output), standard tier, uncached input.
``cost_engine`` derives its per-1K table from this module; Agent FinOps
imports it directly. Do not add a second price table anywhere.

Every entry was checked against the provider's official pricing page on
``PRICING_LAST_VERIFIED``. Models that could not be verified there are
deliberately absent: with the default ``block`` unknown-model policy an
unpriced model is denied before dispatch, never charged as $0. Add private or
gateway-specific pricing with ``KAZENAI_PRICING_JSON``.

Resolution uses longest safe prefix match for versioned ids and bounded
substring match for dotted provider paths (Bedrock), but rejects embedded keys
in arbitrary wrappers.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping, Optional, Tuple

PRICING_LAST_VERIFIED = "2026-10-12"

PRICING_SOURCES: Dict[str, str] = {
    "openai": "https://developers.openai.com/api/docs/pricing",
    "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
    "google": "https://ai.google.dev/gemini-api/docs/pricing",
    "deepseek": "https://api-docs.deepseek.com/quick_start/pricing",
}

MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # --- OpenAI -----------------------------------------------------------
    "gpt-6-astra": (10.00, 50.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    # gpt-5.5 / gpt-5.4 / *-pro: short-context (<272K input) rates; the
    # long-context surcharge is not modelled.
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.5-pro": (30.00, 180.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.2-pro": (21.00, 168.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-pro": (15.00, 120.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-05-13": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),
    "o1-pro": (150.00, 600.00),
    "o3": (2.00, 8.00),
    "o3-pro": (20.00, 80.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # --- Anthropic --------------------------------------------------------
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    # Haiku 5.5: prompts up to 100K tokens; longer prompts bill $0.50/$2.50.
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    # --- Google Gemini (paid tier, text input, <=200K prompt) -------------
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    # 3.6/3.8 flash rates rise to $1.50/$7.50 on 2027-01-01.
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    # --- DeepSeek ---------------------------------------------------------
    # Peak-hour rates (2x off-peak): a budget bound must not under-project.
    "deepseek-flash": (0.30, 1.20),
    "deepseek-v4-pro": (1.32, 3.96),
}

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
    """Return the canonical pricing-table key for *model*, or None if unknown."""
    table = pricing if pricing is not None else MODEL_PRICING
    norm = normalize_model_id(model)
    keys = {k.lower(): k for k in table}
    if norm in keys:
        return keys[norm]

    best = ""
    best_canonical: Optional[str] = None
    for k_lower, k_orig in keys.items():
        if not norm.startswith(k_lower) or len(k_lower) <= len(best):
            continue
        suffix = norm[len(k_lower):]
        if not allow_version_suffix and _is_version_bump(suffix):
            continue
        if len(norm) == len(k_lower) or norm[len(k_lower)] in "-_.":
            best = k_lower
            best_canonical = k_orig
    if best_canonical:
        return best_canonical

    if "." not in norm:
        return None

    best = ""
    best_canonical = None
    for k_lower, k_orig in keys.items():
        idx = norm.find(k_lower)
        if idx < 0:
            continue
        before_ok = idx == 0 or norm[idx - 1] in "."
        after_idx = idx + len(k_lower)
        after_ok = after_idx == len(norm) or norm[after_idx] in "-_.:"
        if not allow_version_suffix and _is_version_bump(norm[after_idx:]):
            continue
        if before_ok and after_ok and len(k_lower) > len(best):
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
    """Compute USD cost from token counts; None when model is not in the pricing table."""
    if not model or (input_tokens <= 0 and output_tokens <= 0):
        return 0.0
    key = resolve_model_pricing_key(model, pricing)
    if key is None:
        return None
    table = pricing if pricing is not None else MODEL_PRICING
    input_rate, output_rate = table[key]
    return (input_tokens / 1_000_000.0) * input_rate + (output_tokens / 1_000_000.0) * output_rate


def default_max_output_tokens(model: str) -> Optional[int]:
    # Resolve against the limit keys themselves so a family prefix (``gpt-5``)
    # bounds every priced member without one row per model.
    limit_keys = {key: (0.0, 0.0) for key in _DEFAULT_MAX_OUTPUT}
    key = resolve_model_pricing_key(model, pricing=limit_keys, allow_version_suffix=True)
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
