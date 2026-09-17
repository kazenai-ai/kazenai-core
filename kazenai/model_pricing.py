"""Canonical FinOps LLM model pricing (Loop 25 — single source of truth).

Rates are USD per 1M tokens (input, output). Resolution uses longest safe
prefix match for versioned ids and bounded substring match for dotted
provider paths (Bedrock), but rejects embedded keys in arbitrary wrappers.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

# Keep aligned with kazenai-agent-finops historical MODEL_PRICING table.
FINOPS_MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (5.00, 15.00),
    "gemini-2.0-flash": (0.075, 0.30),
    "gemini-2.5-pro": (3.50, 10.50),
    "deepseek-v3": (0.27, 1.10),
    "deepseek-v3.2": (0.27, 1.10),
    "deepseek.v3.2": (0.27, 1.10),
    "llama-3.3-70b": (0.59, 0.79),
    "llama-3.3-70b-instruct": (0.59, 0.79),
    "llama-3.1-8b-instruct": (0.08, 0.08),
    "llama-3.1-8b": (0.08, 0.08),
    "gpt-oss-120b": (0.15, 0.60),
    "amazon.nova-pro": (0.80, 3.20),
    "amazon.nova-lite": (0.06, 0.24),
    "amazon.nova-micro": (0.035, 0.14),
    "meta.llama3-70b": (0.99, 0.99),
    "meta.llama3-8b": (0.20, 0.20),
    "pioneer-pro": (1.00, 4.00),
    "pioneer-fast": (0.20, 0.80),
    "Llama-3.3-70B": (0.59, 0.79),
    "Mistral-7B": (0.08, 0.08),
    "Qwen2.5-72B": (0.29, 0.39),
    # Bedrock Qwen3 VL 235B (plan reviewer / plan critic default in local stack).
    "qwen3-vl-235b": (0.50, 1.50),
    "command-r-plus": (3.00, 15.00),
    "command-r": (0.50, 1.50),
    "llama3-70b": (0.59, 0.79),
    "mixtral-8x7b": (0.27, 0.27),
}

_DEFAULT_MAX_OUTPUT: Dict[str, int] = {
    "gpt-4o": 4096,
    "gpt-4o-mini": 16384,
    "claude-sonnet-4-6": 8192,
    "claude-opus-4-7": 8192,
    "claude-haiku-4-5": 8192,
    "gemini-2.0-flash": 8192,
    "gemini-2.5-pro": 8192,
}


def normalize_model_id(model: str) -> str:
    raw = str(model or "").strip().lower()
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    # HF Inference Router provider suffix (e.g. meta-llama/...:novita).
    if ":" in raw:
        raw = raw.split(":", 1)[0]
    return raw


def resolve_model_pricing_key(
    model: str,
    pricing: Mapping[str, Tuple[float, float]] | None = None,
) -> Optional[str]:
    """Return the canonical pricing-table key for *model*, or None if unknown."""
    table = pricing if pricing is not None else FINOPS_MODEL_PRICING
    norm = normalize_model_id(model)
    keys = {k.lower(): k for k in table}
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
    table = pricing if pricing is not None else FINOPS_MODEL_PRICING
    input_rate, output_rate = table[key]
    return (input_tokens / 1_000_000.0) * input_rate + (output_tokens / 1_000_000.0) * output_rate


def default_max_output_tokens(model: str) -> Optional[int]:
    key = resolve_model_pricing_key(model)
    if key and key in _DEFAULT_MAX_OUTPUT:
        return _DEFAULT_MAX_OUTPUT[key]
    return None
