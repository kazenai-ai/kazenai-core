from __future__ import annotations

import os
from typing import Mapping, Optional

class UnknownModelError(ValueError):
    """Raised when an unrecognized model is used under block policy."""


# Conservative fallback (USD per 1k tokens) when policy=estimate.
_ESTIMATE_INPUT_PER_1K = 0.015
_ESTIMATE_OUTPUT_PER_1K = 0.075


def _deployment_mode() -> str:
    from kazenai.deployment import resolve_deployment_mode

    return resolve_deployment_mode()


_PRODUCTION_LIKE = {"saas", "production", "prod", "staging"}
_POLICIES = {"warn", "block", "estimate"}


def unknown_model_policy() -> str:
    """Return warn | block | estimate. Fails closed (``block``) by default.

    An unpriced model must never silently cost $0 against a budget, so the
    default is ``block`` in every environment. ``estimate`` (conservative
    flat rate) and ``warn`` (report $0, log) are explicit opt-ins via
    ``KAZENAI_UNKNOWN_MODEL_POLICY``; ``warn`` is ignored in production-like
    deployments, where it would let spend past the cap unseen.
    """
    explicit = os.getenv("KAZENAI_UNKNOWN_MODEL_POLICY", "").strip().lower()
    if explicit not in _POLICIES:
        return "block"
    if explicit == "warn" and _deployment_mode() in _PRODUCTION_LIKE:
        return "block"
    return explicit


def resolve_model_key(
    model: str,
    known_keys,
    aliases: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Single source of truth for "what pricing key does this model resolve to?".

    Resolution order for prices: exact key, then an explicit alias. Nothing is
    inherited. ``-latest`` and a dated snapshot that is not in the alias table
    are unknown. ``default_max_output_tokens`` may still family-match; this
    function must not.

    Both ``is_known_model`` and ``TokenCostEngine.price`` delegate here so that
    "known" and "priced" can never disagree.
    """
    key = str(model or "").strip().lower()
    if not key:
        return None
    keys = set(known_keys)
    if key in keys:
        return key
    if aliases:
        canonical = aliases.get(key)
        if canonical and canonical in keys:
            return canonical
    return None


def is_known_model(
    model: str,
    known_keys: set[str],
    aliases: Optional[Mapping[str, str]] = None,
) -> bool:
    return resolve_model_key(model, known_keys, aliases) is not None


def enforce_unknown_model(
    model: str,
    *,
    known_keys: set[str],
    aliases: Optional[Mapping[str, str]] = None,
) -> None:
    if is_known_model(model, known_keys, aliases):
        return
    policy = unknown_model_policy()
    if policy == "block":
        raise UnknownModelError(
            f"Unknown model {model!r} has no price, so KazenAI cannot bound its spend "
            "and blocks it (unknown-model policy: block, the default). Add pricing via "
            "KAZENAI_PRICING_JSON, pin a priced model, or opt in to "
            "KAZENAI_UNKNOWN_MODEL_POLICY=estimate."
        )


def estimate_unknown_model_pricing():
    """Return (input_per_1k, output_per_1k) for conservative unknown-model estimates."""
    return _ESTIMATE_INPUT_PER_1K, _ESTIMATE_OUTPUT_PER_1K
