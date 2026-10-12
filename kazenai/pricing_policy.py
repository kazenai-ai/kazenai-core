from __future__ import annotations

import os
from typing import Mapping, Optional

from .model_pricing import _is_version_bump


class UnknownModelError(ValueError):
    """Raised when an unrecognized model is used under block policy."""


# Conservative fallback (USD per 1k tokens) when policy=estimate.
_ESTIMATE_INPUT_PER_1K = 0.015
_ESTIMATE_OUTPUT_PER_1K = 0.075


def _deployment_mode() -> str:
    return os.getenv(
        "KAZENAI_DEPLOYMENT_MODE",
        os.getenv("KAZENAI_ENV", "development"),
    ).strip().lower()


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

    Resolution order: exact key → alias → longest-prefix match (the query must
    be a versioned/suffixed extension of a known base key, at a token boundary).
    Returns the resolved pricing key, or None if the model is unknown.

    Both ``is_known_model`` and ``TokenCostEngine.price`` delegate here so that
    "known" and "priced" can never disagree (see P0-2): a model is known iff it
    resolves to a real pricing row. Substring matching is intentionally NOT used
    — it is the loosest and most dangerous form (a proxy/gateway-prefixed name
    like "my-gpt-4o-proxy" must be treated as unknown, not silently priced).
    """
    key = str(model or "").strip().lower()
    if not key:
        return None
    if key in known_keys:
        return key
    if aliases:
        canonical = aliases.get(key)
        if canonical and canonical in known_keys:
            return canonical
    # Longest-prefix match in the key.startswith(k) direction only, requiring a
    # token boundary after the prefix so "gpt-4omega" can't match "gpt-4o".
    best = ""
    for k in known_keys:
        if not k or len(k) <= len(best) or not key.startswith(k):
            continue
        if not key[len(k)].isalnum() and not _is_version_bump(key[len(k):]):
            best = k
    return best or None


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
