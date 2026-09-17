"""Shared input-bounding primitives for KazenAI request models.

These helpers give every service a consistent way to cap the size and shape of
untrusted input at the validation boundary, instead of each service inventing
its own (or omitting bounds entirely).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Conservative service-wide defaults; individual fields can override.
DEFAULT_MAX_STR = 10_000
DEFAULT_MAX_ITEMS = 200
DEFAULT_MAX_JSON_BYTES = 256_000
DEFAULT_MAX_JSON_DEPTH = 12


def bounded_str(max_length: int = DEFAULT_MAX_STR, *, min_length: int = 0, **kwargs: Any):
    """A pydantic ``Field`` for a length-bounded string."""
    return Field(min_length=min_length, max_length=max_length, **kwargs)


def bounded_list(max_items: int = DEFAULT_MAX_ITEMS, *, min_items: int = 0, **kwargs: Any):
    """A pydantic ``Field`` for a length-bounded list."""
    return Field(min_length=min_items, max_length=max_items, **kwargs)


def json_size_bytes(value: Any) -> int:
    """Serialized UTF-8 size of *value*. Raises ValueError if not serializable."""
    try:
        return len(json.dumps(value, default=str).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not JSON-serializable: {exc}") from exc


def json_depth(value: Any, _depth: int = 0, *, limit: int = DEFAULT_MAX_JSON_DEPTH) -> int:
    """Maximum nesting depth of *value*, short-circuiting once *limit* is passed."""
    if _depth > limit:
        return _depth
    if isinstance(value, dict):
        return max((json_depth(v, _depth + 1, limit=limit) for v in value.values()), default=_depth)
    if isinstance(value, (list, tuple)):
        return max((json_depth(v, _depth + 1, limit=limit) for v in value), default=_depth)
    return _depth


def enforce_json_bounds(
    value: Any,
    *,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
    max_depth: int = DEFAULT_MAX_JSON_DEPTH,
    field: str = "value",
) -> Any:
    """Validate that *value* is within size/depth limits; raise ValueError otherwise.

    Returns the value unchanged so it can be used inline in pydantic validators.
    """
    size = json_size_bytes(value)
    if size > max_bytes:
        raise ValueError(f"{field} too large ({size} bytes; max {max_bytes})")
    if json_depth(value, limit=max_depth) > max_depth:
        raise ValueError(f"{field} nested too deeply (max depth {max_depth})")
    return value


class ValidatedRequest(BaseModel):
    """Base model for request payloads at runtime execution boundaries.

    - ``extra="forbid"`` rejects unexpected fields (fail closed on unknown input).
    - ``str_strip_whitespace`` normalizes leading/trailing whitespace.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
