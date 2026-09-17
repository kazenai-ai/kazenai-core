"""Shared input-validation surface for KazenAI services.

Canonical home for prompt-injection detection and request-bounding helpers.
Import from here rather than re-implementing per service.
"""

from __future__ import annotations

from .bounds import (
    DEFAULT_MAX_ITEMS,
    DEFAULT_MAX_JSON_BYTES,
    DEFAULT_MAX_JSON_DEPTH,
    DEFAULT_MAX_STR,
    ValidatedRequest,
    bounded_list,
    bounded_str,
    enforce_json_bounds,
    json_depth,
    json_size_bytes,
)
from .injection import (
    BLOCK_THRESHOLD,
    InjectionAttemptDetected,
    InjectionPattern,
    InjectionResult,
    PromptInjectionDetector,
    get_detector,
    scan_and_raise_or_http,
)

__all__ = [
    "BLOCK_THRESHOLD",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_MAX_JSON_BYTES",
    "DEFAULT_MAX_JSON_DEPTH",
    "DEFAULT_MAX_STR",
    "InjectionAttemptDetected",
    "InjectionPattern",
    "InjectionResult",
    "PromptInjectionDetector",
    "ValidatedRequest",
    "bounded_list",
    "bounded_str",
    "enforce_json_bounds",
    "get_detector",
    "json_depth",
    "json_size_bytes",
    "scan_and_raise_or_http",
]
