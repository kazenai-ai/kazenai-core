"""SDK business attribution context (FINAL_1_b P1-3).

Attribution is **metadata only**. It cannot grant access, choose a workspace, or
lower an authorized amount. Access-control identity remains org/workspace + principal
(see ``RunContext``); economics use opaque ``business_subject_ref`` + ``feature_id`` +
``workflow_id``.

Missing fields resolve to explicit ``attribution_state`` of ``unattributed`` or
``unknown`` per contracts rules — never collapse into a fabricated customer.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, fields, replace
from typing import Any, Iterator, Mapping, Optional

ATTRIBUTION_VERSION = "business.attribution.v1"

_ATTR: contextvars.ContextVar[Optional["AttributionContext"]] = contextvars.ContextVar(
    "kazenai_attribution_context",
    default=None,
)


class AttributionConflict(ValueError):
    """Raised when nested attribution sets the same field to a different value."""

    def __init__(self, field: str, existing: Any, new: Any) -> None:
        self.field = field
        self.existing = existing
        self.new = new
        super().__init__(
            f"conflicting attribution for {field!r}: existing={existing!r} new={new!r}"
        )


@dataclass(frozen=True)
class AttributionContext:
    """Opaque economic / execution attribution for a request scope.

    All fields are optional strings. Values must not contain credentials, prompts,
    or response bodies — only stable identifiers and opaque refs.
    """

    business_subject_ref: Optional[str] = None
    feature_id: Optional[str] = None
    workflow_id: Optional[str] = None
    operation_id: Optional[str] = None
    attempt_id: Optional[str] = None

    def attribution_state(self) -> str:
        """Explicit defaulting: attributed / unknown / unattributed."""
        if self.business_subject_ref:
            return "attributed"
        if self.feature_id or self.workflow_id:
            return "unknown"
        return "unattributed"

    def normalized(self) -> "AttributionContext":
        """Strip empty strings to None so missing stays explicitly unattributed."""

        def _clean(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        return AttributionContext(
            business_subject_ref=_clean(self.business_subject_ref),
            feature_id=_clean(self.feature_id),
            workflow_id=_clean(self.workflow_id),
            operation_id=_clean(self.operation_id),
            attempt_id=_clean(self.attempt_id),
        )

    def to_dict(self) -> dict[str, Any]:
        """Safe metadata dict for FinOps reserve / receipts (no secrets)."""
        ctx = self.normalized()
        out: dict[str, Any] = {
            "attribution_version": ATTRIBUTION_VERSION,
            "attribution_state": ctx.attribution_state(),
        }
        for name in (
            "business_subject_ref",
            "feature_id",
            "workflow_id",
            "operation_id",
            "attempt_id",
        ):
            value = getattr(ctx, name)
            if value is not None:
                out[name] = value
        return out

    def __repr__(self) -> str:
        # Never dump unexpected kwargs / secrets; only declared metadata fields.
        parts = [f"{f.name}={getattr(self, f.name)!r}" for f in fields(self)]
        return f"AttributionContext({', '.join(parts)})"


def get_attribution() -> Optional[AttributionContext]:
    return _ATTR.get()


def _merge_nested(parent: AttributionContext, child: AttributionContext) -> AttributionContext:
    """Merge nested attribution; identical values OK, conflicting values raise."""
    updates: dict[str, Any] = {}
    for f in fields(AttributionContext):
        p = getattr(parent, f.name)
        c = getattr(child, f.name)
        if c is None or c == "":
            continue
        if p is not None and p != "" and p != c:
            raise AttributionConflict(f.name, p, c)
        updates[f.name] = c
    return replace(parent, **updates).normalized() if updates else parent.normalized()


@contextlib.contextmanager
def use_attribution(
    business_subject_ref: Optional[str] = None,
    feature_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    *,
    ctx: Optional[AttributionContext] = None,
) -> Iterator[AttributionContext]:
    """Install attribution for the current ContextVar scope (concurrency-safe).

    Nesting with the same values is allowed. Nesting with different values for the
    same field raises :class:`AttributionConflict`. Does not mutate process globals.
    """
    incoming = (
        ctx
        if ctx is not None
        else AttributionContext(
            business_subject_ref=business_subject_ref,
            feature_id=feature_id,
            workflow_id=workflow_id,
            operation_id=operation_id,
            attempt_id=attempt_id,
        )
    ).normalized()
    parent = _ATTR.get()
    effective = _merge_nested(parent, incoming) if parent is not None else incoming
    token = _ATTR.set(effective)
    try:
        yield effective
    finally:
        _ATTR.reset(token)


def resolve_attribution(
    *,
    business_subject_ref: Optional[str] = None,
    feature_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    feature: Optional[str] = None,
) -> AttributionContext:
    """Prefer ambient AttributionContext, then fall back to explicit kwargs.

    ``feature`` is accepted for compatibility and maps to ``feature_id`` when
    ``feature_id`` is absent.
    """
    ambient = get_attribution()
    kwargs_feature = feature_id if feature_id is not None else feature
    kwargs_ctx = AttributionContext(
        business_subject_ref=business_subject_ref,
        feature_id=kwargs_feature,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
    ).normalized()
    if ambient is None:
        return kwargs_ctx
    # Ambient wins per-field; fill gaps from kwargs.
    return AttributionContext(
        business_subject_ref=ambient.business_subject_ref or kwargs_ctx.business_subject_ref,
        feature_id=ambient.feature_id or kwargs_ctx.feature_id,
        workflow_id=ambient.workflow_id or kwargs_ctx.workflow_id,
        operation_id=ambient.operation_id or kwargs_ctx.operation_id,
        attempt_id=ambient.attempt_id or kwargs_ctx.attempt_id,
    ).normalized()


def attribution_for_reserve_body(
    *,
    business_subject_ref: Optional[str] = None,
    feature_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    feature: Optional[str] = None,
) -> dict[str, Any]:
    """Fields safe to attach to a FinOps reserve / check body."""
    ctx = resolve_attribution(
        business_subject_ref=business_subject_ref,
        feature_id=feature_id,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
        feature=feature,
    )
    body: dict[str, Any] = {"attribution_state": ctx.attribution_state()}
    # Compat: legacy ``feature`` alias when feature_id present.
    if ctx.feature_id:
        body["feature_id"] = ctx.feature_id
        body["feature"] = ctx.feature_id
    if ctx.business_subject_ref:
        body["business_subject_ref"] = ctx.business_subject_ref
    if ctx.workflow_id:
        body["workflow_id"] = ctx.workflow_id
    if ctx.operation_id:
        body["operation_id"] = ctx.operation_id
    if ctx.attempt_id:
        body["attempt_id"] = ctx.attempt_id
    return body


def assert_metadata_only(mapping: Mapping[str, Any]) -> None:
    """Reject credential-shaped keys from attribution payloads (defense in depth)."""
    forbidden = (
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
        "cookie",
        "messages",
        "prompt",
        "content",
        "inputs",
        "outputs",
    )
    for key in mapping:
        lowered = str(key).strip().lower().replace("-", "_")
        if any(part in lowered for part in forbidden):
            raise ValueError(f"attribution must not carry credential/content field {key!r}")
