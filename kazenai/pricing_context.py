"""Task-local pricing scope. No catalog data and no I/O."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Optional

CURRENT_PRICING: ContextVar[Any] = ContextVar("kazenai_current_pricing", default=None)
CALL_PROVIDER: ContextVar[str] = ContextVar("kazenai_call_provider", default="")
CALL_SURFACE: ContextVar[str] = ContextVar("kazenai_call_surface", default="")
CALL_CLIENT: ContextVar[Any] = ContextVar("kazenai_call_client", default=None)
CALL_ORG: ContextVar[str] = ContextVar("kazenai_call_org", default="")
OBSERVE_UNBOUNDED: ContextVar[tuple] = ContextVar("kazenai_observe_unbounded", default=())
# id() of the request kwargs this scope was armed for. Projection reuses the
# scope only for that same request object, never for an unrelated later call.
CALL_KWARGS_ID: ContextVar[int] = ContextVar("kazenai_call_kwargs_id", default=0)

ALL_VARS = (CURRENT_PRICING, CALL_PROVIDER, CALL_SURFACE, CALL_CLIENT, CALL_ORG, OBSERVE_UNBOUNDED, CALL_KWARGS_ID)


def current_pricing() -> Any:
    return CURRENT_PRICING.get()


def bind_call(
    *,
    provider: str,
    surface: str,
    client: Any,
    org_id: str,
    observe_unbounded: tuple = (),
    kwargs: Any = None,
) -> list:
    """Install the scope used by pre-call projection. Returns reset tokens."""
    return [
        CALL_PROVIDER.set(provider),
        CALL_SURFACE.set(surface),
        CALL_CLIENT.set(client),
        CALL_ORG.set(str(org_id or "")),
        OBSERVE_UNBOUNDED.set(tuple(observe_unbounded or ())),
        CALL_KWARGS_ID.set(id(kwargs) if kwargs is not None else 0),
    ]


def reset_call(tokens: list) -> None:
    if len(tokens) > 5:
        CALL_KWARGS_ID.reset(tokens[5])
    for var, token in (
        (OBSERVE_UNBOUNDED, tokens[4]),
        (CALL_ORG, tokens[3]),
        (CALL_CLIENT, tokens[2]),
        (CALL_SURFACE, tokens[1]),
        (CALL_PROVIDER, tokens[0]),
    ):
        var.reset(token)


def pricing_scope_active(kwargs: Any = None) -> bool:
    """True inside an armed call. With ``kwargs``, only for the request it was armed for."""
    if not CALL_PROVIDER.get():
        return False
    if kwargs is None:
        return True
    armed_for = CALL_KWARGS_ID.get()
    return bool(armed_for) and armed_for == id(kwargs)


def scope_provider() -> str:
    return CALL_PROVIDER.get()


def scope_surface() -> str:
    return CALL_SURFACE.get()


def scope_client() -> Any:
    return CALL_CLIENT.get()


def scope_org() -> str:
    return CALL_ORG.get()


def scope_observe_unbounded() -> tuple:
    return tuple(OBSERVE_UNBOUNDED.get() or ())


def set_current_pricing(pricing: Optional[Any]) -> None:
    CURRENT_PRICING.set(pricing)
