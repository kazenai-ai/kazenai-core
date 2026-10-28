"""Request-level backstop for monitored OpenAI / Anthropic clients.

``monitor()`` guards the SDK methods it knows (``chat.completions.create``,
``responses.parse``, ``messages.stream`` ...) by replacing them on the resource
objects. A reference taken *before* ``monitor()`` -- a bound ``create``, a
``with_raw_response`` wrapper, ``client.request`` itself -- still points at the
original method and would reach the provider without a budget check.

The gate runs at two layers of the monitored client, with one policy:

- ``client.request()``, which every SDK method calls when it runs (a saved
  ``create`` or ``with_raw_response`` wrapper included), so callers get a plain
  ``UnsupportedModeError``;
- the client's own gated view of its ``httpx`` client, which every request reaches
  when it is sent, including a ``client.request`` saved before ``monitor()``. The
  SDK reports a refusal there as ``APIConnectionError`` (its ``__cause__`` is the
  ``UnsupportedModeError``); nothing is sent either way.

A request passes when:

- a guarded wrapper of *this* client is dispatching it (it has reserved first).
  The authorization is a per-gate capability, so a guarded call on one client
  never authorizes a request on another; or
- it cannot spend: reads, deletes, and a short list of free endpoints; or
- the caller named its endpoint in ``allow_unmetered`` (sent without a check).

Anything else raises ``UnsupportedModeError`` before it is sent.

Boundary: this covers everything sent through a monitored client. It does not
stop code that holds the provider key and builds a new client, sends raw HTTP,
or keeps a reference to the SDK's private ``_client`` taken before ``monitor()``.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from .enforcement import UnsupportedModeError

# Capabilities of the gates whose guarded wrappers are dispatching right now.
_AUTHORIZED: ContextVar[FrozenSet[object]] = ContextVar("kazenai_dispatch_authorized", default=frozenset())

# Methods that never start paid work.
_SAFE_METHODS = frozenset({"get", "head", "options", "delete"})

# POST endpoints that do not bill model usage. ``*`` matches one path segment.
_FREE_POSTS: Tuple[str, ...] = (
    # OpenAI
    "/files",
    "/uploads",
    "/uploads/*/parts",
    "/uploads/*/complete",
    "/uploads/*/cancel",
    "/moderations",
    "/responses/input_tokens",
    "/responses/*/cancel",
    "/batches/*/cancel",
    "/fine_tuning/jobs/*/cancel",
    # Anthropic
    "/messages/count_tokens",
    "/files",
    "/messages/batches/*/cancel",
)

GATE_ATTR = "_kazenai_dispatch_gate"


def _normalize_path(path: Any) -> str:
    text = str(path or "").split("?", 1)[0].split("#", 1)[0]
    if not text.startswith("/"):
        text = "/" + text
    return text.rstrip("/") or "/"


def _matches(path: str, pattern: str) -> bool:
    pattern = _normalize_path(pattern)
    if "*" not in pattern:
        return path == pattern
    have = path.split("/")
    want = pattern.split("/")
    if len(have) != len(want):
        return False
    return all(w == "*" or w == h for h, w in zip(have, want))


def _prefix_match(path: str, prefix: str) -> bool:
    prefix = _normalize_path(prefix)
    return path == prefix or path.startswith(prefix + "/")


class DispatchGate:
    """One monitored client's gate. ``authorized()`` is its dispatch capability."""

    def __init__(self, *, base_path: str = "/", allow_unmetered: Sequence[str] = ()) -> None:
        self._capability = object()
        self.base_path = _normalize_path(base_path)
        self.allow_unmetered = tuple(str(p) for p in allow_unmetered or ())

    @contextmanager
    def authorized(self) -> Iterator[None]:
        """Requests this gate sees inside the block were checked by a guarded wrapper."""
        token = _AUTHORIZED.set(_AUTHORIZED.get() | {self._capability})
        try:
            yield
        finally:
            _AUTHORIZED.reset(token)

    def endpoint(self, url_path: str) -> str:
        """The API endpoint for a request path, without the client's base path or /v1."""
        path = _normalize_path(url_path)
        if self.base_path != "/" and (path == self.base_path or path.startswith(self.base_path + "/")):
            path = _normalize_path(path[len(self.base_path):])
        if path == "/v1" or path.startswith("/v1/"):
            path = _normalize_path(path[3:])
        return path

    def check(self, method: str, url_path: str) -> None:
        if self._capability in _AUTHORIZED.get():
            return
        method = str(method or "").lower()
        if method in _SAFE_METHODS:
            return
        endpoint = self.endpoint(url_path)
        if any(_matches(endpoint, p) for p in _FREE_POSTS):
            return
        if any(_prefix_match(endpoint, p) for p in self.allow_unmetered):
            return
        raise UnsupportedModeError(
            f"{method.upper() or 'request'} {endpoint} on a monitored client did not go through a "
            "KazenAI budget check, so it was not sent. This happens when a provider method, "
            "with_raw_response wrapper or client.request was saved before monitor(), when another "
            "monitored client's call triggers it, or when the endpoint is not a supported Control "
            "surface (for example embeddings, images, audio, batches). Read the method from the "
            "client after monitor(), or pass "
            f"monitor(..., allow_unmetered=({endpoint!r},)) to send it without a budget check.",
            error_code="unguarded_dispatch",
        )


_GATED_CLASSES: dict[tuple[type, bool], type] = {}


def _gated_class(base: type, is_async: bool) -> type:
    """A subclass of the SDK's httpx client type whose ``send`` checks the gate.

    It must stay an ``httpx.Client`` / ``AsyncClient`` (the SDK checks that when a
    client is copied) and must not inherit the SDK wrapper's ``__del__``, which
    would close the shared connection pool when the view is dropped.
    """
    key = (base, is_async)
    cls = _GATED_CLASSES.get(key)
    if cls is not None:
        return cls

    if is_async:

        async def send(self: Any, request: Any, **kwargs: Any) -> Any:
            self._kazenai_gate.check(request.method, request.url.path)
            return await base.send(self, request, **kwargs)

    else:

        def send(self: Any, request: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            self._kazenai_gate.check(request.method, request.url.path)
            return base.send(self, request, **kwargs)

    def __del__(self: Any) -> None:
        # The view shares the original client's state; closing is the owner's job.
        return None

    cls = type(f"KazenAIGated{base.__name__}", (base,), {"send": send, "__del__": __del__})
    _GATED_CLASSES[key] = cls
    return cls


def _gated_view(inner: Any, gate: DispatchGate, *, is_async: bool) -> Any:
    """A view of ``inner`` sharing its connection pool and settings, gated on send."""
    base = getattr(inner, "_kazenai_gated_base", None)
    if base is not None:
        inner = base  # re-gating (a clone or a second monitor()): wrap the real client
    # Our attributes live on a per-gate subclass: instance attributes would land in
    # the __dict__ shared with ``inner``.
    cls = type(
        f"KazenAIGated{type(inner).__name__}",
        (_gated_class(type(inner), is_async),),
        {"_kazenai_gate": gate, "_kazenai_gated_base": inner},
    )
    view = object.__new__(cls)
    view.__dict__ = inner.__dict__
    return view


def install_dispatch_gate(
    client: Any,
    patches: List[Tuple[Any, str, Any]],
    *,
    allow_unmetered: Optional[Iterable[str]] = None,
    is_async: bool = False,
) -> Optional[DispatchGate]:
    """Gate every request ``client`` sends; returns the gate (None if unsupported).

    The original ``httpx`` client is recorded in ``patches`` so ``undo()`` restores it.
    """
    inner = getattr(client, "_client", None)
    if inner is None or not callable(getattr(inner, "send", None)):
        return None
    base_url = getattr(client, "base_url", None)
    gate = DispatchGate(
        base_path=str(getattr(base_url, "path", "/") or "/"),
        allow_unmetered=tuple(allow_unmetered or ()),
    )
    try:
        view = _gated_view(inner, gate, is_async=is_async)
        setattr(client, "_client", view)
        setattr(client, GATE_ATTR, gate)
    except Exception:
        return None
    patches.append((client, "_client", inner))
    _gate_request_method(client, gate, patches, is_async=is_async)
    return gate


def _options_path(options: Any) -> str:
    url = str(getattr(options, "url", "") or "")
    return urlsplit(url).path if "://" in url else url


def _gate_request_method(client: Any, gate: DispatchGate, patches: List[Tuple[Any, str, Any]], *, is_async: bool) -> None:
    """Check at ``client.request()`` too, so the common paths raise a plain error."""
    orig = getattr(client, "request", None)
    if not callable(orig):
        return
    base_path = gate.base_path

    def _check(options: Any) -> None:
        path = _options_path(options)
        # Relative SDK paths ("/embeddings") are under the client's base path.
        if base_path != "/" and not path.startswith(base_path + "/"):
            path = base_path + "/" + path.lstrip("/")
        gate.check(str(getattr(options, "method", "") or ""), path)

    if is_async:

        async def gated_request(cast_to: Any, options: Any, *args: Any, **kwargs: Any) -> Any:
            _check(options)
            return await orig(cast_to, options, *args, **kwargs)

        wrapper: Any = gated_request
    else:

        def gated_request(cast_to: Any, options: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            _check(options)
            return orig(cast_to, options, *args, **kwargs)

        wrapper = gated_request
    try:
        setattr(wrapper, "__kazenai_wrapped__", True)
        setattr(client, "request", wrapper)
    except Exception:
        return
    patches.append((client, "request", orig))
