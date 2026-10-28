"""Kept-alive HTTP connections from the SDK to Agent FinOps.

A guarded call makes up to three requests to Agent FinOps in a row: reserve,
``provider_started`` and ``usage_known``. A new TCP and TLS connection for each
one costs two extra network round trips per request, which is most of the added
latency when the app runs far from Agent FinOps. These helpers keep connections
open and reuse them:

- sync callers share one ``httpx.Client`` per process;
- async callers share one ``httpx.AsyncClient`` per event loop, because an async
  client cannot move between loops. ``asyncio.run`` closes it on the way out.

:func:`urlopen` keeps the ``urllib.request.urlopen`` contract the guard was
written against: HTTP 4xx and 5xx raise ``urllib.error.HTTPError`` with the body
readable, transport failures raise ``urllib.error.URLError``, and a read timeout
raises ``TimeoutError``. Every deny, conflict and fail-closed path therefore
behaves as before. Sync requests trust the same certificates as urllib (the
system store, or ``SSL_CERT_FILE``); async requests keep httpx's defaults.

A reserve, settle or GET that fails because the server closed a kept-alive
connection is sent once more. Agent FinOps answers a repeated request with the
same idempotency key as a duplicate of the first. The deprecated legacy routes
are never resent.

``KAZENAI_FINOPS_KEEPALIVE_S`` (default 15) is how long an idle connection may
be reused. It stays below common NAT and firewall idle timeouts, which drop a
connection without telling either end. ``0`` opens a new connection for every
request, as before 1.2.0. It is read once, when the first pool is built.

A forked child (for example a gunicorn worker) starts with empty pools and never
touches its parent's sockets.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import email.message
import io
import os
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, AsyncIterator, Optional

import httpx

DEFAULT_KEEPALIVE_S = 15.0

# Failures that mean the connection broke under the request, not that the server
# answered or took too long. Timeouts are not resent: the fail-closed bound holds.
_BROKEN_CONNECTION = (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)
_RESENDABLE_PATHS = ("/v1/budget/reserve", "/v1/budget/settle")

_lock = threading.Lock()
_sync_client: Optional[httpx.Client] = None
# One entry per running event loop: id(loop) -> (loop, factory, client, closer).
_async_pools: dict[int, tuple[Any, Any, Any, Any]] = {}
# Pools a forked child inherited. Kept referenced so the child never closes them.
_inherited: list[Any] = []


def keepalive_s() -> float:
    raw = os.getenv("KAZENAI_FINOPS_KEEPALIVE_S", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_KEEPALIVE_S
    except ValueError:
        value = DEFAULT_KEEPALIVE_S
    return value if value > 0 else 0.0


def _limits() -> httpx.Limits:
    idle = keepalive_s()
    # No cap on open connections: a burst opens extra ones, as before, and only
    # the idle ones above the keep-alive count are closed.
    if idle <= 0:
        return httpx.Limits(max_connections=None, max_keepalive_connections=0)
    return httpx.Limits(max_connections=None, max_keepalive_connections=20, keepalive_expiry=idle)


def _resendable(method: str, url: str) -> bool:
    if method.upper() == "GET":
        return True
    path = urllib.parse.urlsplit(url).path.rstrip("/")
    return path.endswith(_RESENDABLE_PATHS)


def _sync() -> httpx.Client:
    global _sync_client
    client = _sync_client
    if client is None:
        with _lock:
            if _sync_client is None:
                _sync_client = httpx.Client(
                    verify=ssl.create_default_context(),
                    limits=_limits(),
                    trust_env=True,
                )
            client = _sync_client
    return client


def _message(headers: httpx.Headers) -> email.message.Message:
    msg = email.message.Message()
    for name, value in headers.multi_items():
        msg[name] = value
    return msg


class _Response:
    """The parts of ``http.client.HTTPResponse`` the guard reads."""

    def __init__(self, resp: httpx.Response) -> None:
        self.status = resp.status_code
        self.code = resp.status_code
        self.reason = resp.reason_phrase
        self.headers = _message(resp.headers)
        self.url = str(resp.url)
        self._body = io.BytesIO(resp.content)

    def read(self, amt: Optional[int] = None) -> bytes:
        return self._body.read() if amt is None or amt < 0 else self._body.read(amt)

    def getcode(self) -> int:
        return self.status

    def close(self) -> None:
        self._body.close()

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def urlopen(req: urllib.request.Request, timeout: float) -> _Response:
    """``urllib.request.urlopen`` for Agent FinOps requests, over a kept-alive connection."""
    method = req.get_method()
    url = req.full_url
    headers = dict(req.header_items())

    def send() -> httpx.Response:
        return _sync().request(method, url, content=req.data, headers=headers, timeout=timeout)

    try:
        try:
            resp = send()
        except _BROKEN_CONNECTION:
            if not _resendable(method, url):
                raise
            resp = send()
    except httpx.ConnectTimeout as exc:
        raise urllib.error.URLError(TimeoutError("timed out")) from exc
    except httpx.TimeoutException as exc:
        raise TimeoutError("timed out") from exc
    except httpx.TransportError as exc:
        raise urllib.error.URLError(exc) from exc
    if resp.status_code >= 400:
        raise urllib.error.HTTPError(
            url, resp.status_code, resp.reason_phrase, _message(resp.headers), io.BytesIO(resp.content)
        )
    return _Response(resp)


async def _close_with_loop(client: Any) -> AsyncIterator[None]:
    """Parked async generator: ``loop.shutdown_asyncgens()`` resumes it to close the pool."""
    try:
        yield
    finally:
        _forget_loop_client(client)
        close = getattr(client, "aclose", None)
        if close is not None:
            with contextlib.suppress(Exception):
                await close()


def _forget_loop_client(client: Any) -> None:
    with _lock:
        for key, entry in list(_async_pools.items()):
            if entry[2] is client:
                del _async_pools[key]


async def _loop_client() -> Any:
    """The pooled async client for the running loop, or None outside asyncio."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    factory = httpx.AsyncClient
    with _lock:
        for key, entry in list(_async_pools.items()):
            if entry[0].is_closed():
                # Closed without shutdown_asyncgens; its sockets go with the objects.
                del _async_pools[key]
        entry = _async_pools.get(id(loop))
        if entry is not None and entry[0] is loop and entry[1] is factory:
            return entry[2]
        client = factory(limits=_limits(), trust_env=True)
        closer = _close_with_loop(client)
        _async_pools[id(loop)] = (loop, factory, client, closer)
    with contextlib.suppress(StopAsyncIteration):
        await closer.__anext__()
    return client


class _Session:
    """``get``/``post`` on the pooled client with this call's timeout."""

    def __init__(self, client: Any, timeout: Any) -> None:
        self._client = client
        self._timeout = timeout

    async def _send(self, method: str, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", self._timeout)
        send = getattr(self._client, method.lower())
        try:
            return await send(url, **kwargs)
        except _BROKEN_CONNECTION:
            if not _resendable(method, url):
                raise
            return await send(url, **kwargs)

    async def get(self, url: str, **kwargs: Any) -> Any:
        return await self._send("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> Any:
        return await self._send("POST", url, **kwargs)


@contextlib.asynccontextmanager
async def async_session(timeout: Any) -> AsyncIterator[_Session]:
    """Async requests to Agent FinOps over this loop's kept-alive connections.

    Outside asyncio (for example under trio) each use gets its own client, as before.
    """
    client = await _loop_client()
    if client is not None:
        yield _Session(client, timeout)
        return
    async with httpx.AsyncClient(timeout=timeout) as own:
        yield _Session(own, timeout)


def reset_pools() -> None:
    """Close the sync pool and forget the async ones (tests)."""
    global _sync_client
    with _lock:
        client, _sync_client = _sync_client, None
        _async_pools.clear()
    if client is not None:
        with contextlib.suppress(Exception):
            client.close()


def _forget_all() -> None:
    global _sync_client
    if _sync_client is not None:
        _inherited.append(_sync_client)
    _inherited.extend(_async_pools.values())
    _sync_client = None
    _async_pools.clear()


def _after_fork_in_child() -> None:
    # Another parent thread may have held the lock at fork time: start a new one.
    global _lock
    _lock = threading.Lock()
    _forget_all()


def _close_at_exit() -> None:
    client = _sync_client
    if client is not None:
        with contextlib.suppress(Exception):
            client.close()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
atexit.register(_close_at_exit)
