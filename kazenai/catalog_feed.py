"""Live price catalog for processes connected to Agent FinOps.

The bundled snapshot is always available and is what an SDK without FinOps
uses. A process configured with ``KAZENAI_FINOPS_URL`` (and an API key) also
fetches the signed catalog FinOps serves at ``GET /v1/pricing/catalog``,
verifies it against the pinned KazenAI pricing keys, and installs it. FinOps
prices every reservation and settlement itself and keeps the higher figure, so
this feed keeps local refusals (new models, expired promotions) current; it is
never the only thing standing between a request and an under-reservation.

* Never blocks the call path on the network: the first priced call loads a
  verified on-disk cache (if any) and starts one background refresh.
* Never raises into the call path. A bad or missing catalog leaves the
  previous catalog active.
* ``KAZENAI_PRICING_LIVE=0`` turns the feed off. ``KAZENAI_PRICING_REFRESH_S``
  (default 900) sets the refresh interval. ``KAZENAI_PRICING_CACHE_DIR`` (default
  ``$XDG_CACHE_HOME/kazenai`` or ``~/.cache/kazenai``) holds the verified copy
  (mode 0600 in a 0700 directory owned by this user, else no cache is used).
"""

from __future__ import annotations

import logging
import os
import stat
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from . import catalog_document as cd
from . import outbound_http

_log = logging.getLogger("kazenai.catalog_feed")

CACHE_NAME = "price-catalog.signed.json"
DEFAULT_REFRESH_S = 900.0
FETCH_TIMEOUT_S = 5.0

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"pid": None, "cache_loaded": False, "last_attempt": 0.0, "etag": None, "thread": None}


def _reset_after_fork() -> None:
    _STATE.update({"pid": None, "cache_loaded": False, "last_attempt": 0.0, "etag": None, "thread": None})


if hasattr(os, "register_at_fork"):  # pragma: no branch
    os.register_at_fork(after_in_child=_reset_after_fork)


def _refresh_interval() -> float:
    try:
        return max(30.0, float(os.getenv("KAZENAI_PRICING_REFRESH_S", str(DEFAULT_REFRESH_S))))
    except ValueError:
        return DEFAULT_REFRESH_S


def _endpoint() -> tuple[str, Optional[str]]:
    from .spine.guard import _api_key, _finops_url

    return (str(_finops_url() or "").rstrip("/"), _api_key())


def enabled() -> bool:
    if os.getenv("KAZENAI_PRICING_LIVE", "1").strip() == "0":
        return False
    base, _key = _endpoint()
    if not base:
        return False
    try:
        if not cd.trusted_keys():
            return False
        import cryptography  # noqa: F401
    except (cd.CatalogDocumentError, ImportError):
        return False
    return True


def cache_path() -> Optional[str]:
    root = os.getenv("KAZENAI_PRICING_CACHE_DIR", "").strip()
    if not root:
        base = os.getenv("XDG_CACHE_HOME", "").strip() or os.path.join(os.path.expanduser("~"), ".cache")
        root = os.path.join(base, "kazenai")
    try:
        os.makedirs(root, mode=0o700, exist_ok=True)
        info = os.stat(root)
    except OSError:
        return None
    if info.st_uid != os.geteuid() or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        _log.warning("price catalog cache dir is not private to this user; not caching")
        return None
    return os.path.join(root, CACHE_NAME)


def _load_cache() -> None:
    path = cache_path()
    if path is None or not os.path.exists(path):
        return
    try:
        with open(path, "rb") as handle:
            raw = handle.read(cd.MAX_ENVELOPE_BYTES + 1)
        cd.accept_envelope(raw)
    except (OSError, cd.CatalogDocumentError) as exc:
        # Expired, older than the bundled snapshot, or tampered: ignore it.
        _log.info("cached price catalog not used: %s", exc)


def _write_cache(raw: bytes) -> None:
    path = cache_path()
    if path is None:
        return
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, path)
    except OSError:
        _log.debug("could not write the price catalog cache", exc_info=True)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def refresh_now() -> str:
    """Fetch, verify, and install the catalog FinOps serves. Returns the outcome."""
    base, key = _endpoint()
    if not base:
        return "unconfigured"
    request = urllib.request.Request(f"{base}/v1/pricing/catalog", method="GET")
    request.add_header("Accept", "application/json")
    if key:
        request.add_header("X-API-Key", str(key))
    etag = _STATE.get("etag")
    if etag:
        request.add_header("If-None-Match", str(etag))
    try:
        # No redirects: the SDK key goes only to the configured FinOps URL.
        with outbound_http.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:  # noqa: S310 - configured FinOps URL
            raw = response.read(cd.MAX_ENVELOPE_BYTES + 1)
            new_etag = response.headers.get("ETag")
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return "not_modified"
        if exc.code in (204, 404):
            return "none_published"
        _log.info("price catalog fetch refused by FinOps: HTTP %s", exc.code)
        return "http_error"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        _log.info("price catalog fetch failed: %s", type(exc).__name__)
        return "unreachable"
    try:
        cd.accept_envelope(raw)
    except cd.CatalogDocumentError as exc:
        _log.warning("price catalog from FinOps refused: %s", exc)
        return "refused"
    _STATE["etag"] = new_etag
    _write_cache(raw)
    return "installed"


def _run_refresh() -> None:
    try:
        refresh_now()
    except Exception:  # pragma: no cover - the call path must never see this
        _log.debug("price catalog refresh failed", exc_info=True)


def ensure_current() -> None:
    """Hot-path hook: load the cache once and keep one background refresh going."""
    try:
        if not enabled():
            return
        with _LOCK:
            if _STATE["pid"] != os.getpid():
                _reset_after_fork()
                _STATE["pid"] = os.getpid()
            if not _STATE["cache_loaded"]:
                _STATE["cache_loaded"] = True
                _load_cache()
            thread = _STATE["thread"]
            now = time.monotonic()
            due = now - float(_STATE["last_attempt"] or 0.0) >= _refresh_interval() or not _STATE["last_attempt"]
            if due and (thread is None or not thread.is_alive()):
                _STATE["last_attempt"] = now
                worker = threading.Thread(target=_run_refresh, name="kazenai-price-catalog", daemon=True)
                _STATE["thread"] = worker
                worker.start()
    except Exception:  # pragma: no cover
        _log.debug("price catalog feed check failed", exc_info=True)


def reset_for_tests() -> None:
    with _LOCK:
        thread = _STATE.get("thread")
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        _reset_after_fork()
