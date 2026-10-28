"""An open tier breaker (FinOps 409 ``tier_breaker``) is a deny, never a fail-open hold.

1.4: admission for the org/provider/model is refused until an operator resolves
the mismatch. A fail-open reservation profile covers FinOps being unreachable;
it must not turn a breaker refusal into a zero hold and a dispatch.
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import httpx
import pytest

from kazenai.enforcement import BudgetExceeded
from kazenai.spine import guard

_BREAKER = {
    "detail": {
        "lifecycle_version": "reservation.lifecycle.v1",
        "allowed": False,
        "error_code": "tier_breaker",
        "reason": "admission is closed until an operator settles the tier mismatch",
    }
}
_CAPS = {
    "ok": True,
    "effective_lifecycle": {"profile": "control", "lifecycle_version": "reservation.lifecycle.v1"},
    "capabilities": {"reservation_settlement": True},
}


@pytest.fixture
def fail_open(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops.test")
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "k")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.delenv("KAZENAI_FINOPS_ENFORCEMENT_MODE", raising=False)


class _Resp:
    def __init__(self, body: dict):
        self._body = body

    def read(self):
        return json.dumps(self._body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_sync_reserve_treats_an_open_breaker_as_a_deny(fail_open):
    def _urlopen(req, timeout=0):  # noqa: ARG001
        url = getattr(req, "full_url", None) or req.get_full_url()
        if "/lifecycle" in str(url):
            return _Resp(_CAPS)
        raise urllib.error.HTTPError(str(url), 409, "Conflict", {}, io.BytesIO(json.dumps(_BREAKER).encode()))

    with patch("kazenai.spine.transport.urlopen", side_effect=_urlopen):
        with pytest.raises(BudgetExceeded, match="tier"):
            guard.reserve_budget(org_id="o", workspace_id="w", run_id="r")


def test_async_reserve_treats_an_open_breaker_as_a_deny(fail_open):
    import asyncio

    def _urlopen(req, timeout=0):  # noqa: ARG001
        return _Resp(_CAPS)

    def _handler(request: httpx.Request) -> httpx.Response:
        if "/lifecycle" in str(request.url):
            return httpx.Response(200, json=_CAPS)
        return httpx.Response(409, json=_BREAKER)

    real_client = httpx.AsyncClient

    def _client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(_handler)
        return real_client(*args, **kwargs)

    with patch("kazenai.spine.transport.urlopen", side_effect=_urlopen), patch(
        "httpx.AsyncClient", side_effect=_client
    ):
        with pytest.raises(BudgetExceeded, match="tier"):
            asyncio.run(guard.reserve_budget_async(org_id="o", workspace_id="w", run_id="r"))
