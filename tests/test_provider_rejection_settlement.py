"""A definitive provider refusal settles at $0 instead of holding budget as pending.

Found live on 2026-10-12: a 404 "model not found" left the reservation in
pending_reconciliation, so a typo'd model name tied up workflow budget until
reconciliation. Timeouts, connection errors and 5xx stay pending because the
provider may have done billable work.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from kazenai import monitor
from kazenai.provider_errors import is_definitive_provider_rejection
from kazenai.spine.guard import ReservationHandle


class _Status(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.status_code = code


@pytest.mark.parametrize("code", [400, 401, 403, 404, 413, 422, 429])
def test_definitive_4xx_are_rejections(code):
    assert is_definitive_provider_rejection(_Status(code))


@pytest.mark.parametrize("code", [408, 409, 499, 500, 502, 503, 529])
def test_ambiguous_and_server_errors_are_not(code):
    assert not is_definitive_provider_rejection(_Status(code))


def test_errors_without_a_status_are_not_rejections():
    assert not is_definitive_provider_rejection(TimeoutError("read timeout"))
    assert not is_definitive_provider_rejection(ConnectionError("reset"))


@pytest.fixture
def lifecycle(monkeypatch):
    """Hold a lifecycle reservation for every call and record settlement events."""
    events: list[dict[str, Any]] = []
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL"):
        monkeypatch.delenv(key, raising=False)

    import sys

    monitor_module = sys.modules["kazenai.monitor"]  # `kazenai.monitor` is also the exported function
    handle = ReservationHandle(reserved_cost_usd=0.05, reservation_id="rsv_test", lifecycle=True)
    monkeypatch.setattr(monitor_module, "_try_reserve_budget", lambda **_: handle)

    async def _reserve_async(**_):
        return handle

    monkeypatch.setattr(monitor_module, "_try_reserve_budget_async", _reserve_async)

    def _signal(**kwargs):
        events.append({"event": kwargs.get("event"), "actual": kwargs.get("actual_cost_usd")})
        return True

    async def _signal_async(**kwargs):
        return _signal(**kwargs)

    monkeypatch.setattr("kazenai.spine.guard.signal_reservation_event", _signal)
    monkeypatch.setattr("kazenai.spine.guard.signal_reservation_event_async", _signal_async)
    return events


def _openai(status: int, *, use_async: bool = False):
    openai = pytest.importorskip("openai")
    body = {"error": {"message": "boom", "type": "invalid_request_error", "code": str(status)}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body, request=request)

    if use_async:
        return monitor(
            openai.AsyncOpenAI(api_key="t", base_url="https://o.test/v1", max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))),
            max_budget_usd=1.0,
        )
    return monitor(
        openai.OpenAI(api_key="t", base_url="https://o.test/v1", max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(handler))),
        max_budget_usd=1.0,
    )


def _call(client, *, use_async: bool, stream: bool = False):
    kwargs = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}], "stream": stream}
    if use_async:
        async def _go():
            return await client.chat.completions.create(**kwargs)

        return asyncio.run(_go())
    return client.chat.completions.create(**kwargs)


@pytest.mark.parametrize("use_async", [False, True])
def test_model_not_found_settles_zero_not_pending(lifecycle, use_async):
    client = _openai(404, use_async=use_async)
    with pytest.raises(Exception):
        _call(client, use_async=use_async)
    settles = [e for e in lifecycle if e["event"] in ("usage_known", "outcome_unknown")]
    assert settles == [{"event": "usage_known", "actual": 0.0}], lifecycle
    assert client._kazenai_enforcement.snapshot()["pending_projected_usd"] == 0.0


@pytest.mark.parametrize("use_async", [False, True])
def test_server_error_stays_pending(lifecycle, use_async):
    client = _openai(500, use_async=use_async)
    with pytest.raises(Exception):
        _call(client, use_async=use_async)
    settles = [e for e in lifecycle if e["event"] in ("usage_known", "outcome_unknown")]
    assert settles == [{"event": "outcome_unknown", "actual": None}], lifecycle


def test_stream_rejected_before_any_token_settles_zero(lifecycle):
    client = _openai(404)
    with pytest.raises(Exception):
        _call(client, use_async=False, stream=True)
    assert {"event": "usage_known", "actual": 0.0} in lifecycle, lifecycle
    assert not any(e["event"] == "outcome_unknown" for e in lifecycle), lifecycle
