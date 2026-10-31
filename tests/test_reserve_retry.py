"""A reservation that gets no answer is sent once more (kazenai 1.2.1).

Phase 9 on staging: the first call after a quiet spell raised ``BudgetUnavailable``
while the next seven went through. One attempt with a 0.8 s budget that also had to
open the connection was too tight. Now a timeout, a refused or dropped connection, or
a 502/503/504 gets one more attempt with the same body, so the same idempotency key:
Agent FinOps answers a repeat with the reservation it already made. A decision (402),
or any other answer, is never retried. The default timeout is 2 s.
Agent FinOps runs on an ``httpx.MockTransport`` through the SDK's own transport.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from typing import Any

import httpx
import pytest

from kazenai import monitor
from kazenai.enforcement import BudgetExceeded, BudgetUnavailable
from kazenai.spine import guard, org_policy, transport

openai = pytest.importorskip("openai")

_CHAT = {
    "id": "c1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}

_DENIED = {"detail": {"allowed": False, "error_code": "invalid_amount", "reason": "insufficient remaining capacity on org"}}


class FinOps:
    """Agent FinOps whose reserve answers follow ``script`` (then succeed)."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.reserve_bodies: list[dict[str, Any]] = []
        self.reservations: dict[str, str] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "effective_lifecycle": {"profile": "control", "lifecycle_version": "reservation.lifecycle.v1"},
                    "capabilities": {"reservation_settlement": True, "legacy_check_reconcile": False},
                },
            )
        if request.url.path.endswith("/reserve"):
            body = json.loads(request.content or b"{}")
            self.reserve_bodies.append(body)
            step = self.script.pop(0) if self.script else 200
            if isinstance(step, type) and issubclass(step, Exception):
                raise step("no answer", request=request)
            if step == "admitted-then-lost":
                # FinOps admits, but the answer never reaches the SDK.
                self.reservations.setdefault(body["idempotency_key"], f"rsv_{uuid.uuid4().hex}")
                raise httpx.ReadTimeout("answer lost", request=request)
            if step == 402:
                return httpx.Response(402, json=_DENIED)
            if step != 200:
                return httpx.Response(step, json={"detail": "upstream"})
            rid = self.reservations.setdefault(body["idempotency_key"], f"rsv_{uuid.uuid4().hex}")
            return httpx.Response(
                200,
                json={
                    "allowed": True,
                    "reservation_id": rid,
                    "decision_id": "dec_1",
                    "lifecycle_version": "reservation.lifecycle.v1",
                    "call_id": body.get("call_id"),
                    "attempt": body.get("attempt") or 1,
                    "reserved_usd_micros": int(body.get("estimated_usd_micros") or 0),
                    "settlement_capability": "kzcap1.test." + "A" * 43,
                },
            )
        return httpx.Response(200, json={"result": "applied"})


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for key in ("KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL", "KAZENAI_DENY_STREAMING", "KAZENAI_FINOPS_RESERVE_TIMEOUT_S"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops.test")
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "test")
    monkeypatch.setattr(guard, "_reserve_retry_delay_s", lambda: 0.0)
    org_policy.reset_for_tests()
    guard._LIFECYCLE_CAP_CACHE.clear()
    transport.reset_pools()
    yield
    org_policy.reset_for_tests()
    guard._LIFECYCLE_CAP_CACHE.clear()
    transport.reset_pools()


def _route(monkeypatch, finops: FinOps) -> None:
    monkeypatch.setattr(transport, "_sync_client", httpx.Client(transport=httpx.MockTransport(finops)))

    @contextlib.asynccontextmanager
    async def async_session(timeout: Any):
        async with httpx.AsyncClient(transport=httpx.MockTransport(finops)) as client:
            yield transport._Session(client, timeout)

    monkeypatch.setattr(transport, "async_session", async_session)


def _call(finops: FinOps, *, is_async: bool) -> list[str]:
    hits: list[str] = []

    def provider(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        return httpx.Response(200, json=_CHAT)

    kw = dict(api_key="t", base_url="https://openai.test/v1", max_retries=0)
    if is_async:
        raw = openai.AsyncOpenAI(**kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)))
    else:
        raw = openai.OpenAI(**kw, http_client=httpx.Client(transport=httpx.MockTransport(provider)))
    client = monitor(raw, org_id="org-retry", workspace_id="ws", agent_id=f"retry-{uuid.uuid4().hex[:6]}", max_budget_usd=5.0)
    ask = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"hi {uuid.uuid4().hex}"}])
    if is_async:
        asyncio.run(ask)
    return hits


SIDES = pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])


@SIDES
@pytest.mark.parametrize("first", [httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError, 502, 503, 504])
def test_one_retry_after_no_answer(monkeypatch, is_async, first):
    finops = FinOps([first])
    _route(monkeypatch, finops)
    assert _call(finops, is_async=is_async) == ["/v1/chat/completions"]
    assert len(finops.reserve_bodies) == 2
    # Same body, so the same idempotency key and call identity.
    assert finops.reserve_bodies[0] == finops.reserve_bodies[1]


@SIDES
def test_an_admission_whose_answer_was_lost_is_not_reserved_twice(monkeypatch, is_async):
    finops = FinOps(["admitted-then-lost"])
    _route(monkeypatch, finops)
    assert _call(finops, is_async=is_async) == ["/v1/chat/completions"]
    keys = {b["idempotency_key"] for b in finops.reserve_bodies}
    assert len(finops.reserve_bodies) == 2 and len(keys) == 1
    assert len(finops.reservations) == 1


@SIDES
def test_still_no_answer_after_the_retry_is_unavailable(monkeypatch, is_async):
    finops = FinOps([httpx.ReadTimeout, httpx.ReadTimeout])
    _route(monkeypatch, finops)
    with pytest.raises(BudgetUnavailable):
        _call(finops, is_async=is_async)
    assert len(finops.reserve_bodies) == 2  # never a third attempt


@SIDES
@pytest.mark.parametrize("answer", [402, 500, 429, 403])
def test_a_decision_or_other_answer_is_not_retried(monkeypatch, is_async, answer):
    finops = FinOps([answer])
    _route(monkeypatch, finops)
    with pytest.raises(BudgetExceeded if answer == 402 else BudgetUnavailable):
        _call(finops, is_async=is_async)
    assert len(finops.reserve_bodies) == 1


def test_default_timeout_is_two_seconds(monkeypatch):
    assert guard._reserve_timeout_s() == 2.0
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVE_TIMEOUT_S", "0.5")
    assert guard._reserve_timeout_s() == 0.5
