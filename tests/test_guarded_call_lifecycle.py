"""guarded_llm_call / aguarded_llm_call / guarded_embedding_call follow the
reservation lifecycle: reserve -> provider_started -> call -> one settlement.

Regressions for the 2026-10-25 audit: the wrappers settled ``usage_known`` from
``reserved`` (FinOps refuses that transition), left reservations held after a
provider error, priced an un-awaited coroutine at $0, skipped the enforcement
owner check on the async path, and settled embeddings twice in legacy mode.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import kazenai.spine.guard as guard
from kazenai.sinks import MemorySink
from kazenai.spine.guard import ReservationHandle

MODEL = "gpt-4o-mini"
USAGE = {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500}


class _Resp:
    def __init__(self, usage: dict[str, int] | None = USAGE) -> None:
        self.usage = usage


class _Status(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.status_code = code


def _expected_cost() -> float:
    cost = guard._guard_llm_price(MODEL, _Resp())[1]
    assert cost and cost > 0
    return cost


@pytest.fixture
def ledger(monkeypatch):
    """Lifecycle reservation plus a log of every lifecycle call, in order."""
    log: list[tuple[str, Any]] = []
    handle = ReservationHandle(
        reserved_cost_usd=0.05, reservation_id="rsv_1", lifecycle=True, settlement_capability="cap"
    )
    state = {"handle": handle, "settle_ack": True, "start_raises": None}

    def reserve(**_):
        log.append(("reserve", None))
        return state["handle"]

    async def reserve_async(**kwargs):
        return reserve(**kwargs)

    def signal(**kwargs):
        event = kwargs["event"]
        if event == "provider_started" and state["start_raises"] is not None:
            log.append(("provider_started_failed", None))
            raise state["start_raises"]
        log.append((event, kwargs.get("actual_cost_usd")))
        return True

    async def signal_async(**kwargs):
        return signal(**kwargs)

    def reconcile(**kwargs):
        log.append(("settle", kwargs["actual_cost_usd"]))
        return state["settle_ack"]

    async def reconcile_async(**kwargs):
        return reconcile(**kwargs)

    monkeypatch.setattr(guard, "reserve_budget", reserve)
    monkeypatch.setattr(guard, "reserve_budget_async", reserve_async)
    monkeypatch.setattr(guard, "signal_reservation_event", signal)
    monkeypatch.setattr(guard, "signal_reservation_event_async", signal_async)
    monkeypatch.setattr(guard, "reconcile_budget", reconcile)
    monkeypatch.setattr(guard, "reconcile_budget_async", reconcile_async)
    monkeypatch.setattr(guard, "finops_reservation_fail_closed", lambda: True)
    return log, state


def _kw(**extra):
    return {"model": MODEL, "org_id": "org_1", "run_id": "run_1", "event_sink": MemorySink(), **extra}


def test_sync_records_provider_started_before_dispatch_then_settles_once(ledger):
    log, _ = ledger

    def provider():
        log.append(("provider", None))
        return _Resp()

    out = guard.guarded_llm_call(provider, **_kw())
    assert isinstance(out, _Resp)
    assert [e for e, _ in log] == ["reserve", "provider_started", "provider", "settle"]
    assert log[-1][1] == pytest.approx(_expected_cost())


def test_sync_server_error_keeps_hold_pending(ledger):
    log, _ = ledger

    def provider():
        raise _Status(500)

    with pytest.raises(_Status):
        guard.guarded_llm_call(provider, **_kw())
    assert [e for e, _ in log] == ["reserve", "provider_started", "outcome_unknown"]


def test_sync_definitive_refusal_settles_zero(ledger):
    log, _ = ledger

    def provider():
        raise _Status(404)

    with pytest.raises(_Status):
        guard.guarded_llm_call(provider, **_kw())
    assert log[-1] == ("usage_known", 0.0)


def test_sync_interrupt_marks_outcome_unknown(ledger):
    log, _ = ledger

    def provider():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        guard.guarded_llm_call(provider, **_kw())
    assert log[-1] == ("outcome_unknown", None)


def test_sync_provider_started_failure_refuses_without_dispatch(ledger):
    log, state = ledger
    state["start_raises"] = guard.BudgetUnavailable("finops down")
    called = []

    with pytest.raises(guard.BudgetUnavailable):
        guard.guarded_llm_call(lambda: called.append(1) or _Resp(), **_kw())
    assert called == []
    assert [e for e, _ in log] == ["reserve", "provider_started_failed", "cancel_before_call"]


def test_sync_unacked_settlement_keeps_exact_amount(ledger):
    log, state = ledger
    state["settle_ack"] = False
    guard.guarded_llm_call(lambda: _Resp(), **_kw())
    assert log[-2:] == [("settle", pytest.approx(_expected_cost())), ("usage_known", pytest.approx(_expected_cost()))]


def test_sync_missing_usage_is_pending_not_zero(ledger):
    log, _ = ledger
    guard.guarded_llm_call(lambda: _Resp(usage=None), **_kw())
    assert log[-1] == ("outcome_unknown", None)
    assert not any(e == "settle" for e, _ in log)


def test_async_lambda_returning_coroutine_is_awaited_before_settlement(ledger):
    log, _ = ledger

    async def create():
        log.append(("provider", None))
        return _Resp()

    async def go():
        return await guard.aguarded_llm_call(lambda: create(), **_kw())

    out = asyncio.run(go())
    assert isinstance(out, _Resp)
    assert [e for e, _ in log] == ["reserve", "provider_started", "provider", "settle"]
    assert log[-1][1] == pytest.approx(_expected_cost())


def test_async_callable_object_is_awaited(ledger):
    log, _ = ledger

    class Caller:
        async def __call__(self):
            log.append(("provider", None))
            return _Resp()

    out = asyncio.run(guard.aguarded_llm_call(Caller(), **_kw()))
    assert isinstance(out, _Resp)
    assert log[-1][1] == pytest.approx(_expected_cost())


def test_async_cancellation_marks_outcome_unknown(ledger):
    log, _ = ledger

    async def provider():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(guard.aguarded_llm_call(provider, **_kw()))
    assert log[-1] == ("outcome_unknown", None)


def test_gateway_owned_handle_is_never_signalled(ledger):
    log, state = ledger
    state["handle"] = ReservationHandle(reserved_cost_usd=0.0, reservation_id="rsv_gateway", lifecycle=True)
    guard.guarded_llm_call(lambda: _Resp(), **_kw())
    asyncio.run(guard.aguarded_llm_call(lambda: _Resp(), **_kw()))
    assert [e for e, _ in log] == ["reserve", "reserve"]


def test_async_reserve_honours_gateway_enforcement_owner(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_OWNER", "gateway")
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://127.0.0.1:9")

    def no_network(*_a, **_k):
        raise AssertionError("gateway-owned attempts must not reserve again")

    monkeypatch.setattr(guard._transport, "async_session", no_network)
    monkeypatch.setattr(guard._transport, "urlopen", no_network)
    sync = guard.reserve_budget(org_id="o", workspace_id="w", run_id="r")
    handle = asyncio.run(guard.reserve_budget_async(org_id="o", workspace_id="w", run_id="r"))
    assert isinstance(handle, ReservationHandle) and isinstance(sync, ReservationHandle)
    assert float(handle) == 0.0 and handle.settlement_capability is None
    assert (handle.reservation_id, handle.lifecycle) == (sync.reservation_id, sync.lifecycle)


def test_embedding_settles_once_with_embedding_price_in_legacy_mode(ledger):
    log, state = ledger
    state["handle"] = 0.01  # legacy /v1/budget/check reservation

    class Emb:
        usage = {"prompt_tokens": 400, "total_tokens": 400}

    guard.guarded_embedding_call(lambda: Emb(), texts=["a" * 1600], **_kw())
    settles = [amount for e, amount in log if e == "settle"]
    assert settles == [pytest.approx(guard._embedding_cost(MODEL, Emb(), ["a" * 1600]))]


def test_unpriceable_result_keeps_hold_pending(ledger, monkeypatch):
    log, _ = ledger

    def boom(*_a, **_k):
        raise guard.BudgetUnavailable("no price")

    monkeypatch.setattr(guard, "_extract_usage_cost", boom)
    with pytest.raises(guard.BudgetUnavailable):
        guard.guarded_llm_call(lambda: _Resp(), **_kw())
    assert log[-1] == ("outcome_unknown", None)
