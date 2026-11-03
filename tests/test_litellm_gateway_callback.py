"""The LiteLLM gateway callback names the customer, holds a real bound, prices cache
tokens at their own rates and settles streams and cache hits (kazenai 1.2.2).

LiteLLM is stubbed here so these run in every CI job; ``test_litellm_gateway_proxy.py``
proves the same behaviour through a real proxy. FinOps is ``FakeFinOps`` on loopback.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import time
import types
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kazenai.cost_engine import TokenCostEngine
from kazenai.integrations.litellm_control import GATEWAY_OWNER, GatewayAdmissionDenied, GatewayFinOpsClient
from kazenai.model_pricing import default_max_output_tokens

from test_litellm_control import FakeFinOps, _serve_finops

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"


@pytest.fixture()
def finops() -> FakeFinOps:
    return FakeFinOps(capacity_micros=100_000_000)


@pytest.fixture()
def logger(monkeypatch, tmp_path, finops):
    stub = types.ModuleType("litellm")
    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:  # noqa: N801
        pass

    custom_logger.CustomLogger = CustomLogger
    monkeypatch.setitem(sys.modules, "litellm", stub)
    monkeypatch.setitem(sys.modules, "litellm.integrations", integrations)
    monkeypatch.setitem(sys.modules, "litellm.integrations.custom_logger", custom_logger)
    monkeypatch.delitem(sys.modules, "kazenai.integrations.litellm_gateway", raising=False)

    server = _serve_finops(finops)
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_OWNER", GATEWAY_OWNER)
    monkeypatch.setenv("KAZENAI_FINOPS_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "k")
    monkeypatch.setenv("KAZENAI_ORG_ID", "org-gw")
    monkeypatch.setenv("KAZENAI_WORKSPACE_ID", "ws-gw")
    monkeypatch.setenv("KAZENAI_GATEWAY_PENDING_DIR", str(tmp_path))
    module = importlib.import_module("kazenai.integrations.litellm_gateway")
    yield module.KazenGatewayControlLogger(finops=GatewayFinOpsClient())
    server.shutdown()
    sys.modules.pop("kazenai.integrations.litellm_gateway", None)


def _request(model: str = OPENAI_MODEL, **extra: Any) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_call_id": "call-1",
        **extra,
    }


def _events(finops: FakeFinOps, event: str) -> list[dict[str, Any]]:
    return [e for e in finops.events if e.get("event") == event]


def _reserved_micros(finops: FakeFinOps) -> int:
    return int(_events(finops, "reserve")[-1]["estimated_usd_micros"])


def _rates(model: str):
    row = TokenCostEngine()._row(model)
    assert row is not None, model
    return row.rates


# ---------------------------------------------------------------------------
# Customer ID
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"user": "customer-42"},
        {"metadata": {"user_api_key_end_user_id": "customer-42"}},
        {"litellm_metadata": {"user_api_key_end_user_id": "customer-42"}},
        {"litellm_params": {"metadata": {"user_api_key_end_user_id": "customer-42"}}},
        {"user": "other", "metadata": {"user_api_key_end_user_id": "customer-42"}},
    ],
)
async def test_the_customer_reaches_the_reservation(logger, finops, extra):
    await logger.async_pre_call_deployment_hook(_request(**extra))
    assert _events(finops, "reserve")[0]["business_subject_ref"] == "customer-42"


@pytest.mark.asyncio
async def test_a_call_with_no_customer_sends_no_customer_field(logger, finops):
    await logger.async_pre_call_deployment_hook(_request())
    assert "business_subject_ref" not in _events(finops, "reserve")[0]


# ---------------------------------------------------------------------------
# The hold is a real bound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_hold_covers_the_requested_output(logger, finops):
    await logger.async_pre_call_deployment_hook(_request(max_tokens=16_000))
    assert _reserved_micros(finops) >= int(16_000 * _rates(OPENAI_MODEL).output)


@pytest.mark.asyncio
async def test_a_provider_prefix_gets_the_same_hold(logger, finops):
    await logger.async_pre_call_deployment_hook(_request(max_tokens=16_000))
    bare = _reserved_micros(finops)
    await logger.async_pre_call_deployment_hook(_request(model=f"openai/{OPENAI_MODEL}", max_tokens=16_000))
    assert _reserved_micros(finops) == bare


@pytest.mark.asyncio
async def test_a_request_with_no_limit_gets_the_models_bound(logger, finops):
    bound = default_max_output_tokens(OPENAI_MODEL)
    assert bound
    kwargs = _request()
    out = await logger.async_pre_call_deployment_hook(kwargs)
    assert out["max_tokens"] == bound
    assert _reserved_micros(finops) >= int(bound * _rates(OPENAI_MODEL).output)


@pytest.mark.asyncio
async def test_a_limit_above_the_bound_is_lowered(logger):
    bound = default_max_output_tokens(OPENAI_MODEL)
    out = await logger.async_pre_call_deployment_hook(_request(max_tokens=bound * 100))
    assert out["max_tokens"] == bound


@pytest.mark.asyncio
async def test_a_limit_below_the_bound_is_left_alone(logger):
    out = await logger.async_pre_call_deployment_hook(_request(max_tokens=64))
    assert out["max_tokens"] == 64


@pytest.mark.asyncio
async def test_several_choices_multiply_the_output_in_the_hold(logger, finops):
    await logger.async_pre_call_deployment_hook(_request(max_tokens=1_000, n=4))
    assert _reserved_micros(finops) >= int(4_000 * _rates(OPENAI_MODEL).output)


@pytest.mark.asyncio
async def test_the_hold_counts_the_whole_prompt_and_its_tools(logger, finops):
    tool = {"type": "function", "function": {"name": "f", "description": "x" * 30_000}}
    kwargs = _request(
        messages=[{"role": "user", "content": "y" * 30_000}],
        tools=[tool],
        max_tokens=1,
    )
    await logger.async_pre_call_deployment_hook(kwargs)
    assert _reserved_micros(finops) >= int(60_000 * _rates(OPENAI_MODEL).input)


@pytest.mark.asyncio
async def test_litellm_internals_do_not_inflate_the_hold(logger, finops):
    await logger.async_pre_call_deployment_hook(_request(max_tokens=64))
    plain = _reserved_micros(finops)
    noisy = _request(
        max_tokens=64,
        api_key="k" * 50_000,
        litellm_logging_obj=object(),
        metadata={"blob": "z" * 50_000},
    )
    await logger.async_pre_call_deployment_hook(noisy)
    assert _reserved_micros(finops) == plain


@pytest.mark.asyncio
async def test_a_responses_call_is_bounded_with_its_own_parameter(logger):
    kwargs = {"model": OPENAI_MODEL, "input": "hi", "litellm_call_id": "call-1"}
    out = await logger.async_pre_call_deployment_hook(kwargs, "aresponses")
    assert out["max_output_tokens"] == default_max_output_tokens(OPENAI_MODEL)
    assert "max_tokens" not in out


@pytest.mark.asyncio
async def test_a_non_chat_call_gets_no_output_parameter(logger, finops):
    kwargs = {"model": "text-embedding-3-small", "input": ["hi"], "litellm_call_id": "call-1"}
    out = await logger.async_pre_call_deployment_hook(kwargs, "aembedding")
    assert "max_tokens" not in out
    assert "max_completion_tokens" not in out
    assert _events(finops, "reserve")


@pytest.mark.asyncio
async def test_sdk_owned_mode_leaves_the_request_alone(logger, finops, monkeypatch):
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_OWNER", "sdk")
    kwargs = _request()
    out = await logger.async_pre_call_deployment_hook(kwargs)
    assert "max_tokens" not in out
    assert not finops.events


# ---------------------------------------------------------------------------
# Cache tokens at their own rates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_cached_tokens_bill_at_the_cached_rate(logger, finops):
    kwargs = _request(max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    usage = {
        "prompt_tokens": 1_000,
        "completion_tokens": 100,
        "total_tokens": 1_100,
        "prompt_tokens_details": {"cached_tokens": 900},
    }
    await logger.async_post_call_success_deployment_hook(kwargs, SimpleNamespace(usage=usage))
    rates = _rates(OPENAI_MODEL)
    expected = Decimal(100) * rates.input + Decimal(900) * rates.cached_input + Decimal(100) * rates.output
    assert int(_events(finops, "usage_known")[0]["actual_usd_micros"]) == int(expected.to_integral_value(rounding="ROUND_CEILING"))


def _anthropic_expected() -> int:
    rates = _rates(ANTHROPIC_MODEL)
    expected = (
        Decimal(100) * rates.input
        + Decimal(400) * rates.cached_input
        + Decimal(1_000) * rates.cache_write_5m
        + Decimal(50) * rates.output
    )
    return int(expected.to_integral_value(rounding="ROUND_CEILING"))


@pytest.mark.asyncio
async def test_litellm_normalized_anthropic_usage_counts_each_token_once(logger, finops):
    """LiteLLM's ``prompt_tokens`` already includes cache reads and cache writes."""
    kwargs = _request(model=f"anthropic/{ANTHROPIC_MODEL}", max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    usage = {
        "prompt_tokens": 1_500,
        "completion_tokens": 50,
        "total_tokens": 1_550,
        "prompt_tokens_details": {"cached_tokens": 400},
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 1_000,
    }
    await logger.async_post_call_success_deployment_hook(kwargs, SimpleNamespace(usage=usage))
    assert int(_events(finops, "usage_known")[0]["actual_usd_micros"]) == _anthropic_expected()


@pytest.mark.asyncio
async def test_raw_anthropic_usage_counts_each_token_once(logger, finops):
    """On ``/v1/messages``, Anthropic's ``input_tokens`` is only the uncached part."""
    kwargs = _request(model=f"anthropic/{ANTHROPIC_MODEL}", max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs, "anthropic_messages")
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 1_000,
    }
    await logger.async_post_call_success_deployment_hook(kwargs, SimpleNamespace(usage=usage))
    assert int(_events(finops, "usage_known")[0]["actual_usd_micros"]) == _anthropic_expected()


# ---------------------------------------------------------------------------
# What the deployment hook never sees: streams and cache hits
# ---------------------------------------------------------------------------

_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


@pytest.mark.asyncio
async def test_a_finished_stream_settles_from_its_usage(logger, finops):
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    assert not _events(finops, "usage_known")

    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)

    settled = _events(finops, "usage_known")
    assert len(settled) == 1
    assert int(settled[0]["actual_usd_micros"]) > 0


@pytest.mark.asyncio
async def test_a_stream_settles_at_the_price_a_plain_call_settles_at(logger, finops):
    plain = _request(max_tokens=100)
    await logger.async_pre_call_deployment_hook(plain)
    await logger.async_post_call_success_deployment_hook(plain, SimpleNamespace(usage=_USAGE))

    stream = _request(stream=True, max_tokens=100, litellm_call_id="call-2")
    await logger.async_pre_call_deployment_hook(stream)
    await logger.async_log_success_event(stream, SimpleNamespace(usage=_USAGE), None, None)

    amounts = [e["actual_usd_micros"] for e in _events(finops, "usage_known")]
    assert len(amounts) == 2
    assert amounts[0] == amounts[1]


@pytest.mark.asyncio
async def test_a_plain_call_settles_once_through_both_hooks(logger, finops):
    kwargs = _request(max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    response = SimpleNamespace(usage=_USAGE)
    await logger.async_post_call_success_deployment_hook(kwargs, response)
    await logger.async_log_success_event(kwargs, response, None, None)
    assert len(_events(finops, "usage_known")) == 1


@pytest.mark.asyncio
async def test_a_cache_hit_settles_at_zero(logger, finops):
    """LiteLLM answers a cache hit before the provider and before the success hook."""
    kwargs = _request(max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    kwargs["cache_hit"] = True
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)

    settled = _events(finops, "usage_known")
    assert len(settled) == 1
    assert int(settled[0]["actual_usd_micros"]) == 0
    assert settled[0]["usage_evidence"]["cache_hit"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["metadata", "litellm_params"])
async def test_a_client_disconnect_stays_pending(logger, finops, where):
    """The provider may have kept generating, so the partial count isn't the charge."""
    kwargs = _request(stream=True, max_tokens=100, litellm_params={"metadata": {}})
    await logger.async_pre_call_deployment_hook(kwargs)
    block = kwargs["metadata"] if where == "metadata" else kwargs["litellm_params"]["metadata"]
    block["client_disconnected"] = True

    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)

    assert not _events(finops, "usage_known")
    assert _events(finops, "outcome_unknown")[0]["usage_evidence"]["error"] == "client_disconnected"


@pytest.mark.asyncio
async def test_a_stream_with_no_usage_stays_pending_never_zero(logger, finops):
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=None), None, None)
    assert not _events(finops, "usage_known")
    assert _events(finops, "outcome_unknown")[0]["usage_evidence"]["error"] == "missing_usage"


@pytest.mark.asyncio
async def test_a_stream_that_fails_midway_stays_pending(logger, finops):
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    kwargs["exception"] = RuntimeError("connection reset")
    await logger.async_log_failure_event(kwargs, None, None, None)
    assert not _events(finops, "usage_known")
    assert len(_events(finops, "outcome_unknown")) == 1


@pytest.mark.asyncio
async def test_a_failure_the_deployment_hook_settled_is_not_touched_again(logger, finops):
    kwargs = _request(max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)

    class Rejected(Exception):
        status_code = 400

    await logger.async_post_call_failure_deployment_hook(kwargs, Rejected("bad request"))
    assert int(_events(finops, "usage_known")[0]["actual_usd_micros"]) == 0

    await logger.async_log_failure_event(kwargs, None, None, None)
    assert not _events(finops, "outcome_unknown")


@pytest.mark.asyncio
async def test_events_for_a_call_it_never_reserved_do_nothing(logger, finops):
    kwargs = _request(stream=True)
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)
    await logger.async_log_failure_event(kwargs, None, None, None)
    assert not finops.events


@pytest.mark.asyncio
async def test_a_settled_call_is_forgotten(logger):
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    assert logger._by_litellm_call
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)
    assert not logger._by_litellm_call


# ---------------------------------------------------------------------------
# FinOps calls stay off the proxy's event loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finops_calls_do_not_block_the_event_loop(logger, monkeypatch):
    original = GatewayFinOpsClient._post

    def slow(self, path, body):
        time.sleep(0.5)
        return original(self, path, body)

    monkeypatch.setattr(GatewayFinOpsClient, "_post", slow)
    gaps: list[float] = []
    done = asyncio.Event()

    async def heartbeat() -> None:
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)
    done.set()
    await beat
    # On the event loop, each 0.5 s call would stall it for the whole 0.5 s.
    assert max(gaps) < 0.4, f"the event loop stalled for {max(gaps):.3f}s"


# ---------------------------------------------------------------------------
# A settlement that failed to send is resent
# ---------------------------------------------------------------------------


def _unreachable(monkeypatch) -> None:
    def down(self, path, body):
        raise GatewayAdmissionDenied("FinOps unavailable", code="finops_unavailable", http_status=503)

    monkeypatch.setattr(GatewayFinOpsClient, "_post", down)


async def _flush(logger) -> None:
    logger._last_flush = float("-inf")
    logger._schedule_flush()
    await logger._flush_task


@pytest.mark.asyncio
async def test_an_unsent_settlement_is_resent_once_finops_is_back(logger, finops):
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    with pytest.MonkeyPatch.context() as patch:
        _unreachable(patch)
        await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)
        assert not _events(finops, "usage_known")
        assert logger.finops.store.list_pending()

    await _flush(logger)
    assert len(_events(finops, "usage_known")) == 1
    assert not logger.finops.store.list_pending()


@pytest.mark.asyncio
async def test_the_first_call_after_a_restart_resends_what_was_left(logger, finops, monkeypatch):
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    with pytest.MonkeyPatch.context() as patch:
        _unreachable(patch)
        await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)

    restarted = type(logger)(finops=GatewayFinOpsClient())
    await restarted.async_pre_call_deployment_hook(_request(max_tokens=100, litellm_call_id="call-2"))
    await restarted._flush_task
    assert len(_events(finops, "usage_known")) == 1


@pytest.mark.asyncio
async def test_resending_runs_at_most_once_per_interval(logger, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(logger.finops, "flush_pending", lambda: calls.append(1) or {"applied": 0, "failed": 0})
    for index in range(3):
        await logger.async_pre_call_deployment_hook(_request(max_tokens=100, litellm_call_id=f"call-{index}"))
        await logger._flush_task
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_settlement_finops_refuses_is_set_aside_not_retried_forever(logger, finops, tmp_path):
    store = logger.finops.store
    store.enqueue_pending(body={"idempotency_key": "gone", "reservation_id": "rsv_gone", "event": "usage_known"})
    assert logger.finops.flush_pending()["quarantined"] == 1
    assert not store.list_pending()
    assert len(os.listdir(store.quarantine_dir)) == 1


def test_an_old_settlement_is_set_aside(logger, finops, monkeypatch):
    store = logger.finops.store
    store.enqueue_pending(body={"idempotency_key": "old", "reservation_id": "rsv_old", "event": "usage_known"})
    path = store.list_pending()[0]["_path"]
    payload = json.loads(Path(path).read_text())
    payload["enqueued_at"] = time.time() - 8 * 24 * 3600
    Path(path).write_text(json.dumps(payload))
    assert logger.finops.flush_pending()["quarantined"] == 1
    assert not finops.events


@pytest.mark.asyncio
async def test_a_recorded_unknown_outcome_leaves_the_queue_but_keeps_the_hold(logger, finops):
    """FinOps answers ``pending`` to ``outcome_unknown``: that is the record, not a failure."""
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=None), None, None)
    assert len(_events(finops, "outcome_unknown")) == 1
    assert not logger.finops.store.list_pending()
    reservation = _events(finops, "reserve")[-1]["reservation_id"]
    assert finops.reservations[reservation]["state"] == "pending_reconciliation"


@pytest.mark.asyncio
async def test_a_usage_reply_of_pending_keeps_the_settlement_queued(logger, finops, monkeypatch):
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    monkeypatch.setattr(GatewayFinOpsClient, "_post", lambda self, path, body: (200, {"result": "pending"}))
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)
    assert len(logger.finops.store.list_pending()) == 1


# ---------------------------------------------------------------------------
# Regional and other non-first-party endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_non_first_party_endpoint_holds_and_settles_with_the_uplift(logger, finops):
    plain = _request(max_tokens=100)
    await logger.async_pre_call_deployment_hook(plain)
    await logger.async_post_call_success_deployment_hook(plain, SimpleNamespace(usage=_USAGE))
    base_hold = _reserved_micros(finops)
    base_settled = int(_events(finops, "usage_known")[-1]["actual_usd_micros"])

    regional = _request(max_tokens=100, api_base="https://eu.api.openai.com/v1", litellm_call_id="call-2")
    await logger.async_pre_call_deployment_hook(regional)
    assert _reserved_micros(finops) == pytest.approx(base_hold * 1.1, abs=2)
    await logger.async_post_call_success_deployment_hook(regional, SimpleNamespace(usage=_USAGE))
    assert int(_events(finops, "usage_known")[-1]["actual_usd_micros"]) == pytest.approx(base_settled * 1.1, abs=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("api_base", [None, "https://api.openai.com/v1"])
async def test_the_first_party_endpoint_has_no_uplift(logger, finops, api_base):
    await logger.async_pre_call_deployment_hook(_request(max_tokens=100))
    base_hold = _reserved_micros(finops)
    extra = {"api_base": api_base} if api_base else {}
    await logger.async_pre_call_deployment_hook(_request(max_tokens=100, litellm_call_id="call-2", **extra))
    assert _reserved_micros(finops) == base_hold


@pytest.mark.asyncio
async def test_an_operator_can_declare_an_endpoint_at_list_price(logger, finops, monkeypatch):
    await logger.async_pre_call_deployment_hook(_request(max_tokens=100))
    base_hold = _reserved_micros(finops)
    monkeypatch.setenv("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS", json.dumps({"llm.internal": "1"}))
    await logger.async_pre_call_deployment_hook(
        _request(max_tokens=100, api_base="https://llm.internal/v1", litellm_call_id="call-2")
    )
    assert _reserved_micros(finops) == base_hold


@pytest.mark.asyncio
async def test_a_stream_settles_with_the_uplift_frozen_at_reservation(logger, finops):
    plain = _request(max_tokens=100)
    await logger.async_pre_call_deployment_hook(plain)
    await logger.async_post_call_success_deployment_hook(plain, SimpleNamespace(usage=_USAGE))
    base_settled = int(_events(finops, "usage_known")[-1]["actual_usd_micros"])

    stream = _request(stream=True, max_tokens=100, api_base="https://eu.api.openai.com/v1", litellm_call_id="call-2")
    await logger.async_pre_call_deployment_hook(stream)
    logged = {key: value for key, value in stream.items() if key != "api_base"}
    await logger.async_log_success_event(logged, SimpleNamespace(usage=_USAGE), None, None)
    assert int(_events(finops, "usage_known")[-1]["actual_usd_micros"]) == pytest.approx(base_settled * 1.1, abs=2)


@pytest.mark.asyncio
async def test_a_bad_endpoint_setting_refuses_the_call(logger, finops, monkeypatch):
    monkeypatch.setenv("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS", "not json")
    with pytest.raises(GatewayAdmissionDenied):
        await logger.async_pre_call_deployment_hook(_request(max_tokens=100, api_base="https://llm.internal/v1"))
    assert not finops.events


# ---------------------------------------------------------------------------
# Anthropic 1-hour cache writes
# ---------------------------------------------------------------------------


def _one_hour_expected() -> int:
    rates = _rates(ANTHROPIC_MODEL)
    expected = (
        Decimal(100) * rates.input
        + Decimal(400) * rates.cached_input
        + Decimal(600) * rates.cache_write_5m
        + Decimal(400) * rates.cache_write_1h
        + Decimal(50) * rates.output
    )
    return int(expected.to_integral_value(rounding="ROUND_CEILING"))


@pytest.mark.asyncio
async def test_litellm_one_hour_cache_writes_bill_at_their_rate(logger, finops):
    kwargs = _request(model=f"anthropic/{ANTHROPIC_MODEL}", max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    usage = {
        "prompt_tokens": 1_500,
        "completion_tokens": 50,
        "total_tokens": 1_550,
        "prompt_tokens_details": {
            "cached_tokens": 400,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 600, "ephemeral_1h_input_tokens": 400},
        },
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 1_000,
    }
    await logger.async_post_call_success_deployment_hook(kwargs, SimpleNamespace(usage=usage))
    assert int(_events(finops, "usage_known")[0]["actual_usd_micros"]) == _one_hour_expected()


@pytest.mark.asyncio
async def test_raw_anthropic_one_hour_cache_writes_bill_at_their_rate(logger, finops):
    kwargs = _request(model=f"anthropic/{ANTHROPIC_MODEL}", max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs, "anthropic_messages")
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 1_000,
        "cache_creation": {"ephemeral_5m_input_tokens": 600, "ephemeral_1h_input_tokens": 400},
    }
    await logger.async_post_call_success_deployment_hook(kwargs, SimpleNamespace(usage=usage))
    assert int(_events(finops, "usage_known")[0]["actual_usd_micros"]) == _one_hour_expected()


# ---------------------------------------------------------------------------
# A stream the provider never finished or reported usage for
# ---------------------------------------------------------------------------


class _Wrapper:
    """What the proxy hands the iterator hook: an async stream that keeps its chunks."""

    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks

    def __aiter__(self):
        async def run():
            for chunk in self.chunks:
                yield chunk

        return run()


def _chunk(*, usage: Any = None, finish: Any = None) -> SimpleNamespace:
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(finish_reason=finish)])


async def _stream_through(logger, kwargs, chunks) -> list[Any]:
    wrapper = chunks if isinstance(chunks, _Wrapper) else _Wrapper(chunks)
    return [chunk async for chunk in logger.async_post_call_streaming_iterator_hook({}, wrapper, kwargs)]


@pytest.mark.asyncio
async def test_a_stream_the_provider_finished_and_reported_settles(logger, finops):
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    chunks = [_chunk(), _chunk(finish="stop"), _chunk(usage=_USAGE)]
    assert await _stream_through(logger, kwargs, chunks) == chunks
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)
    assert len(_events(finops, "usage_known")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param([_chunk(), _chunk(finish="stop")], id="finished-no-usage"),
        pytest.param([_chunk()], id="cut-off"),
        pytest.param([_chunk(usage=_USAGE)], id="usage-but-never-finished"),
        pytest.param([], id="empty"),
    ],
)
async def test_litellms_own_count_is_never_settled_as_the_charge(logger, finops, chunks):
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    await _stream_through(logger, kwargs, chunks)
    # LiteLLM still logs success, with usage from its own tokenizer.
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)
    assert not _events(finops, "usage_known")
    assert _events(finops, "outcome_unknown")[0]["usage_evidence"]["error"] == "missing_usage"


@pytest.mark.asyncio
async def test_a_stream_it_could_not_inspect_settles_and_says_so(logger, finops):
    """No iterator hook ran, or LiteLLM's stream keeps no chunks: label it, don't strand every stream."""
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)
    assert _events(finops, "usage_known")[0]["usage_evidence"]["usage_source"] == "unverified"


@pytest.mark.asyncio
async def test_stream_evidence_is_forgotten_after_settling(logger):
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    await _stream_through(logger, kwargs, [_chunk(finish="stop"), _chunk(usage=_USAGE)])
    assert logger._stream_evidence
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)
    assert not logger._stream_evidence


# ---------------------------------------------------------------------------
# Requiring a customer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_call_with_no_customer_is_refused_when_required(logger, finops, monkeypatch):
    monkeypatch.setenv("KAZENAI_REQUIRE_CUSTOMER_ID", "1")
    with pytest.raises(GatewayAdmissionDenied) as refused:
        await logger.async_pre_call_deployment_hook(_request(max_tokens=100))
    assert refused.value.code == "customer_id_required"
    assert refused.value.http_status == 400
    assert not finops.events


@pytest.mark.asyncio
async def test_a_call_naming_a_customer_passes_when_required(logger, finops, monkeypatch):
    monkeypatch.setenv("KAZENAI_REQUIRE_CUSTOMER_ID", "1")
    await logger.async_pre_call_deployment_hook(_request(max_tokens=100, user="customer-42"))
    assert _events(finops, "reserve")[0]["business_subject_ref"] == "customer-42"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [402, 422])
async def test_a_customer_id_finops_rejects_is_the_callers_error(logger, monkeypatch, status):
    monkeypatch.setattr(
        GatewayFinOpsClient,
        "_post",
        lambda self, path, body: (
            status,
            {"allowed": False, "error_code": "invalid_attribution", "reason": "business_subject_ref has invalid characters"},
        ),
    )
    with pytest.raises(GatewayAdmissionDenied) as refused:
        await logger.async_pre_call_deployment_hook(_request(max_tokens=100, user="bad id"))
    assert refused.value.code == "invalid_attribution"
    assert refused.value.http_status == 400
    assert "invalid characters" in str(refused.value)


# ---------------------------------------------------------------------------
# Pricing that can't run is refused, not held at a guess
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_pricing_config_refuses_the_call(logger, finops, monkeypatch):
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")
    monkeypatch.delenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", raising=False)
    with pytest.raises(GatewayAdmissionDenied) as refused:
        await logger.async_pre_call_deployment_hook(_request(max_tokens=100))
    assert "KAZENAI_TIER_MISMATCH_MULTIPLIER" in str(refused.value)
    assert not finops.events


@pytest.mark.asyncio
async def test_the_success_hook_leaves_an_open_stream_to_its_success_event(logger, finops):
    """LiteLLM's Responses route calls the hook with the stream just opened, before any usage."""
    kwargs = _request(stream=True, max_tokens=100)
    await logger.async_pre_call_deployment_hook(kwargs)
    await logger.async_post_call_success_deployment_hook(kwargs, SimpleNamespace(usage=None))
    assert not _events(finops, "outcome_unknown")
    await logger.async_log_success_event(kwargs, SimpleNamespace(usage=_USAGE), None, None)
    assert len(_events(finops, "usage_known")) == 1
    assert not _events(finops, "outcome_unknown")


@pytest.mark.asyncio
async def test_a_responses_usage_object_bills_the_cached_rate(logger, finops):
    kwargs = {"model": OPENAI_MODEL, "input": "hi", "max_output_tokens": 100, "litellm_call_id": "call-1"}
    await logger.async_pre_call_deployment_hook(kwargs, "aresponses")
    usage = SimpleNamespace(
        input_tokens=1_000,
        output_tokens=100,
        total_tokens=1_100,
        input_tokens_details=SimpleNamespace(cached_tokens=900),
    )
    await logger.async_post_call_success_deployment_hook(kwargs, SimpleNamespace(usage=usage))
    rates = _rates(OPENAI_MODEL)
    expected = Decimal(100) * rates.input + Decimal(900) * rates.cached_input + Decimal(100) * rates.output
    assert int(_events(finops, "usage_known")[0]["actual_usd_micros"]) == int(expected.to_integral_value(rounding="ROUND_CEILING"))
