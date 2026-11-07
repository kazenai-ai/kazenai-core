"""SDK 1.2.4: one request before each call (start_on_admit).

When Agent FinOps advertises ``capabilities.start_on_admit``, the reserve asks FinOps
to record "dispatch may occur" itself and the SDK sends no provider_started. An older
FinOps keeps the separate start. A reserve whose answer never arrives is abandoned in
the background: the same reserve is replayed to find the hold, and admission_abandoned
releases it. Concurrent first calls share one capabilities fetch.
"""

from __future__ import annotations

import asyncio
import threading
import uuid

import httpx
import pytest

from fast_finops import FastFinOps
from kazenai import monitor
from kazenai.enforcement import BudgetUnavailable
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


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "test")
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "1")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVE_TIMEOUT_S", "0.5")
    org_policy.reset_for_tests()
    guard._LIFECYCLE_CAP_CACHE.clear()
    transport.reset_pools()
    yield
    guard.drain_settlements(5.0)
    org_policy.reset_for_tests()
    guard._LIFECYCLE_CAP_CACHE.clear()
    transport.reset_pools()


def _client(monkeypatch, finops: FastFinOps, *, is_async: bool = False):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", finops.url)

    def provider(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_CHAT)

    if is_async:
        raw = openai.AsyncOpenAI(api_key="t", base_url="https://openai.test/v1", max_retries=0,
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)))
    else:
        raw = openai.OpenAI(api_key="t", base_url="https://openai.test/v1", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(provider)))
    return monitor(raw, org_id="org-start", workspace_id="ws", agent_id=f"sa-{uuid.uuid4().hex[:6]}", max_budget_usd=5.0)


def _ask(client) -> str:
    return client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": f"q {uuid.uuid4().hex}"}]
    ).choices[0].message.content


async def _aask(client) -> str:
    resp = await client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": f"q {uuid.uuid4().hex}"}]
    )
    return resp.choices[0].message.content


def test_start_on_admit_sends_no_provider_started(monkeypatch):
    finops = FastFinOps(start_on_admit=True)
    try:
        assert _ask(_client(monkeypatch, finops)) == "ok"
        assert finops.events() == ["reserve", "usage_known"]
        assert finops.settles("reserve")[0]["start_on_admit"] is True
    finally:
        finops.stop()


def test_start_on_admit_sends_no_provider_started_async(monkeypatch):
    finops = FastFinOps(start_on_admit=True)
    try:
        assert asyncio.run(_aask(_client(monkeypatch, finops, is_async=True))) == "ok"
        assert finops.events() == ["reserve", "usage_known"]
        assert finops.settles("reserve")[0]["start_on_admit"] is True
    finally:
        finops.stop()


def test_an_older_finops_keeps_the_separate_start(monkeypatch):
    finops = FastFinOps(start_on_admit=False)
    try:
        assert _ask(_client(monkeypatch, finops)) == "ok"
        assert finops.events() == ["reserve", "provider_started", "usage_known"]
        assert "start_on_admit" not in finops.settles("reserve")[0]
    finally:
        finops.stop()


@pytest.mark.parametrize("is_async", [False, True])
def test_an_unanswered_reserve_is_abandoned_in_the_background(monkeypatch, is_async):
    monkeypatch.setattr(guard, "_ABANDON_DELAYS_S", (0.05, 0.05))
    finops = FastFinOps(start_on_admit=True)
    finops.slow["reserve"] = 1.5  # longer than the 0.5 s timeout, for both attempts
    finops.slow_reserve_attempts = 2
    try:
        client = _client(monkeypatch, finops, is_async=is_async)
        with pytest.raises(BudgetUnavailable):
            asyncio.run(_aask(client)) if is_async else _ask(client)
        assert guard.drain_settlements(10.0)
        reserves = finops.settles("reserve")
        assert len(reserves) == 3  # two attempts, then the background replay
        assert len({r["idempotency_key"] for r in reserves}) == 1
        abandoned = finops.settles("admission_abandoned")
        assert len(abandoned) == 1
        assert abandoned[0]["settlement_capability"].startswith("kzcap1.")
        assert finops.settles("usage_known") == [] and finops.settles("provider_started") == []
    finally:
        finops.stop()


def test_concurrent_first_calls_share_one_capabilities_fetch_async(monkeypatch):
    finops = FastFinOps(start_on_admit=True)
    finops.slow["lifecycle"] = 0.3
    try:
        client = _client(monkeypatch, finops, is_async=True)

        async def burst():
            return await asyncio.gather(*(_aask(client) for _ in range(20)))

        assert asyncio.run(burst()) == ["ok"] * 20
        assert finops.lifecycle_gets == 1
    finally:
        finops.stop()


def test_concurrent_first_calls_share_one_capabilities_fetch_threads(monkeypatch):
    finops = FastFinOps(start_on_admit=True)
    finops.slow["lifecycle"] = 0.3
    try:
        client = _client(monkeypatch, finops)
        answers: list[str] = []
        threads = [threading.Thread(target=lambda: answers.append(_ask(client))) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert answers == ["ok"] * 10
        assert finops.lifecycle_gets == 1
    finally:
        finops.stop()
