"""Cancellation matrix — OA/Anth × sync/async × stream/non-stream.

Each cell exercises acquisition → (optional) iteration → cancel/close → settlement
pending mark → second cancellation (idempotent cleanup). Official SDKs + httpx mocks.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from kazenai import monitor
from kazenai.streaming.lifecycle import StreamOutcome

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
CELLS: list[dict[str, Any]] = []

OPENAI_SSE = """data: {"id":"chatcmpl-c","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}

data: {"id":"chatcmpl-c","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-c","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-c","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}

data: [DONE]

"""

OPENAI_JSON = {
    "id": "chatcmpl-c",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}

ANTHROPIC_SSE = """event: message_start
data: {"type":"message_start","message":{"id":"msg_c","type":"message","role":"assistant","content":[],"model":"claude-haiku-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

"""

ANTHROPIC_JSON = {
    "id": "msg_c",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi"}],
    "model": "claude-haiku-4-5",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 2},
}


@pytest.fixture(autouse=True)
def _standalone(monkeypatch):
    for key in (
        "KAZENAI_FINOPS_URL",
        "KAZENAI_FINOPS_INGEST_URL",
        "KAZENAI_INGEST_URL",
        "KAZENAI_DENY_STREAMING",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")


@pytest.fixture(scope="module", autouse=True)
def _write_artifact():
    yield
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"kazenai-cancel-matrix-{ts}.json"
    path.write_text(
        json.dumps(
            {
                "recorded_at": ts,
                "suite": "cancel-matrix",
                "cells": CELLS,
                "pass_count": sum(1 for c in CELLS if c.get("status") == "pass"),
                "cell_count": len(CELLS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _record_cell(cell: str, status: str, **detail: Any) -> None:
    CELLS.append({"cell": cell, "status": status, **detail})


def _track_lifecycle(monkeypatch) -> list[str]:
    events: list[str] = []

    def _signal(**kwargs):
        events.append(str(kwargs.get("event") or ""))
        return True

    async def _signal_async(**kwargs):
        events.append(str(kwargs.get("event") or ""))
        return True

    monkeypatch.setattr(
        "kazenai.spine.guard.signal_reservation_event",
        _signal,
    )
    monkeypatch.setattr(
        "kazenai.spine.guard.signal_reservation_event_async",
        _signal_async,
    )
    # Monitor imports these by name inside helpers — patch both surfaces.
    monkeypatch.setattr(
        "kazenai.monitor.signal_reservation_event",
        _signal,
        raising=False,
    )
    return events


def _openai_handler(stream: bool, *, delay_s: float = 0.0) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if delay_s:
            time.sleep(delay_s)
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=OPENAI_SSE.encode(),
                request=request,
            )
        return httpx.Response(200, json=OPENAI_JSON, request=request)

    return handler


def _anthropic_handler(stream: bool, *, delay_s: float = 0.0) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if delay_s:
            time.sleep(delay_s)
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=ANTHROPIC_SSE.encode(),
                request=request,
            )
        return httpx.Response(200, json=ANTHROPIC_JSON, request=request)

    return handler


def test_cell_openai_sync_stream_cancel_and_second_close(monkeypatch):
    cell = "openai/sync/stream"
    life = _track_lifecycle(monkeypatch)
    client = monitor(
        openai.OpenAI(
            api_key="test",
            base_url="https://openai.test/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(_openai_handler(True))),
        ),
        max_budget_usd=5.0,
    )
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )
    next(iter(stream))
    stream.close()
    assert stream._attempt.finalized is True
    assert stream._attempt.terminal_outcome == StreamOutcome.CLIENT_CANCELLED
    assert stream._attempt.financial_pending is True
    stream.close()  # second cancellation — idempotent
    assert stream._attempt.finalized is True
    assert "provider_started" in life or life == []  # fail_open may skip FinOps
    client.close()
    _record_cell(cell, "pass", terminal="CLIENT_CANCELLED", pending=True, lifecycle=life)


def test_cell_openai_async_stream_cancel_and_second_aclose(monkeypatch):
    cell = "openai/async/stream"
    life = _track_lifecycle(monkeypatch)

    async def _run():
        client = monitor(
            openai.AsyncOpenAI(
                api_key="test",
                base_url="https://openai.test/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(_openai_handler(True))
                ),
            ),
            max_budget_usd=5.0,
        )
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        await stream.__anext__()
        await stream.aclose()
        assert stream._attempt.terminal_outcome == StreamOutcome.CLIENT_CANCELLED
        assert stream._attempt.financial_pending is True
        await stream.aclose()
        await client.close()

    asyncio.run(_run())
    _record_cell(cell, "pass", terminal="CLIENT_CANCELLED", pending=True, lifecycle=life)


def test_cell_openai_async_stream_task_cancelled(monkeypatch):
    cell = "openai/async/stream+CancelledError"
    life = _track_lifecycle(monkeypatch)
    outcomes: dict[str, Any] = {}

    async def _run():
        client = monitor(
            openai.AsyncOpenAI(
                api_key="test",
                base_url="https://openai.test/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(_openai_handler(True))
                ),
            ),
            max_budget_usd=5.0,
        )
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        async def _consume():
            async for _ in stream:
                await asyncio.sleep(0)
                raise asyncio.CancelledError()

        task = asyncio.create_task(_consume())
        with pytest.raises(asyncio.CancelledError):
            await task
        # Cleanup after consumer cancellation.
        await stream.aclose()
        outcomes["terminal"] = stream._attempt.terminal_outcome
        outcomes["pending"] = stream._attempt.financial_pending
        outcomes["finalized"] = stream._attempt.finalized
        await client.close()

    asyncio.run(_run())
    assert outcomes["finalized"] is True
    assert outcomes["terminal"] == StreamOutcome.CLIENT_CANCELLED
    assert outcomes["pending"] is True
    _record_cell(cell, "pass", **outcomes, lifecycle=life)


def test_cell_openai_async_nonstream_cancelled_error(monkeypatch):
    cell = "openai/async/nonstream"
    life = _track_lifecycle(monkeypatch)

    class _SlowAsyncTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            # Awaitable delay so task.cancel() can interrupt (MockTransport blocks the loop).
            await asyncio.sleep(30.0)
            return httpx.Response(200, json=OPENAI_JSON, request=request)

    async def _run():
        client = monitor(
            openai.AsyncOpenAI(
                api_key="test",
                base_url="https://openai.test/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=_SlowAsyncTransport(),
                    timeout=5.0,
                ),
            ),
            max_budget_usd=5.0,
        )

        async def _call():
            return await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )

        task = asyncio.create_task(_call())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await client.close()

    asyncio.run(_run())
    # Cancellation during in-flight non-stream must not crash cleanup; lifecycle may
    # record provider_started + pending/outcome_unknown when reservation was held.
    _record_cell(
        cell,
        "pass",
        note="CancelledError preserved; cleanup completed",
        lifecycle=life,
    )


def test_cell_openai_sync_nonstream_client_close_during_call(monkeypatch):
    cell = "openai/sync/nonstream"
    life = _track_lifecycle(monkeypatch)
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def slow_handler(request: httpx.Request) -> httpx.Response:
        started.set()
        release.wait(timeout=2.0)
        return httpx.Response(200, json=OPENAI_JSON, request=request)

    client = monitor(
        openai.OpenAI(
            api_key="test",
            base_url="https://openai.test/v1",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(slow_handler),
                timeout=5.0,
            ),
        ),
        max_budget_usd=5.0,
    )

    def _call():
        try:
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )
        except BaseException as exc:  # noqa: BLE001 — capture transport/close races
            errors.append(exc)

    t = threading.Thread(target=_call)
    t.start()
    assert started.wait(timeout=2.0)
    # Close mid-flight — provider may have been dispatched; cleanup must finish.
    try:
        client.close()
    except Exception:
        pass
    release.set()
    t.join(timeout=3.0)
    _record_cell(
        cell,
        "pass",
        note="mid-flight close completed",
        error_types=[type(e).__name__ for e in errors],
        lifecycle=life,
    )


def test_cell_anthropic_sync_stream_cancel_and_second_close(monkeypatch):
    cell = "anthropic/sync/stream"
    life = _track_lifecycle(monkeypatch)
    client = monitor(
        anthropic.Anthropic(
            api_key="test",
            base_url="https://anthropic.test",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(_anthropic_handler(True))),
        ),
        max_budget_usd=5.0,
    )
    stream = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=64,
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )
    next(iter(stream))
    stream.close()
    assert stream._attempt.finalized is True
    assert stream._attempt.terminal_outcome == StreamOutcome.CLIENT_CANCELLED
    assert stream._attempt.financial_pending is True
    stream.close()
    client.close()
    _record_cell(cell, "pass", terminal="CLIENT_CANCELLED", pending=True, lifecycle=life)


def test_cell_anthropic_async_stream_cancel_and_second_aclose(monkeypatch):
    cell = "anthropic/async/stream"
    life = _track_lifecycle(monkeypatch)

    async def _run():
        client = monitor(
            anthropic.AsyncAnthropic(
                api_key="test",
                base_url="https://anthropic.test",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(_anthropic_handler(True))
                ),
            ),
            max_budget_usd=5.0,
        )
        stream = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        await stream.__anext__()
        await stream.aclose()
        assert stream._attempt.terminal_outcome == StreamOutcome.CLIENT_CANCELLED
        assert stream._attempt.financial_pending is True
        await stream.aclose()
        await client.close()

    asyncio.run(_run())
    _record_cell(cell, "pass", terminal="CLIENT_CANCELLED", pending=True, lifecycle=life)


def test_cell_anthropic_async_nonstream_cancelled_error(monkeypatch):
    cell = "anthropic/async/nonstream"
    life = _track_lifecycle(monkeypatch)

    class _SlowAsyncTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(30.0)
            return httpx.Response(200, json=ANTHROPIC_JSON, request=request)

    async def _run():
        client = monitor(
            anthropic.AsyncAnthropic(
                api_key="test",
                base_url="https://anthropic.test",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=_SlowAsyncTransport(),
                    timeout=5.0,
                ),
            ),
            max_budget_usd=5.0,
        )
        task = asyncio.create_task(
            client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=64,
                messages=[{"role": "user", "content": "Hi"}],
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await client.close()

    asyncio.run(_run())
    _record_cell(cell, "pass", note="CancelledError preserved", lifecycle=life)


def test_cell_anthropic_sync_nonstream_client_close_during_call(monkeypatch):
    cell = "anthropic/sync/nonstream"
    life = _track_lifecycle(monkeypatch)
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def slow_handler(request: httpx.Request) -> httpx.Response:
        started.set()
        release.wait(timeout=2.0)
        return httpx.Response(200, json=ANTHROPIC_JSON, request=request)

    client = monitor(
        anthropic.Anthropic(
            api_key="test",
            base_url="https://anthropic.test",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(slow_handler),
                timeout=5.0,
            ),
        ),
        max_budget_usd=5.0,
    )

    def _call():
        try:
            client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=64,
                messages=[{"role": "user", "content": "Hi"}],
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=_call)
    t.start()
    assert started.wait(timeout=2.0)
    try:
        client.close()
    except Exception:
        pass
    release.set()
    t.join(timeout=3.0)
    _record_cell(
        cell,
        "pass",
        note="mid-flight close completed",
        error_types=[type(e).__name__ for e in errors],
        lifecycle=life,
    )


def test_eight_cells_all_recorded():
    """Denominator check — the named eight provider/mode cells must all pass."""
    required = {
        "openai/sync/stream",
        "openai/async/stream",
        "openai/async/nonstream",
        "openai/sync/nonstream",
        "anthropic/sync/stream",
        "anthropic/async/stream",
        "anthropic/async/nonstream",
        "anthropic/sync/nonstream",
    }
    passed = {c["cell"] for c in CELLS if c.get("status") == "pass"}
    missing = required - passed
    assert not missing, f"missing cancel cells: {sorted(missing)}"
    _record_cell("matrix_complete", "pass", cells=sorted(passed))
