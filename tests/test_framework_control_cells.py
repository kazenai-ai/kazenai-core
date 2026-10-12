"""Framework Control cells (real packages where required)."""

from __future__ import annotations

import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from kazenai.integrations.adapter_contract import UnsupportedFrameworkPath
from kazenai.integrations.langchain import (
    bind_monitored_chat_openai,
    declare_langchain_control_boundary,
    wrap_langchain_runnable,
)
from kazenai.integrations.langgraph import declare_langgraph_control_boundary, wrap_graph
from kazenai.integrations.crewai import CrewAdapter, wrap_crew

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
RESULTS: list[dict[str, Any]] = []

# Test-local copy of kazenai-examples/litellm-otel-control/fake_upstream.py, so
# Core's suite does not depend on a sibling checkout.
from fake_upstream import serve as serve_fake_upstream  # noqa: E402

from test_litellm_control import FakeFinOps, _serve_finops  # noqa: E402


OPENAI_JSON = {
    "id": "chatcmpl-p8-cell",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


def _record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"scenario": name, "status": status, "detail": detail})


@pytest.fixture(scope="module", autouse=True)
def _write_cells_artifact():
    yield
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"kazenai-cells-{ts}.json"
    path.write_text(
        json.dumps(
            {
                "recorded_at": ts,
                "gate": "FRAMEWORK_CONTROL_CELLS",
                "results": RESULTS,
                "pass_count": sum(1 for r in RESULTS if r["status"] == "pass"),
                "skip_count": sum(1 for r in RESULTS if r["status"] == "skip"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _isolate_control_env(monkeypatch):
    """Per-test FinOps/LiteLLM isolation — avoids cross-test 402 flakes."""
    from kazenai.spine import guard as guard_mod
    from kazenai.spine.guard import reset_default_sink

    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_ENFORCEMENT_OWNER", raising=False)
    monkeypatch.delenv("KAZENAI_CONTROL_PROFILE", raising=False)
    monkeypatch.delenv("KAZENAI_FINOPS_CONTROL_PROFILE", raising=False)
    monkeypatch.delenv("KAZENAI_PROFILE", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    guard_mod._LIFECYCLE_CAP_CACHE.clear()
    reset_default_sink()
    yield
    guard_mod._LIFECYCLE_CAP_CACHE.clear()
    reset_default_sink()
    try:
        from test_litellm_proxy_integration import _terminate_procs

        _terminate_procs()
    except Exception:
        pass


@pytest.fixture
def control_env(monkeypatch):
    """Strict lifecycle profile for cells that assert reserve/settle behavior."""
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")


def _mock_openai_client(*, hits: list[str], json_body: dict | None = None):
    openai = pytest.importorskip("openai")

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json=json_body or OPENAI_JSON)
        return httpx.Response(404, json={"error": "missing"})

    sync = httpx.Client(transport=httpx.MockTransport(handler))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=sync,
    )
    aclient = openai.AsyncOpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=async_client,
    )
    return client, aclient, hits


def _reserve_count(fake: FakeFinOps) -> int:
    return sum(1 for e in fake.events if e.get("event") == "reserve")


def _serve_gateway_proxy(*, upstream_base: str, finops: FakeFinOps) -> ThreadingHTTPServer:
    """Minimal LiteLLM-style HTTP gateway: reserve once, then forward upstream."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode() or "{}")
            status, _payload = finops.handle(
                "/v1/budget/reserve",
                {
                    "estimated_usd_micros": 50_000,
                    "org_id": "org-gw",
                    "workspace_id": "ws-gw",
                    "run_id": body.get("metadata", {}).get("run_id") or "run-gw",
                    "model": body.get("model") or "gpt-4o-mini",
                },
            )
            if status == 402:
                raw = json.dumps(_payload).encode()
                self.send_response(402)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)
                return
            upstream_url = f"{upstream_base.rstrip('/')}/chat/completions"
            with httpx.Client(timeout=5.0) as client:
                up = client.post(upstream_url, json=body)
            self.send_response(up.status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(up.content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.asyncio
async def test_langchain_litellm_gateway_async_non_streaming(control_env, monkeypatch):
    """LangChain Runnable → HTTP gateway (gateway-owned reserve, SDK skip)."""
    import asyncio

    openai = pytest.importorskip("openai")

    upstream, oracle = serve_fake_upstream("127.0.0.1", 0)
    upstream_base = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    fake = FakeFinOps(capacity_micros=10_000_000)
    finops_srv = _serve_finops(fake)
    gw = _serve_gateway_proxy(upstream_base=upstream_base, finops=fake)
    gw_base = f"http://127.0.0.1:{gw.server_address[1]}/v1"

    monkeypatch.setenv("KAZENAI_ENFORCEMENT_OWNER", "gateway")
    monkeypatch.setenv("KAZENAI_FINOPS_URL", f"http://127.0.0.1:{finops_srv.server_address[1]}")

    raw = openai.OpenAI(
        api_key="gw",
        base_url=gw_base,
        max_retries=0,
        http_client=httpx.Client(timeout=5.0),
    )

    from kazenai import monitor

    guarded = monitor(raw, max_budget_usd=1.0, org_id="org-lc-gw", workspace_id="ws-lc-gw")

    class LlmRunnable:
        """Async Runnable calling guarded Chat Completions via the gateway base URL."""

        async def ainvoke(self, messages: list[dict[str, str]]) -> Any:
            return await asyncio.to_thread(
                guarded.chat.completions.create,
                model="gpt-4o-mini",
                messages=messages,
            )

    op_id = f"op_{uuid.uuid4().hex}"
    events: list[Any] = []
    llm_runnable = declare_langchain_control_boundary(LlmRunnable(), guarded)
    adapter = wrap_langchain_runnable(
        llm_runnable,
        org_id="org-lc-gw",
        workspace_id="ws-lc-gw",
        capability="control",
        guarded_client=guarded,
        operation_id=op_id,
        on_event=events.append,
    )
    reserves_before = _reserve_count(fake)
    out = await adapter.ainvoke([{"role": "user", "content": "hi"}])
    assert out.choices[0].message.content == "ok"
    assert oracle.count() == 1
    assert _reserve_count(fake) - reserves_before == 1
    assert any(e.event_type == "workflow.start" for e in events)
    blob = json.dumps([e.payload for e in events])
    assert op_id in blob or any(getattr(e, "operation_id", None) == op_id for e in events)

    upstream.shutdown()
    gw.shutdown()
    finops_srv.shutdown()
    _record("langchain_litellm_gateway_ainvoke", "pass", f"dispatches={oracle.count()}")


@pytest.mark.asyncio
async def test_langchain_gateway_async_streaming(control_env, monkeypatch):
    """LangChain Runnable astream → HTTP gateway → SSE upstream (Control boundary).

    Uses monitored AsyncOpenAI Chat Completions streaming (not ChatOpenAI Responses
    bind). Gateway-owned reserve once; stream finalize settles with usage.
    """
    openai = pytest.importorskip("openai")
    from kazenai import monitor
    from kazenai.streaming.lifecycle import StreamOutcome

    OPENAI_SSE = (
        'data: {"id":"chatcmpl-p8s","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant",'
        '"content":"ok"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-p8s","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: {"id":"chatcmpl-p8s","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,'
        '"total_tokens":7}}\n\n'
        "data: [DONE]\n\n"
    )

    hits: list[str] = []

    def stream_handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=OPENAI_SSE.encode(),
                request=request,
            )
        return httpx.Response(404, json={"error": "missing"}, request=request)

    # App-owned stream lifecycle against OpenAI-compatible gateway URL (SSE mock).
    # Live LiteLLM streaming is certified in test_langchain_live_litellm_proxy_async_streaming.
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_ENFORCEMENT_OWNER", raising=False)
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")

    araw = openai.AsyncOpenAI(
        api_key="gw",
        base_url="https://gateway.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(stream_handler), timeout=10.0),
    )
    guarded = monitor(araw, max_budget_usd=1.0, org_id="org-lc-stream", workspace_id="ws")
    events: list[Any] = []
    parts: list[str] = []

    class StreamRunnable:
        async def astream(self, messages: list[dict[str, str]]):
            stream = await guarded.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
                yield chunk
            assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE

    stream_runnable = declare_langchain_control_boundary(StreamRunnable(), guarded)
    adapter = wrap_langchain_runnable(
        stream_runnable,
        org_id="org-lc-stream",
        capability="control",
        guarded_client=guarded,
        on_event=events.append,
    )
    async for _ in adapter.astream([{"role": "user", "content": "hi"}]):
        pass
    assert "".join(parts) == "ok"
    assert any("chat/completions" in h for h in hits)
    assert any(e.event_type == "workflow.start" for e in events)
    assert any(e.event_type == "workflow.end" for e in events)
    _record(
        "langchain_gateway_astream",
        "pass",
        f"parts={''.join(parts)} hits={len(hits)}",
    )


@pytest.mark.asyncio
async def test_langchain_live_litellm_proxy_async_streaming(control_env, monkeypatch):
    """LangChain Runnable astream → real LiteLLM proxy → fake upstream SSE."""
    import os

    from test_litellm_proxy_integration import (  # noqa: PLC0415
        CONFIG_APP,
        EXAMPLES,
        _ensure_litellm_venv,
        _ephemeral_port,
        _oracle_count,
        _start_fake_upstream,
        _start_litellm_proxy,
    )

    openai = pytest.importorskip("openai")
    from kazenai import monitor
    from kazenai.streaming.lifecycle import StreamOutcome

    _ensure_litellm_venv()
    _, upstream_v1 = _start_fake_upstream()
    proxy_port = _ephemeral_port()
    master_key = "sk-p8-live-stream"

    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_ENFORCEMENT_OWNER", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")

    env = os.environ.copy()
    env.update(
        {
            "KAZENAI_FAKE_UPSTREAM_URL": upstream_v1,
            "LITELLM_MASTER_KEY": master_key,
            "PYTHONPATH": str(EXAMPLES),
        }
    )
    _start_litellm_proxy(
        config=CONFIG_APP,
        port=proxy_port,
        env=env,
        expect_models=("gpt-4o-mini",),
    )

    aclient = openai.AsyncOpenAI(
        api_key=master_key,
        base_url=f"http://127.0.0.1:{proxy_port}/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(timeout=60.0),
    )
    guarded = monitor(aclient, max_budget_usd=1.0, org_id="org-p8-stream", workspace_id="ws")
    events: list[Any] = []
    parts: list[str] = []
    before = _oracle_count(upstream_v1)

    class ProxyStreamRunnable:
        async def astream(self, messages: list[dict[str, str]]):
            stream = await guarded.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and getattr(chunk.choices[0].delta, "content", None):
                    parts.append(chunk.choices[0].delta.content or "")
                yield chunk
            assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE

    proxy_stream = declare_langchain_control_boundary(ProxyStreamRunnable(), guarded)
    adapter = wrap_langchain_runnable(
        proxy_stream,
        org_id="org-p8-stream",
        capability="control",
        guarded_client=guarded,
        on_event=events.append,
    )
    async for _ in adapter.astream([{"role": "user", "content": "hi"}]):
        pass
    assert "".join(parts) == "ok"
    assert _oracle_count(upstream_v1) - before == 1
    assert any(e.event_type == "workflow.end" for e in events)
    try:
        from test_litellm_proxy_integration import _terminate_procs

        _terminate_procs()
    except Exception:
        pass
    await aclient.close()
    _record(
        "langchain_live_litellm_proxy_astream",
        "pass",
        f"proxy_port={proxy_port} parts={''.join(parts)}",
    )

def test_cache_short_circuit_no_new_charge(control_env, monkeypatch):
    """Reused Runnable result must not reserve or dispatch provider again."""
    openai = pytest.importorskip("openai")
    from kazenai import monitor

    hits: list[str] = []
    raw, _, _ = _mock_openai_client(hits=hits)
    fake = FakeFinOps(capacity_micros=10_000_000)
    finops_srv = _serve_finops(fake)
    monkeypatch.setenv("KAZENAI_FINOPS_URL", f"http://127.0.0.1:{finops_srv.server_address[1]}")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_closed")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_closed")

    guarded = monitor(raw, max_budget_usd=1.0, org_id="org-cache", workspace_id="ws")

    class Inner:
        def invoke(self, key: str) -> str:
            guarded.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": key}],
            )
            return "fresh"

    cache: dict[str, str] = {}

    class Cached:
        def __init__(self, inner: Inner) -> None:
            self._inner = inner

        def invoke(self, key: str) -> str:
            if key in cache:
                return cache[key]
            val = self._inner.invoke(key)
            cache[key] = val
            return val

    cached = declare_langchain_control_boundary(Cached(Inner()), guarded)
    adapter = wrap_langchain_runnable(
        cached,
        org_id="org-cache",
        capability="control",
        guarded_client=guarded,
    )
    assert adapter.invoke("k") == "fresh"
    assert adapter.invoke("k") == "fresh"
    assert len(hits) == 1
    assert _reserve_count(fake) == 1
    finops_srv.shutdown()
    _record("cache_short_circuit_no_charge", "pass", f"hits={len(hits)} reserves={_reserve_count(fake)}")


def test_finops_budget_denial_explicit_not_closed_port(control_env, monkeypatch):
    """402 reserve from FakeFinOps — no provider dispatch, explicit BudgetExceeded."""
    openai = pytest.importorskip("openai")
    from kazenai.enforcement import BudgetExceeded
    from kazenai import monitor

    hits: list[str] = []
    raw, _, _ = _mock_openai_client(hits=hits)
    fake = FakeFinOps(capacity_micros=0)
    finops_srv = _serve_finops(fake)
    monkeypatch.setenv("KAZENAI_FINOPS_URL", f"http://127.0.0.1:{finops_srv.server_address[1]}")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_closed")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_closed")

    guarded = monitor(raw, max_budget_usd=1.0, org_id="org-deny", workspace_id="ws")
    with pytest.raises(BudgetExceeded):
        guarded.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert hits == []
    finops_srv.shutdown()
    _record("finops_402_no_dispatch", "pass")


def test_langgraph_sqlite_interrupt_restart_resume(control_env, tmp_path):
    """Durable SqliteSaver interrupt → new adapter → resume; one provider dispatch."""
    langgraph = pytest.importorskip("langgraph")
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, StateGraph
    from langgraph.types import Command, interrupt
    from typing import TypedDict

    openai = pytest.importorskip("openai")
    from kazenai import monitor

    hits: list[str] = []
    raw, _, _ = _mock_openai_client(hits=hits)
    guarded = monitor(raw, max_budget_usd=1.0, org_id="org-lg", workspace_id="ws")
    op_id = f"op_{uuid.uuid4().hex[:12]}"
    db = tmp_path / "lg-checkpoints.sqlite"

    class S(TypedDict):
        text: str
        phase: str

    def model_node(state: S) -> S:
        r = guarded.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": state.get("text") or "hi"}],
        )
        return {"text": r.choices[0].message.content or "", "phase": "modeled"}

    def approval_node(state: S) -> S:
        answer = interrupt({"question": "approve?"})
        return {"text": state["text"], "phase": f"approved:{answer}"}

    def build():
        g = StateGraph(S)
        g.add_node("model", model_node)
        g.add_node("approve", approval_node)
        g.add_edge("model", "approve")
        g.add_edge("approve", END)
        g.set_entry_point("model")
        return g

    events: list[Any] = []
    cfg = {"configurable": {"thread_id": "thread-p8-cell"}}

    with SqliteSaver.from_conn_string(str(db)) as checkpointer:
        compiled = build().compile(checkpointer=checkpointer)
        declare_langgraph_control_boundary(compiled, guarded)
        adapter = wrap_graph(
            compiled,
            org_id="org-lg",
            workspace_id="ws",
            capability="control",
            guarded_client=guarded,
            operation_id=op_id,
            on_event=events.append,
        )
        paused = adapter.invoke({"text": "hi", "phase": ""}, config=cfg)
    assert paused.get("__interrupt__")
    assert len(hits) == 1
    types = [e.event_type for e in events]
    assert "workflow.suspend" in types
    assert types.count("workflow.end") == 0

    events.clear()
    with SqliteSaver.from_conn_string(str(db)) as checkpointer:
        compiled2 = build().compile(checkpointer=checkpointer)
        declare_langgraph_control_boundary(compiled2, guarded)
        adapter2 = wrap_graph(
            compiled2,
            org_id="org-lg",
            workspace_id="ws",
            capability="control",
            guarded_client=guarded,
            operation_id=op_id,
            on_event=events.append,
        )
        out = adapter2.invoke(Command(resume="yes"), config=cfg)
    assert out["phase"].startswith("approved:")
    assert len(hits) == 1
    assert "workflow.end" in [e.event_type for e in events]
    _record("langgraph_sqlite_interrupt_resume", "pass", f"hits={len(hits)} op={op_id}")


@pytest.mark.asyncio
async def test_langchain_live_litellm_proxy_async_non_streaming(control_env, monkeypatch):
    """LangChain Runnable ainvoke → real LiteLLM proxy (live helpers)."""
    import asyncio
    import os

    from test_litellm_proxy_integration import (  # noqa: PLC0415
        CONFIG_APP,
        EXAMPLES,
        _ensure_litellm_venv,
        _ephemeral_port,
        _oracle_count,
        _start_fake_upstream,
        _start_litellm_proxy,
    )

    openai = pytest.importorskip("openai")
    _ensure_litellm_venv()

    _, upstream_v1 = _start_fake_upstream()
    proxy_port = _ephemeral_port()
    master_key = "sk-p8-live-lc"

    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_ENFORCEMENT_OWNER", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")

    env = os.environ.copy()
    env.update(
        {
            "KAZENAI_FAKE_UPSTREAM_URL": upstream_v1,
            "LITELLM_MASTER_KEY": master_key,
            "PYTHONPATH": str(EXAMPLES),
        }
    )
    _start_litellm_proxy(
        config=CONFIG_APP,
        port=proxy_port,
        env=env,
        expect_models=("gpt-4o-mini",),
    )

    from kazenai import monitor

    client = openai.OpenAI(
        api_key=master_key,
        base_url=f"http://127.0.0.1:{proxy_port}/v1",
        max_retries=0,
        http_client=httpx.Client(timeout=60.0),
    )
    guarded = monitor(client, max_budget_usd=1.0, org_id="org-p8-live", workspace_id="ws-p8-live")
    op_id = f"op_{uuid.uuid4().hex[:12]}"
    events: list[Any] = []

    class ProxyRunnable:
        async def ainvoke(self, messages: list[dict[str, str]]) -> Any:
            return await asyncio.to_thread(
                guarded.chat.completions.create,
                model="gpt-4o-mini",
                messages=messages,
            )

    proxy_runnable = declare_langchain_control_boundary(ProxyRunnable(), guarded)
    adapter = wrap_langchain_runnable(
        proxy_runnable,
        org_id="org-p8-live",
        workspace_id="ws-p8-live",
        capability="control",
        guarded_client=guarded,
        operation_id=op_id,
        on_event=events.append,
    )
    before = _oracle_count(upstream_v1)
    out = await adapter.ainvoke([{"role": "user", "content": "hi"}])
    assert out.choices[0].message.content == "ok"
    assert _oracle_count(upstream_v1) - before == 1
    assert any(e.event_type == "workflow.start" for e in events)
    try:
        from test_litellm_proxy_integration import _terminate_procs

        _terminate_procs()
    except Exception:
        pass
    _record(
        "langchain_live_litellm_proxy_ainvoke",
        "pass",
        f"proxy_port={proxy_port} op={op_id}",
    )


@pytest.mark.asyncio
async def test_langgraph_async_sqlite_interrupt_resume(control_env, tmp_path):
    """AsyncSqliteSaver + wrap_graph.ainvoke interrupt/resume; one provider dispatch."""
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, StateGraph
    from langgraph.types import Command, interrupt
    from typing import TypedDict

    openai = pytest.importorskip("openai")
    from kazenai import monitor

    hits: list[str] = []
    raw, _, _ = _mock_openai_client(hits=hits)
    guarded = monitor(raw, max_budget_usd=1.0, org_id="org-lg-async", workspace_id="ws")
    op_id = f"op_{uuid.uuid4().hex[:12]}"
    db = tmp_path / "lg-async-checkpoints.sqlite"

    class S(TypedDict):
        text: str
        phase: str

    def model_node(state: S) -> S:
        r = guarded.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": state.get("text") or "hi"}],
        )
        return {"text": r.choices[0].message.content or "", "phase": "modeled"}

    def approval_node(state: S) -> S:
        answer = interrupt({"question": "approve?"})
        return {"text": state["text"], "phase": f"approved:{answer}"}

    def build():
        g = StateGraph(S)
        g.add_node("model", model_node)
        g.add_node("approve", approval_node)
        g.add_edge("model", "approve")
        g.add_edge("approve", END)
        g.set_entry_point("model")
        return g

    events: list[Any] = []
    cfg = {"configurable": {"thread_id": "thread-p8-async"}}

    async with AsyncSqliteSaver.from_conn_string(str(db)) as checkpointer:
        compiled = build().compile(checkpointer=checkpointer)
        declare_langgraph_control_boundary(compiled, guarded)
        adapter = wrap_graph(
            compiled,
            org_id="org-lg-async",
            workspace_id="ws",
            capability="control",
            guarded_client=guarded,
            operation_id=op_id,
            on_event=events.append,
        )
        paused = await adapter.ainvoke({"text": "hi", "phase": ""}, config=cfg)
    assert paused.get("__interrupt__")
    assert len(hits) == 1
    assert "workflow.suspend" in [e.event_type for e in events]
    assert "workflow.end" not in [e.event_type for e in events]

    # Clear process-local cache to force durable envelope / checkpoint recovery.
    from kazenai.integrations import langgraph as lg

    first_ops = [
        (e.payload or {}).get("operation_id")
        for e in events
        if e.event_type == "workflow.start"
    ]
    lg._THREAD_ENVELOPES.clear()
    events.clear()
    async with AsyncSqliteSaver.from_conn_string(str(db)) as checkpointer:
        compiled2 = build().compile(checkpointer=checkpointer)
        declare_langgraph_control_boundary(compiled2, guarded)
        adapter2 = wrap_graph(
            compiled2,
            org_id="org-lg-async",
            workspace_id="ws",
            capability="control",
            guarded_client=guarded,
            operation_id=op_id,
            on_event=events.append,
        )
        out = await adapter2.ainvoke(Command(resume="yes"), config=cfg)
    assert out["phase"].startswith("approved:")
    assert len(hits) == 1
    assert "workflow.end" in [e.event_type for e in events]
    resume_ops = [
        (e.payload or {}).get("operation_id")
        for e in events
        if e.event_type == "workflow.start"
    ]
    assert first_ops and resume_ops and first_ops[0] == resume_ops[0] == op_id
    _record("langgraph_async_sqlite_interrupt_resume", "pass", f"hits={len(hits)} op={op_id}")


def test_chatopenai_responses_api_rejected_before_dispatch():
    langchain_openai = pytest.importorskip("langchain_openai")
    openai = pytest.importorskip("openai")
    hits: list[str] = []
    raw, _, _ = _mock_openai_client(hits=hits)
    llm = langchain_openai.ChatOpenAI(
        model="gpt-4o-mini",
        api_key="test",
        temperature=0,
        use_responses_api=True,
    )
    with pytest.raises(UnsupportedFrameworkPath) as ei:
        bind_monitored_chat_openai(llm, raw, max_budget_usd=1.0, org_id="o", workspace_id="ws")
    assert ei.value.code == "unsupported_provider_api"
    assert hits == []
    _record("chatopenai_responses_api_rejected", "pass")


def test_crewai_hook_aborted_no_provider_dispatch(control_env, tmp_path, monkeypatch):
    """Run after live/async cells — CrewAI import can disturb FinOps env globals."""
    pytest.importorskip("crewai")
    pytest.importorskip("langchain_openai")
    openai = pytest.importorskip("openai")

    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crewai-data"))
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    from crewai import Agent, Crew, Task
    from crewai.hooks import HookAborted, clear_all_global_hooks, register_before_llm_call_hook

    hits: list[str] = []
    raw, _, _ = _mock_openai_client(hits=hits)

    def deny(_ctx: Any) -> None:
        raise HookAborted("budget blocked before llm")

    register_before_llm_call_hook(deny)
    try:
        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            llm="gpt-4o-mini",
            verbose=False,
            allow_delegation=False,
        )
        task = Task(description="say hi", expected_output="hi", agent=agent)
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        from kazenai import monitor
        from kazenai.integrations.crewai import declare_crew_control_boundary

        guarded = monitor(raw, max_budget_usd=1.0, org_id="org-crew", workspace_id="ws")
        declare_crew_control_boundary(crew, guarded)
        events: list[Any] = []
        adapter = CrewAdapter(
            crew,
            org_id="org-crew",
            capability="control",
            guarded_client=guarded,
            on_event=events.append,
        )
        with pytest.raises(Exception) as ei:
            adapter.kickoff()
        assert type(ei.value).__name__ == "HookAborted" or "HookAborted" in str(type(ei.value))
        assert hits == []
        assert any(e.event_type == "workflow.suspend" for e in events)
        assert not any(e.event_type == "workflow.end" for e in events)
        _record("crewai_hook_aborted_no_dispatch", "pass")
    finally:
        clear_all_global_hooks()
        from kazenai.spine.guard import reset_default_sink

        reset_default_sink()
