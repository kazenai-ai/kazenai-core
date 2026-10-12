"""Real framework packages + fake OpenAI HTTP (importorskip)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

openai = pytest.importorskip("openai")
langchain_openai = pytest.importorskip("langchain_openai")
langgraph = pytest.importorskip("langgraph")

from kazenai import monitor
from kazenai.integrations.adapter_contract import UnsupportedFrameworkPath
from kazenai.integrations.langchain import (
    KazenCallbackHandler,
    bind_monitored_chat_openai,
    wrap_langchain_runnable,
    wrap_model_call_middleware,
)
from kazenai.integrations.langgraph import wrap_graph

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


OPENAI_JSON = {
    "id": "chatcmpl-p8",
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


@pytest.fixture
def fake_openai(monkeypatch):
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.delenv("KAZENAI_ENFORCEMENT_OWNER", raising=False)
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json=OPENAI_JSON)
        return httpx.Response(404, json={"error": "missing"})

    transport = httpx.MockTransport(handler)
    client = openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
    )
    return monitor(client, max_budget_usd=1.0, org_id="org-p8", workspace_id="ws-p8"), hits


def test_langchain_chatopenai_bind_invoke_control(monkeypatch):
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json=OPENAI_JSON)
        return httpx.Response(404, json={"error": "missing"})

    raw = openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    ChatOpenAI = langchain_openai.ChatOpenAI
    model = ChatOpenAI(model="gpt-4o-mini", api_key="test", temperature=0, max_retries=0)
    bind_monitored_chat_openai(
        model,
        raw,
        max_budget_usd=1.0,
        org_id="org-p8",
        workspace_id="ws-p8",
    )
    bound = getattr(model, "_kazenai_bound_guarded_client")
    out = model.invoke([{"role": "user", "content": "hi"}])
    assert "hello" in str(out.content)
    assert any("chat/completions" in h for h in hits)

    adapter = wrap_langchain_runnable(
        model,
        org_id="org-p8",
        capability="control",
        guarded_client=bound,
    )
    assert adapter.invoke([{"role": "user", "content": "again"}])
    assert len([h for h in hits if "chat/completions" in h]) >= 2


def test_langchain_control_rejects_decoy_guarded_client(fake_openai):
    guarded, _hits = fake_openai
    with pytest.raises(UnsupportedFrameworkPath) as ei:
        wrap_langchain_runnable(
            _EchoRunnable(),
            org_id="org-p8",
            capability="control",
            guarded_client=guarded,
        )
    assert ei.value.code == "unguarded_control_boundary"


def test_langchain_real_callback_emits_events():
    from langchain_core.runnables import RunnableLambda

    events: list[Any] = []
    cb = KazenCallbackHandler(org_id="org-p8", on_event=lambda e: events.append(e))
    assert getattr(cb, "ignore_chain") is False
    RunnableLambda(lambda x: x).invoke("hi", config={"callbacks": [cb]})
    assert events
    assert events[0].schema_version == "1.2"
    assert any(e.event_type == "workflow.start" for e in events)


def test_langchain_agent_middleware_is_agent_middleware():
    from langchain.agents.middleware.types import AgentMiddleware

    mw = wrap_model_call_middleware(org_id="org-p8")
    assert isinstance(mw, AgentMiddleware)
    assert mw.__class__.wrap_tool_call is AgentMiddleware.wrap_tool_call


class _EchoRunnable:
    def invoke(self, x):
        return x

    async def ainvoke(self, x):
        return x

    def stream(self, x):
        yield x

    async def astream(self, x):
        yield x


def test_langgraph_compiled_graph_modes(fake_openai):
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    guarded, hits = fake_openai

    class S(TypedDict):
        text: str

    def node(state: S) -> S:
        # Model call at guarded boundary inside node
        r = guarded.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": state["text"]}],
        )
        return {"text": r.choices[0].message.content or ""}

    g = StateGraph(S)
    g.add_node("n", node)
    g.add_edge("n", END)
    g.set_entry_point("n")
    compiled = g.compile()
    from kazenai.integrations.langgraph import declare_langgraph_control_boundary

    declare_langgraph_control_boundary(compiled, guarded)
    adapter = wrap_graph(
        compiled,
        org_id="org-p8",
        workspace_id="ws-p8",
        capability="control",
        guarded_client=guarded,
    )
    out = adapter.invoke({"text": "hi"}, config={"configurable": {"thread_id": "t-p8"}})
    assert out["text"] == "hello"
    assert hits


def test_budget_denial_no_dispatch(monkeypatch):
    """Explicit FinOps 402 denial (not a closed port / transport ambiguity)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from kazenai.enforcement import BudgetExceeded

    class _DenyFinOps:
        def handle(self, path: str, body: dict) -> tuple[int, dict]:
            if path.endswith("/lifecycle"):
                return 200, {
                    "capabilities": {"reservation_settlement": True},
                    "effective_lifecycle": {
                        "profile": "control",
                        "accounting_mode": "reservation_settlement",
                        "failure_policy": "fail_closed",
                    },
                }
            if path.endswith("/reserve"):
                return 402, {"allowed": False, "reason": "capacity", "error_code": "budget_exhausted"}
            return 404, {"error": "not_found"}

    deny = _DenyFinOps()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: ANN001
            return

        def do_GET(self):  # noqa: N802
            status, payload = deny.handle(self.path, {})
            raw = __import__("json").dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            body = __import__("json").loads(self.rfile.read(n).decode() or "{}")
            status, payload = deny.handle(self.path, body)
            raw = __import__("json").dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    monkeypatch.setenv("KAZENAI_FINOPS_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_closed")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_closed")
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        return httpx.Response(200, json=OPENAI_JSON)

    client = openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    guarded = monitor(client, max_budget_usd=1.0, org_id="org-deny", workspace_id="ws")
    with pytest.raises(BudgetExceeded):
        guarded.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert hits == []
    server.shutdown()


@pytest.fixture(scope="module", autouse=True)
def _artifact():
    yield
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"kazenai-real-frameworks-{ts}.json"
    path.write_text(
        json.dumps(
            {
                "recorded_at": ts,
                "status": "executed",
                "packages": {
                    "openai": getattr(openai, "__version__", "?"),
                    "langchain_openai": getattr(langchain_openai, "__version__", "?"),
                    "langgraph": getattr(langgraph, "__version__", "?"),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
