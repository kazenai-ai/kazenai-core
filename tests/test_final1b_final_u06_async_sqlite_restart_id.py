"""U06: AsyncSqliteSaver restart must keep the same operation_id across process cache loss."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, TypedDict

import pytest

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
RESULTS: list[dict[str, Any]] = []


def _record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"scenario": name, "status": status, "detail": detail})


@pytest.fixture(scope="module", autouse=True)
def _write_artifact():
    yield
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"final1b-final-u06-async-sqlite-restart-{ts}.json"
    path.write_text(
        json.dumps(
            {
                "recorded_at": ts,
                "loop": "FINAL_1_b_final",
                "phase": "u06-async-sqlite-restart-id",
                "results": RESULTS,
                "pass_count": sum(1 for r in RESULTS if r["status"] == "pass"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_async_sqlite_restart_keeps_operation_id(tmp_path, monkeypatch):
    """Clear process-local cache between saver sessions — durable envelope must win."""
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, StateGraph
    from langgraph.types import Command, interrupt

    from kazenai.integrations import langgraph as lg
    from kazenai.integrations.langgraph import wrap_graph

    monkeypatch.setenv("KAZENAI_LANGGRAPH_ENVELOPE_DIR", str(tmp_path / "envelopes"))
    db = tmp_path / "restart.sqlite"
    monkeypatch.setenv("KAZENAI_LANGGRAPH_STORE_ID", str(db.resolve()))
    lg._THREAD_ENVELOPES.clear()

    class S(TypedDict):
        answer: str

    def gate(state: S) -> S:
        return {"answer": interrupt("approve?")}

    def build():
        g = StateGraph(S)
        g.add_node("gate", gate)
        g.set_entry_point("gate")
        g.add_edge("gate", END)
        return g

    events: list[Any] = []
    cfg = {"configurable": {"thread_id": f"async-restart-{uuid.uuid4().hex[:8]}"}}
    org = "org-u06-restart"

    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        compiled = build().compile(checkpointer=saver)
        adapter = wrap_graph(
            compiled,
            org_id=org,
            workspace_id="ws",
            project_id="p",
            agent_id="ag",
            on_event=events.append,
        )
        paused = await adapter.ainvoke({"answer": ""}, config=cfg)
    assert paused.get("__interrupt__")
    starts = [
        (e.payload or {}).get("operation_id")
        for e in events
        if e.event_type == "workflow.start"
    ]
    assert len(starts) == 1 and starts[0]
    op_first = starts[0]

    # Simulate process restart: drop only the in-memory cache.
    lg._THREAD_ENVELOPES.clear()
    events.clear()

    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        compiled2 = build().compile(checkpointer=saver)
        adapter2 = wrap_graph(
            compiled2,
            org_id=org,
            workspace_id="ws",
            project_id="p",
            agent_id="ag",
            on_event=events.append,
        )
        out = await adapter2.ainvoke(Command(resume="yes"), config=cfg)
    assert out["answer"] == "yes"
    starts2 = [
        (e.payload or {}).get("operation_id")
        for e in events
        if e.event_type == "workflow.start"
    ]
    assert len(starts2) == 1
    assert starts2[0] == op_first, f"operation_id drifted across restart: {op_first} vs {starts2[0]}"
    assert "workflow.end" in [e.event_type for e in events]
    _record(
        "async_sqlite_restart_id_equality",
        "pass",
        f"operation_id={op_first}",
    )


@pytest.mark.asyncio
async def test_async_sqlite_restart_without_constructor_operation_id(tmp_path, monkeypatch):
    """Same ID equality when constructor does not seed operation_id."""
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, StateGraph
    from langgraph.types import Command, interrupt

    from kazenai.integrations import langgraph as lg
    from kazenai.integrations.langgraph import wrap_graph

    monkeypatch.setenv("KAZENAI_LANGGRAPH_ENVELOPE_DIR", str(tmp_path / "envelopes2"))
    db = tmp_path / "restart2.sqlite"
    monkeypatch.setenv("KAZENAI_LANGGRAPH_STORE_ID", str(db.resolve()))
    lg._THREAD_ENVELOPES.clear()

    class S(TypedDict):
        text: str

    def node(state: S) -> S:
        if state.get("text") == "wait":
            return {"text": interrupt("go?")}
        return state

    def build():
        g = StateGraph(S)
        g.add_node("n", node)
        g.set_entry_point("n")
        g.add_edge("n", END)
        return g

    events: list[Any] = []
    cfg = {"configurable": {"thread_id": "no-seed-op"}}

    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        adapter = wrap_graph(
            build().compile(checkpointer=saver),
            org_id="org-b",
            workspace_id="ws",
            on_event=events.append,
        )
        await adapter.ainvoke({"text": "wait"}, config=cfg)
    op_a = next(
        (e.payload or {}).get("operation_id")
        for e in events
        if e.event_type == "workflow.start"
    )
    lg._THREAD_ENVELOPES.clear()
    events.clear()
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        adapter2 = wrap_graph(
            build().compile(checkpointer=saver),
            org_id="org-b",
            workspace_id="ws",
            on_event=events.append,
        )
        out = await adapter2.ainvoke(Command(resume="ok"), config=cfg)
    op_b = next(
        (e.payload or {}).get("operation_id")
        for e in events
        if e.event_type == "workflow.start"
    )
    assert out["text"] == "ok"
    assert op_a == op_b
    _record("async_sqlite_restart_unseeded_id_equality", "pass", f"{op_a}")
