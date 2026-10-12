"""Optional paid framework smokes — LangChain, LangGraph, LiteLLM — never in default CI.

Every cell runs real provider calls (OpenAI and Anthropic) through the
integration under test and checks the same contract as the provider smoke:
a real response, a recorded cost equal to the pricing table applied to the
provider-reported input/output tokens, and denial before dispatch.

Cells that use the shared spend authority run against a real Agent FinOps
service (Control profile, Postgres) and read its ledger directly.

Required environment:

    KAZENAI_LIVE_FRAMEWORK_SMOKE=1
    OPENAI_API_KEY, ANTHROPIC_API_KEY
    KAZENAI_LIVE_FINOPS_URL        e.g. http://127.0.0.1:18099 (Control profile)
    KAZENAI_LIVE_FINOPS_API_KEY    admin key for KAZENAI_LIVE_ORG_ID (creates budget limits)
    KAZENAI_LIVE_FINOPS_SDK_KEY    an ordinary (e.g. developer-role) key the SDK and gateway use;
                                   they send X-Kazen-Act-As-Org for the key's own org
    KAZENAI_LIVE_FINOPS_DB_URL     postgresql:// URL of that FinOps database (read-only use)
    KAZENAI_LIVE_LITELLM_BIN       path to a `litellm` proxy executable with kazenai installed

Optional: KAZENAI_LIVE_OPENAI_MODEL (gpt-4o-mini), KAZENAI_LIVE_ANTHROPIC_MODEL
(claude-haiku-4-5), KAZENAI_LIVE_ANTHROPIC_MODEL_2 (claude-haiku-4-5),
KAZENAI_LIVE_ORG_ID (org-live), KAZENAI_LIVE_WORKSPACE_ID (ws-live).

Evidence (models, versions, token usage, costs, ledger rows — no keys, prompts
or outputs) is written to ``artifacts/live-framework-smoke-<ts>.json``.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

pytestmark = pytest.mark.live

_RESULTS: List[Dict[str, Any]] = []
_FINDINGS: List[str] = []
EXAMPLES = Path(__file__).resolve().parents[2] / "kazenai-examples" / "litellm-otel-control"


# --------------------------------------------------------------------------- setup


def _enabled() -> bool:
    return os.getenv("KAZENAI_LIVE_FRAMEWORK_SMOKE", "").strip() in {"1", "true", "yes", "on"}


def _openai_model() -> str:
    return os.getenv("KAZENAI_LIVE_OPENAI_MODEL", "gpt-4o-mini")


def _anthropic_model() -> str:
    return os.getenv("KAZENAI_LIVE_ANTHROPIC_MODEL", "claude-haiku-4-5")


def _anthropic_model_2() -> str:
    return os.getenv("KAZENAI_LIVE_ANTHROPIC_MODEL_2", "claude-haiku-4-5")


def _key(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        pytest.skip(f"{name} not set")
    return value


def _finops_cfg() -> Dict[str, str]:
    cfg = {
        "url": (os.getenv("KAZENAI_LIVE_FINOPS_URL") or "").rstrip("/"),
        "api_key": os.getenv("KAZENAI_LIVE_FINOPS_API_KEY") or "",
        "sdk_key": os.getenv("KAZENAI_LIVE_FINOPS_SDK_KEY") or "",
        "db": os.getenv("KAZENAI_LIVE_FINOPS_DB_URL") or "",
        "org": os.getenv("KAZENAI_LIVE_ORG_ID", "org-live"),
        "ws": os.getenv("KAZENAI_LIVE_WORKSPACE_ID", "ws-live"),
    }
    if not (cfg["url"] and cfg["api_key"] and cfg["sdk_key"] and cfg["db"]):
        pytest.skip("KAZENAI_LIVE_FINOPS_URL / _API_KEY / _SDK_KEY / _DB_URL not set")
    return cfg


@pytest.fixture(autouse=True)
def _gate(monkeypatch):
    if not _enabled():
        pytest.skip("Set KAZENAI_LIVE_FRAMEWORK_SMOKE=1 to run paid framework smokes")
    for key in (
        "KAZENAI_FINOPS_URL",
        "KAZENAI_FINOPS_INGEST_URL",
        "KAZENAI_INGEST_URL",
        "KAZENAI_ENFORCEMENT_OWNER",
        "KAZENAI_CONTROL_PROFILE",
        "KAZENAI_FINOPS_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_LANGGRAPH_ENVELOPE_DIR", tempfile.mkdtemp(prefix="kz-lg-env-"))


def _local_mode(monkeypatch) -> None:
    """monitor() local caps only; no shared authority."""
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")


def _finops_mode(monkeypatch, cfg: Dict[str, str]) -> None:
    """monitor() reserves and settles against the real FinOps Control authority."""
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")
    monkeypatch.setenv("KAZENAI_FINOPS_URL", cfg["url"])
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", cfg["sdk_key"])
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_closed")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_closed")


@pytest.fixture(scope="module", autouse=True)
def _write_evidence():
    yield
    if not _RESULTS:
        return
    target = Path(os.getenv("KAZENAI_LIVE_EVIDENCE_DIR", "") or Path(__file__).resolve().parents[1] / "artifacts")
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versions = {}
    for dist in ("kazenai", "openai", "anthropic", "langchain", "langchain-core", "langchain-openai", "langgraph", "langgraph-checkpoint-sqlite"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = "not-installed"
    payload = {
        "evidence": "live_framework_smoke",
        "recorded_at_utc": stamp,
        "versions": versions,
        "litellm_version": os.getenv("KAZENAI_LIVE_LITELLM_VERSION", ""),
        "models": {"openai": _openai_model(), "anthropic": _anthropic_model(), "anthropic_2": _anthropic_model_2()},
        "cells": sorted(_RESULTS, key=lambda r: r["cell"]),
        "findings": _FINDINGS,
        "all_passed": all(r.get("passed") for r in _RESULTS),
    }
    (target / f"live-framework-smoke-{stamp}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


class _Cell:
    def __init__(self, name: str, **fields: Any) -> None:
        self.fields: Dict[str, Any] = {"cell": name, "passed": False, **fields}
        self.started = time.perf_counter()

    def __enter__(self) -> Dict[str, Any]:
        return self.fields

    def __exit__(self, exc_type, exc, tb) -> None:
        self.fields["latency_ms"] = round((time.perf_counter() - self.started) * 1000, 1)
        if exc is not None:
            self.fields["error"] = f"{exc_type.__name__}: {str(exc)[:300]}"
        else:
            self.fields["passed"] = True
        _RESULTS.append(self.fields)


# --------------------------------------------------------------------------- pricing / ledger


def _price(model: str, inp: int, out: int) -> float:
    from kazenai.model_pricing import model_cost_usd_per_million

    cost = model_cost_usd_per_million(model, input_tokens=inp, output_tokens=out)
    assert cost is not None, f"{model} not in the pricing table"
    return cost


def _recorded(*clients: Any) -> float:
    total = 0.0
    for client in clients:
        enf = getattr(client, "_kazenai_enforcement", None)
        if enf is not None:
            total += float(enf.snapshot()["cumulative_cost_usd"])
    return total


def _create_limit(cfg: Dict[str, str], *, dimension: str, dimension_id: str, limit_usd: float) -> None:
    import httpx

    body = {
        "workspace_id": cfg["ws"],
        "dimension": dimension,
        "dimension_id": dimension_id,
        "window": "daily",
        "limit_usd": limit_usd,
        "audit_reason": "live framework smoke",
    }
    resp = httpx.post(
        f"{cfg['url']}/v1/org/{cfg['org']}/budget-limits",
        json=body,
        headers={"X-API-Key": cfg["api_key"]},
        timeout=10,
    )
    assert resp.status_code in (200, 201), resp.text


def _ledger(cfg: Dict[str, str], where: str, params: tuple) -> List[Dict[str, Any]]:
    import psycopg2
    import psycopg2.extras

    url = cfg["db"].replace("postgresql+asyncpg://", "postgresql://")
    with psycopg2.connect(url) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT reservation_id, state, reserved_usd_micros, actual_usd_micros, run_id, feature, "
            "feature_id, workflow_id, business_subject_ref, decision_id "
            f"FROM reservations WHERE org_id = %s AND {where} ORDER BY created_at",
            (cfg["org"], *params),
        )
        return [dict(r) for r in cur.fetchall()]


def _receipts(cfg: Dict[str, str], *, workflow_id: str = "", run_id: str = "") -> List[Dict[str, Any]]:
    import psycopg2
    import psycopg2.extras

    url = cfg["db"].replace("postgresql+asyncpg://", "postgresql://")
    with psycopg2.connect(url) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT decision_id, decision, reason_code, reservation_id, receipt_json "
            "FROM economic_decision_receipts WHERE org_id = %s ORDER BY created_at",
            (cfg["org"],),
        )
        rows = [dict(r) for r in cur.fetchall()]
    out = []
    for row in rows:
        blob = json.dumps(row.get("receipt_json") or {})
        if (workflow_id and workflow_id in blob) or (run_id and run_id in blob):
            out.append(row)
    return out


def _micros(usd: float) -> int:
    import math

    return int(math.ceil(round(usd * 1_000_000, 6)))


# --------------------------------------------------------------------------- LangChain


def _bound_chat_openai(*, api_key: str, budget: float = 1.0, shared: Any = None, **attribution: Any):
    from langchain_openai import ChatOpenAI
    from openai import AsyncOpenAI, OpenAI

    from kazenai.integrations.langchain import bind_monitored_chat_openai

    chat = ChatOpenAI(model=_openai_model(), api_key=api_key, max_retries=0, stream_usage=True)
    bind_monitored_chat_openai(
        chat,
        OpenAI(api_key=api_key, max_retries=0),
        async_client=AsyncOpenAI(api_key=api_key, max_retries=0),
        **({"budget": shared} if shared is not None else {"max_budget_usd": budget}),
        soft_pause_pct=100.0,
        loop_anomaly_threshold=1.0,
        **attribution,
    )
    return chat, chat._kazenai_bound_guarded_client, getattr(chat, "_kazenai_bound_guarded_async_client", None)


def _bound_chat_anthropic(*, api_key: str, model: str, budget: float = 1.0, shared: Any = None, **attribution: Any):
    import anthropic
    from langchain_anthropic import ChatAnthropic

    from kazenai.integrations.langchain import bind_monitored_chat_anthropic

    chat = ChatAnthropic(model=model, api_key=api_key, max_retries=0, max_tokens=400)
    bind_monitored_chat_anthropic(
        chat,
        anthropic.Anthropic(api_key=api_key, max_retries=0),
        async_client=anthropic.AsyncAnthropic(api_key=api_key, max_retries=0),
        soft_pause_pct=100.0,
        loop_anomaly_threshold=1.0,
        **({"budget": shared} if shared is not None else {"max_budget_usd": budget}),
        **attribution,
    )
    return chat, chat._kazenai_bound_guarded_client, chat._kazenai_bound_guarded_async_client


def _usage_of(message: Any) -> tuple[int, int]:
    meta = getattr(message, "usage_metadata", None) or {}
    assert meta, "LangChain message carries no usage_metadata"
    return int(meta["input_tokens"]), int(meta["output_tokens"])


@pytest.mark.parametrize("mode", ["invoke", "stream", "ainvoke", "astream"])
def test_langchain_chat_openai_control(monkeypatch, mode):
    from kazenai.integrations.langchain import wrap_langchain_runnable

    _local_mode(monkeypatch)
    key = _key("OPENAI_API_KEY")
    with _Cell(f"langchain.chat_openai.{mode}", model=_openai_model(), framework="langchain") as cell:
        chat, guarded, guarded_async = _bound_chat_openai(api_key=key, workflow_id="lc-live", feature_id=f"lc-{mode}")
        adapter = wrap_langchain_runnable(chat, capability="control", guarded_client=guarded)
        prompt = [{"role": "user", "content": "Reply with the single word: ok"}]
        if mode == "invoke":
            msg = adapter.invoke(prompt)
        elif mode == "ainvoke":
            msg = asyncio.run(adapter.ainvoke(prompt))
        elif mode == "stream":
            msg = None
            for chunk in adapter.stream(prompt):
                msg = chunk if msg is None else msg + chunk
        else:

            async def _go():
                acc = None
                async for chunk in adapter.astream(prompt):
                    acc = chunk if acc is None else acc + chunk
                return acc

            msg = asyncio.run(_go())
        assert str(msg.content).strip()
        inp, out = _usage_of(msg)
        expected = _price(_openai_model(), inp, out)
        recorded = _recorded(guarded, guarded_async)
        cell.update(input_tokens=inp, output_tokens=out, expected_cost_usd=expected, recorded_cost_usd=recorded)
        assert recorded == pytest.approx(expected, rel=1e-6, abs=1e-9)

        # Too-small budget: denied before dispatch (invalid key proves no provider call).
        from kazenai import BudgetExceeded

        deny_chat, deny_guarded, _ = _bound_chat_openai(api_key="sk-invalid-never-dispatched", budget=1e-9)
        deny = wrap_langchain_runnable(deny_chat, capability="control", guarded_client=deny_guarded)
        with pytest.raises(BudgetExceeded):
            if mode in ("invoke", "stream"):
                deny.invoke(prompt)
            else:
                asyncio.run(deny.ainvoke(prompt))
        cell["denied_before_dispatch"] = True


@pytest.mark.parametrize("mode", ["invoke", "stream", "ainvoke", "astream"])
def test_langchain_chat_anthropic_control(monkeypatch, mode):
    from kazenai import BudgetExceeded
    from kazenai.integrations.langchain import wrap_langchain_runnable

    _local_mode(monkeypatch)
    key = _key("ANTHROPIC_API_KEY")
    model = _anthropic_model()
    with _Cell(f"langchain.chat_anthropic.{mode}", model=model, framework="langchain") as cell:
        chat, guarded, guarded_async = _bound_chat_anthropic(api_key=key, model=model, workflow_id="lc-anthropic")
        adapter = wrap_langchain_runnable(chat, capability="control", guarded_client=guarded)
        prompt = [{"role": "user", "content": "Reply with the single word: ok"}]
        if mode == "invoke":
            msg = adapter.invoke(prompt)
        elif mode == "ainvoke":
            msg = asyncio.run(adapter.ainvoke(prompt))
        elif mode == "stream":
            msg = None
            for chunk in adapter.stream(prompt):
                msg = chunk if msg is None else msg + chunk
        else:

            async def _go():
                acc = None
                async for chunk in adapter.astream(prompt):
                    acc = chunk if acc is None else acc + chunk
                return acc

            msg = asyncio.run(_go())
        assert str(msg.content).strip()
        inp, out = _usage_of(msg)
        expected = _price(model, inp, out)
        recorded = _recorded(guarded, guarded_async)
        cell.update(input_tokens=inp, output_tokens=out, expected_cost_usd=expected, recorded_cost_usd=recorded)
        assert recorded == pytest.approx(expected, rel=1e-6, abs=1e-9)

        deny_chat, deny_guarded, _ = _bound_chat_anthropic(api_key="sk-ant-invalid-never-dispatched", model=model, budget=1e-9)
        deny = wrap_langchain_runnable(deny_chat, capability="control", guarded_client=deny_guarded)
        with pytest.raises(BudgetExceeded):
            if mode in ("invoke", "stream"):
                deny.invoke(prompt)
            else:
                asyncio.run(deny.ainvoke(prompt))
        cell["denied_before_dispatch"] = True


# --------------------------------------------------------------------------- LangGraph


def test_langgraph_checkpoint_interrupt_resume(monkeypatch, tmp_path):
    from typing import TypedDict

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, StateGraph

    from kazenai.integrations.langgraph import declare_langgraph_control_boundary, wrap_graph

    _local_mode(monkeypatch)
    key = _key("OPENAI_API_KEY")
    with _Cell("langgraph.checkpoint_resume", model=_openai_model(), framework="langgraph") as cell:
        chat, guarded, guarded_async = _bound_chat_openai(api_key=key, workflow_id="lg-resume")
        calls: List[tuple[int, int]] = []

        class S(TypedDict, total=False):
            first: str
            second: str

        def first(state: S) -> S:
            msg = chat.invoke([{"role": "user", "content": "Reply with the single word: one"}])
            calls.append(_usage_of(msg))
            return {"first": str(msg.content)}

        def second(state: S) -> S:
            msg = chat.invoke([{"role": "user", "content": "Reply with the single word: two"}])
            calls.append(_usage_of(msg))
            return {"second": str(msg.content)}

        def build(saver):
            g = StateGraph(S)
            g.add_node("first", first)
            g.add_node("second", second)
            g.set_entry_point("first")
            g.add_edge("first", "second")
            g.add_edge("second", END)
            compiled = g.compile(checkpointer=saver, interrupt_before=["second"])
            declare_langgraph_control_boundary(compiled, guarded)
            return wrap_graph(compiled, capability="control", guarded_client=guarded)

        cfg = {"configurable": {"thread_id": f"t-{uuid.uuid4().hex[:8]}"}}
        db = str(tmp_path / "ckpt.sqlite")
        with SqliteSaver.from_conn_string(db) as saver:
            build(saver).invoke({}, config=cfg)
        assert len(calls) == 1, "graph should pause before the second node"
        with SqliteSaver.from_conn_string(db) as saver:  # fresh adapter, durable checkpoint
            out = build(saver).invoke(None, config=cfg)
        assert out.get("first") and out.get("second")
        assert len(calls) == 2, f"resume must not re-run the first node: {len(calls)} calls"
        expected = sum(_price(_openai_model(), i, o) for i, o in calls)
        recorded = _recorded(guarded, guarded_async)
        cell.update(provider_calls=len(calls), expected_cost_usd=expected, recorded_cost_usd=recorded)
        assert recorded == pytest.approx(expected, rel=1e-6, abs=1e-9)


# --------------------------------------------------------------------------- multi-model workflow


def _multi_model_graph(
    *, openai_key: str, anthropic_key: str, attribution: Dict[str, Any], steps: List[Dict[str, Any]], shared: Any = None
):
    """plan (OpenAI) → draft (Anthropic model 1) → review (Anthropic model 2)."""
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    from kazenai.integrations.langgraph import declare_langgraph_control_boundary, wrap_graph

    chat, guarded, guarded_async = _bound_chat_openai(api_key=openai_key, shared=shared, feature_id="plan", **attribution)
    draft_chat, draft_client, draft_async = _bound_chat_anthropic(
        api_key=anthropic_key, model=_anthropic_model(), shared=shared, feature_id="draft", **attribution
    )
    review_chat, review_client, review_async = _bound_chat_anthropic(
        api_key=anthropic_key, model=_anthropic_model_2(), shared=shared, feature_id="review", **attribution
    )

    class S(TypedDict, total=False):
        topic: str
        plan: str
        draft: str
        review: str

    def plan(state: S) -> S:
        msg = chat.invoke([{"role": "user", "content": f"List 3 short bullet points for a note about {state['topic']}."}])
        steps.append({"step": "plan", "model": _openai_model(), "usage": _usage_of(msg)})
        return {"plan": str(msg.content)}

    def _anthropic_step(step_chat: Any, model: str, name: str, prompt: str) -> str:
        msg = step_chat.invoke([{"role": "user", "content": prompt}])
        steps.append({"step": name, "model": model, "usage": _usage_of(msg)})
        return str(msg.content)

    def draft(state: S) -> S:
        return {"draft": _anthropic_step(draft_chat, _anthropic_model(), "draft", f"Write a 120-word note from this plan:\n{state['plan']}")}

    def review(state: S) -> S:
        return {"review": _anthropic_step(review_chat, _anthropic_model_2(), "review", f"Give two short suggestions to improve:\n{state['draft']}")}

    g = StateGraph(S)
    g.add_node("plan", plan)
    g.add_node("draft", draft)
    g.add_node("review", review)
    g.set_entry_point("plan")
    g.add_edge("plan", "draft")
    g.add_edge("draft", "review")
    g.add_edge("review", END)
    compiled = g.compile()
    # Every provider client the nodes use is declared on the graph.
    declare_langgraph_control_boundary(compiled, guarded, draft_client, review_client)
    graph = wrap_graph(compiled, capability="control", guarded_clients=[guarded, draft_client, review_client])
    clients = {"plan": (guarded, guarded_async), "draft": (draft_client, draft_async), "review": (review_client, review_async)}
    return graph, clients


def test_multi_model_workflow_local_attribution(monkeypatch):
    """Three models, two providers, one LangGraph workflow — local caps only."""
    _local_mode(monkeypatch)
    okey, akey = _key("OPENAI_API_KEY"), _key("ANTHROPIC_API_KEY")
    with _Cell(
        "workflow.multi_model.local",
        models=[_openai_model(), _anthropic_model(), _anthropic_model_2()],
        framework="langgraph",
    ) as cell:
        steps: List[Dict[str, Any]] = []
        graph, clients = _multi_model_graph(
            openai_key=okey, anthropic_key=akey, attribution={"workflow_id": "wf-local"}, steps=steps
        )
        out = graph.invoke({"topic": "budgeting AI agents"})
        assert out.get("review")
        per_step = []
        for step in steps:
            inp, outp = step["usage"]
            expected = _price(step["model"], inp, outp)
            recorded = _recorded(*clients[step["step"]])
            per_step.append({**step, "expected_cost_usd": expected, "recorded_cost_usd": recorded})
            assert recorded == pytest.approx(expected, rel=1e-6, abs=1e-9), step["step"]
        cell["steps"] = per_step
        cell["workflow_total_usd"] = sum(s["recorded_cost_usd"] for s in per_step)
        enforcements = {id(c._kazenai_enforcement) for pair in clients.values() for c in pair if c is not None}
        cell["independent_local_budgets"] = len(enforcements)
        _FINDINGS.append(
            "Local mode without SharedBudget: each monitor() client keeps its own budget "
            f"({len(enforcements)} independent local budgets in a 3-step workflow)."
        )


def test_multi_model_workflow_shared_local_budget(monkeypatch):
    """Same 3-model workflow, one SharedBudget, no FinOps: runs until denied mid-workflow."""
    from kazenai import BudgetExceeded, SharedBudget

    _local_mode(monkeypatch)
    okey, akey = _key("OPENAI_API_KEY"), _key("ANTHROPIC_API_KEY")
    limit_usd = float(os.getenv("KAZENAI_LIVE_WORKFLOW_LIMIT_USD", "0.02"))
    with _Cell(
        "workflow.multi_model.shared_local_budget",
        models=[_openai_model(), _anthropic_model(), _anthropic_model_2()],
        framework="langgraph",
        shared_budget_usd=limit_usd,
    ) as cell:
        budget = SharedBudget(max_usd=limit_usd, name="live-workflow")
        steps: List[Dict[str, Any]] = []
        completed_passes, denied_at = 0, None
        wallets = set()
        for _ in range(12):
            graph, clients = _multi_model_graph(
                openai_key=okey, anthropic_key=akey, attribution={"workflow_id": "wf-local-shared"}, steps=steps, shared=budget
            )
            wallets |= {id(c._kazenai_enforcement) for pair in clients.values() for c in pair if c is not None}
            before = len(steps)
            try:
                graph.invoke({"topic": "budgeting AI agents"})
                completed_passes += 1
            except BudgetExceeded:
                denied_at = ["plan", "draft", "review"][len(steps) - before]
                break
        assert wallets == {id(budget.enforcement)}, "every client must draw from the one SharedBudget"
        assert completed_passes >= 1 and denied_at is not None
        expected = sum(_price(s["model"], *s["usage"]) for s in steps)
        assert budget.spent_usd == pytest.approx(expected, rel=1e-6, abs=1e-9)
        assert budget.spent_usd <= limit_usd
        cell.update(completed_passes=completed_passes, denied_step=denied_at, dispatched_steps=len(steps),
                    spent_usd=budget.spent_usd, local_budgets=len(wallets))


def test_multi_model_workflow_shared_finops_budget(monkeypatch):
    """Same workflow, one FinOps workflow budget across models: runs until denied mid-workflow."""
    from kazenai import BudgetExceeded

    cfg = _finops_cfg()
    _finops_mode(monkeypatch, cfg)
    okey, akey = _key("OPENAI_API_KEY"), _key("ANTHROPIC_API_KEY")
    workflow_id = f"wf-multi-{uuid.uuid4().hex[:8]}"
    limit_usd = float(os.getenv("KAZENAI_LIVE_WORKFLOW_LIMIT_USD", "0.02"))
    _create_limit(cfg, dimension="workflow", dimension_id=workflow_id, limit_usd=limit_usd)
    with _Cell(
        "workflow.multi_model.shared_finops_budget",
        models=[_openai_model(), _anthropic_model(), _anthropic_model_2()],
        framework="langgraph",
        workflow_id=workflow_id,
        workflow_limit_usd=limit_usd,
    ) as cell:
        attribution = {"org_id": cfg["org"], "workspace_id": cfg["ws"], "workflow_id": workflow_id}
        steps: List[Dict[str, Any]] = []
        completed_passes = 0
        denied_at: Optional[str] = None
        for _ in range(12):
            graph, _clients = _multi_model_graph(openai_key=okey, anthropic_key=akey, attribution=attribution, steps=steps)
            before = len(steps)
            try:
                graph.invoke({"topic": "budgeting AI agents"})
                completed_passes += 1
            except BudgetExceeded:
                denied_at = ["plan", "draft", "review"][len(steps) - before]
                break
        assert completed_passes >= 1, "at least one full multi-model pass should fit the budget"
        assert denied_at is not None, "the shared workflow budget should eventually deny a step"

        rows = _ledger(cfg, "workflow_id = %s", (workflow_id,))
        assert len(rows) == len(steps), "exactly one reservation per dispatched step"
        assert all(r["state"] == "settled" for r in rows), [r["state"] for r in rows]
        settled_micros = sum(int(r["actual_usd_micros"] or 0) for r in rows)
        assert settled_micros <= _micros(limit_usd)
        # Per-step attribution: settled spend by feature equals the price of what each step used.
        by_feature: Dict[str, int] = {}
        for r in rows:
            by_feature[r["feature_id"] or r["feature"]] = by_feature.get(r["feature_id"] or r["feature"], 0) + int(r["actual_usd_micros"] or 0)
        expected_by_feature: Dict[str, float] = {}
        for s in steps:
            expected_by_feature[s["step"]] = expected_by_feature.get(s["step"], 0.0) + _price(s["model"], *s["usage"])
        for feature, usd in expected_by_feature.items():
            assert abs(by_feature.get(feature, 0) - usd * 1_000_000) <= len(steps) + 1, (feature, by_feature, expected_by_feature)
        receipts = _receipts(cfg, workflow_id=workflow_id)
        denies = [r for r in receipts if r["decision"] == "deny"]
        assert denies, "the denied step must leave a deny receipt"
        cell.update(
            completed_passes=completed_passes,
            denied_step=denied_at,
            dispatched_steps=len(steps),
            reservations=len(rows),
            settled_usd=settled_micros / 1_000_000,
            settled_by_step_usd={k: v / 1_000_000 for k, v in by_feature.items()},
            receipts={"allow": sum(1 for r in receipts if r["decision"] == "allow"), "deny": len(denies)},
            deny_reason=denies[-1]["reason_code"],
        )


# --------------------------------------------------------------------------- LiteLLM proxy


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_proxy(*, gateway_owned: bool, cfg: Dict[str, str], tmp: Path) -> tuple[subprocess.Popen, str, str]:
    binary = (os.getenv("KAZENAI_LIVE_LITELLM_BIN") or "").strip()
    if not binary:
        pytest.skip("KAZENAI_LIVE_LITELLM_BIN not set")
    master = f"sk-litellm-{uuid.uuid4().hex}"
    port = _free_port()
    lines = [
        "model_list:",
        f"  - model_name: {_openai_model()}",
        "    litellm_params:",
        f"      model: openai/{_openai_model()}",
        "      api_key: os.environ/OPENAI_API_KEY",
        f"  - model_name: {_anthropic_model()}",
        "    litellm_params:",
        f"      model: anthropic/{_anthropic_model()}",
        "      api_key: os.environ/ANTHROPIC_API_KEY",
        "router_settings:",
        "  num_retries: 0",
        "  allowed_fails: 0",
        "litellm_settings:",
        "  num_retries: 0",
        "  turn_off_message_logging: true",
        "  drop_params: false",
    ]
    if gateway_owned:
        lines.append("  callbacks: kazenai_litellm_callback.proxy_handler_instance")
    lines += ["general_settings:", "  master_key: os.environ/LITELLM_MASTER_KEY"]
    config = tmp / f"proxy-{'gateway' if gateway_owned else 'app'}.yaml"
    config.write_text("\n".join(lines) + "\n")
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": "/usr/bin:/bin",
        "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
        "LITELLM_MASTER_KEY": master,
        "PYTHONPATH": str(EXAMPLES),
        "LITELLM_LOG": "ERROR",
    }
    if gateway_owned:
        env.update(
            KAZENAI_ENFORCEMENT_OWNER="gateway",
            KAZENAI_FINOPS_URL=cfg["url"],
            KAZENAI_FINOPS_API_KEY=cfg["sdk_key"],
            KAZENAI_ORG_ID=cfg["org"],
            KAZENAI_WORKSPACE_ID=cfg["ws"],
            KAZENAI_CONTROL_PROFILE="1",
            KAZENAI_ENV="dev",
            KAZENAI_GATEWAY_PENDING_DIR=str(tmp / "gateway-pending"),
        )
    log = open(tmp / f"proxy-{port}.log", "w")
    proc = subprocess.Popen([binary, "--config", str(config), "--host", "127.0.0.1", "--port", str(port)], env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    import httpx

    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(f"LiteLLM proxy exited: {(tmp / f'proxy-{port}.log').read_text()[-2000:]}")
        try:
            if httpx.get(f"{base}/health/liveliness", timeout=1).status_code == 200:
                return proc, base, master
        except Exception:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("LiteLLM proxy did not become ready")


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.mark.parametrize("which", ["openai", "anthropic"])
def test_litellm_app_owned_accounting(monkeypatch, tmp_path, which):
    """monitor(OpenAI(base_url=LiteLLM)) owns reserve/settle; the gateway adds no accounting."""
    from openai import OpenAI

    from kazenai import BudgetExceeded, monitor

    cfg = _finops_cfg()
    _key("OPENAI_API_KEY"), _key("ANTHROPIC_API_KEY")
    _finops_mode(monkeypatch, cfg)
    model = _openai_model() if which == "openai" else _anthropic_model()
    proc, base, master = _start_proxy(gateway_owned=False, cfg=cfg, tmp=tmp_path)
    try:
        with _Cell(f"litellm.app_owned.{which}", model=model, framework="litellm") as cell:
            workflow_id = f"wf-lite-app-{uuid.uuid4().hex[:8]}"
            _create_limit(cfg, dimension="workflow", dimension_id=workflow_id, limit_usd=1.0)
            client = monitor(
                OpenAI(base_url=base, api_key=master, max_retries=0),
                org_id=cfg["org"], workspace_id=cfg["ws"], workflow_id=workflow_id, feature_id="litellm-app",
                max_budget_usd=1.0, soft_pause_pct=100.0, loop_anomaly_threshold=1.0,
            )
            from kazenai.model_pricing import output_limit_param

            kwargs = {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ok"}], output_limit_param(model): 256}
            resp = client.chat.completions.create(**kwargs)
            assert resp.choices[0].message.content
            usage_ns = (resp.usage.prompt_tokens, resp.usage.completion_tokens)
            # A different prompt: an identical repeat to the same client is (correctly) a loop.
            stream_kwargs = {**kwargs, "messages": [{"role": "user", "content": "Reply with the single word: yes"}]}
            stream = client.chat.completions.create(stream=True, **stream_kwargs)
            assert "".join((c.choices[0].delta.content or "") if c.choices else "" for c in stream).strip()
            u = stream._attempt.observed_usage
            usage_s = (u.prompt_tokens, u.completion_tokens)
            rows = _ledger(cfg, "workflow_id = %s", (workflow_id,))
            assert len(rows) == 2, f"exactly one reservation per call (no gateway double count): {len(rows)}"
            assert all(r["state"] == "settled" for r in rows)
            expected = [_price(model, *usage_ns), _price(model, *usage_s)]
            actual = [int(r["actual_usd_micros"] or 0) for r in rows]
            for got, want in zip(actual, expected):
                assert abs(got - want * 1_000_000) <= 1, (actual, expected)
            assert _recorded(client) == pytest.approx(sum(expected), rel=1e-6, abs=1e-9)
            cell.update(usage_non_stream=usage_ns, usage_stream=usage_s, expected_cost_usd=expected, settled_usd_micros=actual, reservations=len(rows))

            deny_wf = f"wf-lite-deny-{uuid.uuid4().hex[:8]}"
            _create_limit(cfg, dimension="workflow", dimension_id=deny_wf, limit_usd=0.000001)
            deny = monitor(
                OpenAI(base_url=base, api_key=master, max_retries=0),
                org_id=cfg["org"], workspace_id=cfg["ws"], workflow_id=deny_wf, max_budget_usd=1.0,
            )
            with pytest.raises(BudgetExceeded):
                deny.chat.completions.create(**kwargs)
            assert _ledger(cfg, "workflow_id = %s", (deny_wf,)) == []
            assert any(r["decision"] == "deny" for r in _receipts(cfg, workflow_id=deny_wf))
            cell["denied_before_dispatch"] = True
    finally:
        _stop(proc)


@pytest.mark.parametrize("which", ["openai", "anthropic"])
def test_litellm_gateway_owned_accounting(monkeypatch, tmp_path, which):
    """The LiteLLM callback reserves/settles; a monitored app in gateway-owner mode adds nothing."""
    from openai import APIStatusError, OpenAI

    from kazenai import monitor

    cfg = _finops_cfg()
    _key("OPENAI_API_KEY"), _key("ANTHROPIC_API_KEY")
    model = _openai_model() if which == "openai" else _anthropic_model()
    proc, base, master = _start_proxy(gateway_owned=True, cfg=cfg, tmp=tmp_path)
    try:
        with _Cell(f"litellm.gateway_owned.{which}", model=model, framework="litellm") as cell:
            from kazenai.model_pricing import output_limit_param

            # The gateway reserves with workflow_id = operation id from request metadata.
            op_id = f"wf-lite-gw-{uuid.uuid4().hex[:8]}"
            _create_limit(cfg, dimension="workflow", dimension_id=op_id, limit_usd=1.0)
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                output_limit_param(model): 256,
                "extra_body": {"metadata": {"kazenai_operation_id": op_id, "run_id": op_id}},
            }
            # A monitored app pointed at a gateway-owned proxy must not reserve too.
            monkeypatch.setenv("KAZENAI_ENFORCEMENT_OWNER", "gateway")
            monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")
            monkeypatch.setenv("KAZENAI_FINOPS_URL", cfg["url"])
            monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", cfg["sdk_key"])
            client = monitor(OpenAI(base_url=base, api_key=master, max_retries=0), org_id=cfg["org"], workspace_id=cfg["ws"], max_budget_usd=1.0)
            resp = client.chat.completions.create(**kwargs)
            assert resp.choices[0].message.content
            usage = (resp.usage.prompt_tokens, resp.usage.completion_tokens)
            time.sleep(1.0)  # post-call hook settles asynchronously in the proxy
            rows = _ledger(cfg, "workflow_id = %s", (op_id,))
            assert len(rows) == 1, f"one gateway reservation, no SDK double count: {len(rows)}"
            assert rows[0]["state"] == "settled", rows[0]["state"]
            expected = _price(model, *usage)
            assert abs(int(rows[0]["actual_usd_micros"] or 0) - expected * 1_000_000) <= 1, (rows[0], expected)
            cell.update(usage=usage, expected_cost_usd=expected, settled_usd_micros=rows[0]["actual_usd_micros"], reservations=1)

            deny_op = f"wf-lite-gw-deny-{uuid.uuid4().hex[:8]}"
            _create_limit(cfg, dimension="workflow", dimension_id=deny_op, limit_usd=0.000001)
            raw = OpenAI(base_url=base, api_key=master, max_retries=0)
            with pytest.raises(APIStatusError) as denied:
                raw.chat.completions.create(**{**kwargs, "extra_body": {"metadata": {"kazenai_operation_id": deny_op, "run_id": deny_op}}})
            assert _ledger(cfg, "workflow_id = %s", (deny_op,)) == []
            cell.update(denied_before_dispatch=True, deny_http_status=denied.value.status_code)
    finally:
        _stop(proc)


def test_provider_rejection_releases_finops_hold(monkeypatch):
    """A priced model the account can't use (404) settles at $0 instead of holding budget."""
    import anthropic

    from kazenai import monitor

    cfg = _finops_cfg()
    _finops_mode(monkeypatch, cfg)
    akey = _key("ANTHROPIC_API_KEY")
    model = os.getenv("KAZENAI_LIVE_UNAVAILABLE_MODEL", "claude-sonnet-4-5")
    workflow_id = f"wf-reject-{uuid.uuid4().hex[:8]}"
    _create_limit(cfg, dimension="workflow", dimension_id=workflow_id, limit_usd=1.0)
    with _Cell("finops.provider_rejection_releases_hold", model=model) as cell:
        client = monitor(
            anthropic.Anthropic(api_key=akey, max_retries=0),
            org_id=cfg["org"], workspace_id=cfg["ws"], workflow_id=workflow_id, max_budget_usd=1.0,
        )
        try:
            client.messages.create(model=model, max_tokens=64, messages=[{"role": "user", "content": "hi"}])
        except anthropic.NotFoundError as exc:
            cell["provider_status"] = exc.status_code
        else:
            pytest.skip(f"{model} is available on this account; pick an unavailable priced model")
        rows = _ledger(cfg, "workflow_id = %s", (workflow_id,))
        assert len(rows) == 1
        assert rows[0]["state"] == "settled", rows[0]["state"]
        assert int(rows[0]["actual_usd_micros"] or 0) == 0
        assert client._kazenai_enforcement.snapshot()["pending_projected_usd"] == 0.0
        cell.update(state=rows[0]["state"], reserved_usd_micros=rows[0]["reserved_usd_micros"], settled_usd_micros=0)
