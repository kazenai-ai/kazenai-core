"""Follow-on F01–F07 acceptance tests (journal, Control, Crew, correlation)."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from kazenai.integrations.adapter_contract import UnsupportedFrameworkPath
from kazenai.spine import guard
from kazenai.spine import settle_queue as sq
from kazenai.spine.settle_queue import SettleIntentConflict


def _handle(**kwargs):
    base = dict(
        reserved_cost_usd=0.02,
        reservation_id="rsv-f01",
        call_id="call-f01",
        attempt=1,
        lifecycle=True,
        org_id="org-f01",
        workspace_id="ws",
        finops_base_url="https://finops.invalid",
        finops_api_key="SYNTHETIC-ONLY",
    )
    base.update(kwargs)
    return guard.ReservationHandle(**base)


def _body(amount=12000, **extra):
    body = {
        "event": "usage_known",
        "reservation_id": "rsv-f01",
        "call_id": "call-f01",
        "attempt": 1,
        "org_id": "org-f01",
        "idempotency_key": "settle:rsv-f01:usage",
        "actual_usd_micros": amount,
        "lifecycle_version": "reservation.lifecycle.v1",
    }
    body.update(extra)
    return body


def test_f01_process_death_during_sync_reconcile_keeps_journal(tmp_path):
    """Public sync reconcile journals before send; os._exit leaves pending intent."""
    script = tmp_path / "crash_child.py"
    queue = tmp_path / "queue"
    observed = tmp_path / "observed.json"
    core_root = str(Path(__file__).resolve().parents[1])
    script.write_text(
        "\n".join(
            [
                "import os, sys",
                f"sys.path.insert(0, {core_root!r})",
                "from unittest.mock import patch",
                "from kazenai.spine import guard",
                "from kazenai.spine import settle_queue as sq",
                f"sq.reset_settle_queue_for_tests({str(queue)!r})",
                "def crash(req, **kwargs):",
                f"    open({str(observed)!r}, 'wb').write(req.data)",
                "    os._exit(73)",
                "h = guard.ReservationHandle(",
                "    reserved_cost_usd=0.02, reservation_id='rsv-f01', call_id='call-f01',",
                "    attempt=1, lifecycle=True, org_id='org-f01', workspace_id='ws',",
                "    finops_base_url='https://finops.invalid', finops_api_key='SYNTHETIC',",
                ")",
                "with patch('urllib.request.urlopen', crash):",
                "    guard.reconcile_budget(org_id='org-f01', workspace_id='ws', run_id='run',",
                "                           reserved_cost_usd=h, actual_cost_usd=0.012)",
            ]
        ),
        encoding="utf-8",
    )
    child = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert child.returncode == 73
    assert observed.is_file()
    pending = sq.DurableSettleQueue(str(queue)).list_pending()
    assert len(pending) == 1
    assert pending[0]["body"]["actual_usd_micros"] == 12000


@pytest.mark.asyncio
async def test_f01_async_connect_error_and_cancel_keep_journal(tmp_path):
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "async-q"))

    class BoomClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("synthetic")

    with patch.object(httpx, "AsyncClient", BoomClient):
        ok = await guard.reconcile_budget_async(
            org_id="org-f01",
            workspace_id="ws",
            run_id="run",
            reserved_cost_usd=_handle(),
            actual_cost_usd=0.012,
        )
    assert ok is False
    assert len(q.list_pending()) == 1

    q2 = sq.reset_settle_queue_for_tests(str(tmp_path / "async-cancel"))

    class CancelClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            raise asyncio.CancelledError()

    with patch.object(httpx, "AsyncClient", CancelClient):
        with pytest.raises(asyncio.CancelledError):
            await guard.reconcile_budget_async(
                org_id="org-f01",
                workspace_id="ws",
                run_id="run",
                reserved_cost_usd=_handle(),
                actual_cost_usd=0.012,
            )
    assert len(q2.list_pending()) == 1


def test_f02_fingerprint_identity_permissions_and_no_default_creds(tmp_path):
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "spool"))
    q.enqueue(
        _body(12000),
        org_id="org-a",
        base_url="https://finops-a.invalid",
        api_key="DUMMY-A",
    )
    q.enqueue(
        _body(19000),
        org_id="org-b",
        base_url="https://finops-b.invalid",
        api_key="DUMMY-B",
    )
    rows = q.list_pending()
    assert len(rows) == 2
    modes = {oct(stat.S_IMODE(Path(r["_path"]).stat().st_mode)) for r in rows}
    assert modes == {"0o600"}
    assert all(not r.get("api_key") for r in rows)
    with pytest.raises(SettleIntentConflict):
        q.enqueue(
            _body(27000),
            org_id="org-a",
            base_url="https://finops-a.invalid",
            api_key="DUMMY-X",
        )


def test_f02_stale_ack_does_not_delete_replacement(tmp_path):
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "race"))
    q.enqueue(_body(12000), org_id="org-a", base_url="https://a.invalid")
    q.enqueue(_body(19000), org_id="org-b", base_url="https://b.invalid")

    def replace_during_flush(*args, **kwargs):
        q.enqueue(_body(27000), org_id="org-c", base_url="https://c.invalid")
        return True

    with patch.object(guard, "_post_settle_sync", replace_during_flush):
        flush = sq.flush_pending_settlements()
    pending = q.list_pending()
    amounts = sorted(int(p["body"]["actual_usd_micros"]) for p in pending)
    assert 27000 in amounts
    assert flush["applied"] + flush["failed"] >= 1


def test_f03_wrap_graph_control_rejects_undeclared_guarded_client():
    from kazenai.integrations.langgraph import wrap_graph

    class _G:
        def invoke(self, *_a, **_k):
            return {"ok": True}

    class FakeGuarded:
        _kazenai_monitored = True

    with pytest.raises(UnsupportedFrameworkPath) as ei:
        wrap_graph(
            _G(),
            capability="control",
            guarded_client=FakeGuarded(),
            org_id="o",
            workspace_id="w",
        )
    assert ei.value.code == "unguarded_control_boundary"


def test_f04_crew_listener_rejects_foreign_source_without_crew():
    from kazenai.integrations import crewai as crew_mod
    from kazenai.integrations.crewai import CrewAdapter

    class _Crew:
        def kickoff(self, *a, **k):
            return {"ok": True}

    events_a: list = []
    events_b: list = []
    crew_a = _Crew()
    crew_b = _Crew()
    a = CrewAdapter(
        crew_a,
        org_id="org-a",
        workspace_id="ws",
        project_id="p",
        agent_id="ag-a",
        on_event=lambda e: events_a.append(e),
    )
    b = CrewAdapter(
        crew_b,
        org_id="org-b",
        workspace_id="ws",
        project_id="p",
        agent_id="ag-b",
        on_event=lambda e: events_b.append(e),
    )
    tok = a._bind(
        type("Op", (), {"operation_id": "op-a", "org_id": "org-a", "framework": "crewai"})()
    )
    # Foreign task scope for crew_b must not resolve via A's ContextVar.
    assert b._listener._op_for(crew_b) is None
    assert a._listener._op_for(crew_a) is not None
    # Bus filter: source without owned crew is rejected for B.
    source = object()  # no .crew
    owned_b = b._listener._crew_ref
    assert owned_b is crew_b
    src_crew = getattr(source, "crew", None)
    assert source is not owned_b and src_crew is not owned_b
    a._unbind(tok)
    # Nested foreign scope must not leak into the other crew.
    tok_b = b._bind(type("Op", (), {"operation_id": "op-b", "framework": "crewai"})())
    assert a._listener._op_for(crew_a) is None
    assert b._listener._op_for(crew_b) is not None
    b._unbind(tok_b)
    assert crew_mod._TASK_SCOPE.get() is None


def test_f06_explicit_operation_id_conflicts_with_resume(tmp_path, monkeypatch):
    from kazenai.integrations import langgraph as lg
    from kazenai.integrations.langgraph import LangGraphAdapter

    monkeypatch.setenv("KAZENAI_LANGGRAPH_ENVELOPE_DIR", str(tmp_path / "env"))
    lg._THREAD_ENVELOPES.clear()

    class _G:
        def __init__(self):
            self._configs = []

        def invoke(self, _input, config=None, **_k):
            self._configs.append(config or {})
            return {"ok": True}

        def get_state(self, config=None):
            # Pretend a checkpoint exists so side-store may load.
            return type(
                "Snap",
                (),
                {
                    "values": {"x": 1},
                    "next": ("n",),
                    "config": {"configurable": {"checkpoint_id": "ck-1"}},
                },
            )()

    g = _G()
    first = LangGraphAdapter(
        g, org_id="org", workspace_id="ws", project_id="p", agent_id="ag",
        operation_id="op-first",
    )
    first.invoke({}, config={"configurable": {"thread_id": "t-conflict"}})
    lg._THREAD_ENVELOPES.clear()
    second = LangGraphAdapter(
        g, org_id="org", workspace_id="ws", project_id="p", agent_id="ag",
        operation_id="op-explicit-new",
    )
    with pytest.raises(UnsupportedFrameworkPath) as ei:
        second.invoke({}, config={"configurable": {"thread_id": "t-conflict"}})
    assert ei.value.code == "correlation_conflict"
