"""Tests for kazenai.spine guarded_llm_call."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from kazenai.enforcement import BudgetExceeded, BudgetUnavailable
from kazenai.spine.guard import guarded_llm_call, reconcile_budget, reserve_budget
from kazenai.sinks import MemorySink


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in (
        "KAZENAI_FINOPS_URL",
        "KAZENAI_FINOPS_INGEST_URL",
        "KAZENAI_FINOPS_API_KEY",
        "KAZENAI_ENFORCEMENT_MODE",
        "KAZENAI_FINOPS_RESERVATION_MODE",
        "KAZENAI_FINOPS_RESERVE_TIMEOUT_S",
        "KAZENAI_ENV",
        "KAZENAI_FINOPS_CONTROL_PROFILE",
        "KAZENAI_CONTROL_PROFILE",
        "KAZENAI_PROFILE",
        "KAZENAI_FINOPS_STRICT_BUDGET",
        "KAZENAI_AGENTLENS_MIRROR",
        "KAZENAI_AGENTLENS_INGEST_URL",
        "KAZENAI_AGENTLENS_URL",
        "KAZENAI_LENS_INGEST_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_reserve_raises_budget_unavailable_without_finops_url(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_closed")
    with pytest.raises(BudgetUnavailable, match="not configured"):
        reserve_budget(org_id="o", workspace_id="w", run_id="r")


def test_reserve_fail_open_in_dev_without_url(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    assert reserve_budget(org_id="o", workspace_id="w", run_id="r") == 0.0


def test_reserve_uses_configurable_timeout(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVE_TIMEOUT_S", "2.5")
    captured_timeouts: list[float | None] = []
    reserve_resp = MagicMock()
    reserve_resp.read.return_value = json.dumps({"estimated_cost_usd": 0.01}).encode()
    reserve_resp.__enter__ = lambda s: reserve_resp
    reserve_resp.__exit__ = MagicMock(return_value=False)

    def fake_urlopen(_req, timeout=None):
        captured_timeouts.append(timeout)
        return reserve_resp

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=fake_urlopen):
        assert reserve_budget(org_id="o", workspace_id="w", run_id="r") == pytest.approx(0.01)

    assert captured_timeouts == [2.5]


def test_reserve_sends_explicit_estimated_cost(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    captured_payloads: list[dict] = []
    reserve_resp = MagicMock()
    reserve_resp.read.return_value = json.dumps({"estimated_cost_usd": 0.25}).encode()
    reserve_resp.__enter__ = lambda s: reserve_resp
    reserve_resp.__exit__ = MagicMock(return_value=False)

    def fake_urlopen(req, timeout=None):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        return reserve_resp

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=fake_urlopen):
        assert reserve_budget(
            org_id="o",
            workspace_id="w",
            run_id="r",
            estimated_cost_usd=0.25,
            increment_step=False,
        ) == pytest.approx(0.25)

    assert captured_payloads == [
        {
            "org_id": "o",
            "workspace_id": "w",
            "run_id": "r",
            "reserve": True,
            "increment_step": False,
            "input_tokens": 1000,
            "estimated_cost_usd": 0.25,
            "projected_spend_usd": 0.25,
        }
    ]


def test_reserve_forwards_feature_slug(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    captured_payloads: list[dict] = []
    reserve_resp = MagicMock()
    reserve_resp.read.return_value = json.dumps({"estimated_cost_usd": 0.1}).encode()
    reserve_resp.__enter__ = lambda s: reserve_resp
    reserve_resp.__exit__ = MagicMock(return_value=False)

    def fake_urlopen(req, timeout=None):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        return reserve_resp

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=fake_urlopen):
        reserve_budget(
            org_id="o",
            workspace_id="prod",
            run_id="r",
            feature="support-bot",
            increment_step=False,
        )

    assert captured_payloads[0]["feature"] == "support-bot"


def test_reserve_omits_empty_feature(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    captured_payloads: list[dict] = []
    reserve_resp = MagicMock()
    reserve_resp.read.return_value = json.dumps({"estimated_cost_usd": 0.0}).encode()
    reserve_resp.__enter__ = lambda s: reserve_resp
    reserve_resp.__exit__ = MagicMock(return_value=False)

    def fake_urlopen(req, timeout=None):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        return reserve_resp

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=fake_urlopen):
        reserve_budget(org_id="o", workspace_id="prod", run_id="r", feature="")

    assert "feature" not in captured_payloads[0]


def test_reconcile_forwards_feature_slug(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    captured_payloads: list[dict] = []
    reconcile_resp = MagicMock()
    reconcile_resp.__enter__ = lambda s: reconcile_resp
    reconcile_resp.__exit__ = MagicMock(return_value=False)

    def fake_urlopen(req, timeout=None):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        return reconcile_resp

    class _ImmediateThread:
        def __init__(self, target=None, daemon=True, **kwargs):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    with patch("kazenai.spine.guard.threading.Thread", _ImmediateThread):
        with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=fake_urlopen):
            reconcile_budget(
                org_id="o",
                workspace_id="prod",
                run_id="r",
                reserved_cost_usd=0.1,
                actual_cost_usd=0.08,
                feature="support-bot",
            )

    assert captured_payloads[0]["feature"] == "support-bot"


def test_guarded_llm_call_forwards_feature_to_finops_reserve(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")

    captured_payloads: list[dict] = []
    reserve_resp = MagicMock()
    reserve_resp.read.return_value = json.dumps({"estimated_cost_usd": 0.6}).encode()
    reserve_resp.__enter__ = lambda s: reserve_resp
    reserve_resp.__exit__ = MagicMock(return_value=False)
    reconcile_resp = MagicMock()
    reconcile_resp.__enter__ = lambda s: reconcile_resp
    reconcile_resp.__exit__ = MagicMock(return_value=False)

    def fake_urlopen(req, timeout=None):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        if "reconcile" in req.full_url:
            return reconcile_resp
        return reserve_resp

    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=fake_urlopen):
        with patch("kazenai.spine.guard._extract_usage_cost", return_value=(15, 0.002)):
            guarded_llm_call(
                lambda: resp,
                model="gpt-4o-mini",
                org_id="org",
                workspace_id="prod",
                run_id="run-feature",
                feature="support-bot",
                projected_cost_usd=0.6,
                increment_step=False,
                event_sink=MemorySink(max_events=10),
            )

    assert captured_payloads[0]["feature"] == "support-bot"
    time.sleep(0.1)
    assert len(captured_payloads) >= 2
    assert captured_payloads[1]["feature"] == "support-bot"


def test_guarded_llm_call_feature_cap_denied_before_provider(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    captured_payloads: list[dict] = []
    sink = MemorySink(max_events=10)

    import urllib.error

    def fake_urlopen(req, timeout=None):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        raise urllib.error.HTTPError(
            url=req.full_url,
            code=402,
            msg="Payment Required",
            hdrs={},
            fp=MagicMock(
                read=MagicMock(
                    return_value=json.dumps(
                        {
                            "allowed": False,
                            "level_blocked": "feature",
                            "feature": "support-bot",
                            "reason": "feature budget exhausted",
                        }
                    ).encode()
                )
            ),
        )

    provider_called = []

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(BudgetExceeded, match="feature budget exhausted"):
            guarded_llm_call(
                lambda: provider_called.append(True),
                model="gpt-4o-mini",
                org_id="org",
                workspace_id="prod",
                run_id="run-deny",
                feature="support-bot",
                projected_cost_usd=0.6,
                event_sink=sink,
            )

    assert captured_payloads[0]["feature"] == "support-bot"
    assert provider_called == []
    assert any(e.event_type == "finops.budget.denied" for e in sink.events)


def test_reserve_non_402_http_error_fails_open_in_dev(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_ENV", "dev")

    import urllib.error

    err = urllib.error.HTTPError(
        url="http://finops:8090/v1/budget/check",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=MagicMock(read=MagicMock(return_value=b"{}")),
    )

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=err):
        assert reserve_budget(org_id="o", workspace_id="w", run_id="r") == 0.0


def test_reserve_402_with_unreadable_body_uses_default_reason(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")

    import urllib.error

    bad_fp = MagicMock()
    bad_fp.read.side_effect = RuntimeError("bad body")
    err = urllib.error.HTTPError(
        url="http://finops:8090/v1/budget/check",
        code=402,
        msg="Payment Required",
        hdrs={},
        fp=bad_fp,
    )

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=err):
        with pytest.raises(BudgetExceeded, match="budget exhausted"):
            reserve_budget(org_id="o", workspace_id="w", run_id="r")


def test_reserve_non_402_http_error_fails_closed_by_default(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")

    import urllib.error

    err = urllib.error.HTTPError(
        url="http://finops:8090/v1/budget/check",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=MagicMock(read=MagicMock(return_value=b"{}")),
    )

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=err):
        with pytest.raises(BudgetUnavailable, match="HTTP 503"):
            reserve_budget(org_id="o", workspace_id="w", run_id="r")


def test_reserve_timeout_fails_open_in_dev(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_ENV", "dev")

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=TimeoutError("slow")):
        assert reserve_budget(org_id="o", workspace_id="w", run_id="r") == 0.0


def test_default_sink_wires_finops_and_lens_mirrors(monkeypatch):
    from kazenai.spine import guard as guard_mod
    from kazenai.sinks import HttpSink, MultiSink

    guard_mod.reset_default_sink()
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "finops-key")
    monkeypatch.setenv("KAZENAI_AGENTLENS_MIRROR", "on")
    monkeypatch.setenv("KAZENAI_AGENTLENS_URL", "http://lens:8790/v1/events")
    monkeypatch.setenv("KAZENAI_AGENTLENS_API_KEY", "lens-key")

    sink = guard_mod._default_sink()
    sink2 = guard_mod._default_sink()
    assert sink is sink2

    assert isinstance(sink, MultiSink)
    http_sinks = [s for s in sink._sinks if isinstance(s, HttpSink)]
    assert len(http_sinks) == 2
    assert [s._cfg.ingest_url for s in http_sinks] == ["http://finops:8090", "http://lens:8790"]
    assert [s._cfg.api_key for s in http_sinks] == ["finops-key", "lens-key"]
    guard_mod.reset_default_sink()


def test_guarded_llm_call_computes_cost_and_emits_events(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")

    sink = MemorySink(max_events=50)
    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    with patch("kazenai.spine.guard.urllib.request.urlopen") as mock_urlopen:
        reserve_resp = MagicMock()
        reserve_resp.read.return_value = json.dumps({"estimated_cost_usd": 0.01}).encode()
        reserve_resp.__enter__ = lambda s: reserve_resp
        reserve_resp.__exit__ = MagicMock(return_value=False)
        reconcile_resp = MagicMock()
        reconcile_resp.__enter__ = lambda s: reconcile_resp
        reconcile_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.side_effect = [reserve_resp, reconcile_resp]

        with patch("kazenai.spine.guard._extract_usage_cost", return_value=(150, 0.002)):
            out = guarded_llm_call(
                lambda: resp,
                model="gpt-4o-mini",
                org_id="org",
                workspace_id="ws",
                run_id="run-1",
                agent_id="brain",
                surface="test",
                feature="unit",
                event_sink=sink,
            )

    assert out is resp
    types = [e.event_type for e in sink.events]
    assert "finops.budget.reserve" in types
    assert "model.call" in types
    model_ev = [e for e in sink.events if e.event_type == "model.call"][0]
    assert model_ev.cost_usd == pytest.approx(0.002)


def test_guarded_llm_call_denied_on_402(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    sink = MemorySink(max_events=10)

    import urllib.error

    err = urllib.error.HTTPError(
        url="http://finops:8090/v1/budget/check",
        code=402,
        msg="Payment Required",
        hdrs={},
        fp=MagicMock(read=MagicMock(return_value=json.dumps({"reason": "exhausted"}).encode())),
    )

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=err):
        with pytest.raises(BudgetExceeded):
            guarded_llm_call(
                lambda: "never",
                model="gpt-4o-mini",
                org_id="org",
                workspace_id="ws",
                run_id="run-1",
                event_sink=sink,
            )

    assert any(e.event_type == "finops.budget.denied" for e in sink.events)
