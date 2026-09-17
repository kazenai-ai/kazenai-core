"""Tests for pre-call FinOps budget reservation wiring in monitor.py."""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from io import BytesIO
from unittest.mock import MagicMock, call, patch

from kazenai.enforcement import BudgetExceeded, BudgetUnavailable
from kazenai.monitor import _try_reconcile_budget, _try_reserve_budget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_http_response(status: int, body: dict) -> MagicMock:
    """Build a mock urllib response context manager."""
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    resp.read.return_value = json.dumps(body).encode("utf-8")
    resp.status = status
    return resp


def _make_http_error(code: int, body: dict) -> urllib.error.HTTPError:
    err = urllib.error.HTTPError(
        url="http://localhost:8090/v1/budget/check",
        code=code,
        msg="error",
        hdrs={},  # type: ignore[arg-type]
        fp=BytesIO(json.dumps(body).encode("utf-8")),
    )
    return err


# ---------------------------------------------------------------------------
# test_reserve_called_before_llm_call
# ---------------------------------------------------------------------------

class TestReserveCalledBeforeLLMCall(unittest.TestCase):
    """Verify that _try_reserve_budget fires before the LLM API is invoked."""

    def test_reserve_called_before_llm_call(self):
        call_order = []

        mock_resp = _make_http_response(200, {"estimated_cost_usd": 0.0, "allowed": True})

        def fake_urlopen(req, timeout=None):
            call_order.append("reserve")
            return mock_resp

        def fake_llm_call(*args, **kwargs):
            call_order.append("llm")
            mock_resp_obj = MagicMock()
            mock_resp_obj.usage = None
            return mock_resp_obj

        with patch.dict("os.environ", {"KAZENAI_FINOPS_URL": "http://localhost:8090"}):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                # Simulate the sequence monitor.py follows:
                reserved = _try_reserve_budget(
                    org_id="test-org", workspace_id="default", run_id="run-001"
                )
                fake_llm_call()  # LLM call happens AFTER reserve

        self.assertEqual(call_order, ["reserve", "llm"])
        self.assertIsNotNone(reserved)

    def test_reserve_timeout_is_configurable(self):
        captured_timeouts = []
        mock_resp = _make_http_response(200, {"estimated_cost_usd": 0.0, "allowed": True})

        def fake_urlopen(req, timeout=None):
            captured_timeouts.append(timeout)
            return mock_resp

        with patch.dict(
            "os.environ",
            {
                "KAZENAI_FINOPS_URL": "http://localhost:8090",
                "KAZENAI_FINOPS_RESERVE_TIMEOUT_S": "2.5",
            },
        ):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                _try_reserve_budget(
                    org_id="test-org", workspace_id="default", run_id="run-timeout"
                )

        self.assertEqual(captured_timeouts, [2.5])


# ---------------------------------------------------------------------------
# test_budget_exceeded_raises_before_api_call
# ---------------------------------------------------------------------------

class TestBudgetExceededRaisesBeforeApiCall(unittest.TestCase):
    """When FinOps returns 402, BudgetExceeded is raised before LLM is called."""

    def test_budget_exceeded_raises_before_api_call(self):
        llm_called = []

        def fake_urlopen(req, timeout=None):
            raise _make_http_error(402, {"reason": "daily budget exhausted", "allowed": False})

        def fake_llm():
            llm_called.append(True)

        with patch.dict("os.environ", {"KAZENAI_FINOPS_URL": "http://localhost:8090"}):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                with self.assertRaises(BudgetExceeded) as ctx:
                    _try_reserve_budget(
                        org_id="test-org", workspace_id="default", run_id="run-002"
                    )
                    fake_llm()  # must NOT be reached

        self.assertIn("daily budget exhausted", str(ctx.exception))
        self.assertEqual(llm_called, [], "LLM must not be called when budget is exceeded")


# ---------------------------------------------------------------------------
# test_finops_timeout_fails_open
# ---------------------------------------------------------------------------

class TestFinopsTimeoutFailsOpen(unittest.TestCase):
    """When FinOps service is slow / unreachable, agent continues (fail-open)."""

    def test_finops_timeout_fails_open(self):
        def slow_urlopen(req, timeout=None):
            raise TimeoutError("connection timed out")

        # Fail-open on a FinOps timeout is now a dev-only resilience path: it reserves $0
        # (not None) instead of raising. Production defaults fail closed.
        dev_env = {
            "KAZENAI_FINOPS_URL": "http://localhost:8090",
            "KAZENAI_ENV": "dev",
            "KAZENAI_ENFORCEMENT_MODE": "fail_open",
            "KAZENAI_FINOPS_RESERVATION_MODE": "fail_open",
        }
        with patch.dict("os.environ", dev_env, clear=True):
            with patch("urllib.request.urlopen", side_effect=slow_urlopen):
                result = _try_reserve_budget(
                    org_id="test-org", workspace_id="default", run_id="run-003"
                )
        self.assertEqual(result, 0.0, "dev/fail-open timeout reserves $0 rather than raising")

        # Production default: the same timeout must fail closed (raise BudgetUnavailable).
        prod_env = {"KAZENAI_FINOPS_URL": "http://localhost:8090", "KAZENAI_ENV": "production"}
        with patch.dict("os.environ", prod_env, clear=True):
            with patch("urllib.request.urlopen", side_effect=slow_urlopen):
                with self.assertRaises(BudgetUnavailable):
                    _try_reserve_budget(
                        org_id="test-org", workspace_id="default", run_id="run-003"
                    )


# ---------------------------------------------------------------------------
# test_reconcile_fires_after_call
# ---------------------------------------------------------------------------

class TestReconcileFiresAfterCall(unittest.TestCase):
    """_try_reconcile_budget fires a POST to /v1/budget/reconcile after the call."""

    def test_reconcile_fires_after_call(self):
        reconcile_calls = []

        mock_resp = _make_http_response(200, {"status": "ok"})

        def fake_urlopen(req, timeout=None):
            if "reconcile" in req.full_url:
                reconcile_calls.append(json.loads(req.data.decode()))
            return mock_resp

        with patch.dict("os.environ", {"KAZENAI_FINOPS_URL": "http://localhost:8090"}):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                _try_reconcile_budget(
                    org_id="test-org",
                    workspace_id="default",
                    run_id="run-004",
                    reserved_cost_usd=0.0,
                    actual_cost_usd=0.004,
                )
                # Give the daemon thread time to fire
                time.sleep(0.1)

        self.assertEqual(len(reconcile_calls), 1)
        payload = reconcile_calls[0]
        self.assertEqual(payload["org_id"], "test-org")
        self.assertAlmostEqual(payload["actual_cost_usd"], 0.004, places=5)

    def test_no_finops_url_skips_reserve(self):
        """No FinOps URL: dev/fail-open skips the reserve ($0, no HTTP); prod fails closed."""
        urlopen_called = []

        dev_env = {
            "KAZENAI_ENV": "dev",
            "KAZENAI_ENFORCEMENT_MODE": "fail_open",
            "KAZENAI_FINOPS_RESERVATION_MODE": "fail_open",
        }
        with patch.dict("os.environ", dev_env, clear=True):
            with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: urlopen_called.append(1)):
                result = _try_reserve_budget(
                    org_id="test-org", workspace_id="default", run_id="run-005"
                )

        self.assertEqual(result, 0.0)
        self.assertEqual(urlopen_called, [])

        # Production default: a missing FinOps URL must fail closed.
        with patch.dict("os.environ", {"KAZENAI_ENV": "production"}, clear=True):
            with self.assertRaises(BudgetUnavailable):
                _try_reserve_budget(
                    org_id="test-org", workspace_id="default", run_id="run-005"
                )


class TestWorkspaceIdScoping(unittest.TestCase):
    """FinOps reserve must use workspace_id, not project_id."""

    def test_reserve_payload_uses_explicit_workspace_id(self):
        captured: list[dict] = []

        mock_resp = _make_http_response(200, {"estimated_cost_usd": 0.0, "allowed": True})

        def fake_urlopen(req, timeout=None):
            captured.append(json.loads(req.data.decode("utf-8")))
            return mock_resp

        with patch.dict("os.environ", {"KAZENAI_FINOPS_URL": "http://localhost:8090"}):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                _try_reserve_budget(
                    org_id="test-org",
                    workspace_id="billing-ws-99",
                    run_id="run-ws-001",
                )

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["workspace_id"], "billing-ws-99")
        self.assertNotEqual(captured[0]["workspace_id"], "repo-project-alpha")


class TestActAsOrgHeader(unittest.TestCase):
    """Service-credential callers must assert the tenant org via X-Kazen-Act-As-Org.

    FinOps org binding (Loop 25 H4) rejects cross-org budget calls from a service key
    without this header (403), which makes enforcement fail closed. The spine guard
    must send it on both reserve and reconcile.
    """

    def test_reserve_sends_act_as_org_header(self):
        captured: list[urllib.request.Request] = []
        mock_resp = _make_http_response(200, {"estimated_cost_usd": 0.0, "allowed": True})

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            return mock_resp

        with patch.dict(
            "os.environ",
            {"KAZENAI_FINOPS_URL": "http://localhost:8090", "KAZENAI_FINOPS_API_KEY": "svc-key"},
        ):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                _try_reserve_budget(org_id="tenant-org", workspace_id="default", run_id="run-actas")

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].headers.get("X-kazen-act-as-org"), "tenant-org")
        self.assertEqual(captured[0].headers.get("X-api-key"), "svc-key")

    def test_reconcile_sends_act_as_org_header(self):
        captured: list[urllib.request.Request] = []
        mock_resp = _make_http_response(200, {})

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            return mock_resp

        with patch.dict(
            "os.environ",
            {"KAZENAI_FINOPS_URL": "http://localhost:8090", "KAZENAI_FINOPS_API_KEY": "svc-key"},
        ):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                _try_reconcile_budget(
                    org_id="tenant-org",
                    workspace_id="default",
                    run_id="run-actas",
                    reserved_cost_usd=0.01,
                    actual_cost_usd=0.004,
                )

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].headers.get("X-kazen-act-as-org"), "tenant-org")


if __name__ == "__main__":
    unittest.main()
