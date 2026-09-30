"""FINAL_1_b — ReservationHandle carries decision_id + attribution through Core."""

from __future__ import annotations

import json
from unittest.mock import patch

from kazenai.spine.guard import ReservationHandle, reserve_budget


def test_lifecycle_reserve_preserves_decision_id(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops.test")
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "k")

    payload = {
        "reservation_id": "rsv_1",
        "call_id": "call_1",
        "attempt": 1,
        "reserved_usd_micros": 10_000,
        "decision_id": "dec_xyz",
        "decision_receipt": {
            "decision_id": "dec_xyz",
            "attribution": {
                "economics": {
                    "business_subject_ref": "cust-9",
                    "feature_id": "assist",
                    "workflow_id": "wf-9",
                }
            },
        },
    }

    class _Resp:
        def read(self):
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("kazenai.spine.guard.urllib.request.urlopen", return_value=_Resp()):
        handle = reserve_budget(org_id="o", workspace_id="w", run_id="r")
    assert isinstance(handle, ReservationHandle)
    assert handle.decision_id == "dec_xyz"
    assert handle.reservation_id == "rsv_1"
    assert handle.business_subject_ref == "cust-9"
    assert handle.feature_id == "assist"
    assert handle.workflow_id == "wf-9"
