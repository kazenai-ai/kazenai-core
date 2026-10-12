"""FINAL_1_b_final P3 — SDK diagnostics and client/server capability mismatch."""

from __future__ import annotations

import pytest

from kazenai.lifecycle_config import (
    ClientServerLifecycleMismatch,
    assert_compatible_with_server,
    sdk_lifecycle_diagnostics,
)


def test_sdk_lifecycle_diagnostics_control(monkeypatch) -> None:
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    diag = sdk_lifecycle_diagnostics()
    assert diag["component"] == "kazenai-sdk"
    assert diag["effective_lifecycle"]["profile"] == "control"
    assert diag["capabilities"]["reservation_settlement"] is True


def test_assert_compatible_with_matching_server(monkeypatch) -> None:
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    cfg = assert_compatible_with_server(
        {
            "effective_lifecycle": {
                "profile": "control",
                "lifecycle_version": "reservation.lifecycle.v1",
            },
            "capabilities": {"reservation_settlement": True},
        }
    )
    assert cfg.is_control


def test_client_control_server_legacy_mismatches(monkeypatch) -> None:
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    with pytest.raises(ClientServerLifecycleMismatch) as excinfo:
        assert_compatible_with_server(
            {
                "effective_lifecycle": {"profile": "legacy"},
                "capabilities": {"reservation_settlement": False},
            }
        )
    assert excinfo.value.category == "capability_mismatch"
    assert excinfo.value.client["profile"] == "control"
