"""Reservation-bound lifecycle snapshot shape."""

from __future__ import annotations

from kazenai.lifecycle_config import resolve_effective_lifecycle


def test_receipt_block_immutable_shape_under_env_change(monkeypatch) -> None:
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    bound = resolve_effective_lifecycle().to_receipt_block()
    assert bound["profile"] == "control"

    # Env change after admit must not rewrite a stored snapshot object.
    monkeypatch.delenv("KAZENAI_FINOPS_CONTROL_PROFILE", raising=False)
    monkeypatch.setenv("KAZENAI_PROFILE", "legacy")
    live = resolve_effective_lifecycle().to_receipt_block()
    assert live["profile"] == "legacy"
    assert bound["profile"] == "control"
    assert bound["lifecycle_version"] == "reservation.lifecycle.v1"
