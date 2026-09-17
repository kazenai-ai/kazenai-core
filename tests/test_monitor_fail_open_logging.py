"""P3-3 — fail-open helper paths log unexpected exceptions."""

from __future__ import annotations

import importlib
import logging

import pytest

from kazenai.enforcement import BudgetExceeded

monitor_mod = importlib.import_module("kazenai.monitor")
_extract_tool_names = monitor_mod._extract_tool_names
_log_fail_open = monitor_mod._log_fail_open


class _BadToolsIterable:
    def __iter__(self):
        raise RuntimeError("injected fault")


def test_extract_tool_names_logs_injected_fault(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(monitor_mod, "structlog", None)
    caplog.set_level(logging.WARNING, logger="kazenai.monitor")
    result = _extract_tool_names({"tools": _BadToolsIterable()})
    assert result is None
    assert any(
        "fail-open" in record.message and "extract_tool_names" in record.message
        for record in caplog.records
    )


def test_log_fail_open_skips_expected_enforcement_blocks(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(monitor_mod, "structlog", None)
    _log_fail_open("budget_check", BudgetExceeded("cap hit"))
    assert not caplog.records
