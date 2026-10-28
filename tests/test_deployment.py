"""Tests for saas fail-safe deployment defaults."""

from __future__ import annotations

import os

import pytest

from kazenai.deployment import (
    enforcement_fail_closed,
    finops_reservation_fail_closed,
    is_dev_env,
    is_saas_deployment,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in (
        "KAZENAI_DEPLOYMENT_MODE",
        "KAZENAI_ENV",
        "KAZENAI_FINOPS_RESERVATION_MODE",
        "KAZENAI_ENFORCEMENT_MODE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_is_saas_deployment():
    os.environ["KAZENAI_DEPLOYMENT_MODE"] = "saas"
    assert is_saas_deployment() is True
    os.environ["KAZENAI_DEPLOYMENT_MODE"] = "development"
    assert is_saas_deployment() is False


def test_finops_reservation_fail_closed_in_saas(monkeypatch):
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "staging")
    assert finops_reservation_fail_closed() is True


def test_finops_reservation_fail_open_in_dev(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    assert finops_reservation_fail_closed() is False


def test_enforcement_fail_closed_by_default(monkeypatch):
    assert enforcement_fail_closed() is True


def test_enforcement_fail_open_only_in_dev(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_ENV", "development")
    assert enforcement_fail_closed() is True
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    assert is_dev_env() is True
    assert enforcement_fail_closed() is False
