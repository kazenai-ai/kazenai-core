"""Tests for access_list context fencing (Loop 24 P1-7)."""

from __future__ import annotations

from kazenai.orchestration.context_fence import build_worker_context, normalize_access_list


def test_normalize_access_list_all_string():
    assert normalize_access_list("all") == "all"


def test_build_worker_context_empty_access_list():
    prior = [{"subtask": "step0"}, {"subtask": "step1"}]
    out = build_worker_context([], prior, messages=[{"role": "user", "content": "base"}])
    assert out == [{"role": "user", "content": "base"}]


def test_build_worker_context_index_fence():
    prior = [{"subtask": "step0"}, {"subtask": "step1"}]
    out = build_worker_context([0], prior, messages=[{"role": "user", "content": "base"}])
    assert len(out) == 2
    assert out[1]["content"] == "step0"


def test_build_worker_context_all():
    prior = [{"subtask": "step0"}, {"subtask": "step1"}]
    out = build_worker_context("all", prior, messages=[{"role": "user", "content": "base"}])
    assert len(out) == 3
