"""FINAL_1_b P1-3 — SDK attribution ContextVar API."""

from __future__ import annotations

import concurrent.futures
import json

import pytest

from kazenai.attribution import (
    AttributionConflict,
    AttributionContext,
    attribution_for_reserve_body,
    get_attribution,
    use_attribution,
)
from kazenai.enforcement import UnsupportedModeError
from kazenai.monitor import monitor


def test_parallel_contexts_different_subjects():
    results: dict[str, str] = {}

    def worker(subject: str) -> None:
        with use_attribution(business_subject_ref=subject, feature_id="chat"):
            ctx = get_attribution()
            assert ctx is not None
            results[subject] = ctx.business_subject_ref or ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, [f"cust-{i}" for i in range(8)]))

    assert results["cust-0"] == "cust-0"
    assert results["cust-7"] == "cust-7"
    assert get_attribution() is None


def test_nested_same_ok_conflict_raises():
    with use_attribution(business_subject_ref="acct-a", feature_id="search"):
        with use_attribution(business_subject_ref="acct-a", workflow_id="wf-v1"):
            ctx = get_attribution()
            assert ctx is not None
            assert ctx.feature_id == "search"
            assert ctx.workflow_id == "wf-v1"
        with pytest.raises(AttributionConflict):
            with use_attribution(business_subject_ref="acct-b"):
                pass


def test_async_envelope_rejection():
    class Asyncish:
        async def create(self):  # pragma: no cover - never called
            return None

    class FakeAsyncOpenAI:
        def __init__(self) -> None:
            self.chat = type("C", (), {"completions": Asyncish()})()

    with pytest.raises(UnsupportedModeError):
        monitor(FakeAsyncOpenAI())


def test_no_credential_leakage_in_repr_and_dict():
    canary = "sk-CANARY_SECRET_attrib_9f3a"
    ctx = AttributionContext(
        business_subject_ref="cust-opaque",
        feature_id="assist",
        workflow_id="main",
    )
    blob = repr(ctx) + json.dumps(ctx.to_dict())
    assert canary not in blob
    assert "api_key" not in blob
    assert "authorization" not in ctx.to_dict()
    assert ctx.attribution_state() == "attributed"
    assert AttributionContext().attribution_state() == "unattributed"
    assert AttributionContext(feature_id="x").attribution_state() == "unknown"


def test_reserve_body_prefers_context_over_kwargs():
    with use_attribution(business_subject_ref="from-ctx", feature_id="ctx-feat"):
        body = attribution_for_reserve_body(
            business_subject_ref="from-kw",
            feature_id="kw-feat",
            workflow_id="wf-1",
        )
    assert body["business_subject_ref"] == "from-ctx"
    assert body["feature_id"] == "ctx-feat"
    assert body["workflow_id"] == "wf-1"
    assert body["attribution_state"] == "attributed"
