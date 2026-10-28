"""Optional paid OpenAI Responses API smokes — never run in default CI.

Covers ``responses.parse`` / ``responses.create`` (non-stream and ``stream=True``)
and the ``responses.stream()`` helper through ``monitor()``, sync and async.
Each cell checks that:

1. the live provider answers through the monitored client,
2. KazenAI records a cost equal to the pricing table applied to the
   provider-reported ``input_tokens`` / ``output_tokens``, and
3. a budget too small for the call raises ``BudgetExceeded`` before dispatch
   (proved with an invalid key: had the call reached the provider it would fail
   authentication instead). A model with a long-context price band uses the real
   key for this check, because pricing it first calls OpenAI's free input-token
   counter; a generated answer would still fail the check.

A refusal cell checks that ``background=True`` is refused before dispatch.

Enable explicitly (keys stay outside the repo):

```bash
KAZENAI_LIVE_PROVIDER_SMOKE=1 KAZENAI_LIVE_OPENAI_MODEL=gpt-6-luna \\
  pytest -m live tests/test_live_responses_smoke.py -q
```

``KAZENAI_LIVE_RESPONSES_CELLS`` (comma-separated cell names) limits the run.
Evidence (model, SDK versions, latency, token usage, recorded cost and the run
total; no keys, prompts or outputs) is written to
``artifacts/live-responses-smoke-<ts>.json`` or ``KAZENAI_LIVE_EVIDENCE_DIR``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.live

_INPUT = "Name one primary colour. Answer with the colour only."
# Reasoning models count hidden reasoning in max_output_tokens; leave room for the answer.
_MAX_OUTPUT_TOKENS = 2048
_BUDGET_USD = 0.25
_RESULTS: List[Dict[str, Any]] = []


class Colour(BaseModel):
    colour: str


def _live_enabled() -> bool:
    return os.getenv("KAZENAI_LIVE_PROVIDER_SMOKE", "").strip() in {"1", "true", "yes", "on"}


def _model() -> str:
    return os.getenv("KAZENAI_LIVE_OPENAI_MODEL", "gpt-6-luna")


def _selected(cell: str) -> bool:
    raw = os.getenv("KAZENAI_LIVE_RESPONSES_CELLS", "").strip()
    return not raw or cell in {c.strip() for c in raw.split(",")}


@pytest.fixture(autouse=True)
def _require_live_gate(monkeypatch):
    if not _live_enabled():
        pytest.skip("Set KAZENAI_LIVE_PROVIDER_SMOKE=1 to run paid provider smokes")
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL"):
        monkeypatch.delenv(key, raising=False)


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not-installed"


@pytest.fixture(scope="module", autouse=True)
def _write_evidence():
    yield
    if not _RESULTS:
        return
    out_dir = os.getenv("KAZENAI_LIVE_EVIDENCE_DIR", "").strip()
    target = Path(out_dir) if out_dir else Path(__file__).resolve().parents[1] / "artifacts"
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    totals: Dict[str, float] = {}
    for row in _RESULTS:
        totals[row["model"]] = round(totals.get(row["model"], 0.0) + float(row.get("recorded_cost_usd") or 0.0), 8)
    payload = {
        "evidence": "live_responses_smoke",
        "recorded_at_utc": stamp,
        "kazenai_version": _version("kazenai"),
        "sdk_versions": {"openai": _version("openai")},
        "cells": sorted(_RESULTS, key=lambda r: r["cell"]),
        "total_recorded_cost_usd": totals,
        "all_passed": all(r.get("passed") for r in _RESULTS),
    }
    (target / f"live-responses-smoke-{stamp}.json").write_text(json.dumps(payload, indent=2) + "\n")


def _key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


def _monitored(client: Any, *, budget: float = _BUDGET_USD) -> Any:
    from kazenai import monitor

    return monitor(
        client,
        agent_id="live-responses-smoke",
        max_budget_usd=budget,
        soft_pause_pct=100.0,
        loop_anomaly_threshold=1.0,
        business_subject_ref="smoke:openai-responses",
        feature_id="live_responses_smoke",
        workflow_id="paid_smoke.v1",
    )


def _recorded_cost(client: Any) -> float:
    return float(client._kazenai_enforcement.snapshot()["cumulative_cost_usd"])


def _expected_cost(model: str, usage: Tuple[int, int]) -> float:
    from kazenai.model_pricing import model_cost_usd_per_million

    cost = model_cost_usd_per_million(model, input_tokens=usage[0], output_tokens=usage[1])
    assert cost is not None, f"{model} is not in the pricing table"
    return cost


def _resp_usage(resp: Any) -> Tuple[int, int]:
    return int(resp.usage.input_tokens), int(resp.usage.output_tokens)


def _stream_usage(stream: Any) -> Tuple[int, int]:
    usage = stream._attempt.observed_usage
    assert usage.authoritative and usage.input_tokens is not None and usage.output_tokens is not None
    return int(usage.input_tokens), int(usage.output_tokens)


def _kwargs(model: str) -> Dict[str, Any]:
    return {"model": model, "input": _INPUT, "max_output_tokens": _MAX_OUTPUT_TOKENS}


# --- calls: each returns (client, text, (input_tokens, output_tokens), extra) -----------


def _parse(use_async: bool) -> Tuple[Any, str, Tuple[int, int], Dict[str, Any]]:
    import openai

    kwargs = dict(_kwargs(_model()), text_format=Colour)
    if not use_async:
        client = _monitored(openai.OpenAI(api_key=_key(), max_retries=0))
        resp = client.responses.parse(**kwargs)
    else:
        client = _monitored(openai.AsyncOpenAI(api_key=_key(), max_retries=0))
        resp = asyncio.run(client.responses.parse(**kwargs))
    parsed = resp.output_parsed
    assert isinstance(parsed, Colour) and parsed.colour.strip(), f"no parsed output (status={resp.status})"
    return client, parsed.colour, _resp_usage(resp), {"status": resp.status}


def _create(use_async: bool) -> Tuple[Any, str, Tuple[int, int], Dict[str, Any]]:
    import openai

    kwargs = _kwargs(_model())
    if not use_async:
        client = _monitored(openai.OpenAI(api_key=_key(), max_retries=0))
        resp = client.responses.create(**kwargs)
    else:
        client = _monitored(openai.AsyncOpenAI(api_key=_key(), max_retries=0))
        resp = asyncio.run(client.responses.create(**kwargs))
    return client, resp.output_text, _resp_usage(resp), {"status": resp.status}


def _create_stream(use_async: bool) -> Tuple[Any, str, Tuple[int, int], Dict[str, Any]]:
    import openai

    kwargs = dict(_kwargs(_model()), stream=True)
    if not use_async:
        client = _monitored(openai.OpenAI(api_key=_key(), max_retries=0))
        stream = client.responses.create(**kwargs)
        text = "".join(e.delta for e in stream if e.type == "response.output_text.delta")
    else:
        client = _monitored(openai.AsyncOpenAI(api_key=_key(), max_retries=0))

        async def _go():
            s = await client.responses.create(**kwargs)
            parts = [e.delta async for e in s if e.type == "response.output_text.delta"]
            return s, "".join(parts)

        stream, text = asyncio.run(_go())
    attempt = stream._attempt
    assert attempt.cost_confidence == "exact", attempt.cost_confidence
    return client, text, _stream_usage(stream), {"outcome": str(attempt.terminal_outcome)}


def _stream_helper(use_async: bool) -> Tuple[Any, str, Tuple[int, int], Dict[str, Any]]:
    import openai

    kwargs = _kwargs(_model())
    if not use_async:
        client = _monitored(openai.OpenAI(api_key=_key(), max_retries=0))
        with client.responses.stream(**kwargs) as stream:
            final = stream.get_final_response()
    else:
        client = _monitored(openai.AsyncOpenAI(api_key=_key(), max_retries=0))

        async def _go():
            async with client.responses.stream(**kwargs) as s:
                return s, await s.get_final_response()

        stream, final = asyncio.run(_go())
    assert stream._attempt.cost_confidence == "exact", stream._attempt.cost_confidence
    return client, final.output_text, _resp_usage(final), {"status": final.status}


_CALLS: Dict[str, Callable[[bool], Tuple[Any, str, Tuple[int, int], Dict[str, Any]]]] = {
    "parse": _parse,
    "create": _create,
    "create_stream": _create_stream,
    "stream_helper": _stream_helper,
}


def _denied_before_dispatch(kind: str, use_async: bool) -> bool:
    import openai

    from kazenai import BudgetExceeded

    kwargs = _kwargs(_model())
    if kind == "parse":
        kwargs["text_format"] = Colour
    if kind == "create_stream":
        kwargs["stream"] = True
    method = "stream" if kind == "stream_helper" else ("parse" if kind == "parse" else "create")
    # A model with a long-context price band is priced from OpenAI's exact input count,
    # so the SDK calls responses.input_tokens.count first. That call is free and generates
    # nothing, but with an invalid key it fails (input_counter_failed) before the budget
    # check. Use the real key there: BudgetExceeded must still come before any generation,
    # and a generated answer would fail pytest.raises.
    from kazenai.price_catalog import model_declares_long_band

    key = _key() if model_declares_long_band("openai", _model()) else "sk-invalid-never-dispatched"
    if not use_async:
        client = _monitored(openai.OpenAI(api_key=key, max_retries=0), budget=1e-9)
        with pytest.raises(BudgetExceeded):
            result = getattr(client.responses, method)(**kwargs)
            if method == "stream":
                result.__enter__()
        return True
    client = _monitored(openai.AsyncOpenAI(api_key=key, max_retries=0), budget=1e-9)

    async def _go():
        with pytest.raises(BudgetExceeded):
            result = getattr(client.responses, method)(**kwargs)
            if method == "stream":
                await result.__aenter__()
            else:
                await result

    asyncio.run(_go())
    return True


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", list(_CALLS))
def test_live_responses_cell(kind: str, use_async: bool) -> None:
    cell = f"openai.responses.{kind}.{'async' if use_async else 'sync'}"
    if not _selected(cell):
        pytest.skip(f"{cell} not selected")
    model = _model()
    started = time.perf_counter()
    row: Dict[str, Any] = {"cell": cell, "model": model, "passed": False}
    try:
        client, text, usage, extra = _CALLS[kind](use_async)
        recorded = _recorded_cost(client)
        expected = _expected_cost(model, usage)
        row.update(
            input_tokens=usage[0],
            output_tokens=usage[1],
            recorded_cost_usd=recorded,
            expected_cost_usd=expected,
            output_nonempty=bool(text.strip()),
            **extra,
        )
        assert text.strip(), f"{cell}: empty output"
        assert recorded > 0
        assert recorded == pytest.approx(expected, rel=1e-6, abs=1e-9)
        row["denied_before_dispatch"] = _denied_before_dispatch(kind, use_async)
        row["passed"] = True
    finally:
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        _RESULTS.append(row)


def test_live_background_mode_refused_before_dispatch() -> None:
    cell = "openai.responses.background_refused.sync"
    if not _selected(cell):
        pytest.skip(f"{cell} not selected")
    import openai

    from kazenai.enforcement import UnsupportedModeError

    started = time.perf_counter()
    # A real key: had the request been sent, it would have queued a billed background response.
    client = _monitored(openai.OpenAI(api_key=_key(), max_retries=0))
    with pytest.raises(UnsupportedModeError) as info:
        client.responses.create(background=True, **_kwargs(_model()))
    assert info.value.error_code == "responses_background_unsupported"
    assert _recorded_cost(client) == 0.0
    _RESULTS.append(
        {
            "cell": cell,
            "model": _model(),
            "passed": True,
            "recorded_cost_usd": 0.0,
            "refused_before_dispatch": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    )
