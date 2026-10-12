"""Optional paid OpenAI/Anthropic smokes — never run in default CI.

Covers the headline surfaces through ``monitor()``: OpenAI Chat Completions and
Anthropic Messages, sync and async, non-streaming and streaming (8 cells).
Each cell checks that:

1. the live provider answers through the monitored client,
2. KazenAI records a non-zero cost equal to the pricing table applied to the
   provider-reported input/output usage (streaming and non-streaming), and
3. a budget too small for the call raises ``BudgetExceeded`` before dispatch
   (proved with an invalid key: had the call reached the provider it would
   fail authentication instead).

Enable explicitly (keys stay outside the repo):

```bash
set -a; source ~/.config/kazenai/live-smoke.env; set +a
KAZENAI_LIVE_PROVIDER_SMOKE=1 pytest -m live tests/test_live_provider_smoke.py -q
```

Spend is a few cents at most: a small output limit (``max_completion_tokens``
for OpenAI reasoning families such as gpt-5/gpt-6/o-series, ``max_tokens``
otherwise) and models set by ``KAZENAI_LIVE_OPENAI_MODEL`` /
``KAZENAI_LIVE_ANTHROPIC_MODEL`` (defaults: gpt-4o-mini, claude-haiku-4-5).
Evidence (models, SDK versions, latency, token usage, recorded cost — no keys,
no prompts or outputs) is written to ``artifacts/live-provider-smoke-<ts>.json``
or ``KAZENAI_LIVE_EVIDENCE_DIR``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytestmark = pytest.mark.live

_PROMPT = [{"role": "user", "content": "Reply with the single word: ok"}]
# Room for reasoning/adaptive-thinking models to still emit a visible answer.
_MAX_TOKENS = 256
_BUDGET_USD = 0.05
_RESULTS: List[Dict[str, Any]] = []


def _live_enabled() -> bool:
    return os.getenv("KAZENAI_LIVE_PROVIDER_SMOKE", "").strip() in {"1", "true", "yes", "on"}


def _openai_model() -> str:
    return os.getenv("KAZENAI_LIVE_OPENAI_MODEL", "gpt-4o-mini")


def _anthropic_model() -> str:
    return os.getenv("KAZENAI_LIVE_ANTHROPIC_MODEL", "claude-haiku-4-5")


def _hermetic_local_env(monkeypatch) -> None:
    # Local hard cap only; the shared FinOps authority is not part of this smoke.
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _require_live_gate(monkeypatch):
    if not _live_enabled():
        pytest.skip("Set KAZENAI_LIVE_PROVIDER_SMOKE=1 to run paid provider smokes")
    _hermetic_local_env(monkeypatch)


@pytest.fixture(scope="module", autouse=True)
def _write_evidence():
    yield
    if not _RESULTS:
        return
    out_dir = os.getenv("KAZENAI_LIVE_EVIDENCE_DIR", "").strip()
    target = Path(out_dir) if out_dir else Path(__file__).resolve().parents[1] / "artifacts"
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "evidence": "live_provider_smoke",
        "recorded_at_utc": stamp,
        "kazenai_version": _version("kazenai"),
        "sdk_versions": {"openai": _version("openai"), "anthropic": _version("anthropic")},
        "cells": sorted(_RESULTS, key=lambda r: r["cell"]),
        "all_passed": all(r.get("passed") for r in _RESULTS),
    }
    (target / f"live-provider-smoke-{stamp}.json").write_text(json.dumps(payload, indent=2) + "\n")


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _key(env: str) -> str:
    key = (os.getenv(env) or "").strip()
    if not key:
        pytest.skip(f"{env} not set")
    return key


def _monitored(client: Any, provider: str, *, budget: float = _BUDGET_USD) -> Any:
    from kazenai import monitor

    return monitor(
        client,
        agent_id=f"live-smoke-{provider}",
        max_budget_usd=budget,
        soft_pause_pct=100.0,
        loop_anomaly_threshold=1.0,
        business_subject_ref=f"smoke:{provider}",
        feature_id="live_provider_smoke",
        workflow_id="paid_smoke.v1",
    )


def _recorded_cost(client: Any) -> float:
    return float(client._kazenai_enforcement.snapshot()["cumulative_cost_usd"])


def _expected_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    from kazenai.model_pricing import model_cost_usd_per_million

    cost = model_cost_usd_per_million(model, input_tokens=input_tokens, output_tokens=output_tokens)
    assert cost is not None, f"{model} is not in the pricing table"
    return cost


def _stream_usage(stream: Any) -> Any:
    """(input, output) tokens the provider reported at stream end, as KazenAI observed them."""
    usage = stream._attempt.observed_usage
    inp = usage.prompt_tokens if usage.prompt_tokens is not None else usage.input_tokens
    out = usage.completion_tokens if usage.completion_tokens is not None else usage.output_tokens
    assert usage.authoritative and inp is not None and out is not None, "no final usage on the stream"
    return int(inp), int(out)


def _record(cell: str, model: str, started: float, **fields: Any) -> None:
    _RESULTS.append(
        {"cell": cell, "model": model, "latency_ms": round((time.perf_counter() - started) * 1000, 1), **fields}
    )


# --- OpenAI -----------------------------------------------------------------


def _openai_kwargs(model: str) -> Dict[str, Any]:
    from kazenai.model_pricing import output_limit_param

    return {"model": model, "messages": _PROMPT, output_limit_param(model): _MAX_TOKENS}


def _openai_call(stream: bool, use_async: bool) -> Dict[str, Any]:
    openai = pytest.importorskip("openai")
    key = _key("OPENAI_API_KEY")
    model = _openai_model()
    kwargs = _openai_kwargs(model)
    started = time.perf_counter()

    if not use_async:
        client = _monitored(openai.OpenAI(api_key=key, max_retries=0), "openai")
        if stream:
            stream = client.chat.completions.create(stream=True, **kwargs)
            text = "".join((c.choices[0].delta.content or "") if c.choices else "" for c in stream)
            usage = _stream_usage(stream)
        else:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            usage = (resp.usage.prompt_tokens, resp.usage.completion_tokens)
    else:
        client = _monitored(openai.AsyncOpenAI(api_key=key, max_retries=0), "openai")

        async def _go():
            if stream:
                parts = []
                resp_stream = await client.chat.completions.create(stream=True, **kwargs)
                async for c in resp_stream:
                    if c.choices:
                        parts.append(c.choices[0].delta.content or "")
                return "".join(parts), _stream_usage(resp_stream)
            resp = await client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or "", (resp.usage.prompt_tokens, resp.usage.completion_tokens)

        text, usage = asyncio.run(_go())
    return {"client": client, "model": model, "text": text, "usage": usage, "started": started}


def _openai_denied_before_dispatch(use_async: bool) -> bool:
    openai = pytest.importorskip("openai")
    from kazenai import BudgetExceeded

    kwargs = _openai_kwargs(_openai_model())
    if not use_async:
        client = _monitored(openai.OpenAI(api_key="sk-invalid-never-dispatched", max_retries=0), "openai", budget=1e-9)
        with pytest.raises(BudgetExceeded):
            client.chat.completions.create(**kwargs)
        return True
    client = _monitored(openai.AsyncOpenAI(api_key="sk-invalid-never-dispatched", max_retries=0), "openai", budget=1e-9)

    async def _go():
        with pytest.raises(BudgetExceeded):
            await client.chat.completions.create(**kwargs)

    asyncio.run(_go())
    return True


# --- Anthropic --------------------------------------------------------------


def _anthropic_call(stream: bool, use_async: bool) -> Dict[str, Any]:
    anthropic = pytest.importorskip("anthropic")
    key = _key("ANTHROPIC_API_KEY")
    model = _anthropic_model()
    kwargs = {"model": model, "messages": _PROMPT, "max_tokens": _MAX_TOKENS}
    started = time.perf_counter()

    def _delta_text(event: Any) -> str:
        delta = getattr(event, "delta", None)
        return getattr(delta, "text", "") or ""

    if not use_async:
        client = _monitored(anthropic.Anthropic(api_key=key, max_retries=0), "anthropic")
        if stream:
            stream = client.messages.create(stream=True, **kwargs)
            text = "".join(_delta_text(e) for e in stream)
            usage = _stream_usage(stream)
        else:
            resp = client.messages.create(**kwargs)
            text = "".join(getattr(b, "text", "") for b in resp.content)
            usage = (resp.usage.input_tokens, resp.usage.output_tokens)
    else:
        client = _monitored(anthropic.AsyncAnthropic(api_key=key, max_retries=0), "anthropic")

        async def _go():
            if stream:
                parts = []
                resp_stream = await client.messages.create(stream=True, **kwargs)
                async for e in resp_stream:
                    parts.append(_delta_text(e))
                return "".join(parts), _stream_usage(resp_stream)
            resp = await client.messages.create(**kwargs)
            return "".join(getattr(b, "text", "") for b in resp.content), (
                resp.usage.input_tokens,
                resp.usage.output_tokens,
            )

        text, usage = asyncio.run(_go())
    return {"client": client, "model": model, "text": text, "usage": usage, "started": started}


def _anthropic_denied_before_dispatch(use_async: bool) -> bool:
    anthropic = pytest.importorskip("anthropic")
    from kazenai import BudgetExceeded

    kwargs = {"model": _anthropic_model(), "messages": _PROMPT, "max_tokens": _MAX_TOKENS}
    if not use_async:
        client = _monitored(anthropic.Anthropic(api_key="sk-ant-invalid-never-dispatched", max_retries=0), "anthropic", budget=1e-9)
        with pytest.raises(BudgetExceeded):
            client.messages.create(**kwargs)
        return True
    client = _monitored(
        anthropic.AsyncAnthropic(api_key="sk-ant-invalid-never-dispatched", max_retries=0), "anthropic", budget=1e-9
    )

    async def _go():
        with pytest.raises(BudgetExceeded):
            await client.messages.create(**kwargs)

    asyncio.run(_go())
    return True


# --- The 8 cells ------------------------------------------------------------

_CELLS = [
    ("openai.chat_completions.sync", "openai", False, False),
    ("openai.chat_completions.sync.stream", "openai", False, True),
    ("openai.chat_completions.async", "openai", True, False),
    ("openai.chat_completions.async.stream", "openai", True, True),
    ("anthropic.messages.sync", "anthropic", False, False),
    ("anthropic.messages.sync.stream", "anthropic", False, True),
    ("anthropic.messages.async", "anthropic", True, False),
    ("anthropic.messages.async.stream", "anthropic", True, True),
]


@pytest.mark.parametrize(("cell", "provider", "use_async", "stream"), _CELLS, ids=[c[0] for c in _CELLS])
def test_live_surface(cell, provider, use_async, stream):
    call = _openai_call if provider == "openai" else _anthropic_call
    deny = _openai_denied_before_dispatch if provider == "openai" else _anthropic_denied_before_dispatch
    fields: Dict[str, Any] = {"provider": provider, "async": use_async, "stream": stream, "passed": False}
    model = _openai_model() if provider == "openai" else _anthropic_model()
    started = time.perf_counter()
    try:
        out = call(stream, use_async)
        started = out["started"]
        assert out["text"].strip(), "empty response from provider"
        recorded = _recorded_cost(out["client"])
        assert recorded > 0, "no cost recorded for a completed live call"
        fields["recorded_cost_usd"] = round(recorded, 8)
        # Streamed and non-streamed calls alike: the recorded cost must equal the
        # pricing table applied to the provider-reported input/output split.
        in_tok, out_tok = out["usage"]
        expected = _expected_cost(out["model"], in_tok, out_tok)
        assert recorded == pytest.approx(expected, rel=1e-6, abs=1e-9)
        fields.update(input_tokens=in_tok, output_tokens=out_tok, expected_cost_usd=round(expected, 8))
        fields["denied_before_dispatch"] = deny(use_async)
        fields["passed"] = True
    finally:
        _record(cell, model, started, **fields)
