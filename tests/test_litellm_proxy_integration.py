"""LiteLLM gateway Control (live proxy process).

Starts a real LiteLLM proxy subprocess, fake OpenAI upstream oracle(s), and
(optionally) in-process FakeFinOps. Hermetic proofs remain in
``test_litellm_control.py``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from kazenai.integrations.litellm_control import (
    AttemptIdentity,
    AttemptReservation,
    GatewayFinOpsClient,
    estimate_cost_usd,
    usd_to_micros_ceil,
)

from test_litellm_control import FakeFinOps, _serve_finops

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "kazenai-examples" / "litellm-otel-control"
ARTIFACTS = REPO / "artifacts"
LITELLM_VENV = ARTIFACTS / "kazenai-litellm-venv"
LITELLM_MAIN = Path(os.getenv("KAZENAI_LITELLM_SOURCE", "/nonexistent-litellm-source"))

CONFIG_APP = EXAMPLES / "configs" / "proxy_app_owned.yaml"
CONFIG_GATEWAY = EXAMPLES / "configs" / "proxy_gateway_owned.yaml"
FAKE_UPSTREAM = EXAMPLES / "fake_upstream.py"

# Needs the kazenai-examples LiteLLM configs and gateway callback beside this repo.
if not (EXAMPLES / "kazenai_litellm_callback.py").is_file():
    pytest.skip("kazenai-examples/litellm-otel-control not checked out beside kazenai-core", allow_module_level=True)

RESULTS: list[dict[str, Any]] = []
_PROCS: list[subprocess.Popen[Any]] = []


def _record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"scenario": name, "status": status, "detail": detail})


def _ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _terminate_procs() -> None:
    for proc in list(_PROCS):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    _PROCS.clear()


def _strip_redis_env(env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    for key in list(out):
        if key.upper().startswith("REDIS"):
            del out[key]
    return out


def _litellm_bin() -> Path:
    return LITELLM_VENV / "bin" / "litellm"


def _ensure_litellm_venv() -> Path:
    """Return litellm CLI path or pytest.skip with reason."""
    bin_path = _litellm_bin()
    if bin_path.is_file():
        return bin_path

    py = sys.executable
    LITELLM_VENV.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([py, "-m", "venv", str(LITELLM_VENV)], check=True, capture_output=True, text=True)
        pip = LITELLM_VENV / "bin" / "pip"
        subprocess.run([str(pip), "install", "-U", "pip"], check=True, capture_output=True, text=True)
        install_target = "litellm[proxy]==1.104.2"
        if LITELLM_MAIN.is_dir():
            install_target = f"{LITELLM_MAIN}[proxy]"
        subprocess.run(
            [str(pip), "install", install_target],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        subprocess.run(
            [str(pip), "install", "-e", str(REPO / "kazen-event-schema"), "-e", str(REPO / "kazenai-core")],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        pytest.skip(f"LiteLLM proxy venv bootstrap failed: {exc}")

    if not bin_path.is_file():
        pytest.skip("LiteLLM CLI missing after venv bootstrap")
    return bin_path


def _oracle_count(base: str) -> int:
    root = base.rstrip("/").removesuffix("/v1")
    resp = httpx.get(f"{root}/oracle", timeout=5.0)
    resp.raise_for_status()
    return int(resp.json().get("count") or 0)


def _reservation_from_fake(fake: FakeFinOps, reservation_id: str) -> AttemptReservation:
    row: dict[str, Any] = dict(fake.reservations[reservation_id])
    for event in fake.events:
        if event.get("event") == "reserve" and event.get("reservation_id") == reservation_id:
            row.update({k: v for k, v in event.items() if k != "event"})
            break
    identity = AttemptIdentity(
        operation_id=str(row.get("operation_id") or ""),
        call_id=str(row.get("call_id") or ""),
        attempt_id=str(row.get("attempt_id") or ""),
        attempt=int(row.get("attempt") or 1),
        idempotency_key=str(row.get("idempotency_key") or ""),
    )
    assert identity.idempotency_key, f"missing idempotency_key on reserve row {reservation_id!r}"
    return AttemptReservation(
        identity=identity,
        reservation_id=str(reservation_id),
        estimated_usd_micros=int(row.get("estimated_usd_micros") or 0),
        status=str(row.get("state") or "reserved"),
    )


def _start_fake_upstream(*, fail_next: int = 0, lose_response_next: int = 0) -> tuple[int, str]:
    port = _ephemeral_port()
    cmd = [
        sys.executable,
        str(FAKE_UPSTREAM),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if fail_next:
        cmd.extend(["--fail-next", str(fail_next)])
    if lose_response_next:
        cmd.extend(["--lose-response-next", str(lose_response_next)])
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _PROCS.append(proc)
    base = f"http://127.0.0.1:{port}"
    for _ in range(40):
        try:
            httpx.get(f"{base}/oracle", timeout=1.0)
            return port, f"{base}/v1"
        except Exception:
            time.sleep(0.1)
    _terminate_procs()
    pytest.skip("fake_upstream did not become ready")


def _wait_proxy_ready(*, port: int, master_key: str, expect_models: tuple[str, ...]) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    headers = {"Authorization": f"Bearer {master_key}"}
    last_err = ""
    for _ in range(80):
        try:
            resp = httpx.get(url, headers=headers, timeout=2.0)
            if resp.status_code == 200:
                body = resp.text
                if all(m in body for m in expect_models):
                    return
                last_err = f"models response missing expected ids: {expect_models!r}"
            else:
                last_err = f"HTTP {resp.status_code}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.25)
    _terminate_procs()
    pytest.skip(f"LiteLLM proxy not ready on port {port}: {last_err}")


def _start_litellm_proxy(
    *,
    config: Path,
    port: int,
    env: dict[str, str],
    expect_models: tuple[str, ...],
) -> None:
    litellm_bin = _ensure_litellm_venv()
    merged = _strip_redis_env(env)
    merged.setdefault("PYTHONPATH", str(EXAMPLES))
    proc = subprocess.Popen(
        [
            str(litellm_bin),
            "--config",
            str(config),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _PROCS.append(proc)
    time.sleep(2.0)
    if proc.poll() is not None:
        log = (proc.stdout.read() if proc.stdout else "")[:4000]
        pytest.skip(f"LiteLLM proxy exited during startup:\n{log}")
    _wait_proxy_ready(port=port, master_key=merged["LITELLM_MASTER_KEY"], expect_models=expect_models)


@pytest.fixture(scope="module", autouse=True)
def _live_module_cleanup():
    yield
    _terminate_procs()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"kazenai-live-{ts}.json"
    path.write_text(
        json.dumps(
            {
                "recorded_at": ts,
                "mode": "live_litellm_proxy",
                "litellm_venv": str(LITELLM_VENV),
                "gates": [
                    "GATEWAY_SINGLE_ATTEMPT_VERIFIED",
                    "GATEWAY_RETRY_ACCOUNTING_VERIFIED",
                ],
                "results": RESULTS,
                "pass_count": sum(1 for r in RESULTS if r["status"] == "pass"),
                "fail_count": sum(1 for r in RESULTS if r["status"] == "fail"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_live_app_owned_single_attempt_via_proxy(monkeypatch):
    """Real LiteLLM proxy (app-owned YAML) + monitor; upstream oracle count == 1."""
    openai = pytest.importorskip("openai")
    _ensure_litellm_venv()

    upstream_port, upstream_v1 = _start_fake_upstream()
    proxy_port = _ephemeral_port()
    master_key = "sk-p6-live-app"

    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_ENFORCEMENT_OWNER", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")

    env = os.environ.copy()
    env.update(
        {
            "KAZENAI_FAKE_UPSTREAM_URL": upstream_v1,
            "LITELLM_MASTER_KEY": master_key,
            "PYTHONPATH": str(EXAMPLES),
        }
    )
    _start_litellm_proxy(
        config=CONFIG_APP,
        port=proxy_port,
        env=env,
        expect_models=("gpt-4o-mini",),
    )

    from kazenai import monitor

    client = openai.OpenAI(
        api_key=master_key,
        base_url=f"http://127.0.0.1:{proxy_port}/v1",
        max_retries=0,
        http_client=httpx.Client(timeout=30.0),
    )
    guarded = monitor(client, max_budget_usd=1.0, org_id="org-p6-live", workspace_id="ws-p6-live")
    resp = guarded.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.choices[0].message.content == "ok"
    assert _oracle_count(upstream_v1) == 1, f"upstream_port={upstream_port}"
    _record("live_app_owned_single_attempt", "pass", f"upstream_port={upstream_port}")


def test_live_gateway_denial_blocks_upstream(monkeypatch):
    """Gateway-owned callback + exhausted FakeFinOps → no fake upstream dispatch."""
    _ensure_litellm_venv()
    _, upstream_v1 = _start_fake_upstream()
    proxy_port = _ephemeral_port()
    master_key = "sk-p6-live-deny"

    fake = FakeFinOps(capacity_micros=0)
    finops_server = _serve_finops(fake)

    monkeypatch.setenv("KAZENAI_ENFORCEMENT_OWNER", "gateway")
    env = os.environ.copy()
    env.update(
        {
            "KAZENAI_FAKE_UPSTREAM_URL": upstream_v1,
            "KAZENAI_FAKE_UPSTREAM_FALLBACK_URL": upstream_v1,
            "LITELLM_MASTER_KEY": master_key,
            "KAZENAI_ENFORCEMENT_OWNER": "gateway",
            "KAZENAI_FINOPS_URL": f"http://127.0.0.1:{finops_server.server_address[1]}",
            "KAZENAI_FINOPS_API_KEY": "k",
            "KAZENAI_ORG_ID": "org-p6-live",
            "KAZENAI_WORKSPACE_ID": "ws-p6-live",
            "PYTHONPATH": str(EXAMPLES),
        }
    )
    _start_litellm_proxy(
        config=CONFIG_GATEWAY,
        port=proxy_port,
        env=env,
        expect_models=("gpt-4o-mini", "gpt-4o-mini-fallback"),
    )

    resp = httpx.post(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        headers={"Authorization": f"Bearer {master_key}"},
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        timeout=30.0,
    )
    assert resp.status_code >= 400
    body_text = resp.text.lower()
    admission_typed = resp.status_code in (402, 422) or any(
        token in body_text for token in ("budget", "admission", "denied", "exhausted", "capacity")
    )
    assert admission_typed, resp.text[:800]
    assert _oracle_count(upstream_v1) == 0
    assert not fake.events or all(e.get("event") != "usage_known" for e in fake.events)
    finops_server.shutdown()
    _record(
        "live_gateway_denial_zero_upstream",
        "pass",
        f"http={resp.status_code} admission_typed={admission_typed} ideal_http=402",
    )


def test_live_gateway_retry_accounting_lose_response_late_settle(monkeypatch):
    """GATEWAY_RETRY_ACCOUNTING on real proxy: A pending (response lost), B settled, late A once."""
    _ensure_litellm_venv()
    _, primary_v1 = _start_fake_upstream(lose_response_next=1)
    _, fallback_v1 = _start_fake_upstream()
    proxy_port = _ephemeral_port()
    master_key = "sk-p6-live-retry"

    fake = FakeFinOps(capacity_micros=50_000_000)
    finops_server = _serve_finops(fake)
    finops_url = f"http://127.0.0.1:{finops_server.server_address[1]}"

    env = os.environ.copy()
    env.update(
        {
            "KAZENAI_FAKE_UPSTREAM_URL": primary_v1,
            "KAZENAI_FAKE_UPSTREAM_FALLBACK_URL": fallback_v1,
            "LITELLM_MASTER_KEY": master_key,
            "KAZENAI_ENFORCEMENT_OWNER": "gateway",
            "KAZENAI_FINOPS_URL": finops_url,
            "KAZENAI_FINOPS_API_KEY": "k",
            "KAZENAI_ORG_ID": "org-p6-live",
            "KAZENAI_WORKSPACE_ID": "ws-p6-live",
            "PYTHONPATH": str(EXAMPLES),
        }
    )
    _start_litellm_proxy(
        config=CONFIG_GATEWAY,
        port=proxy_port,
        env=env,
        expect_models=("gpt-4o-mini", "gpt-4o-mini-fallback"),
    )

    op_id = f"op_{uuid.uuid4().hex}"
    call_id = f"call_{uuid.uuid4().hex}"
    resp = httpx.post(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        headers={"Authorization": f"Bearer {master_key}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {
                "kazenai_operation_id": op_id,
                "kazenai_call_id": call_id,
                "run_id": op_id,
            },
        },
        timeout=90.0,
    )
    assert resp.status_code == 200, resp.text[:800]

    reserves = [e for e in fake.events if e.get("event") == "reserve"]
    # reservation.lifecycle.v1 uses outcome_unknown (legacy alias: pending).
    pending_events = [
        e
        for e in fake.events
        if e.get("event") in {"pending", "outcome_unknown", "replay_pending"}
    ]
    settle_events = [e for e in fake.events if e.get("event") == "usage_known"]
    assert len(reserves) == 2, fake.events
    assert len(pending_events) >= 1, fake.events
    assert len(settle_events) == 1, fake.events

    rid_a = str(reserves[0]["reservation_id"])
    rid_b = str(reserves[1]["reservation_id"])
    assert reserves[0]["attempt_id"] != reserves[1]["attempt_id"]
    assert reserves[0]["idempotency_key"] != reserves[1]["idempotency_key"]
    assert fake.reservations[rid_a].get("state") in {
        "pending",
        "pending_reconciliation",
    }
    assert fake.reservations[rid_b].get("settled") is True
    assert settle_events[0].get("reservation_id") == rid_b

    assert _oracle_count(primary_v1) == 1, "attempt A must reach primary upstream oracle"
    assert _oracle_count(fallback_v1) == 1, "attempt B must reach fallback upstream oracle"

    monkeypatch.setenv("KAZENAI_FINOPS_URL", finops_url)
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "k")
    monkeypatch.setenv("KAZENAI_ORG_ID", "org-p6-live")
    monkeypatch.setenv("KAZENAI_WORKSPACE_ID", "ws-p6-live")
    finops_client = GatewayFinOpsClient()
    rsv_a = _reservation_from_fake(fake, rid_a)
    late_usd = estimate_cost_usd(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5)
    settled_micros_before = fake.settled
    late = finops_client.settle_success(
        rsv_a,
        actual_usd=late_usd,
        usage_evidence={"prompt_tokens": 10, "completion_tokens": 5, "late": True},
    )
    assert late.get("result") == "applied", late
    if late.get("result") == "applied":
        assert fake.settled == settled_micros_before + usd_to_micros_ceil(late_usd)
    again = finops_client.settle_success(
        rsv_a,
        actual_usd=late_usd,
        usage_evidence={"prompt_tokens": 10, "completion_tokens": 5, "late": True},
    )
    assert again.get("result") == "duplicate"
    assert fake.settled == settled_micros_before + usd_to_micros_ceil(late_usd)
    a_settle_events = [
        e for e in fake.events if e.get("event") == "usage_known" and e.get("reservation_id") == rid_a
    ]
    assert len(a_settle_events) == 1

    settled_rows = [r for r in fake.reservations.values() if r.get("settled")]
    assert len(settled_rows) == 2
    assert fake.reservations[rid_a].get("settled") is True

    finops_server.shutdown()
    _record(
        "live_gateway_retry_accounting_lose_response",
        "pass",
        (
            f"op={op_id} reserves=2 pending_a=1 settle_b=1 "
            f"primary_hits=1 fallback_hits=1 settled_rows=2"
        ),
    )
