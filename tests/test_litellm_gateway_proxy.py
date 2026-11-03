"""The LiteLLM gateway callback through a real proxy process (kazenai 1.2.2).

A real LiteLLM proxy runs the real callback. The provider is a fake on loopback that
speaks OpenAI and Anthropic, streaming included, and FinOps is ``FakeFinOps``, so no
real model call is made and nothing is spent. LiteLLM's own usage assembly and
Anthropic normalization are what's under test, so they aren't stubbed.

Opt-in (``KAZENAI_LITELLM_PROXY_TESTS=1``), from the open-components harness
(``tests/litellm_proxy_harness.lock``; see ``test_litellm_proxy_integration.py``).
Set ``KAZENAI_LITELLM_BIN`` to use a proxy other than the default harness venv's.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest

from kazenai.cost_engine import TokenCostEngine
from kazenai.model_pricing import default_max_output_tokens

from test_litellm_control import FakeFinOps, _serve_finops

if os.getenv("KAZENAI_LITELLM_PROXY_TESTS") != "1":
    pytest.skip(
        "LiteLLM proxy tests are opt-in (KAZENAI_LITELLM_PROXY_TESTS=1); they run a "
        "third-party proxy under its own terms from the open-components test harness",
        allow_module_level=True,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = ROOT.parent / "artifacts" / "kazenai-litellm-harness-1.106.6-venv" / "bin" / "litellm"
MASTER_KEY = "sk-gateway-proxy-test"
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"
UNDECLARED_ENDPOINT_MODEL = "gpt-4o-mini-regional"
CUSTOMER = "customer-42"

OPENAI_USAGE = {
    "prompt_tokens": 1_000,
    "completion_tokens": 100,
    "total_tokens": 1_100,
    "prompt_tokens_details": {"cached_tokens": 900},
}
RESPONSES_USAGE = {
    "input_tokens": 1_000,
    "output_tokens": 100,
    "total_tokens": 1_100,
    "input_tokens_details": {"cached_tokens": 900},
    "output_tokens_details": {"reasoning_tokens": 0},
}
ANTHROPIC_USAGE = {
    "input_tokens": 100,
    "cache_read_input_tokens": 400,
    "cache_creation_input_tokens": 1_000,
}


class FakeProvider:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def bodies_with(self, marker: str) -> list[dict[str, Any]]:
        with self.lock:
            return [body for body in self.requests if marker in json.dumps(body)]

    def _handler(self):
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def _json(self, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _sse(self, data: dict[str, Any], event: str | None = None) -> None:
                prefix = f"event: {event}\n".encode() if event else b""
                self.wfile.write(prefix + b"data: " + json.dumps(data).encode() + b"\n\n")
                self.wfile.flush()

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                with provider.lock:
                    provider.requests.append(body)
                if self.path.rstrip("/").endswith("/messages"):
                    self._anthropic(body)
                elif self.path.rstrip("/").endswith("/responses"):
                    self._responses(body)
                else:
                    self._openai(body)

            def _responses(self, body: dict[str, Any]) -> None:
                def response(status: str, usage: dict[str, Any] | None = None) -> dict[str, Any]:
                    return {
                        "id": "resp_x",
                        "object": "response",
                        "created_at": 1,
                        "model": body.get("model") or OPENAI_MODEL,
                        "status": status,
                        "output": [],
                        "parallel_tool_calls": False,
                        "tool_choice": "auto",
                        "tools": [],
                        "usage": usage,
                    }

                if not body.get("stream"):
                    self._json(response("completed", RESPONSES_USAGE))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self._sse(
                    {"type": "response.created", "sequence_number": 0, "response": response("in_progress")},
                    event="response.created",
                )
                self._sse(
                    {"type": "response.completed", "sequence_number": 1, "response": response("completed", RESPONSES_USAGE)},
                    event="response.completed",
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def _openai(self, body: dict[str, Any]) -> None:
                base = {"id": "chatcmpl-x", "created": 1, "model": body.get("model") or OPENAI_MODEL}
                if not body.get("stream"):
                    self._json(
                        {
                            **base,
                            "object": "chat.completion",
                            "usage": OPENAI_USAGE,
                            "choices": [
                                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}
                            ],
                        }
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunk = {**base, "object": "chat.completion.chunk"}
                text_of_body = json.dumps(body)
                slow = "slow-stream" in text_of_body
                try:
                    for text in ("a", "b", "c", "d", "e", "f") if slow else ("a",):
                        self._sse(
                            {**chunk, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                        )
                        if slow:
                            time.sleep(0.5)
                    if "cut-stream" in text_of_body:
                        self.close_connection = True
                        return
                    self._sse({**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                    if "no-usage" not in text_of_body and (body.get("stream_options") or {}).get("include_usage"):
                        self._sse({**chunk, "choices": [], "usage": OPENAI_USAGE})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    return

            def _anthropic(self, body: dict[str, Any]) -> None:
                model = body.get("model") or ANTHROPIC_MODEL
                text_of_body = json.dumps(body)
                input_usage = dict(ANTHROPIC_USAGE)
                if "one-hour-cache" in text_of_body:
                    input_usage["cache_creation"] = {
                        "ephemeral_5m_input_tokens": 600,
                        "ephemeral_1h_input_tokens": 400,
                    }
                if not body.get("stream"):
                    self._json(
                        {
                            "id": "msg_x",
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [{"type": "text", "text": "hi"}],
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                            "usage": {**input_usage, "output_tokens": 50},
                        }
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self._sse(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_x",
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {**input_usage, "output_tokens": 1},
                        },
                    },
                    event="message_start",
                )
                self._sse(
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    event="content_block_start",
                )
                self._sse(
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
                    event="content_block_delta",
                )
                if "cut-stream" in text_of_body:
                    self.close_connection = True
                    return
                self._sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")
                self._sse(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 50},
                    },
                    event="message_delta",
                )
                self._sse({"type": "message_stop"}, event="message_stop")

        return Handler


class Stack:
    def __init__(self, base_url: str, finops: FakeFinOps, provider: FakeProvider) -> None:
        self.base_url = base_url
        self.finops = finops
        self.provider = provider
        self.finops_delay_s = 0.0
        self.settle_unavailable = False
        handle = finops.handle

        def controlled(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            if self.finops_delay_s:
                time.sleep(self.finops_delay_s)
            if self.settle_unavailable and path.endswith("/settle") and body.get("event") != "provider_started":
                return 503, {"error": "unavailable"}
            return handle(path, body)

        finops.handle = controlled  # type: ignore[method-assign]

    def request(self, model: str = OPENAI_MODEL, *, operation: str | None = None, **extra: Any) -> tuple[str, dict]:
        op = operation or f"op-{uuid.uuid4().hex}"
        return op, {
            "model": model,
            "messages": [{"role": "user", "content": f"say hi {op}"}],
            "user": CUSTOMER,
            "metadata": {"kazenai_operation_id": op},
            **extra,
        }

    def post(self, body: dict[str, Any]) -> httpx.Response:
        return httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {MASTER_KEY}"},
            timeout=60,
        )

    def consume_stream(self, body: dict[str, Any]) -> str:
        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json={**body, "stream": True},
            headers={"Authorization": f"Bearer {MASTER_KEY}"},
            timeout=60,
        ) as response:
            response.raise_for_status()
            return "".join(response.iter_text())

    def reservation(self, op: str) -> dict[str, Any]:
        with self.finops.lock:
            rows = [e for e in self.finops.events if e.get("event") == "reserve" and e.get("operation_id") == op]
        assert len(rows) == 1, f"one reservation for {op}: {rows}"
        return rows[0]

    def events_for(self, op: str, event: str) -> list[dict[str, Any]]:
        rid = self.reservation(op)["reservation_id"]
        with self.finops.lock:
            return [e for e in self.finops.events if e.get("event") == event and e.get("reservation_id") == rid]

    def wait_for_event(self, op: str, event: str, *, timeout_s: float = 20.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rows = self.events_for(op, event)
            if rows:
                return rows[0]
            time.sleep(0.1)
        raise AssertionError(f"no {event} for {op}: {self.finops.events[-6:]}")

    def wait_settled(self, op: str, *, timeout_s: float = 20.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rows = self.events_for(op, "usage_known")
            if rows:
                return rows[0]
            time.sleep(0.1)
        raise AssertionError(f"no usage_known for {op}: {self.finops.events[-6:]}")


@contextmanager
def _running_stack(work: Path, *, extra_env: dict[str, str] | None = None) -> Iterator[Stack]:
    binary = Path(os.environ.get("KAZENAI_LITELLM_BIN") or DEFAULT_BIN)
    assert binary.is_file(), f"LiteLLM harness not found: {binary} (build it from tests/litellm_proxy_harness.lock)"
    probe = subprocess.run(
        [
            str(binary.parent / "python"),
            "-c",
            "import importlib.util, sys; sys.exit(importlib.util.find_spec('litellm_enterprise') is not None)",
        ],
        check=False,
    )
    assert probe.returncode == 0, "litellm_enterprise is installed in the LiteLLM test harness"

    finops = FakeFinOps(capacity_micros=100_000_000)
    finops_server = _serve_finops(finops)
    provider = FakeProvider()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    provider_port = provider.server.server_address[1]
    config = work / "config.yaml"
    config.write_text(
        f"""model_list:
  - model_name: {OPENAI_MODEL}
    litellm_params:
      model: openai/{OPENAI_MODEL}
      api_base: {provider.base_url}/v1
      api_key: fake-provider-key
      max_retries: 0
  - model_name: {ANTHROPIC_MODEL}
    litellm_params:
      model: anthropic/{ANTHROPIC_MODEL}
      api_base: {provider.base_url}
      api_key: fake-provider-key
      max_retries: 0
  - model_name: {UNDECLARED_ENDPOINT_MODEL}
    litellm_params:
      model: openai/{OPENAI_MODEL}
      api_base: http://localhost:{provider_port}/v1
      api_key: fake-provider-key
      max_retries: 0
router_settings:
  num_retries: 0
litellm_settings:
  telemetry: false
  cache: true
  cache_params:
    type: local
  callbacks: kazenai.integrations.litellm_gateway.proxy_handler_instance
general_settings:
  master_key: {MASTER_KEY}
"""
    )
    # A small explicit environment: no provider keys from the machine or CI reach the proxy.
    env = {
        "PATH": f"{binary.parent}:/usr/bin:/bin",
        "HOME": str(work),
        "TMPDIR": str(work),
        "PYTHONPATH": str(ROOT),
        "LITELLM_MODE": "PRODUCTION",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "DISABLE_SCHEMA_UPDATE": "true",
        "KAZENAI_FINOPS_URL": f"http://127.0.0.1:{finops_server.server_address[1]}",
        "KAZENAI_FINOPS_API_KEY": "fake-finops-key",
        "KAZENAI_ORG_ID": "org-gw",
        "KAZENAI_WORKSPACE_ID": "ws-gw",
        "KAZENAI_ENFORCEMENT_OWNER": "gateway",
        "KAZENAI_TIER_MISMATCH_MULTIPLIER": "1",
        "KAZENAI_GATEWAY_PENDING_DIR": str(work / "pending"),
        "KAZENAI_GATEWAY_FLUSH_INTERVAL_S": "1",
        # The fake provider stands in for first-party APIs. The undeclared host shows the uplift.
        "KAZENAI_ENDPOINT_PRICE_MULTIPLIERS": json.dumps({"127.0.0.1": "1"}),
        **(extra_env or {}),
    }
    log_path = work / "proxy.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [str(binary), "--config", str(config), "--host", "127.0.0.1", "--port", str(port)],
            cwd=work,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 180
            while True:
                assert proc.poll() is None, f"LiteLLM proxy exited:\n{log_path.read_text()[-2000:]}"
                try:
                    if httpx.get(f"{base_url}/health/liveliness", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                assert time.monotonic() < deadline, f"LiteLLM proxy not ready:\n{log_path.read_text()[-2000:]}"
                time.sleep(0.5)
            yield Stack(base_url, finops, provider)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            finops_server.shutdown()
            provider.server.shutdown()


@pytest.fixture(scope="module")
def stack(tmp_path_factory) -> Iterator[Stack]:
    with _running_stack(tmp_path_factory.mktemp("litellm-gateway")) as running:
        yield running


@pytest.fixture(scope="module")
def customer_required_stack(tmp_path_factory) -> Iterator[Stack]:
    work = tmp_path_factory.mktemp("litellm-gateway-strict")
    with _running_stack(work, extra_env={"KAZENAI_REQUIRE_CUSTOMER_ID": "1"}) as running:
        yield running


def _rates(model: str):
    row = TokenCostEngine()._row(model)
    assert row is not None, model
    return row.rates


def _ceil(value: Decimal) -> int:
    return int(value.to_integral_value(rounding="ROUND_CEILING"))


def _openai_expected() -> int:
    rates = _rates(OPENAI_MODEL)
    return _ceil(Decimal(100) * rates.input + Decimal(900) * rates.cached_input + Decimal(100) * rates.output)


def _anthropic_expected() -> int:
    rates = _rates(ANTHROPIC_MODEL)
    return _ceil(
        Decimal(100) * rates.input
        + Decimal(400) * rates.cached_input
        + Decimal(1_000) * rates.cache_write_5m
        + Decimal(50) * rates.output
    )


def test_a_plain_call_names_the_customer_and_bills_cache_tokens_at_their_rate(stack):
    op, body = stack.request()
    assert stack.post(body).status_code == 200
    assert stack.reservation(op)["business_subject_ref"] == CUSTOMER
    assert int(stack.wait_settled(op)["actual_usd_micros"]) == _openai_expected()


def test_a_customer_header_names_the_customer_too(stack):
    op, body = stack.request()
    del body["user"]
    response = httpx.post(
        f"{stack.base_url}/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {MASTER_KEY}", "x-litellm-customer-id": "customer-from-header"},
        timeout=60,
    )
    assert response.status_code == 200
    assert stack.reservation(op)["business_subject_ref"] == "customer-from-header"


def test_a_request_with_no_limit_leaves_with_the_models_bound_and_a_hold_for_it(stack):
    op, body = stack.request()
    assert stack.post(body).status_code == 200
    bound = default_max_output_tokens(OPENAI_MODEL)
    assert bound
    sent = stack.provider.bodies_with(op)
    assert sent and sent[0].get("max_tokens") == bound
    assert int(stack.reservation(op)["estimated_usd_micros"]) >= int(bound * _rates(OPENAI_MODEL).output)


def test_the_hold_follows_the_requested_limit(stack):
    op, body = stack.request(max_tokens=16_000)
    assert stack.post(body).status_code == 200
    assert stack.provider.bodies_with(op)[0].get("max_tokens") == 16_000
    assert int(stack.reservation(op)["estimated_usd_micros"]) >= int(16_000 * _rates(OPENAI_MODEL).output)


@pytest.mark.parametrize("stream_options", [None, {"include_usage": True}])
def test_a_finished_stream_settles_at_the_same_price_as_a_plain_call(stack, stream_options):
    op, body = stack.request(max_tokens=100)
    if stream_options:
        body["stream_options"] = stream_options
    stack.consume_stream(body)
    assert stack.reservation(op)["business_subject_ref"] == CUSTOMER
    settled = stack.wait_settled(op)
    assert int(settled["actual_usd_micros"]) == _openai_expected()
    assert "usage_source" not in settled["usage_evidence"], "the callback could not inspect a real proxy's stream"
    time.sleep(1.0)
    assert len(stack.events_for(op, "usage_known")) == 1


def test_anthropic_through_litellm_counts_each_cache_token_once(stack):
    op, body = stack.request(ANTHROPIC_MODEL, max_tokens=100)
    assert stack.post(body).status_code == 200
    assert int(stack.wait_settled(op)["actual_usd_micros"]) == _anthropic_expected()


def test_an_anthropic_stream_counts_each_cache_token_once(stack):
    op, body = stack.request(ANTHROPIC_MODEL, max_tokens=100)
    stack.consume_stream(body)
    assert int(stack.wait_settled(op)["actual_usd_micros"]) == _anthropic_expected()


def test_a_client_that_leaves_mid_stream_never_gets_a_final_charge(stack):
    op, body = stack.request(max_tokens=100)
    body["messages"][0]["content"] += " slow-stream"
    with httpx.stream(
        "POST",
        f"{stack.base_url}/v1/chat/completions",
        json={**body, "stream": True},
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        timeout=60,
    ) as response:
        for _ in response.iter_bytes():
            break
    time.sleep(6.0)
    assert stack.events_for(op, "usage_known") == []


def test_a_cache_hit_settles_at_zero_and_never_reaches_the_provider(stack):
    op, body = stack.request(max_tokens=100)
    assert stack.post(body).status_code == 200
    stack.wait_settled(op)
    cached_op = f"{op}-cached"
    cached = {**body, "metadata": {"kazenai_operation_id": cached_op}}
    assert stack.post(cached).status_code == 200
    settled = stack.wait_settled(cached_op)
    assert int(settled["actual_usd_micros"]) == 0
    assert settled["usage_evidence"]["cache_hit"] is True
    assert len(stack.provider.bodies_with(op)) == 1


def test_a_streamed_cache_hit_settles_at_zero_and_never_reaches_the_provider(stack):
    op, body = stack.request(max_tokens=100)
    stack.consume_stream(body)
    stack.wait_settled(op)
    cached_op = f"{op}-cached"
    stack.consume_stream({**body, "metadata": {"kazenai_operation_id": cached_op}})
    settled = stack.wait_settled(cached_op)
    assert int(settled["actual_usd_micros"]) == 0
    assert settled["usage_evidence"]["cache_hit"] is True
    assert len(stack.provider.bodies_with(op)) == 1


def test_a_refused_stream_never_reaches_the_provider(stack):
    op, body = stack.request(max_tokens=100)
    with stack.finops.lock:
        capacity = stack.finops.capacity
        stack.finops.capacity = 0
    try:
        response = httpx.post(
            f"{stack.base_url}/v1/chat/completions",
            json={**body, "stream": True},
            headers={"Authorization": f"Bearer {MASTER_KEY}"},
            timeout=60,
        )
    finally:
        with stack.finops.lock:
            stack.finops.capacity = capacity
    assert response.status_code >= 400
    assert stack.provider.bodies_with(op) == []


def test_the_stream_the_client_gets_is_unchanged(stack):
    _op, body = stack.request(max_tokens=100)
    text = stack.consume_stream(body)
    assert '"content":"a"' in text.replace(" ", "")
    assert "[DONE]" in text


@pytest.mark.parametrize("marker", ["no-usage", "cut-stream"])
def test_a_stream_the_provider_never_reported_stays_pending(stack, marker):
    op, body = stack.request(max_tokens=100)
    body["messages"][0]["content"] += f" {marker}"
    stack.consume_stream(body)
    unknown = stack.wait_for_event(op, "outcome_unknown")
    assert unknown["usage_evidence"]["error"] == "missing_usage"
    assert stack.events_for(op, "usage_known") == []


def test_an_anthropic_stream_cut_after_its_first_usage_stays_pending(stack):
    op, body = stack.request(ANTHROPIC_MODEL, max_tokens=100)
    body["messages"][0]["content"] += " cut-stream"
    stack.consume_stream(body)
    stack.wait_for_event(op, "outcome_unknown")
    assert stack.events_for(op, "usage_known") == []


def _one_hour_expected() -> int:
    rates = _rates(ANTHROPIC_MODEL)
    return _ceil(
        Decimal(100) * rates.input
        + Decimal(400) * rates.cached_input
        + Decimal(600) * rates.cache_write_5m
        + Decimal(400) * rates.cache_write_1h
        + Decimal(50) * rates.output
    )


@pytest.mark.parametrize("streamed", [False, True])
def test_one_hour_cache_writes_bill_at_their_rate(stack, streamed):
    op, body = stack.request(ANTHROPIC_MODEL, max_tokens=100)
    body["messages"][0]["content"] += " one-hour-cache"
    if streamed:
        stack.consume_stream(body)
    else:
        assert stack.post(body).status_code == 200
    assert int(stack.wait_settled(op)["actual_usd_micros"]) == _one_hour_expected()


def test_an_undeclared_endpoint_holds_and_settles_with_the_uplift(stack):
    plain_op, plain = stack.request(max_tokens=100)
    assert stack.post(plain).status_code == 200
    base_hold = int(stack.reservation(plain_op)["estimated_usd_micros"])
    base_settled = int(stack.wait_settled(plain_op)["actual_usd_micros"])

    op, body = stack.request(UNDECLARED_ENDPOINT_MODEL, max_tokens=100)
    assert stack.post(body).status_code == 200
    assert int(stack.reservation(op)["estimated_usd_micros"]) == pytest.approx(base_hold * 1.1, abs=2)
    assert int(stack.wait_settled(op)["actual_usd_micros"]) == pytest.approx(base_settled * 1.1, abs=2)


def test_slow_finops_doesnt_make_calls_wait_for_each_other(stack):
    """Each call makes three FinOps round trips. On the event loop, eight would run one by one."""
    stack.finops_delay_s = 0.25
    bodies = [stack.request(max_tokens=100)[1] for _ in range(8)]
    try:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = [response.status_code for response in pool.map(stack.post, bodies)]
        elapsed = time.monotonic() - started
    finally:
        stack.finops_delay_s = 0.0
    assert statuses == [200] * 8
    assert elapsed < 8 * 0.25 * 2, f"eight calls took {elapsed:.1f}s"


def test_a_settlement_that_failed_to_send_is_resent_by_later_traffic(stack):
    op, body = stack.request(max_tokens=100)
    stack.settle_unavailable = True
    try:
        stack.consume_stream(body)
        time.sleep(2.0)
        assert stack.events_for(op, "usage_known") == []
    finally:
        stack.settle_unavailable = False
    # The flush interval here is one second. Any later call drives the resend.
    time.sleep(1.2)
    _later, next_call = stack.request(max_tokens=100)
    assert stack.post(next_call).status_code == 200
    assert int(stack.wait_settled(op)["actual_usd_micros"]) == _openai_expected()


def test_a_call_with_no_customer_gets_a_400_when_one_is_required(customer_required_stack):
    op, body = customer_required_stack.request(max_tokens=100)
    del body["user"]
    response = customer_required_stack.post(body)
    assert response.status_code == 400, response.text
    assert "KAZENAI_REQUIRE_CUSTOMER_ID" in response.text
    assert customer_required_stack.provider.bodies_with(op) == []
    assert customer_required_stack.finops.events == []


def test_a_call_naming_a_customer_passes_when_one_is_required(customer_required_stack):
    op, body = customer_required_stack.request(max_tokens=100)
    assert customer_required_stack.post(body).status_code == 200
    assert customer_required_stack.reservation(op)["business_subject_ref"] == CUSTOMER


def _route_call(stack: Stack, path: str, body: dict[str, Any]) -> tuple[list[dict[str, Any]], httpx.Response]:
    """One call on another LiteLLM route, and the FinOps events it produced."""
    with stack.finops.lock:
        before = len(stack.finops.events)
    response = httpx.post(
        f"{stack.base_url}{path}",
        json=body,
        headers={"Authorization": f"Bearer {MASTER_KEY}", "x-litellm-customer-id": CUSTOMER},
        timeout=60,
    )
    assert response.status_code == 200, response.text
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        with stack.finops.lock:
            events = list(stack.finops.events[before:])
        if any(e.get("event") in {"usage_known", "outcome_unknown"} for e in events):
            break
        time.sleep(0.1)
    time.sleep(1.0)
    with stack.finops.lock:
        return list(stack.finops.events[before:]), response


@pytest.mark.parametrize("streamed", [False, True])
def test_the_messages_route_names_the_customer_and_settles_once(stack, streamed):
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": f"messages route {uuid.uuid4().hex}"}],
        "stream": streamed,
    }
    events, _response = _route_call(stack, "/v1/messages", body)
    assert [e for e in events if e.get("event") == "reserve"][0]["business_subject_ref"] == CUSTOMER
    settled = [e for e in events if e.get("event") == "usage_known"]
    assert len(settled) == 1, events
    assert int(settled[0]["actual_usd_micros"]) == _anthropic_expected()
    assert not [e for e in events if e.get("event") == "outcome_unknown"]


@pytest.mark.parametrize("streamed", [False, True])
def test_the_responses_route_names_the_customer_and_settles_once(stack, streamed):
    body = {
        "model": OPENAI_MODEL,
        "input": f"responses route {uuid.uuid4().hex}",
        "max_output_tokens": 100,
        "stream": streamed,
    }
    events, _response = _route_call(stack, "/v1/responses", body)
    assert [e for e in events if e.get("event") == "reserve"][0]["business_subject_ref"] == CUSTOMER
    settled = [e for e in events if e.get("event") == "usage_known"]
    assert len(settled) == 1, events
    assert int(settled[0]["actual_usd_micros"]) == _openai_expected()
    assert not [e for e in events if e.get("event") == "outcome_unknown"]


def test_the_responses_route_gets_its_own_output_limit(stack):
    marker = f"responses limit {uuid.uuid4().hex}"
    events, _response = _route_call(stack, "/v1/responses", {"model": OPENAI_MODEL, "input": marker})
    sent = stack.provider.bodies_with(marker)
    assert sent and sent[0].get("max_output_tokens") == default_max_output_tokens(OPENAI_MODEL)
    assert "max_tokens" not in sent[0]
    assert [e for e in events if e.get("event") == "usage_known"]
