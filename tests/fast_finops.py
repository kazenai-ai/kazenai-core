"""A local Agent FinOps stand-in for the SDK 1.2.4 lifecycle tests.

Records every request in order. ``start_on_admit`` toggles the capability; ``slow``
maps a request kind ("lifecycle", "reserve", "provider_started", "usage_known", ...) to
seconds of delay; ``slow_reserve_attempts`` delays only the first attempts of each
idempotency key (the reservation is committed before the delay, like a real timeout);
``settle_status`` answers every settle with that status.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from kazenai.spine import transport

CAPABILITY = "kzcap1.test." + "A" * 43


class FastFinOps:
    def __init__(self, *, start_on_admit: bool = True) -> None:
        self.start_on_admit = start_on_admit
        self.slow: dict[str, float] = {}
        self.slow_reserve_attempts = 0
        self.settle_status = 200
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.lifecycle_gets = 0
        self._reservations: dict[str, dict[str, Any]] = {}
        self._attempts: dict[str, int] = {}
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def _send(self, status: int, payload: Any) -> None:
                raw = json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up waiting

            def do_GET(self) -> None:  # noqa: N802
                with server._lock:
                    server.lifecycle_gets += 1
                time.sleep(server.slow.get("lifecycle", 0.0))
                capabilities = {"reservation_settlement": True, "legacy_check_reconcile": False}
                if server.start_on_admit:
                    capabilities["start_on_admit"] = True
                self._send(200, {
                    "ok": True,
                    "effective_lifecycle": {"profile": "control", "lifecycle_version": "reservation.lifecycle.v1"},
                    "capabilities": capabilities,
                })

            def do_POST(self) -> None:  # noqa: N802
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path.endswith("/reserve"):
                    return self._reserve(req)
                if self.path.endswith("/settle"):
                    event = str(req.get("event") or "")
                    with server._lock:
                        server.requests.append((event, req))
                    time.sleep(server.slow.get(event, 0.0))
                    if server.settle_status != 200:
                        return self._send(server.settle_status, {"detail": {"error_code": "sql_unavailable"}})
                    state = {
                        "provider_started": "in_flight",
                        "usage_known": "settled",
                        "admission_abandoned": "cancelled_before_call",
                        "cancel_before_call": "cancelled_before_call",
                        "outcome_unknown": "pending_reconciliation",
                    }.get(event, "settled")
                    result = "pending" if state == "pending_reconciliation" else "applied"
                    return self._send(200, {"result": result, "state": state, "reservation_id": req.get("reservation_id")})
                return self._send(200, {"ok": True})

            def _reserve(self, req: dict[str, Any]) -> None:
                key = str(req.get("idempotency_key") or "")
                with server._lock:
                    server.requests.append(("reserve", req))
                    server._attempts[key] = server._attempts.get(key, 0) + 1
                    attempt_no = server._attempts[key]
                    body = server._reservations.get(key)
                    if body is None:
                        body = server._reservations[key] = {
                            "allowed": True,
                            "reservation_id": f"rsv_{uuid.uuid4().hex}",
                            "decision_id": "dec_1",
                            "lifecycle_version": "reservation.lifecycle.v1",
                            "state": "in_flight" if req.get("start_on_admit") else "reserved",
                            "call_id": req.get("call_id"),
                            "attempt": req.get("attempt") or 1,
                            "reserved_usd_micros": int(req.get("estimated_usd_micros") or 0),
                            "settlement_capability": CAPABILITY,
                        }
                        replay = False
                    else:
                        replay = True
                if attempt_no <= server.slow_reserve_attempts:
                    time.sleep(server.slow.get("reserve", 0.0))
                else:
                    time.sleep(0.0 if replay else server.slow.get("reserve_fast", 0.0))
                self._send(200, dict(body, duplicate=replay))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def events(self) -> list[str]:
        with self._lock:
            return [kind for kind, _ in self.requests]

    def settles(self, event: str) -> list[dict[str, Any]]:
        with self._lock:
            return [body for kind, body in self.requests if kind == event]

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        transport.reset_pools()
