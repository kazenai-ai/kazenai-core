"""Independent fake OpenAI-compatible upstream oracle for P6 gateway proofs.

Tracks each HTTP completion request so tests can assert how many provider
dispatches occurred (hidden retries must be visible here).
"""

from __future__ import annotations

import argparse
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class UpstreamOracle:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.fail_next = 0
        self.lose_response_next = 0
        self.default_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def record(self, entry: dict[str, Any]) -> None:
        with self.lock:
            self.requests.append(entry)

    def count(self) -> int:
        with self.lock:
            return len(self.requests)

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.requests)


ORACLE = UpstreamOracle()


def make_handler(oracle: UpstreamOracle):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = {}
            traceparent = self.headers.get("traceparent") or self.headers.get("Traceparent")
            entry = {
                "path": self.path,
                "model": body.get("model"),
                "traceparent": traceparent,
                "authorization_present": bool(self.headers.get("Authorization")),
            }
            oracle.record(entry)

            with oracle.lock:
                fail = oracle.fail_next > 0
                if fail:
                    oracle.fail_next -= 1
                lose = oracle.lose_response_next > 0
                if lose:
                    oracle.lose_response_next -= 1

            if fail:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"injected upstream failure"}}')
                return

            if lose:
                # Close without a valid completion body — client sees transport loss.
                self.close_connection = True
                return

            model = body.get("model") or "gpt-4o-mini"
            usage = dict(oracle.default_usage)
            if body.get("stream"):
                # OpenAI-compatible SSE with final usage chunk (P8 async stream cells).
                chunks = [
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "ok"},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    },
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": model,
                        "choices": [],
                        "usage": usage,
                    },
                ]
                lines = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                payload = lines.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            resp = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
            payload = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/oracle"):
                body = json.dumps({"count": oracle.count(), "requests": oracle.snapshot()}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def serve(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    oracle: UpstreamOracle | None = None,
) -> tuple[ThreadingHTTPServer, UpstreamOracle]:
    """Start a fake upstream. Each call gets an isolated oracle unless one is passed."""
    o = oracle if oracle is not None else UpstreamOracle()
    server = ThreadingHTTPServer((host, port), make_handler(o))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, o


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument(
        "--fail-next",
        type=int,
        default=0,
        help="Return HTTP 500 for the next N completion requests (live retry proofs).",
    )
    parser.add_argument(
        "--lose-response-next",
        type=int,
        default=0,
        help="Close connection without body for the next N requests (pending exposure).",
    )
    args = parser.parse_args()
    with ORACLE.lock:
        ORACLE.fail_next = max(0, int(args.fail_next))
        ORACLE.lose_response_next = max(0, int(args.lose_response_next))
    server, _ = serve(args.host, args.port, oracle=ORACLE)
    print(f"fake upstream on http://{args.host}:{server.server_address[1]}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
