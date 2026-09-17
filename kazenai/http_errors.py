"""Shared FastAPI error handling — sanitized envelopes, no stack-trace leaks.

Every KazenAI service should call :func:`register_exception_handlers` on its app
so that unhandled exceptions return a uniform, trace-id'd JSON envelope instead
of a framework stack trace (which can leak internals to clients).

FastAPI is imported lazily inside the function so importing this module never
requires a web framework.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import uuid4

_log = logging.getLogger("kazenai.http_errors")


def error_envelope(*, message: str, trace_id: str, kind: str = "internal_error") -> Dict[str, Any]:
    return {
        "type": "error",
        "kind": kind,
        "message": message,
        "trace_id": trace_id,
    }


def register_exception_handlers(app: Any) -> None:
    """Install a catch-all handler that returns a sanitized 500 envelope.

    HTTPException / RequestValidationError keep FastAPI's built-in handling; this
    only catches otherwise-unhandled exceptions so internal details never reach
    the client. The full exception (with traceback) is logged server-side under
    the same trace_id for correlation.
    """
    from fastapi.responses import JSONResponse
    from starlette.requests import Request

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: ANN001
        trace_id = str(uuid4())
        _log.exception(
            "unhandled exception [trace_id=%s] on %s %s",
            trace_id,
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content=error_envelope(
                message="Internal server error. Reference this trace_id when reporting.",
                trace_id=trace_id,
            ),
        )
