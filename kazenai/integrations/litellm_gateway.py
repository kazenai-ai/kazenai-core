"""LiteLLM gateway callback: Agent FinOps limits enforced inside a LiteLLM proxy.

Use this when the gateway, not each application, owns accounting
(``KAZENAI_ENFORCEMENT_OWNER=gateway``). Every deployment attempt is reserved in
Agent FinOps before the provider is called and settled from the reported usage.
Load it in the proxy's ``config.yaml``::

    litellm_settings:
      callbacks: kazenai.integrations.litellm_gateway.proxy_handler_instance

The proxy process needs ``kazenai`` installed and the usual Agent FinOps settings
(``KAZENAI_FINOPS_URL``, ``KAZENAI_FINOPS_API_KEY``,
``KAZENAI_TIER_MISMATCH_MULTIPLIER``). Applications behind the proxy send
``metadata={"kazenai_operation_id": ...}`` and must not reserve again.

Licensing: ``kazenai`` does not depend on or bundle LiteLLM. This module imports
only ``litellm.integrations.custom_logger`` from the LiteLLM you already run,
which is MIT-licensed. LiteLLM's ``enterprise/`` code (the ``litellm-enterprise``
package) is under BerriAI's own license and is not used here.

Import this module only inside the gateway process; ``import kazenai`` never loads it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

from kazenai.integrations.litellm_control import (
    GATEWAY_OWNER,
    GatewayAdmissionDenied,
    GatewayFinOpsClient,
    estimate_cost_usd,
    estimate_reserve_usd,
    extract_usage_tokens,
    inject_traceparent,
    resolve_attempt_from_kwargs,
)
from kazenai.provider_errors import is_definitive_provider_rejection

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError as exc:  # pragma: no cover - gateway-only
    raise ImportError(
        "kazenai.integrations.litellm_gateway requires LiteLLM in the gateway process"
    ) from exc


_log = logging.getLogger("kazenai.litellm_gateway")


class KazenGatewayControlLogger(CustomLogger):
    """Per-deployment-attempt admission via FinOps reserve/start/settle."""

    def __init__(self, finops: GatewayFinOpsClient | None = None) -> None:
        super().__init__()
        self.finops = finops or GatewayFinOpsClient()
        self._attempt_counter = 0
        # attempt_id -> reservation
        self._by_litellm_call: dict[str, str] = {}

    def _next_attempt_number(self, kwargs: Mapping[str, Any]) -> int:
        # fallback_depth / retry headers are diagnostics only — never unique IDs.
        self._attempt_counter += 1
        meta = kwargs.get("metadata") if isinstance(kwargs.get("metadata"), Mapping) else {}
        if isinstance(meta, Mapping) and meta.get("kazenai_attempt_number") is not None:
            try:
                return int(meta["kazenai_attempt_number"])
            except (TypeError, ValueError):
                pass
        return self._attempt_counter

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, object], call_type: Any = None
    ) -> dict | None:
        """Reserve + mark start before provider network dispatch. Exceptions block dispatch."""
        if (os.getenv("KAZENAI_ENFORCEMENT_OWNER") or GATEWAY_OWNER).strip().lower() != GATEWAY_OWNER:
            return kwargs

        model = str(kwargs.get("model") or "unknown")
        messages = kwargs.get("messages") or []
        approx_tokens = 0
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, Mapping):
                    approx_tokens += len(str(m.get("content") or "")) // 4
        approx_tokens = max(approx_tokens, 256)

        att_num = self._next_attempt_number(kwargs)
        base = resolve_attempt_from_kwargs(kwargs, attempt_number=att_num)
        # Fresh attempt_id + idempotency key per deployment hook (fallback must not inherit A's ids).
        from kazenai.integrations.litellm_control import new_attempt_identity

        identity = new_attempt_identity(
            operation_id=base.operation_id,
            call_id=base.call_id,
            attempt=att_num,
        )

        estimated = estimate_reserve_usd(
            model=model, prompt_tokens=approx_tokens, completion_tokens=approx_tokens
        )
        try:
            reservation = self.finops.reserve_attempt(
                identity=identity,
                model=model,
                estimated_usd=estimated,
                run_id=str(
                    (kwargs.get("metadata") or {}).get("run_id")
                    if isinstance(kwargs.get("metadata"), Mapping)
                    else identity.operation_id
                ),
            )
            self.finops.mark_start(reservation)
        except GatewayAdmissionDenied as exc:
            # LiteLLM proxy maps BudgetExceededError to budget_exceeded_status_code;
            # raw GatewayAdmissionDenied surfaces as HTTP 500 on pinned 1.104.x.
            # Soft-import so hermetic tests can stub litellm without exceptions submodule.
            if exc.code == "budget_denied":
                try:
                    import litellm
                    from litellm.exceptions import BudgetExceededError
                except Exception:
                    raise exc
                try:
                    litellm.budget_exceeded_status_code = int(exc.http_status or 402)
                except Exception:
                    pass
                raise BudgetExceededError(
                    current_cost=1.0,
                    max_budget=0.0,
                    message=str(exc),
                ) from exc
            raise
        except Exception as exc:
            # Must not be swallowed as best-effort logging — block dispatch.
            raise GatewayAdmissionDenied(
                f"gateway admission error: {exc}", code="admission_error", http_status=503
            ) from exc

        self.finops.bind_kwargs_metadata(kwargs, reservation)
        litellm_call_id = str(kwargs.get("litellm_call_id") or "")
        if litellm_call_id:
            self._by_litellm_call[litellm_call_id] = identity.attempt_id

        tp = None
        headers = kwargs.get("headers")
        if isinstance(headers, Mapping):
            tp = headers.get("traceparent") or headers.get("Traceparent")
        if not tp and isinstance(kwargs.get("metadata"), Mapping):
            tp = kwargs["metadata"].get("kazenai_traceparent")  # type: ignore[index]
        inject_traceparent(kwargs, str(tp) if tp else None)
        return kwargs

    async def async_post_call_success_deployment_hook(
        self,
        request_data: dict,
        response: Any,
        call_type: Any = None,
    ) -> Any:
        reservation = self._reservation_for(request_data)
        if reservation is None:
            return response
        model = str(request_data.get("model") or reservation.meta.get("model") or "unknown")
        prompt, completion, total = extract_usage_tokens(response)
        if total <= 0:
            # Unknown usage → pending, never false-zero settle.
            self.finops.settle_cancel_or_pending(reservation, dispatched=True, error="missing_usage")
            return response
        actual = estimate_cost_usd(model=model, prompt_tokens=prompt, completion_tokens=completion)
        self.finops.settle_success(
            reservation,
            actual_usd=actual,
            usage_evidence={
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "model": model,
                "cost_confidence": "estimated",
            },
        )
        return response

    async def async_post_call_failure_deployment_hook(
        self,
        request_data: Mapping[str, object],
        exception: Exception,
        call_type: Any = None,
        fallback_depth: int | None = None,
    ) -> None:
        _ = fallback_depth  # diagnostic only
        reservation = self._reservation_for(request_data)
        if reservation is None:
            return
        # A definitive provider refusal (HTTP 4xx: bad request, auth, model not
        # found, rate limited) did no billable work: settle exactly $0 and
        # release the hold instead of leaving it pending.
        if not isinstance(exception, GatewayAdmissionDenied) and is_definitive_provider_rejection(exception):
            self.finops.settle_success(
                reservation,
                actual_usd=0.0,
                usage_evidence={
                    "provider_rejected": True,
                    "status_code": getattr(exception, "status_code", None),
                    "cost_confidence": "exact",
                },
            )
            return
        # Provider may have processed the request even if the response was lost.
        dispatched = not isinstance(exception, GatewayAdmissionDenied)
        self.finops.settle_cancel_or_pending(
            reservation,
            dispatched=dispatched,
            error=f"{type(exception).__name__}:{exception}",
        )

    def _reservation_for(self, request_data: Mapping[str, Any] | dict) -> Any:
        meta = request_data.get("metadata") if isinstance(request_data.get("metadata"), Mapping) else {}
        litellm_params = (
            request_data.get("litellm_params")
            if isinstance(request_data.get("litellm_params"), Mapping)
            else {}
        )
        nested = litellm_params.get("metadata") if isinstance(litellm_params.get("metadata"), Mapping) else {}
        attempt_id = ""
        for block in (meta, nested, request_data):
            if isinstance(block, Mapping) and block.get("kazenai_attempt_id"):
                attempt_id = str(block["kazenai_attempt_id"])
                break
        if not attempt_id:
            call_id = str(request_data.get("litellm_call_id") or "")
            attempt_id = self._by_litellm_call.get(call_id, "")
        if not attempt_id:
            return None
        return self.finops.get_open(attempt_id)


# LiteLLM proxy loads callbacks by dotted path to a CustomLogger *instance*
# (class references are rejected at startup on pinned 1.104+).
proxy_handler_instance = KazenGatewayControlLogger()

# Back-compat alias for docs that reference the class name.
kazen_gateway_control = KazenGatewayControlLogger
