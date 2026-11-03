"""LiteLLM gateway callback: Agent FinOps limits enforced inside a LiteLLM proxy.

Use this when the gateway, not each application, owns accounting
(``KAZENAI_ENFORCEMENT_OWNER=gateway``). Every deployment attempt is reserved in
Agent FinOps before the provider is called and settled from the reported usage.
Load it in the proxy's ``config.yaml``::

    litellm_settings:
      callbacks: kazenai.integrations.litellm_gateway.proxy_handler_instance

The proxy process needs ``kazenai`` installed and the usual Agent FinOps settings
(``KAZENAI_FINOPS_URL``, ``KAZENAI_FINOPS_API_KEY``,
``KAZENAI_TIER_MISMATCH_MULTIPLIER``). Applications behind the proxy name the
customer the usual LiteLLM way (``user``, or its customer headers), send
``metadata={"kazenai_operation_id": ...}`` and must not reserve again.

Chat and Responses calls leave with an output limit, added or lowered to the
model's bound the way ``monitor()`` does it, so the hold covers what the call can
cost. A stream or a LiteLLM cache hit settles from its success event, because
LiteLLM returns both before the deployment success hook runs. A stream settles only
if the provider finished it and reported usage; otherwise it stays pending.

``KAZENAI_REQUIRE_CUSTOMER_ID=1`` refuses a call that names no customer. Calls to
FinOps run off the event loop, and a settlement that failed to send is resent.

Licensing: ``kazenai`` does not depend on or bundle LiteLLM. This module imports
only ``litellm.integrations.custom_logger`` from the LiteLLM you already run,
which is MIT-licensed. LiteLLM's ``enterprise/`` code (the ``litellm-enterprise``
package) is under BerriAI's own license and is not used here.

Import this module only inside the gateway process; ``import kazenai`` never loads it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from decimal import Decimal
from typing import Any, Mapping

from kazenai.integrations.litellm_control import (
    GATEWAY_OWNER,
    GatewayAdmissionDenied,
    GatewayFinOpsClient,
    client_disconnected,
    customer_id_required,
    endpoint_price_multiplier,
    estimate_cost_from_response,
    estimate_reserve_usd,
    extract_usage_tokens,
    inject_traceparent,
    provider_finished_and_reported,
    reserve_bound_tokens,
    resolve_attempt_from_kwargs,
    resolve_customer_id,
)
from kazenai.cost_engine import _require_multiplier_when_control
from kazenai.provider_errors import is_definitive_provider_rejection

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError as exc:  # pragma: no cover - gateway-only
    raise ImportError(
        "kazenai.integrations.litellm_gateway requires LiteLLM in the gateway process"
    ) from exc


_log = logging.getLogger("kazenai.litellm_gateway")

# How often unsent settlements are retried, driven by the proxy's own traffic.
_FLUSH_INTERVAL_S = float(os.getenv("KAZENAI_GATEWAY_FLUSH_INTERVAL_S") or 30.0)
# Stream evidence normally lives for moments; this bounds a success event that never comes.
_MAX_STREAM_EVIDENCE = 10_000


def _log_flush_outcome(task: "asyncio.Future[dict[str, int]]") -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _log.warning("gateway settlement resend failed: %s", exc)


class KazenGatewayControlLogger(CustomLogger):
    """Per-deployment-attempt admission via FinOps reserve/start/settle."""

    def __init__(self, finops: GatewayFinOpsClient | None = None) -> None:
        super().__init__()
        self.finops = finops or GatewayFinOpsClient()
        self._attempt_counter = 0
        # attempt_id -> reservation
        self._by_litellm_call: dict[str, str] = {}
        # litellm_call_id -> did the provider finish the stream and report usage
        self._stream_evidence: dict[str, bool] = {}
        self._flush_task: "asyncio.Future[dict[str, int]] | None" = None
        self._last_flush = float("-inf")

    def _schedule_flush(self) -> None:
        """Resend unsent settlements in the background, at most once per interval."""
        if self._flush_task is not None and not self._flush_task.done():
            return
        now = time.monotonic()
        if now - self._last_flush < _FLUSH_INTERVAL_S:
            return
        self._last_flush = now
        self._flush_task = asyncio.ensure_future(asyncio.to_thread(self.finops.flush_pending))
        self._flush_task.add_done_callback(_log_flush_outcome)

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

        self._schedule_flush()
        model = str(kwargs.get("model") or "unknown")

        att_num = self._next_attempt_number(kwargs)
        base = resolve_attempt_from_kwargs(kwargs, attempt_number=att_num)
        # Fresh attempt_id + idempotency key per deployment hook (fallback must not inherit A's ids).
        from kazenai.integrations.litellm_control import new_attempt_identity

        identity = new_attempt_identity(
            operation_id=base.operation_id,
            call_id=base.call_id,
            attempt=att_num,
        )

        customer = resolve_customer_id(kwargs)
        try:
            if customer is None and customer_id_required():
                raise GatewayAdmissionDenied(
                    "this call names no customer, and KAZENAI_REQUIRE_CUSTOMER_ID is on: "
                    "send `user` or LiteLLM's customer header",
                    code="customer_id_required",
                    http_status=400,
                )
            # A guessed price would hold a flat $0.01 and leave every call pending.
            _require_multiplier_when_control()
            price_multiplier = endpoint_price_multiplier(kwargs)
            input_bound, output_bound = reserve_bound_tokens(kwargs, call_type=call_type)
            estimated = estimate_reserve_usd(
                model=model,
                prompt_tokens=input_bound,
                completion_tokens=output_bound,
                price_multiplier=price_multiplier,
            )
            reservation = await asyncio.to_thread(
                self.finops.reserve_attempt,
                identity=identity,
                model=model,
                estimated_usd=estimated,
                run_id=str(
                    (kwargs.get("metadata") or {}).get("run_id")
                    if isinstance(kwargs.get("metadata"), Mapping)
                    else identity.operation_id
                ),
                business_subject_ref=customer,
            )
            reservation.meta["price_multiplier"] = format(price_multiplier, "f")
            await asyncio.to_thread(self.finops.mark_start, reservation)
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
            if exc.http_status == 400:
                # The caller's error: raw GatewayAdmissionDenied would surface as HTTP 500.
                try:
                    from litellm.exceptions import BadRequestError
                except Exception:
                    raise exc
                raise BadRequestError(message=str(exc), model=model, llm_provider="kazenai") from exc
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
        # On LiteLLM's Responses route this runs with the stream just opened. Its success event settles it.
        if request_data.get("stream"):
            return response
        reservation = self._reservation_for(request_data)
        if reservation is None:
            return response
        await self._settle_usage(reservation, request_data, response)
        self._forget_call(request_data)
        return response

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: Any, response: Any, request_data: dict
    ) -> Any:
        """Pass the stream through, then note whether the provider finished it and reported usage."""
        async for chunk in response:
            yield chunk
        call_id = str(request_data.get("litellm_call_id") or "")
        reported = provider_finished_and_reported(response)
        if call_id and reported is not None:
            if len(self._stream_evidence) >= _MAX_STREAM_EVIDENCE:
                self._stream_evidence.pop(next(iter(self._stream_evidence)))
            self._stream_evidence[call_id] = reported

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """Settle what the deployment hook never sees: finished streams and cache hits."""
        self._schedule_flush()
        reported = self._stream_evidence.get(str(kwargs.get("litellm_call_id") or ""))
        reservation = self._reservation_for(kwargs)
        if reservation is None:
            self._forget_call(kwargs)
            return
        if kwargs.get("cache_hit"):
            # Answered from LiteLLM's cache: nothing reached the provider.
            await asyncio.to_thread(
                self.finops.settle_success,
                reservation,
                actual_usd=0.0,
                usage_evidence={"cache_hit": True, "cost_confidence": "exact"},
            )
        elif client_disconnected(kwargs):
            # The usage here is LiteLLM's count of what arrived before the client
            # left. The provider may have kept generating, so it isn't the charge.
            await asyncio.to_thread(
                self.finops.settle_cancel_or_pending, reservation, dispatched=True, error="client_disconnected"
            )
        elif kwargs.get("stream") and reported is False:
            # The usage here is LiteLLM's tokenizer count, not a provider's report.
            await asyncio.to_thread(
                self.finops.settle_cancel_or_pending, reservation, dispatched=True, error="missing_usage"
            )
        else:
            await self._settle_usage(
                reservation, kwargs, response_obj, unverified=bool(kwargs.get("stream")) and reported is None
            )
        self._forget_call(kwargs)

    async def async_log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """A stream that failed after dispatch stays pending. Other failures have the deployment hook."""
        if not kwargs.get("stream"):
            return
        reservation = self._reservation_for(kwargs)
        if reservation is None:
            return
        exception = kwargs.get("exception")
        await asyncio.to_thread(
            self.finops.settle_cancel_or_pending,
            reservation,
            dispatched=True,
            error=f"stream_failed:{type(exception).__name__ if exception else 'unknown'}",
        )
        self._forget_call(kwargs)

    async def _settle_usage(
        self, reservation: Any, request: Mapping[str, Any], response: Any, *, unverified: bool = False
    ) -> None:
        model = str(request.get("model") or reservation.meta.get("model") or "unknown")
        prompt, completion, total = extract_usage_tokens(response)
        if total <= 0:
            # Unknown usage → pending, never false-zero settle.
            await asyncio.to_thread(
                self.finops.settle_cancel_or_pending, reservation, dispatched=True, error="missing_usage"
            )
            return
        evidence: dict[str, Any] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "model": model,
            "cost_confidence": "estimated",
        }
        if unverified:
            # This stream keeps no chunks to check, so LiteLLM's usage can't be confirmed.
            evidence["usage_source"] = "unverified"
        await asyncio.to_thread(
            self.finops.settle_success,
            reservation,
            actual_usd=estimate_cost_from_response(
                model=model,
                response=response,
                price_multiplier=Decimal(str(reservation.meta.get("price_multiplier") or "1")),
            ),
            usage_evidence=evidence,
        )

    def _forget_call(self, request: Mapping[str, Any]) -> None:
        call_id = str(request.get("litellm_call_id") or "")
        self._by_litellm_call.pop(call_id, None)
        self._stream_evidence.pop(call_id, None)

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
            await asyncio.to_thread(
                self.finops.settle_success,
                reservation,
                actual_usd=0.0,
                usage_evidence={
                    "provider_rejected": True,
                    "status_code": getattr(exception, "status_code", None),
                    "cost_confidence": "exact",
                },
            )
            self._forget_call(request_data)
            return
        # Provider may have processed the request even if the response was lost.
        dispatched = not isinstance(exception, GatewayAdmissionDenied)
        await asyncio.to_thread(
            self.finops.settle_cancel_or_pending,
            reservation,
            dispatched=dispatched,
            error=f"{type(exception).__name__}:{exception}",
        )
        self._forget_call(request_data)

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
