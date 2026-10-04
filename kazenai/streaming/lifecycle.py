"""Finalize-once reservation/settlement lifecycle for one provider stream attempt."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional


class StreamOutcome(str, Enum):
    COMPLETE = "COMPLETE"
    CUTOFF = "CUTOFF"
    CLIENT_CANCELLED = "CLIENT_CANCELLED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class StreamUsage:
    """Provider-reported usage snapshot (authoritative when present)."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw: Any = None
    authoritative: bool = False

    def merged(self, newer: "StreamUsage") -> "StreamUsage":
        """Merge cumulative/partial provider snapshots without losing fields.

        Anthropic sends input usage in ``message_start`` and cumulative output
        usage later.  OpenAI normally sends one final complete snapshot.  For
        either provider the newest non-null field wins.
        """

        def _pick(new_value: Optional[int], old_value: Optional[int]) -> Optional[int]:
            return new_value if new_value is not None else old_value

        prompt = _pick(newer.prompt_tokens, self.prompt_tokens)
        completion = _pick(newer.completion_tokens, self.completion_tokens)
        input_tokens = _pick(newer.input_tokens, self.input_tokens)
        output_tokens = _pick(newer.output_tokens, self.output_tokens)
        effective_input = prompt if prompt is not None else input_tokens
        effective_output = completion if completion is not None else output_tokens
        # Partial providers may first report input + an initial output count,
        # then update output alone. Recompute the aggregate whenever both
        # components are known so a stale earlier total cannot survive.
        total: Optional[int]
        if effective_input is not None and effective_output is not None:
            total = int(effective_input) + int(effective_output)
        else:
            total = _pick(newer.total_tokens, self.total_tokens)

        raw: Any
        if isinstance(self.raw, dict) or isinstance(newer.raw, dict):
            raw = {}
            if isinstance(self.raw, dict):
                raw.update(self.raw)
            if isinstance(newer.raw, dict):
                raw.update(newer.raw)
        else:
            raw = newer.raw if newer.raw is not None else self.raw

        # Having both counters does not by itself prove the snapshot is final.
        # Anthropic's ``message_start`` can contain input tokens plus an initial
        # output count; only a terminal provider snapshot (for example
        # ``message_delta`` or a final Message) makes that usage authoritative.
        complete_counts = bool(
            total is not None
            or (effective_input is not None and effective_output is not None)
        )
        authoritative = bool(
            complete_counts and (self.authoritative or newer.authoritative)
        )
        return StreamUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            raw=raw,
            authoritative=authoritative,
        )

    def as_openai_style(self) -> Optional[Dict[str, int]]:
        if not self.authoritative and self.total_tokens is None and self.prompt_tokens is None:
            return None
        prompt = self.prompt_tokens if self.prompt_tokens is not None else self.input_tokens
        completion = self.completion_tokens if self.completion_tokens is not None else self.output_tokens
        total = self.total_tokens
        if total is None and prompt is not None and completion is not None:
            total = int(prompt) + int(completion)
        out: Dict[str, int] = {}
        if prompt is not None:
            out["prompt_tokens"] = int(prompt)
            out["input_tokens"] = int(prompt)
        if completion is not None:
            out["completion_tokens"] = int(completion)
            out["output_tokens"] = int(completion)
        if total is not None:
            out["total_tokens"] = int(total)
        return out or None


@dataclass
class StreamAttempt:
    """Owns one stream attempt: hold + reservation until a single terminal finalize."""

    surface: str
    step_ctx: Any
    enforcement: Any
    held_projection: float
    reserved_cost_usd: Any
    model: str
    method: str
    usd_per_1k_tokens: float
    capture_mode: Any
    agent_role: str
    event_sink: Any = None
    finops: Any = None
    endpoint_url: Optional[str] = None
    endpoint_headers: Optional[Mapping[str, str]] = None
    stream_cutoff_usd: Optional[float] = None
    pricing_version: str = "kazenai.cost_engine.v1"
    # Captured at stream start so finalize can emit attribution after ContextVar exit.
    attribution: Optional[Mapping[str, Any]] = None
    reconcile_fn: Optional[Callable[..., Any]] = None
    pending_fn: Optional[Callable[..., Any]] = None
    postcall_cost_fn: Optional[Callable[..., Optional[float]]] = None
    reservation_payload_fn: Optional[Callable[[Any], dict]] = None
    emit_event_fn: Optional[Callable[..., None]] = None
    started_perf: float = field(default_factory=time.perf_counter)
    first_byte_perf: Optional[float] = None
    observed_usage: StreamUsage = field(default_factory=StreamUsage)
    estimated_output_tokens: int = 0
    estimated_spend_usd: float = 0.0
    upstream_close_attempted: bool = False
    upstream_close_ok: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _finalized: bool = False
    terminal_outcome: Optional[StreamOutcome] = None
    financial_pending: bool = False
    cost_confidence: str = "unknown"

    def note_first_byte(self) -> None:
        if self.first_byte_perf is None:
            self.first_byte_perf = time.perf_counter()

    def observe_usage(self, usage: StreamUsage) -> None:
        if (
            usage.authoritative
            or usage.total_tokens is not None
            or usage.prompt_tokens is not None
            or usage.completion_tokens is not None
            or usage.input_tokens is not None
            or usage.output_tokens is not None
            or usage.raw is not None
        ):
            self.observed_usage = self.observed_usage.merged(usage)

    def observe_text_estimate(self, text: str) -> None:
        """Local observation only — never mutates shared FinOps spend."""
        if not text:
            return
        tokens = max(1, len(text) // 4)
        self.estimated_output_tokens += tokens
        if self.usd_per_1k_tokens:
            self.estimated_spend_usd += (tokens * float(self.usd_per_1k_tokens)) / 1000.0

    def should_cutoff(self) -> bool:
        if self.stream_cutoff_usd is None:
            return False
        return self.estimated_spend_usd >= float(self.stream_cutoff_usd)

    def mark_upstream_close(self, *, ok: bool) -> None:
        self.upstream_close_attempted = True
        self.upstream_close_ok = bool(ok)

    def finalize(
        self,
        outcome: StreamOutcome,
        *,
        usage: Optional[StreamUsage] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            self.terminal_outcome = outcome
            if usage is not None:
                self.observe_usage(usage)

            auth = self.observed_usage.authoritative
            usage_dict = self.observed_usage.as_openai_style()
            tokens_used = None
            if usage_dict and "total_tokens" in usage_dict:
                tokens_used = int(usage_dict["total_tokens"])

            cost_usd: Optional[float] = None
            if auth and self.postcall_cost_fn is not None:
                cost_usd = self.postcall_cost_fn(
                    model=self.model,
                    usage=self.observed_usage.raw if self.observed_usage.raw is not None else usage_dict,
                    tokens_used=tokens_used,
                    usd_per_1k_tokens=self.usd_per_1k_tokens or None,
                )

            pending = False
            confidence = "exact"
            settle_cost = 0.0
            if outcome in (StreamOutcome.COMPLETE,) and auth and cost_usd is not None:
                settle_cost = float(cost_usd)
                confidence = "exact"
                pending = False
            elif outcome == StreamOutcome.COMPLETE and not auth:
                # Observable deltas are useful for a local cutoff, but they omit
                # input, hidden reasoning and provider-side work.  Never use them
                # as authoritative settlement.
                settle_cost = 0.0
                confidence = "pending"
                pending = True
            else:
                # Cancel / error / cutoff / unknown: never silently exact zero settlement.
                settle_cost = 0.0
                confidence = "pending"
                pending = True

            if self.reserved_cost_usd is not None and self.reconcile_fn is not None and not pending:
                try:
                    settled = self.reconcile_fn(
                        org_id=self.step_ctx.org_id,
                        workspace_id=self.step_ctx.workspace_id,
                        run_id=self.step_ctx.run_id,
                        reserved_cost_usd=self.reserved_cost_usd,
                        actual_cost_usd=settle_cost,
                    )
                except Exception:
                    settled = False
                if settled is False:
                    pending = True
                    confidence = "exact_usage_pending_settlement"

            if self.reserved_cost_usd is not None and self.pending_fn is not None and pending:
                try:
                    self.pending_fn(
                        org_id=self.step_ctx.org_id,
                        workspace_id=self.step_ctx.workspace_id,
                        run_id=self.step_ctx.run_id,
                        reserved_cost_usd=self.reserved_cost_usd,
                    )
                except Exception:
                    pass

            self.financial_pending = pending
            self.cost_confidence = confidence

            # Exact provider usage is valid for the local counter even when the
            # shared settlement delivery remains pending. Unknown usage never
            # becomes an exact local zero.
            recorded = settle_cost if confidence.startswith("exact") else 0.0
            try:
                self.enforcement.record_call(
                    cost_usd=recorded,
                    released_projection=self.held_projection,
                )
            except Exception:
                pass

            if self.emit_event_fn is not None:
                # _emit_stream_model_call already fails open for telemetry
                # transport errors and deliberately re-raises public control
                # exceptions such as KazenCircuitBreaker.  Do not erase those
                # decisions here.
                self.emit_event_fn(self, settle_cost=settle_cost, tokens_used=tokens_used, error=error)

    @property
    def finalized(self) -> bool:
        return self._finalized
