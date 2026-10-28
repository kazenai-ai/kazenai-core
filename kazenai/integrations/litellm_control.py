"""LiteLLM gateway Control helpers.

Runs in the *gateway* process (or hermetic tests). Does not import LiteLLM.
Application packages must not depend on LiteLLM internals; load the gateway
CustomLogger, ``kazenai.integrations.litellm_gateway``, inside the pinned gateway.

Two modes:
  - application-owned: SDK reserves/settles; gateway retries/fallbacks off
  - gateway-owned: this module reserves/marks start/settles per deployment attempt
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..spine import transport as _transport


LIFECYCLE_VERSION = "reservation.lifecycle.v1"
GATEWAY_OWNER = "gateway"
CONTROL_ATTEMPT_META_KEY = "kazenai_control_attempt"
ACCOUNTING_OWNER_META_KEY = "kazenai_budget_accounting_owner"


class GatewayAdmissionDenied(RuntimeError):
    """Raised from deployment pre-call hooks to block provider network dispatch."""

    def __init__(self, message: str, *, code: str = "admission_denied", http_status: int = 402) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = int(http_status)


@dataclass
class AttemptIdentity:
    operation_id: str
    call_id: str
    attempt_id: str
    attempt: int
    idempotency_key: str


@dataclass
class AttemptReservation:
    identity: AttemptIdentity
    reservation_id: str
    decision_id: str = ""
    estimated_usd_micros: int = 0
    status: str = "reserved"  # reserved | started | settled | pending | cancelled
    settled_usd_micros: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)


def new_attempt_identity(
    *,
    operation_id: str | None = None,
    call_id: str | None = None,
    attempt: int = 1,
    attempt_id: str | None = None,
) -> AttemptIdentity:
    op = (operation_id or "").strip() or f"op_{uuid.uuid4().hex}"
    cid = (call_id or "").strip() or f"call_{uuid.uuid4().hex}"
    att = max(1, int(attempt))
    aid = (attempt_id or "").strip() or f"att_{uuid.uuid4().hex}"
    idem = hashlib.sha256(f"{op}:{cid}:{aid}:{att}".encode("utf-8")).hexdigest()
    return AttemptIdentity(
        operation_id=op,
        call_id=cid,
        attempt_id=aid,
        attempt=att,
        idempotency_key=idem,
    )


def usd_to_micros_ceil(usd: float) -> int:
    value = max(0.0, float(usd))
    micros = int(value * 1_000_000 + 0.999999)
    return max(1, micros) if value > 0 else 0


def micros_to_usd(micros: int) -> float:
    return float(int(micros)) / 1_000_000.0


def extract_usage_tokens(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if usage is None and hasattr(response, "model_dump"):
        try:
            dumped = response.model_dump()
            if isinstance(dumped, Mapping):
                usage = dumped.get("usage")
        except Exception:
            usage = None
    if isinstance(usage, Mapping):
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        return prompt, completion, total
    if usage is not None:
        prompt = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
        completion = int(
            getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
        )
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
        return prompt, completion, total
    return 0, 0, 0


def _canonical_model_name(model: str) -> str:
    """Strip LiteLLM provider prefixes (``openai/gpt-4o-mini`` → ``gpt-4o-mini``)."""
    text = (model or "").strip()
    if "/" in text:
        text = text.rsplit("/", 1)[-1].strip()
    return text or "unknown"


def estimate_cost_usd(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    audio_prompt_tokens: int | None = 0,
) -> float | None:
    """Authoritative local price when known; ``None`` when pricing is unavailable.

    Missing authoritative price must not become an invented final charge.
    Callers should settle ``outcome_unknown`` / pending when this returns None.
    LiteLLM provider prefixes are stripped before lookup. A ``$0.00`` result for
    an unrecognized model is treated as unavailable pricing (not a free call).

    ``audio_prompt_tokens`` is the audio share of ``prompt_tokens``. Gemini 2.5
    bills audio input above text, so it is priced at the audio rate. ``None``
    means unknown: every prompt token is priced at the higher of the two rates
    (a ceiling, used for holds). Settle with :func:`estimate_cost_from_response`,
    which reads the audio share from the provider usage.
    """
    try:
        from kazenai.cost_engine import TokenCostEngine
        from kazenai.price_catalog import cost_micro_usd, micro_usd_to_decimal
        from kazenai.pricing_policy import resolve_model_key

        engine = TokenCostEngine()
        candidates = []
        raw = (model or "").strip() or "unknown"
        canon = _canonical_model_name(raw)
        for candidate in (canon, raw):
            if candidate not in candidates:
                candidates.append(candidate)
        prompt = max(0, int(prompt_tokens))
        for candidate in candidates:
            known = resolve_model_key(candidate, engine.known_models(), {})
            if known is None:
                continue
            row = engine._row(candidate)
            if row is not None:
                rates = row.rates
                if audio_prompt_tokens is None:
                    audio = prompt if rates.audio_input is not None and rates.audio_input > rates.input else 0
                else:
                    audio = min(prompt, max(0, int(audio_prompt_tokens)))
                micros = cost_micro_usd(
                    rates,
                    input_tokens=prompt - audio,
                    audio_input_tokens=audio,
                    output_tokens=max(0, int(completion_tokens)),
                )
                return float(micro_usd_to_decimal(micros))
            cost = engine.cost_usd(
                model=candidate,
                input_tokens=prompt,
                output_tokens=int(completion_tokens),
            )
            if cost is None:
                continue
            return float(cost)
        return None
    except Exception:
        return None


def extract_audio_prompt_tokens(response: Any) -> int:
    """Audio share of the prompt from provider usage (0 when not reported)."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if usage is None:
        return 0
    from kazenai.request_pricing import usage_components

    return int(usage_components(usage).get("audio_input_tokens") or 0)


def estimate_cost_from_response(*, model: str, response: Any) -> float | None:
    """Settlement price from a provider response, audio input included."""
    prompt, completion, _total = extract_usage_tokens(response)
    return estimate_cost_usd(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        audio_prompt_tokens=extract_audio_prompt_tokens(response),
    )


def estimate_reserve_usd(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    audio_prompt_tokens: int | None = None,
) -> float:
    """Conservative pre-dispatch hold. Never used as a final settlement amount.

    Without ``audio_prompt_tokens`` the hold assumes the prompt may be audio and
    uses the higher audio rate where the model publishes one. Pass ``0`` for a
    text-only request to hold at the text rate.
    """
    priced = estimate_cost_usd(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        audio_prompt_tokens=audio_prompt_tokens,
    )
    if priced is not None and priced > 0:
        return float(priced)
    # Bound only — evidence for settle still requires authoritative usage pricing.
    return max(0.01, (int(prompt_tokens) + int(completion_tokens)) * 0.000002)


def gateway_owns_enforcement() -> bool:
    raw = (os.getenv("KAZENAI_ENFORCEMENT_OWNER") or "").strip().lower()
    return raw == GATEWAY_OWNER


def control_attempt_marker(identity: AttemptIdentity) -> dict[str, Any]:
    return {
        CONTROL_ATTEMPT_META_KEY: True,
        ACCOUNTING_OWNER_META_KEY: GATEWAY_OWNER,
        "kazenai_operation_id": identity.operation_id,
        "kazenai_call_id": identity.call_id,
        "kazenai_attempt_id": identity.attempt_id,
        "kazenai_attempt_number": identity.attempt,
    }


def is_control_owned_callback(payload: Mapping[str, Any]) -> bool:
    """True when legacy gateway callback must not charge again for Control attempts."""
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    litellm_params = payload.get("litellm_params") if isinstance(payload.get("litellm_params"), Mapping) else {}
    nested = litellm_params.get("metadata") if isinstance(litellm_params.get("metadata"), Mapping) else {}
    for block in (payload, meta, nested):
        if not isinstance(block, Mapping):
            continue
        if block.get(CONTROL_ATTEMPT_META_KEY) is True:
            return True
        owner = str(block.get(ACCOUNTING_OWNER_META_KEY) or block.get("kazenai_budget_accounting_owner") or "").lower()
        if owner == GATEWAY_OWNER:
            return True
        if block.get("kazenai_attempt_id") and owner in ("", GATEWAY_OWNER):
            # attempt_id alone is weak; require owner or explicit control marker above
            if block.get(CONTROL_ATTEMPT_META_KEY) or owner == GATEWAY_OWNER:
                return True
    return bool(payload.get(CONTROL_ATTEMPT_META_KEY) is True)


def _gateway_pending_dir() -> str:
    explicit = (os.getenv("KAZENAI_GATEWAY_PENDING_DIR") or "").strip()
    if explicit:
        return explicit
    home = os.path.join(os.path.expanduser("~"), ".kazenai", "gateway-pending")
    try:
        os.makedirs(home, exist_ok=True)
        return home
    except OSError:
        import tempfile

        fallback = os.path.join(tempfile.gettempdir(), "kazenai-gateway-pending")
        os.makedirs(fallback, exist_ok=True)
        return fallback


class DurableGatewayStore:
    """File-backed open-attempt + pending-settlement evidence.

    Survives process restart. Not a substitute for FinOps/PostgreSQL authority —
    it only preserves delivery evidence until FinOps acknowledges.
    """

    def __init__(self, root: str | None = None) -> None:
        self.root = root or _gateway_pending_dir()
        self.open_dir = os.path.join(self.root, "open")
        self.pending_dir = os.path.join(self.root, "pending")
        os.makedirs(self.open_dir, exist_ok=True)
        os.makedirs(self.pending_dir, exist_ok=True)

    def _open_path(self, attempt_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in attempt_id)[:180]
        return os.path.join(self.open_dir, f"{safe}.json")

    def _pending_path(self, idempotency_key: str) -> str:
        digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:40]
        return os.path.join(self.pending_dir, f"{digest}.json")

    def put_open(self, reservation: AttemptReservation) -> None:
        path = self._open_path(reservation.identity.attempt_id)
        payload = {
            "reservation_id": reservation.reservation_id,
            "decision_id": reservation.decision_id,
            "estimated_usd_micros": reservation.estimated_usd_micros,
            "status": reservation.status,
            "settled_usd_micros": reservation.settled_usd_micros,
            "meta": reservation.meta,
            "identity": {
                "operation_id": reservation.identity.operation_id,
                "call_id": reservation.identity.call_id,
                "attempt_id": reservation.identity.attempt_id,
                "attempt": reservation.identity.attempt,
                "idempotency_key": reservation.identity.idempotency_key,
            },
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
        os.replace(tmp, path)

    def get_open(self, attempt_id: str) -> Optional[AttemptReservation]:
        path = self._open_path(attempt_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            ident = raw.get("identity") or {}
            return AttemptReservation(
                identity=AttemptIdentity(
                    operation_id=str(ident.get("operation_id") or ""),
                    call_id=str(ident.get("call_id") or ""),
                    attempt_id=str(ident.get("attempt_id") or attempt_id),
                    attempt=int(ident.get("attempt") or 1),
                    idempotency_key=str(ident.get("idempotency_key") or ""),
                ),
                reservation_id=str(raw.get("reservation_id") or ""),
                decision_id=str(raw.get("decision_id") or ""),
                estimated_usd_micros=int(raw.get("estimated_usd_micros") or 0),
                status=str(raw.get("status") or "reserved"),
                settled_usd_micros=raw.get("settled_usd_micros"),
                meta=dict(raw.get("meta") or {}),
            )
        except Exception:
            return None

    def drop_open(self, attempt_id: str) -> None:
        path = self._open_path(attempt_id)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def enqueue_pending(self, *, body: Mapping[str, Any], error: str = "") -> None:
        key = str(body.get("idempotency_key") or uuid.uuid4().hex)
        path = self._pending_path(key)
        payload = {
            "body": dict(body),
            "error": error,
            "enqueued_at": time.time(),
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
        os.replace(tmp, path)

    def list_pending(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            names = os.listdir(self.pending_dir)
        except FileNotFoundError:
            return out
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.pending_dir, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    raw["_path"] = path
                    out.append(raw)
            except Exception:
                continue
        return out

    def ack_pending(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception:
            pass


class GatewayFinOpsClient:
    """Authenticated FinOps lifecycle client for gateway-owned attempts."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        org_id: str | None = None,
        workspace_id: str | None = None,
        timeout_s: float = 5.0,
        durable_store: DurableGatewayStore | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("KAZENAI_FINOPS_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("KAZENAI_FINOPS_API_KEY") or os.getenv("KAZENAI_API_KEY") or ""
        self.org_id = org_id or os.getenv("KAZENAI_ORG_ID") or ""
        self.workspace_id = workspace_id or os.getenv("KAZENAI_WORKSPACE_ID") or "default"
        self.timeout_s = float(timeout_s)
        # Process cache + durable file store.
        self._open: dict[str, AttemptReservation] = {}
        self.store = durable_store or DurableGatewayStore()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.org_id:
            headers["X-Org-Id"] = self.org_id
            # Service credentials assert tenant via act-as (FinOps org binding).
            headers["X-Kazen-Act-As-Org"] = self.org_id
        return headers

    def _post(self, path: str, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if not self.base_url:
            raise GatewayAdmissionDenied("KAZENAI_FINOPS_URL not configured", code="finops_unconfigured", http_status=503)
        data = json.dumps(dict(body)).encode("utf-8")
        req = urllib_request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=self._headers(),
            method="POST",
        )
        try:
            with _transport.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                payload = json.loads(raw) if raw else {}
                return int(resp.status), payload if isinstance(payload, dict) else {}
        except urllib_error.HTTPError as exc:
            detail: dict[str, Any] = {}
            try:
                detail = json.loads(exc.read().decode("utf-8"))
                if isinstance(detail, dict) and isinstance(detail.get("detail"), dict):
                    detail = detail["detail"]
            except Exception:
                detail = {"reason": str(exc)}
            return int(exc.code), detail if isinstance(detail, dict) else {"reason": str(exc)}
        except Exception as exc:
            raise GatewayAdmissionDenied(
                f"FinOps unavailable: {exc}", code="finops_unavailable", http_status=503
            ) from exc

    def reserve_attempt(
        self,
        *,
        identity: AttemptIdentity,
        model: str,
        estimated_usd: float,
        run_id: str = "",
        feature: str = "litellm_gateway",
    ) -> AttemptReservation:
        micros = usd_to_micros_ceil(estimated_usd)
        body = {
            "lifecycle_version": LIFECYCLE_VERSION,
            "call_id": identity.call_id,
            "attempt": identity.attempt,
            "idempotency_key": identity.idempotency_key,
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "estimated_usd_micros": micros,
            "run_id": run_id or identity.operation_id,
            "feature": feature,
            "operation_id": identity.operation_id,
            "attempt_id": identity.attempt_id,
            "workflow_id": identity.operation_id,
        }
        status, payload = self._post("/v1/budget/reserve", body)
        if status == 402 or payload.get("allowed") is False:
            reason = payload.get("reason") or payload.get("error_code") or "budget exhausted"
            raise GatewayAdmissionDenied(f"gateway admission denied: {reason}", code="budget_denied", http_status=402)
        if status >= 400 or not payload.get("reservation_id"):
            reason = payload.get("reason") or payload.get("error_code") or f"HTTP {status}"
            raise GatewayAdmissionDenied(
                f"gateway reserve failed: {reason}", code="reserve_failed", http_status=max(status, 500)
            )
        reservation = AttemptReservation(
            identity=identity,
            reservation_id=str(payload["reservation_id"]),
            decision_id=str(payload.get("decision_id") or ""),
            estimated_usd_micros=micros,
            status="reserved",
            meta={"model": model},
        )
        self._open[identity.attempt_id] = reservation
        try:
            self.store.put_open(reservation)
        except Exception:
            pass
        return reservation

    @staticmethod
    def _result_code(payload: Mapping[str, Any]) -> str:
        return str(payload.get("result") or payload.get("status") or "").strip().lower()

    def mark_start(self, reservation: AttemptReservation) -> None:
        body = {
            "lifecycle_version": LIFECYCLE_VERSION,
            "reservation_id": reservation.reservation_id,
            "call_id": reservation.identity.call_id,
            "attempt": reservation.identity.attempt,
            "idempotency_key": f"{reservation.identity.idempotency_key}:provider_started",
            "event": "provider_started",
            "org_id": self.org_id,
        }
        status, payload = self._post("/v1/budget/settle", body)
        code = self._result_code(payload)
        if status >= 400 or code not in {"applied", "duplicate"}:
            raise GatewayAdmissionDenied(
                f"gateway start mark failed: {payload.get('error') or payload.get('error_code') or status}",
                code="start_failed",
                http_status=503,
            )
        reservation.status = "started"
        try:
            self.store.put_open(reservation)
        except Exception:
            pass

    def settle_success(
        self,
        reservation: AttemptReservation,
        *,
        actual_usd: float | None,
        usage_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Unknown price → durable pending, never invent a final charge.
        if actual_usd is None:
            return self.settle_cancel_or_pending(
                reservation,
                dispatched=True,
                error="authoritative_price_unavailable",
            )
        micros = usd_to_micros_ceil(actual_usd)
        body = {
            "lifecycle_version": LIFECYCLE_VERSION,
            "reservation_id": reservation.reservation_id,
            "call_id": reservation.identity.call_id,
            "attempt": reservation.identity.attempt,
            "idempotency_key": f"{reservation.identity.idempotency_key}:usage_known",
            "event": "usage_known",
            "org_id": self.org_id,
            "actual_usd_micros": micros,
            "usage_evidence": dict(usage_evidence or {}),
        }
        # Persist exact settlement intent *before* the network call so a
        # transport raise or HTTP 5xx cannot erase the known amount.
        reservation.status = "pending"
        reservation.settled_usd_micros = micros
        try:
            self.store.put_open(reservation)
            self.store.enqueue_pending(body=body, error="pre_send_usage_known")
        except Exception:
            pass
        try:
            status, payload = self._post("/v1/budget/settle", body)
        except Exception as exc:
            self._persist_delivery(reservation, body=body, error=f"transport:{type(exc).__name__}")
            return {"result": "pending", "status": "pending", "error": str(exc)}
        code = self._result_code(payload)
        # HTTP 200 + result=pending is NOT a settlement acknowledgment.
        if status >= 400 or code not in {"applied", "duplicate"}:
            self._persist_delivery(
                reservation,
                body=body,
                error=str(payload.get("error") or code or status),
            )
            return {"result": "pending", "status": "pending", "http_status": status, **payload}
        reservation.status = "settled"
        reservation.settled_usd_micros = micros
        try:
            self.store.drop_open(reservation.identity.attempt_id)
            # Ack the pre-send queue on durable success.
            for item in self.store.list_pending():
                b = item.get("body") if isinstance(item.get("body"), dict) else {}
                if str(b.get("idempotency_key") or "") == body["idempotency_key"]:
                    self.store.ack_pending(str(item.get("_path") or ""))
        except Exception:
            pass
        self._open.pop(reservation.identity.attempt_id, None)
        return {"result": code, **payload}

    def settle_cancel_or_pending(
        self,
        reservation: AttemptReservation,
        *,
        dispatched: bool,
        error: str = "",
    ) -> dict[str, Any]:
        event = "outcome_unknown" if dispatched else "cancel_before_call"
        body = {
            "lifecycle_version": LIFECYCLE_VERSION,
            "reservation_id": reservation.reservation_id,
            "call_id": reservation.identity.call_id,
            "attempt": reservation.identity.attempt,
            "idempotency_key": f"{reservation.identity.idempotency_key}:{event}",
            "event": event,
            "org_id": self.org_id,
            "usage_evidence": {"error": error, "dispatched": dispatched},
        }
        reservation.status = "pending" if dispatched else "cancelled"
        # Persist intent before send.
        try:
            self.store.put_open(reservation)
            self.store.enqueue_pending(body=body, error=error or "pre_send")
        except Exception:
            pass
        try:
            status, payload = self._post("/v1/budget/settle", body)
        except Exception as exc:
            self._persist_delivery(reservation, body=body, error=f"transport:{type(exc).__name__}")
            # Transport failure is never a financial acknowledgment — even for cancel.
            return {"result": "pending", "status": "pending", "error": str(exc)}
        code = self._result_code(payload)
        # Only applied/duplicate acknowledge cancel or unknown; HTTP 200 + pending
        # keeps the durable queue entry for retry.
        if status >= 400 or code not in {"applied", "duplicate"}:
            self._persist_delivery(reservation, body=body, error=error or str(code or status))
            return {"result": "pending", "status": "pending", "http_status": status, **payload}
        # Durable cancel/unknown ack: drop open only for cancel_before_call.
        if not dispatched:
            try:
                self.store.drop_open(reservation.identity.attempt_id)
                for item in self.store.list_pending():
                    b = item.get("body") if isinstance(item.get("body"), dict) else {}
                    if str(b.get("idempotency_key") or "") == body["idempotency_key"]:
                        self.store.ack_pending(str(item.get("_path") or ""))
            except Exception:
                pass
            self._open.pop(reservation.identity.attempt_id, None)
        return {"result": code, **payload}

    def _persist_delivery(
        self,
        reservation: AttemptReservation,
        *,
        body: Mapping[str, Any],
        error: str,
    ) -> None:
        """Ensure durable queue retains the exact settlement body."""
        try:
            self.store.put_open(reservation)
            self.store.enqueue_pending(body=body, error=error)
        except Exception:
            pass

    def _signal_pending(self, reservation: AttemptReservation, *, error: str) -> None:
        body = {
            "lifecycle_version": LIFECYCLE_VERSION,
            "reservation_id": reservation.reservation_id,
            "call_id": reservation.identity.call_id,
            "attempt": reservation.identity.attempt,
            "idempotency_key": f"{reservation.identity.idempotency_key}:outcome_unknown_signal",
            "event": "outcome_unknown",
            "org_id": self.org_id,
            "usage_evidence": {"error": error},
        }
        self._persist_delivery(reservation, body=body, error=error)
        try:
            status, payload = self._post("/v1/budget/settle", body)
            code = self._result_code(payload)
            # pending is a server state mark, not a delivery ack — keep the queue.
            if status < 400 and code in {"applied", "duplicate"}:
                return
        except Exception:
            pass

    def flush_pending(self) -> dict[str, int]:
        """Redeliver durable pending settlement bodies after restart.

        Only ``applied`` / ``duplicate`` acknowledge a queued delivery. HTTP 200 with
        ``result=pending`` must keep the queue entry and open reservation so the
        exact known amount can be retried.
        """
        applied = 0
        failed = 0
        for item in self.store.list_pending():
            body = item.get("body") if isinstance(item.get("body"), dict) else None
            path = str(item.get("_path") or "")
            if not body or not path:
                continue
            try:
                status, payload = self._post("/v1/budget/settle", body)
                code = self._result_code(payload)
                if status < 400 and code in {"applied", "duplicate"}:
                    self.store.ack_pending(path)
                    # Finalize matching open record when usage_known settles.
                    event = str(body.get("event") or "")
                    rid_call = str(body.get("call_id") or "")
                    if event == "usage_known":
                        for attempt_id, open_rsv in list(self._open.items()):
                            if (
                                open_rsv.reservation_id == body.get("reservation_id")
                                and open_rsv.identity.call_id == rid_call
                            ):
                                open_rsv.status = "settled"
                                if body.get("actual_usd_micros") is not None:
                                    open_rsv.settled_usd_micros = int(body["actual_usd_micros"])
                                try:
                                    self.store.drop_open(attempt_id)
                                except Exception:
                                    pass
                                self._open.pop(attempt_id, None)
                    applied += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        return {"applied": applied, "failed": failed}

    def get_open(self, attempt_id: str) -> Optional[AttemptReservation]:
        hit = self._open.get(attempt_id)
        if hit is not None:
            return hit
        restored = self.store.get_open(attempt_id)
        if restored is not None:
            self._open[attempt_id] = restored
        return restored

    def bind_kwargs_metadata(self, kwargs: MutableMapping[str, Any], reservation: AttemptReservation) -> None:
        marker = control_attempt_marker(reservation.identity)
        marker["kazenai_reservation_id"] = reservation.reservation_id
        marker["kazenai_decision_id"] = reservation.decision_id
        # Deployment-hook kwargs become the provider request. Never add a
        # top-level ``litellm_params`` key: real providers reject unknown
        # fields ("Unknown parameter: 'litellm_params'"). Annotate it only
        # when LiteLLM already supplies it; ``metadata`` is a LiteLLM field.
        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            meta = litellm_params.setdefault("metadata", {})
            if isinstance(meta, dict):
                meta.update(marker)
        kwargs.setdefault("metadata", {})
        if isinstance(kwargs["metadata"], dict):
            kwargs["metadata"].update(marker)


def resolve_attempt_from_kwargs(kwargs: Mapping[str, Any], *, attempt_number: int | None = None) -> AttemptIdentity:
    meta: dict[str, Any] = {}
    for key in ("metadata",):
        block = kwargs.get(key)
        if isinstance(block, Mapping):
            meta.update(dict(block))
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, Mapping):
        nested = litellm_params.get("metadata")
        if isinstance(nested, Mapping):
            meta.update(dict(nested))
    op = str(meta.get("kazenai_operation_id") or meta.get("operation_id") or kwargs.get("kazenai_operation_id") or "")
    call = str(meta.get("kazenai_call_id") or meta.get("call_id") or "")
    att_id = str(meta.get("kazenai_attempt_id") or "")
    # Prefer explicit attempt number; never treat fallback_depth as unique attempt id.
    att_num = attempt_number
    if att_num is None:
        raw = meta.get("kazenai_attempt_number") or kwargs.get("attempt") or 1
        try:
            att_num = int(raw)
        except (TypeError, ValueError):
            att_num = 1
    # New attempt_id per deployment attempt when not pre-bound.
    return new_attempt_identity(operation_id=op or None, call_id=call or None, attempt=att_num, attempt_id=att_id or None)


def inject_traceparent(kwargs: MutableMapping[str, Any], traceparent: str | None) -> None:
    if not traceparent:
        return
    headers = kwargs.setdefault("headers", {})
    if not isinstance(headers, dict):
        return
    headers.setdefault("traceparent", traceparent)
    # Allowlisted correlation only — never put secrets in baggage.
    meta = kwargs.setdefault("metadata", {})
    if isinstance(meta, dict):
        meta.setdefault("kazenai_traceparent", traceparent)
