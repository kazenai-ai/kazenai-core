"""LangChain integration — callback handler + chain wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..monitor import monitor
from ..schema import KazenEvent, new_id, now_ms


class KazenCallbackHandler:
    """Minimal LangChain-compatible callback (no hard langchain import)."""

    def __init__(
        self,
        *,
        org_id: str = "local",
        project_id: str = "default",
        agent_id: str = "langchain",
        agent_role: str = "langchain",
        run_id: str = "",
        on_event: Optional[Any] = None,
    ) -> None:
        self.org_id = org_id
        self.project_id = project_id
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.run_id = run_id or new_id()
        self._on_event = on_event
        self._parent_step_id: Optional[str] = None

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage = getattr(response, "llm_output", None) or {}
        if isinstance(usage, dict):
            token_usage = usage.get("token_usage") or {}
        else:
            token_usage = {}
        ev = KazenEvent(
            schema_version="1.0",
            ts_ms=now_ms(),
            event_id=new_id(),
            org_id=self.org_id,
            project_id=self.project_id,
            surface="kazenai-core",
            agent_id=self.agent_id,
            agent_role=self.agent_role,
            run_id=self.run_id,
            step_id=new_id(),
            parent_step_id=self._parent_step_id,
            event_type="model.call",
            tokens_used=int(token_usage.get("total_tokens") or 0) if token_usage else None,
            cost_usd=None,
            payload={"framework": "langchain", "kwargs": {k: str(v)[:200] for k, v in list(kwargs.items())[:5]}},
        )
        if self._on_event:
            try:
                self._on_event(ev)
            except Exception:
                pass


def wrap_langchain_runnable(runnable: Any, **monitor_kwargs: Any) -> Any:
    """Wrap any object with monitor() for budget/loop enforcement."""
    return monitor(runnable, **monitor_kwargs)
