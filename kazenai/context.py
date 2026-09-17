from __future__ import annotations

import contextlib
import contextvars
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional


@dataclass(frozen=True)
class RunContext:
    """
    Async-safe run context propagated via contextvars.

    `parent_step_id` enables nested agent call trees.
    `workspace_id` is the FinOps/billing tenant scope (distinct from project_id).
    """

    org_id: str
    project_id: str
    workspace_id: str
    agent_id: str
    run_id: str
    step_id: str
    parent_step_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(
        *,
        org_id: str,
        project_id: str,
        agent_id: str,
        workspace_id: Optional[str] = None,
        run_id: Optional[str] = None,
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RunContext":
        ws = str(workspace_id or "default").strip() or "default"
        return RunContext(
            org_id=org_id,
            project_id=project_id,
            workspace_id=ws,
            agent_id=agent_id,
            run_id=run_id or uuid.uuid4().hex,
            step_id=step_id or uuid.uuid4().hex,
            parent_step_id=None,
            metadata=metadata or {},
        )

    def child_step(self, *, step_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "RunContext":
        return RunContext(
            org_id=self.org_id,
            project_id=self.project_id,
            workspace_id=self.workspace_id,
            agent_id=self.agent_id,
            run_id=self.run_id,
            step_id=step_id or uuid.uuid4().hex,
            parent_step_id=self.step_id,
            metadata=metadata or {},
        )


_CTX: contextvars.ContextVar[Optional[RunContext]] = contextvars.ContextVar("kazenai_run_context", default=None)


def get_current_context() -> Optional[RunContext]:
    return _CTX.get()


@contextlib.contextmanager
def use_context(ctx: RunContext) -> Iterator[RunContext]:
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)


@dataclass(frozen=True)
class BackgroundTaskContext:
    """Serializable auth/identity context for background workers (Celery, asyncio).

    Background tasks run off the request path and would otherwise execute with no
    tenant/identity context, breaking org-scoped telemetry, billing, and audit.
    Carry this explicitly through the task boundary and re-establish a
    :class:`RunContext` at task entry via :meth:`activate`.
    """

    org_id: str
    workspace_id: str = "default"
    agent_id: str = "background"
    project_id: str = ""
    run_id: Optional[str] = None
    user_id: Optional[str] = None

    _KW_PREFIX = "_kazen_ctx_"

    def to_kwargs(self) -> Dict[str, Any]:
        """Render as JSON-safe kwargs to thread through a Celery task signature."""
        return {
            f"{self._KW_PREFIX}org_id": self.org_id,
            f"{self._KW_PREFIX}workspace_id": self.workspace_id,
            f"{self._KW_PREFIX}agent_id": self.agent_id,
            f"{self._KW_PREFIX}project_id": self.project_id,
            f"{self._KW_PREFIX}run_id": self.run_id,
            f"{self._KW_PREFIX}user_id": self.user_id,
        }

    @classmethod
    def pop_kwargs(cls, kwargs: Dict[str, Any]) -> Optional["BackgroundTaskContext"]:
        """Extract (and remove) the context kwargs from *kwargs*, if present."""
        if f"{cls._KW_PREFIX}org_id" not in kwargs:
            return None
        return cls(
            org_id=str(kwargs.pop(f"{cls._KW_PREFIX}org_id", "") or ""),
            workspace_id=str(kwargs.pop(f"{cls._KW_PREFIX}workspace_id", "default") or "default"),
            agent_id=str(kwargs.pop(f"{cls._KW_PREFIX}agent_id", "background") or "background"),
            project_id=str(kwargs.pop(f"{cls._KW_PREFIX}project_id", "") or ""),
            run_id=kwargs.pop(f"{cls._KW_PREFIX}run_id", None),
            user_id=kwargs.pop(f"{cls._KW_PREFIX}user_id", None),
        )

    @contextlib.contextmanager
    def activate(self) -> Iterator[RunContext]:
        ctx = RunContext.new(
            org_id=self.org_id,
            project_id=self.project_id or self.org_id,
            agent_id=self.agent_id,
            workspace_id=self.workspace_id,
            run_id=self.run_id,
            metadata={"user_id": self.user_id} if self.user_id else {},
        )
        with use_context(ctx) as active:
            yield active
