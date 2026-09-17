"""Conductor access_list context fencing (Loop 24 P1-7)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

AccessSpec = Union[Sequence[int], str]


def normalize_access_list(access_list: AccessSpec | None) -> List[int] | str:
    """Normalize access_list semantics for worker context fencing."""
    if access_list is None:
        return []
    if isinstance(access_list, str):
        token = access_list.strip().lower()
        if token == "all":
            return "all"
        return []
    indices: List[int] = []
    for raw in access_list:
        try:
            indices.append(int(raw))
        except (TypeError, ValueError):
            continue
    return sorted(set(i for i in indices if i >= 0))


def build_worker_context(
    access_list: AccessSpec | None,
    prior_steps: Sequence[Dict[str, Any]],
    *,
    messages: Sequence[Dict[str, Any]] | None = None,
) -> List[Dict[str, str]]:
    """Return messages visible to a worker at the current step.

    Semantics (Conductor communication topology):
    - ``[]`` — no prior step context
    - ``"all"`` or ``["all"]`` — full prior step transcript
    - ``[0, 2]`` — only steps at those indices
    """
    normalized = normalize_access_list(access_list)
    base_messages: List[Dict[str, str]] = []
    if messages:
        base_messages = [
            {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
            for m in messages
            if str(m.get("content") or "").strip()
        ]

    if normalized == "all":
        fenced = list(base_messages)
        for step in prior_steps:
            text = str(step.get("content") or step.get("subtask") or step.get("output") or "")
            if text.strip():
                fenced.append({"role": "assistant", "content": text})
        return fenced

    if not normalized:
        return list(base_messages)

    allowed = set(normalized)
    fenced = list(base_messages)
    for idx, step in enumerate(prior_steps):
        if idx not in allowed:
            continue
        text = str(step.get("content") or step.get("subtask") or step.get("output") or "")
        if text.strip():
            fenced.append({"role": "assistant", "content": text})
    return fenced
