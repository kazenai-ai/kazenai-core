"""mem0-compatible memory SDK surface — maps to Brain ingest/retrieve/think.

Prefer ``think()`` for user/team Q&A (cited answer + sources disclosure).
Keep ``search()`` for RAG/context packs only.

For the full company-brain thin client, prefer ``kazenai_brain.Memory`` when available.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx


class KazenMemory:
    """Drop-in shaped like mem0: add(), search(), and think()."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        org_id: str = "default",
        workspace_id: str = "default",
        user_scope: str = "",
    ) -> None:
        self.base_url = (base_url or os.getenv("KAZENAI_BRAIN_URL", "http://127.0.0.1:8790")).rstrip("/")
        self.api_key = api_key or os.getenv("KAZENAI_BRAIN_API_KEY", "")
        self.org_id = org_id or os.getenv("KAZENAI_BRAIN_ORG_ID", "default")
        self.workspace_id = workspace_id or os.getenv("KAZENAI_BRAIN_WORKSPACE_ID", "default")
        self.user_scope = (user_scope or "").strip()

    def _headers(self, *, actor: str = "") -> Dict[str, str]:
        h: Dict[str, str] = {
            "Content-Type": "application/json",
            "X-Kazen-Org-Id": self.org_id,
            "X-Kazen-Workspace-Id": self.workspace_id,
        }
        if self.api_key:
            if self.api_key.count(".") == 2:
                h["Authorization"] = f"Bearer {self.api_key}"
            else:
                h["X-Kazenai-Api-Key"] = self.api_key
                h["X-API-Key"] = self.api_key
                h["Authorization"] = f"Bearer {self.api_key}"
        scoped = (actor or self.user_scope or "").strip()
        if scoped and self.api_key:
            h["X-Kazen-User-Scope"] = scoped
        return h

    def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        content = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        payload = {
            "content": content,
            "source_type": (metadata or {}).get("source_type", "idea_raw"),
            "author": user_id,
            "structured_fields": {"user_id": user_id, **(metadata or {})},
        }
        schema_id = (metadata or {}).get("schema_id")
        if schema_id:
            payload["schema_id"] = schema_id
            payload["entity_id"] = str((metadata or {}).get("entity_id") or user_id)
            payload["governance_label"] = str((metadata or {}).get("governance_label") or "standard")
        resp = httpx.post(
            f"{self.base_url}/v1/ingest",
            json=payload,
            headers=self._headers(actor=user_id),
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Context-pack retrieve via ``POST /v1/retrieve`` (not for user-facing answers)."""
        resp = httpx.post(
            f"{self.base_url}/v1/retrieve",
            json={"query": query, "caller": user_id, "top_k": limit},
            headers=self._headers(actor=user_id),
            timeout=30.0,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("abstain"):
            return []
        return list(body.get("chunks") or [])

    def think(
        self,
        text: str,
        user_id: str,
        *,
        top_k: int = 8,
        rounds: int = 1,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Company-brain Q&A via ``POST /v1/think`` (prefer over ``search`` for answers).

        Returns cited answer payload including ``sources_used`` / ``sources_excluded``
        when Sources exist.
        """
        payload: Dict[str, Any] = {
            "text": text,
            "user_id": user_id,
            "top_k": top_k,
            "rounds": rounds,
            **extra,
        }
        resp = httpx.post(
            f"{self.base_url}/v1/think",
            json=payload,
            headers=self._headers(actor=user_id),
            timeout=60.0,
        )
        resp.raise_for_status()
        body = resp.json()
        return body if isinstance(body, dict) else {"data": body}
