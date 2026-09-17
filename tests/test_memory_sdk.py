"""KazenMemory mem0-shaped SDK tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kazenai.memory import KazenMemory


def test_add_posts_to_brain_ingest():
    mem = KazenMemory(base_url="http://brain:8790", api_key="k", org_id="org-1", workspace_id="ws-1")
    with patch("kazenai.memory.httpx.post") as post:
        post.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        post.return_value.json.return_value = {"node_id": "n1"}
        out = mem.add([{"role": "user", "content": "hello"}], "user-1")
        assert out["node_id"] == "n1"
        assert "v1/ingest" in post.call_args[0][0]
        headers = post.call_args.kwargs["headers"]
        assert headers["X-Kazen-Org-Id"] == "org-1"
        assert headers["X-Kazen-Workspace-Id"] == "ws-1"
        assert headers["X-Kazen-User-Scope"] == "user-1"


def test_search_returns_chunks_with_user_scope():
    mem = KazenMemory(base_url="http://brain:8790", api_key="k")
    with patch("kazenai.memory.httpx.post") as post:
        post.return_value = MagicMock(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"chunks": [{"content": "hit"}]},
        )
        hits = mem.search("query", "user-1", limit=3)
        assert len(hits) == 1
        assert "v1/retrieve" in post.call_args[0][0]
        assert post.call_args.kwargs["headers"]["X-Kazen-User-Scope"] == "user-1"


def test_think_posts_to_brain_think():
    mem = KazenMemory(base_url="http://brain:8790", api_key="k", workspace_id="default")
    with patch("kazenai.memory.httpx.post") as post:
        post.return_value = MagicMock(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "answer": "SLA is 99.9%.",
                "sources_used": ["shared"],
                "sources_excluded": ["internal"],
                "gaps": [],
            },
        )
        out = mem.think("what is the SLA?", "alice", top_k=5)
        assert out["answer"] == "SLA is 99.9%."
        assert out["sources_used"] == ["shared"]
        assert "v1/think" in post.call_args[0][0]
        body = post.call_args.kwargs["json"]
        assert body["text"] == "what is the SLA?"
        assert body["user_id"] == "alice"
        assert post.call_args.kwargs["headers"]["X-Kazen-User-Scope"] == "alice"
