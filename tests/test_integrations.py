from __future__ import annotations

from kazenai.integrations.langchain import KazenCallbackHandler
from kazenai.integrations.crewai import wrap_crew_kickoff
from kazenai.integrations.langgraph import wrap_graph_invoke
from kazenai.retry_queue import RetryQueue


def test_langchain_callback_emits_event() -> None:
    events = []
    h = KazenCallbackHandler(on_event=lambda e: events.append(e))
    h.on_llm_end({"llm_output": {"token_usage": {"total_tokens": 42}}})
    assert len(events) == 1
    assert events[0].event_type == "model.call"


def test_retry_queue_roundtrip(tmp_path) -> None:
    q = RetryQueue(str(tmp_path / "q.sqlite3"))
    q.enqueue({"event_id": "a", "event_type": "model.call"})
    assert q.pending_count() == 1
    batch = q.drain()
    assert len(batch) == 1
    assert q.pending_count() == 0


class _FakeCrew:
    def kickoff(self, inputs=None):
        return {"ok": True}


def test_wrap_crew_kickoff() -> None:
    crew = _FakeCrew()
    fn = wrap_crew_kickoff(crew, agent_id="t", org_id="o", project_id="p")
    assert fn(inputs={}) == {"ok": True}


class _FakeGraph:
    def invoke(self, state):
        return state


def test_wrap_graph_invoke() -> None:
    g = _FakeGraph()
    fn = wrap_graph_invoke(g, agent_id="g", org_id="o", project_id="p")
    assert fn({"x": 1}) == {"x": 1}
