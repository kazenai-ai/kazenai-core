"""monitor(loop_guard=...): a loop hit is recorded by default, refused only under "block" (kazenai 1.2.3).

The official provider SDKs run on an ``httpx.MockTransport``, so every request that
reaches the "provider" is counted. The scenarios are ordinary traffic the loop guard
used to refuse: repeats, a shared system prompt, an agent's tools list, a growing
chat, content parts, and an app-level retry. Since 1.2.3 the history is kept per
customer, so two customers asking the same thing are not a loop in any mode.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import httpx
import pytest

from kazenai import LoopDetected, monitor, patch_openai, use_attribution
from kazenai.loop_detector import LOOP_GUARD_MODES, resolve_loop_guard

openai = pytest.importorskip("openai")

MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"

SUPPORT_SYSTEM = (
    "You are the customer support assistant for Northwind Outdoor, an online shop that sells "
    "tents, sleeping bags, backpacks and hiking boots. Answer in a friendly, concise tone. "
    "Only answer questions about orders, shipping, returns, sizing and product care. "
    "Orders ship within two business days from our warehouse in Denver. Returns are free "
    "within 30 days if the item is unused and in its original packaging. If you do not know "
    "the answer, say so and offer to connect the customer with a human agent by email. "
    "Never promise refunds or discounts. Never ask for card numbers or passwords."
)
QUESTIONS = [
    "Where is my order?",
    "Can I return boots I wore once?",
    "Do your tents come with stakes?",
    "How do I wash a down sleeping bag?",
    "Do you ship to Canada?",
]
CHAT = [
    ("I want to cook dinner for four tonight. Any ideas?",
     "How about a lemon garlic roast chicken with potatoes and green beans? It takes about an hour."),
    ("Sounds good. What temperature for the chicken?",
     "Roast it at 425F (220C) for roughly 60 to 75 minutes, until the thigh reads 165F."),
    ("Should I brine it first?",
     "A quick dry brine helps: salt the skin a few hours ahead and leave it uncovered in the fridge."),
    ("What about the potatoes?",
     "Cut them into chunks, toss with oil and salt, and roast them in the same pan for the last 45 minutes."),
    ("And the green beans?",
     "Blanch them for three minutes, then saute with butter and the leftover garlic."),
    ("Can I make a sauce from the drippings?",
     "Yes. Skim the fat, add a splash of stock and lemon juice, and simmer until it thickens slightly."),
    ("How long should it rest?",
     "Rest the chicken for ten to fifteen minutes before carving so the juices settle."),
    ("Any dessert that is quick?",
     "Try macerated strawberries with whipped cream; it takes ten minutes."),
    ("Thanks. Can you give me a timeline?",
     "Start the brine at noon, oven on at five, chicken in at five fifteen, potatoes at five thirty."),
    ("What wine goes with it?",
     "A chilled Chardonnay or a light Pinot Noir both pair well."),
]
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _user(text: str, system: Optional[str] = None) -> dict[str, Any]:
    messages = [{"role": "system", "content": system}] if system else []
    return {"messages": [*messages, {"role": "user", "content": text}]}


# Each scenario is a list of (request kwargs, customer ID). "growing_chat" and
# "retry" are driven separately below.
SCENARIOS: dict[str, Callable[[], list[tuple[dict[str, Any], Optional[str]]]]] = {
    "same_prompt": lambda: [(_user("What is the capital of France?"), None)] * 4,
    "same_customer": lambda: [(_user("hi"), "cust_alice"), (_user("hi"), "cust_alice")],
    "system_prompt": lambda: [(_user(q, SUPPORT_SYSTEM), "cust_alice") for q in QUESTIONS],
    "tools_list": lambda: [
        ({**_user(f"weather question number {word}"), "tools": [WEATHER_TOOL]}, None)
        for word in ("alpha", "bravo", "charlie", "delta")
    ],
    "content_parts": lambda: [
        ({"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]}, None)
        for text in ("Describe this photo of a cat.", "Translate good morning into German.", "List three primes.")
    ],
}
ALL_SCENARIOS = [*SCENARIOS, "growing_chat", "retry"]


class _Provider:
    """Counts requests; answers in the OpenAI or Anthropic shape."""

    def __init__(self, *, fail_first: bool = False, replies: Optional[list[str]] = None) -> None:
        self.hits = 0
        self.fail_first = fail_first
        self.replies = list(replies or [])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.hits += 1
        if self.fail_first and self.hits == 1:
            return httpx.Response(500, json={"error": {"message": "stub 500", "type": "server_error"}})
        text = self.replies.pop(0) if self.replies else "ok"
        if request.url.path.endswith("/messages"):
            return httpx.Response(200, json={
                "id": "msg_test", "type": "message", "role": "assistant", "model": ANTHROPIC_MODEL,
                "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            })
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })


@pytest.fixture(autouse=True)
def _local_caps_only(monkeypatch, tmp_path):
    for key in (
        "KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_API_KEY", "KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL",
        "KAZENAI_CONTROL_PROFILE", "KAZENAI_ENFORCEMENT_OWNER", "KAZENAI_TIMELINE_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", "1")
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_QUEUE_DIR", str(tmp_path / "settle"))


def _monitored(provider: _Provider, *, is_async: bool, timeline: Path, **kw: Any) -> Any:
    if is_async:
        raw = openai.AsyncOpenAI(
            api_key="test", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider))
        )
    else:
        raw = openai.OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(provider)))
    return monitor(raw, max_budget_usd=1.0, org_id="org-test", timeline_path=str(timeline), **kw)


def _call(client: Any, is_async: bool, kwargs: dict[str, Any], customer: Optional[str]) -> Any:
    def go() -> Any:
        result = client.chat.completions.create(model=MODEL, **kwargs)
        return asyncio.run(result) if is_async else result

    if customer:
        with use_attribution(business_subject_ref=customer):
            return go()
    return go()


def _run(scenario: str, *, is_async: bool, timeline: Path, **kw: Any) -> tuple[list[str], _Provider]:
    """Play one scenario; return per-call outcomes and the provider stub."""
    outcomes: list[str] = []
    if scenario == "growing_chat":
        provider = _Provider(replies=[reply for _, reply in CHAT])
        client = _monitored(provider, is_async=is_async, timeline=timeline, **kw)
        history: list[dict[str, Any]] = [{"role": "system", "content": "You are a helpful home cooking assistant."}]
        for user, _ in CHAT:
            history.append({"role": "user", "content": user})
            try:
                resp = _call(client, is_async, {"messages": list(history)}, None)
            except LoopDetected:
                outcomes.append("LoopDetected")
                break
            history.append({"role": "assistant", "content": resp.choices[0].message.content})
            outcomes.append("allowed")
        return outcomes, provider
    if scenario == "retry":
        provider = _Provider(fail_first=True)
        calls = [(_user("Summarize our refund policy in two sentences."), "cust_alice")] * 2
    else:
        provider = _Provider()
        calls = SCENARIOS[scenario]()
    client = _monitored(provider, is_async=is_async, timeline=timeline, **kw)
    for kwargs, customer in calls:
        try:
            _call(client, is_async, kwargs, customer)
            outcomes.append("allowed")
        except LoopDetected:
            outcomes.append("LoopDetected")
        except openai.InternalServerError:
            outcomes.append("provider_error")
    return outcomes, provider


def _loop_events(timeline: Path) -> list[dict[str, Any]]:
    if not timeline.exists():
        return []
    events = [json.loads(line) for line in timeline.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        ev for ev in events
        if ev.get("event_type") == "finops.loop.anomaly" and ev.get("payload", {}).get("reason") == "repeated_request"
    ]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_default_records_the_loop_and_sends_every_call(scenario, is_async, tmp_path):
    timeline = tmp_path / "timeline.jsonl"
    outcomes, provider = _run(scenario, is_async=is_async, timeline=timeline)

    assert "LoopDetected" not in outcomes
    assert provider.hits == len(outcomes)
    events = _loop_events(timeline)
    assert events, "a loop hit leaves a finops.loop.anomaly event"
    assert {ev["payload"]["action"] for ev in events} == {"recorded"}
    assert {ev["payload"]["loop_guard"] for ev in events} == {"record"}


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_block_refuses_the_loop_before_the_provider(scenario, is_async, tmp_path):
    timeline = tmp_path / "timeline.jsonl"
    outcomes, provider = _run(scenario, is_async=is_async, timeline=timeline, loop_guard="block")

    blocked = outcomes.count("LoopDetected")
    assert blocked >= 1
    assert provider.hits == len(outcomes) - blocked
    events = _loop_events(timeline)
    assert [ev["payload"]["action"] for ev in events] == ["blocked"] * blocked


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_off_skips_the_check_and_writes_nothing(is_async, tmp_path):
    timeline = tmp_path / "timeline.jsonl"
    outcomes, provider = _run("same_prompt", is_async=is_async, timeline=timeline, loop_guard="off")

    assert outcomes == ["allowed"] * 4
    assert provider.hits == 4
    assert _loop_events(timeline) == []


def _play(calls, *, is_async: bool, timeline: Path, **kw: Any) -> tuple[list[str], _Provider]:
    provider = _Provider()
    client = _monitored(provider, is_async=is_async, timeline=timeline, **kw)
    outcomes = []
    for kwargs, customer in calls:
        try:
            _call(client, is_async, kwargs, customer)
            outcomes.append("allowed")
        except LoopDetected:
            outcomes.append("LoopDetected")
    return outcomes, provider


@pytest.mark.parametrize("mode", ["record", "block"])
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_two_customers_sending_the_same_prompt_is_not_a_loop(mode, is_async, tmp_path):
    timeline = tmp_path / "timeline.jsonl"
    calls = [(_user("hi"), "cust_alice"), (_user("hi"), "cust_bob")]
    outcomes, provider = _play(calls, is_async=is_async, timeline=timeline, loop_guard=mode)

    assert outcomes == ["allowed", "allowed"]
    assert provider.hits == 2
    assert _loop_events(timeline) == []


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_each_customer_keeps_its_own_history(is_async, tmp_path):
    # Alice, Bob, Alice: Bob's call doesn't hide Alice's repeat.
    timeline = tmp_path / "timeline.jsonl"
    calls = [(_user(QUESTIONS[0], SUPPORT_SYSTEM), c) for c in ("cust_alice", "cust_bob", "cust_alice")]
    outcomes, provider = _play(calls, is_async=is_async, timeline=timeline, loop_guard="block")

    assert outcomes == ["allowed", "allowed", "LoopDetected"]
    assert provider.hits == 2
    (event,) = _loop_events(timeline)
    assert event["payload"]["business_subject_ref"] == "cust_alice"


def test_concurrent_customers_with_the_same_prompt_are_all_sent(tmp_path):
    provider = _Provider()
    client = _monitored(provider, is_async=True, timeline=tmp_path / "t.jsonl", loop_guard="block")

    async def one(i: int) -> str:
        with use_attribution(business_subject_ref=f"cust_{i}"):
            await client.chat.completions.create(model=MODEL, **_user("Reset my password please"))
        return "allowed"

    async def main() -> list[str]:
        return await asyncio.gather(*(one(i) for i in range(10)))

    assert asyncio.run(main()) == ["allowed"] * 10
    assert provider.hits == 10


def test_monitor_level_customer_keys_the_history(tmp_path):
    timeline = tmp_path / "timeline.jsonl"
    provider = _Provider()
    raw = openai.OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(provider)))
    client = monitor(
        raw, max_budget_usd=1.0, org_id="org-test", timeline_path=str(timeline),
        business_subject_ref="cust_acme", loop_guard="block",
    )
    client.chat.completions.create(model=MODEL, **_user("ping"))
    with pytest.raises(LoopDetected):
        client.chat.completions.create(model=MODEL, **_user("ping"))
    (event,) = _loop_events(timeline)
    assert event["payload"]["business_subject_ref"] == "cust_acme"


def test_the_loop_event_names_the_customer_and_carries_no_cost(tmp_path):
    timeline = tmp_path / "timeline.jsonl"
    _run("same_customer", is_async=False, timeline=timeline)

    (event,) = _loop_events(timeline)
    assert event["cost_usd"] is None
    assert event["payload"]["business_subject_ref"] == "cust_alice"
    assert event["payload"]["method"] == "openai.chat.completions.create"
    assert event["payload"]["model"] == MODEL
    assert event["payload"]["loop_scores"]["h1_jaccard"] == pytest.approx(1.0)


def test_a_blocked_call_keeps_its_own_step_id(tmp_path):
    timeline = tmp_path / "timeline.jsonl"
    provider = _Provider()
    client = _monitored(provider, is_async=False, timeline=timeline, loop_guard="block")
    client.chat.completions.create(model=MODEL, **_user("ping"))
    with pytest.raises(LoopDetected):
        client.chat.completions.create(model=MODEL, **_user("ping"))

    events = [json.loads(line) for line in timeline.read_text(encoding="utf-8").splitlines()]
    call_steps = {ev["step_id"] for ev in events if ev["event_type"] == "model.call"}
    (loop,) = _loop_events(timeline)
    assert loop["step_id"] not in call_steps


@pytest.mark.parametrize("mode", ["block", "record"])
def test_a_clone_keeps_the_loop_guard_mode(mode, tmp_path):
    provider = _Provider()
    client = _monitored(provider, is_async=False, timeline=tmp_path / "t.jsonl", loop_guard=mode)
    clone = client.with_options(timeout=5)
    clone.chat.completions.create(model=MODEL, **_user("same words here"))
    if mode == "block":
        with pytest.raises(LoopDetected):
            clone.chat.completions.create(model=MODEL, **_user("same words here"))
        assert provider.hits == 1
    else:
        clone.chat.completions.create(model=MODEL, **_user("same words here"))
        assert provider.hits == 2


@pytest.mark.parametrize("bad", ["warn", "", None, 1, "blocking"])
def test_monitor_rejects_an_unknown_mode(bad):
    raw = openai.OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(_Provider())))
    with pytest.raises(ValueError, match="loop_guard must be one of"):
        monitor(raw, loop_guard=bad)


def test_resolve_loop_guard_normalises_case_and_space():
    assert resolve_loop_guard(" Block ") == "block"
    assert LOOP_GUARD_MODES == ("record", "block", "off")


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(
            usage=SimpleNamespace(total_tokens=50),
            model_dump=lambda: {"usage": {"total_tokens": 50}},
        )


class _FakeOpenAI:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_patch_openai_records_by_default():
    client = _FakeOpenAI()
    undo = patch_openai(client, org_id="o", project_id="p", agent_id="a")
    try:
        for _ in range(3):
            client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "repeat"}])
    finally:
        undo()
    assert client.chat.completions.calls == 3



def _anthropic_client(provider: _Provider, *, is_async: bool, timeline: Path, **kw: Any) -> Any:
    sdk = pytest.importorskip("anthropic")
    if is_async:
        raw = sdk.AsyncAnthropic(
            api_key="test", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider))
        )
    else:
        raw = sdk.Anthropic(api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(provider)))
    return monitor(raw, max_budget_usd=1.0, org_id="org-test", timeline_path=str(timeline), **kw)


ANTHROPIC_TOOL = {"name": "get_weather", "description": "Weather", "input_schema": {"type": "object", "properties": {}}}
ANTHROPIC_SCENARIOS = {
    "same_prompt": lambda: [{"messages": [{"role": "user", "content": "What is the capital of France?"}]}] * 3,
    "tools_list": lambda: [
        {"messages": [{"role": "user", "content": f"weather question {word}"}], "tools": [ANTHROPIC_TOOL]}
        for word in ("alpha", "bravo", "charlie", "delta")
    ],
    "content_blocks": lambda: [
        {"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]}
        for text in ("Describe a cat.", "List three primes.", "Translate hello into German.")
    ],
}


@pytest.mark.parametrize("mode", ["record", "block"])
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("scenario", list(ANTHROPIC_SCENARIOS))
def test_anthropic_follows_the_same_modes(scenario, is_async, mode, tmp_path):
    timeline = tmp_path / "timeline.jsonl"
    provider = _Provider()
    client = _anthropic_client(provider, is_async=is_async, timeline=timeline, loop_guard=mode)
    outcomes = []
    for kwargs in ANTHROPIC_SCENARIOS[scenario]():
        try:
            result = client.messages.create(model=ANTHROPIC_MODEL, max_tokens=16, **kwargs)
            if is_async:
                asyncio.run(result)
            outcomes.append("allowed")
        except LoopDetected:
            outcomes.append("LoopDetected")

    events = _loop_events(timeline)
    assert events
    if mode == "record":
        assert "LoopDetected" not in outcomes
        assert provider.hits == len(outcomes)
        assert {ev["payload"]["action"] for ev in events} == {"recorded"}
    else:
        blocked = outcomes.count("LoopDetected")
        assert blocked >= 1
        assert provider.hits == len(outcomes) - blocked
        assert [ev["payload"]["action"] for ev in events] == ["blocked"] * blocked
