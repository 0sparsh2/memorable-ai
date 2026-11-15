"""
Zero-code interception against the real OpenAI / Anthropic SDKs.

Memory goes in front of the latest user message by default (``injection="tail"``),
so the rest of the request stays byte-identical across turns and the provider's
prompt cache keeps the conversation prefix; ``injection="system"`` keeps the
v0.2 placement (system prompt / ``system`` / ``instructions``).

No network: each client gets a mock transport that records the request body and
returns a canned response. The transport comes from whichever HTTP library the
installed SDK is built on (``httpx``, or ``httpx2`` in anthropic>=2). These tests check what actually goes over
the wire: memory is injected into the system prompt for OpenAI and into the
``system`` parameter for Anthropic (never a system-role message), it is scoped
to the calling user, and the exchange is recorded afterwards.
"""

import asyncio
import importlib
import json

import httpx
import pytest

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

from memorable_ai.core.models import Turn  # noqa: E402
from memorable_ai.core.scope import suppressed  # noqa: E402

OPENAI_CHAT = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Avoid peanut snacks."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
ANTHROPIC_MSG = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-test",
    "content": [{"type": "text", "text": "Skip the satay."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
RESPONSES = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "gpt-4o-mini",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "m1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Noted.", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


def _sse(events, named=False):
    lines = []
    for ev in events:
        if named:
            lines.append(f"event: {ev['type']}")
        lines.append("data: " + json.dumps(ev))
        lines.append("")
    if not named:
        lines += ["data: [DONE]", ""]
    return ("\n".join(lines) + "\n").encode()


def _http_lib(sdk):
    """The HTTP library an SDK's client is built on: httpx, or httpx2 (anthropic>=2)."""
    try:
        base = importlib.import_module(f"{sdk.__name__}._base_client")
    except ImportError:
        return httpx
    return getattr(base, "httpx2", None) or getattr(base, "httpx", None) or httpx


class Recorder:
    def __init__(self, body=None, stream_body=None):
        self.requests = []
        self.body = body
        self.stream_body = stream_body
        self.lib = httpx  # set by the client factory to the SDK's HTTP library

    def __call__(self, request):
        payload = json.loads(request.content or b"{}")
        self.requests.append(payload)
        if payload.get("stream"):
            return self.lib.Response(200, content=self.stream_body, headers={"content-type": "text/event-stream"})
        return self.lib.Response(200, json=self.body)

    @property
    def last(self):
        return self.requests[-1]


def _mock_client(sdk, recorder, is_async):
    lib = recorder.lib = _http_lib(sdk)
    cls = lib.AsyncClient if is_async else lib.Client
    return cls(transport=lib.MockTransport(recorder))


def _openai(recorder, is_async=False):
    cls = openai.AsyncOpenAI if is_async else openai.OpenAI
    return cls(api_key="test", base_url="http://mock/v1", http_client=_mock_client(openai, recorder, is_async))


def _anthropic(recorder, is_async=False):
    cls = anthropic.AsyncAnthropic if is_async else anthropic.Anthropic
    return cls(api_key="test", base_url="http://mock", http_client=_mock_client(anthropic, recorder, is_async))


@pytest.fixture
def engine(engine_factory):
    eng = engine_factory()
    asyncio.run(eng.add_memory("User is allergic to peanuts", "fact", user_id="alice"))
    asyncio.run(eng.add_memory("User loves Thai food", "preference", user_id="alice"))
    asyncio.run(eng.add_memory("User is allergic to shellfish", "fact", user_id="bob"))
    eng.enable()
    yield eng
    eng.disable()


def _turns(engine, session_id):
    s = engine._storage.get_session()
    try:
        return [(t.role, t.content) for t in s.query(Turn).filter(Turn.session_id == session_id).order_by(Turn.seq)]
    finally:
        s.close()


def test_all_sdk_targets_hook_and_restore(engine_factory):
    from openai.resources.chat.completions import Completions
    from anthropic.resources.messages import Messages

    eng = engine_factory()
    original_openai, original_anthropic = Completions.create, Messages.create
    eng.enable()
    hooked = set(eng.get_stats()["hooked"])
    assert {
        "openai.chat.completions.create",
        "openai.chat.completions.acreate",
        "openai.responses.create",
        "anthropic.messages.create",
        "anthropic.messages.acreate",
        "litellm.completion",
    } <= hooked
    assert Completions.create is not original_openai
    eng.disable()
    assert Completions.create is original_openai
    assert Messages.create is original_anthropic


def test_openai_chat_injects_scoped_memory_and_records(engine):
    rec = Recorder(OPENAI_CHAT)
    client = _openai(rec)
    with engine.scope(user_id="alice", session_id="chat-1"):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a chef."}, {"role": "user", "content": "Which snacks should I avoid? Any allergies?"}],
        )
    assert resp.choices[0].message.content == "Avoid peanut snacks."
    sent = rec.last["messages"]
    assert len(sent) == 2, "no extra messages"
    assert sent[0] == {"role": "system", "content": "You are a chef."}, "system prompt untouched (cache prefix)"
    assert sent[1]["content"].startswith("<memory>")
    assert sent[1]["content"].endswith("Which snacks should I avoid? Any allergies?")
    assert "allergic to peanuts" in sent[1]["content"]
    assert "shellfish" not in sent[1]["content"], "another user's memory leaked"
    assert _turns(engine, "chat-1") == [
        ("user", "Which snacks should I avoid? Any allergies?"),
        ("assistant", "Avoid peanut snacks."),
    ]


def test_openai_user_isolation_via_metadata(engine):
    rec = Recorder(OPENAI_CHAT)
    client = _openai(rec)
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Do I have any food allergies?"}],
        metadata={"user_id": "bob", "session_id": "bob-1"},
    )
    sent = rec.last["messages"]
    assert [m["role"] for m in sent] == ["user"]
    assert "shellfish" in sent[0]["content"]
    assert "peanuts" not in sent[0]["content"]


def test_anonymous_call_sees_no_user_memory(engine):
    rec = Recorder(OPENAI_CHAT)
    _openai(rec).chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "Do I have any food allergies?"}]
    )
    assert rec.last["messages"] == [{"role": "user", "content": "Do I have any food allergies?"}]


def test_anthropic_context_goes_in_latest_user_turn(engine):
    rec = Recorder(ANTHROPIC_MSG)
    client = _anthropic(rec)
    resp = client.messages.create(
        model="claude-test",
        max_tokens=50,
        system="Be brief.",
        messages=[{"role": "user", "content": "Plan a dinner; what should I avoid for my allergies?"}],
        metadata={"user_id": "alice"},
    )
    assert resp.content[0].text == "Skip the satay."
    body = rec.last
    assert all(m["role"] != "system" for m in body["messages"]), "Anthropic rejects system-role messages"
    assert body["system"] == "Be brief.", "system untouched (cache prefix)"
    assert "allergic to peanuts" in body["messages"][-1]["content"]


def test_openai_stream_is_recorded_after_consumption(engine):
    chunks = [
        {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
         "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}]}
        for part in ("Peanut-free ", "trail mix.")
    ]
    rec = Recorder(stream_body=_sse(chunks))
    client = _openai(rec)
    with engine.scope(user_id="alice", session_id="stream-1"):
        stream = client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "Snack idea for my allergy?"}], stream=True
        )
        text = "".join(c.choices[0].delta.content or "" for c in stream)
    assert text == "Peanut-free trail mix."
    assert "allergic to peanuts" in rec.last["messages"][0]["content"]
    assert _turns(engine, "stream-1")[-1] == ("assistant", "Peanut-free trail mix.")


def test_anthropic_stream_is_recorded(engine):
    events = [
        {"type": "message_start", "message": {**ANTHROPIC_MSG, "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "No peanuts"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": ", ever."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
    ]
    rec = Recorder(stream_body=_sse(events, named=True))
    client = _anthropic(rec)
    with engine.scope(user_id="alice", session_id="astream-1"):
        stream = client.messages.create(
            model="claude-test", max_tokens=20, stream=True,
            messages=[{"role": "user", "content": "Remind me what I can't eat"}],
        )
        for _ in stream:
            pass
    assert _turns(engine, "astream-1")[-1] == ("assistant", "No peanuts, ever.")


def test_openai_responses_api_puts_memory_in_input(engine):
    rec = Recorder(RESPONSES)
    client = _openai(rec)
    with engine.scope(user_id="alice", session_id="resp-1"):
        resp = client.responses.create(model="gpt-4o-mini", input="Any allergies I should mention to the waiter?",
                                       instructions="Be polite.")
    assert resp.output_text == "Noted."
    assert rec.last["instructions"] == "Be polite."
    assert "allergic to peanuts" in rec.last["input"]
    assert rec.last["input"].endswith("Any allergies I should mention to the waiter?")
    assert _turns(engine, "resp-1")[-1] == ("assistant", "Noted.")


def test_suppressed_calls_are_not_intercepted(engine):
    rec = Recorder(OPENAI_CHAT)
    with engine.scope(user_id="alice", session_id="internal"), suppressed():
        _openai(rec).chat.completions.create(model="m", messages=[{"role": "user", "content": "allergies?"}])
    assert rec.last["messages"] == [{"role": "user", "content": "allergies?"}]
    assert _turns(engine, "internal") == []


@pytest.mark.asyncio
async def test_async_openai_and_anthropic(engine_factory):
    eng = engine_factory()
    await eng.add_memory("User is allergic to peanuts", "fact", user_id="alice")
    eng.enable()
    try:
        rec_o, rec_a = Recorder(OPENAI_CHAT), Recorder(ANTHROPIC_MSG)
        with eng.scope(user_id="alice", session_id="async-1"):
            await _openai(rec_o, is_async=True).chat.completions.create(
                model="m", messages=[{"role": "user", "content": "What allergies do I have?"}]
            )
            await _anthropic(rec_a, is_async=True).messages.create(
                model="claude-test", max_tokens=10, messages=[{"role": "user", "content": "And my allergies again?"}]
            )
        assert "allergic to peanuts" in rec_o.last["messages"][-1]["content"]
        assert "allergic to peanuts" in rec_a.last["messages"][-1]["content"]
        roles = [r for r, _ in _turns(eng, "async-1")]
        assert roles.count("assistant") == 2
    finally:
        eng.disable()


def test_litellm_completion_is_intercepted(engine):
    litellm = pytest.importorskip("litellm")
    with engine.scope(user_id="alice", session_id="lite-1"):
        resp = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Any allergies on file for me?"}],
            mock_response="You avoid peanuts.",
        )
    assert resp.choices[0].message.content == "You avoid peanuts."
    assert _turns(engine, "lite-1") == [
        ("user", "Any allergies on file for me?"),
        ("assistant", "You avoid peanuts."),
    ]


def test_background_write_mode_flushes(engine_factory):
    eng = engine_factory(write_mode="background")
    eng.enable()
    try:
        rec = Recorder(OPENAI_CHAT)
        with eng.scope(user_id="carol", session_id="bg-1"):
            _openai(rec).chat.completions.create(
                model="m", messages=[{"role": "user", "content": "I live in Lisbon and I love surfing."}]
            )
        eng.flush(timeout=30)
        contents = [m["content"] for m in asyncio.run(eng._storage.get_memories(user_id="carol"))]
        assert "User lives in Lisbon" in contents
    finally:
        eng.disable()


def test_system_injection_mode_keeps_v02_placement(engine):
    engine.config.memory.injection = "system"
    rec_o, rec_a, rec_r = Recorder(OPENAI_CHAT), Recorder(ANTHROPIC_MSG), Recorder(RESPONSES)
    with engine.scope(user_id="alice", session_id="sys-1"):
        _openai(rec_o).chat.completions.create(
            model="m", messages=[{"role": "system", "content": "You are a chef."},
                                 {"role": "user", "content": "Which snacks should I avoid?"}])
        _anthropic(rec_a).messages.create(model="claude-test", max_tokens=10, system="Be brief.",
                                          messages=[{"role": "user", "content": "My allergies?"}])
        _openai(rec_r).responses.create(model="m", input="Allergies to mention?")
    assert rec_o.last["messages"][0]["content"].startswith("You are a chef.")
    assert "allergic to peanuts" in rec_o.last["messages"][0]["content"]
    assert rec_o.last["messages"][1]["content"] == "Which snacks should I avoid?"
    assert rec_a.last["system"].startswith("Be brief.") and "allergic to peanuts" in rec_a.last["system"]
    assert "allergic to peanuts" in rec_r.last["instructions"]


def test_tail_injection_keeps_conversation_prefix_byte_identical(engine):
    """
    The cache argument. Memory changes between turns (new facts are learned). With tail
    placement, call k+1 still starts with call k's request (minus k's memory-carrying user
    turn), so a provider prompt cache keeps the whole conversation. With system placement
    the change lands at the very start and the whole history is re-billed.
    """

    def two_turns(session, new_fact, follow_up):
        rec = Recorder(OPENAI_CHAT)
        client = _openai(rec)
        history = [{"role": "system", "content": "You are a chef."},
                   {"role": "user", "content": "Which snacks should I avoid?"}]
        with engine.scope(user_id="alice", session_id=session):
            client.chat.completions.create(model="m", messages=list(history))
            first = rec.last["messages"]
            asyncio.run(engine.add_memory(new_fact, "fact", user_id="alice"))
            history += [{"role": "assistant", "content": "Avoid peanut snacks."},
                        {"role": "user", "content": follow_up}]
            client.chat.completions.create(model="m", messages=list(history))
            second = rec.last["messages"]
        return first, second

    first, second = two_turns("cache-tail", "User started a keto diet", "Dinner ideas for my keto diet?")
    assert second[: len(first) - 1] == first[:-1], "tail mode: conversation prefix unchanged"
    assert "keto" in second[-1]["content"]

    engine.config.memory.injection = "system"
    first, second = two_turns("cache-system", "User is training for a marathon",
                              "Meals for my marathon training?")
    assert "marathon" in second[0]["content"] and "marathon" not in first[0]["content"]
    assert second[0] != first[0], "system mode: the very first message changed, so nothing after it is cached"


def test_tool_loop_reuses_the_same_memory_block(engine, monkeypatch):
    calls = []
    real = engine._context_for

    async def counting(messages, scope):
        calls.append(1)
        return await real(messages, scope)

    monkeypatch.setattr(engine, "_context_for", counting)
    rec = Recorder(OPENAI_CHAT)
    client = _openai(rec)
    base = [{"role": "user", "content": "Order dinner that is safe for my allergies."}]
    loop = base + [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "menu", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "satay, pad thai, green curry"},
    ]
    with engine.scope(user_id="alice", session_id="loop-1"):
        client.chat.completions.create(model="m", messages=base)
        first = rec.last["messages"][0]["content"]
        client.chat.completions.create(model="m", messages=loop)
        second = rec.last["messages"][0]["content"]
    assert first == second and "allergic to peanuts" in first
    assert len(calls) == 1, "memory was retrieved again inside the tool loop"
