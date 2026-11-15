"""
Zero-code interception against the real OpenAI / Anthropic SDKs.

No network: each client gets an ``httpx.MockTransport`` that records the request
body and returns a canned response. These tests check what actually goes over
the wire: memory is injected into the system prompt for OpenAI and into the
``system`` parameter for Anthropic (never a system-role message), it is scoped
to the calling user, and the exchange is recorded afterwards.
"""

import asyncio
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


class Recorder:
    def __init__(self, body=None, stream_body=None):
        self.requests = []
        self.body = body
        self.stream_body = stream_body

    def __call__(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        self.requests.append(payload)
        if payload.get("stream"):
            return httpx.Response(200, content=self.stream_body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=self.body)

    @property
    def last(self):
        return self.requests[-1]


def _openai(recorder, is_async=False):
    if is_async:
        return openai.AsyncOpenAI(
            api_key="test", base_url="http://mock/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
        )
    return openai.OpenAI(
        api_key="test", base_url="http://mock/v1", http_client=httpx.Client(transport=httpx.MockTransport(recorder))
    )


def _anthropic(recorder, is_async=False):
    if is_async:
        return anthropic.AsyncAnthropic(
            api_key="test", base_url="http://mock",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
        )
    return anthropic.Anthropic(
        api_key="test", base_url="http://mock", http_client=httpx.Client(transport=httpx.MockTransport(recorder))
    )


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
    assert len(sent) == 2, "context is merged into the existing system prompt, not prepended"
    assert sent[0]["content"].startswith("You are a chef.")
    assert "allergic to peanuts" in sent[0]["content"]
    assert "shellfish" not in sent[0]["content"], "another user's memory leaked"
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
    system = rec.last["messages"][0]
    assert system["role"] == "system"
    assert "shellfish" in system["content"]
    assert "peanuts" not in system["content"]


def test_anonymous_call_sees_no_user_memory(engine):
    rec = Recorder(OPENAI_CHAT)
    _openai(rec).chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "Do I have any food allergies?"}]
    )
    assert all(m["role"] != "system" for m in rec.last["messages"])


def test_anthropic_context_goes_in_system_param(engine):
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
    assert body["system"].startswith("Be brief.")
    assert "allergic to peanuts" in body["system"]


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


def test_openai_responses_api_uses_instructions(engine):
    rec = Recorder(RESPONSES)
    client = _openai(rec)
    with engine.scope(user_id="alice", session_id="resp-1"):
        resp = client.responses.create(model="gpt-4o-mini", input="Any allergies I should mention to the waiter?")
    assert resp.output_text == "Noted."
    assert "allergic to peanuts" in rec.last["instructions"]
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
        assert "allergic to peanuts" in rec_o.last["messages"][0]["content"]
        assert "allergic to peanuts" in rec_a.last["system"]
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
