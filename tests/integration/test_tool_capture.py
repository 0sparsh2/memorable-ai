"""
Procedural P1 capture: agent tool loops through the real SDKs (mock transports) land in L0
as ordered message / tool_call / tool_result events, redacted, deduplicated on resend, and
kept out of conversational excerpt retrieval and episode segmentation.
"""

import json

import pytest

from memorable_ai.core.models import Episode, Turn
from tests.integration.test_sdk_interception import ANTHROPIC_MSG, OPENAI_CHAT, Recorder, _anthropic, _openai

SECRET = "sk-proj-" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1"


class Script(Recorder):
    """Recorder that replies with the next canned body on each call."""

    def __init__(self, bodies):
        super().__init__(bodies[0])
        self.bodies = list(bodies)

    def __call__(self, request):
        self.body = self.bodies[min(len(self.requests), len(self.bodies) - 1)]
        return super().__call__(request)


def _rows(engine, session_id):
    s = engine._storage.get_session()
    try:
        return [dict(kind=t.kind, role=t.role, tool=t.tool_name, call=t.tool_call_id, err=t.is_error,
                     payload=t.payload, values=t.salient_values, emb=t.embedding is not None, content=t.content)
                for t in s.query(Turn).filter(Turn.session_id == session_id).order_by(Turn.seq)]
    finally:
        s.close()


def _openai_tool_reply():
    return {**OPENAI_CHAT, "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{
            "id": "call_9", "type": "function",
            "function": {"name": "lookup_order", "arguments": json.dumps({"order_id": "ord_4417"})}}]}}]}


def test_openai_tool_loop_is_captured_in_order_redacted_and_idempotent(engine_factory):
    eng = engine_factory()
    eng.enable()
    try:
        rec = Script([_openai_tool_reply(), OPENAI_CHAT])
        client = _openai(rec)
        history = [{"role": "user", "content": "Where is my order ord_4417?"}]
        with eng.scope(user_id="alice", session_id="loop-1"):
            first = client.chat.completions.create(model="m", messages=list(history))
            history.append(first.choices[0].message.model_dump(exclude_none=True))
            history.append({"role": "tool", "tool_call_id": "call_9",
                            "content": json.dumps({"status": "shipped", "carrier": "DHL", "api_key": SECRET})})
            client.chat.completions.create(model="m", messages=list(history))
        rows = _rows(eng, "loop-1")
        assert [(r["kind"], r["role"]) for r in rows] == [
            ("message", "user"), ("tool_call", "assistant"), ("tool_result", "tool"), ("message", "assistant")]
        call, result = rows[1], rows[2]
        assert call["tool"] == "lookup_order" and call["call"] == "call_9" and call["payload"] == {"order_id": "ord_4417"}
        assert result["tool"] == "lookup_order", "result paired with its call"
        assert result["payload"]["api_key"] == "[REDACTED:secret]" and SECRET not in json.dumps(result["payload"])
        assert "ord_4417" in call["values"] and "DHL" in result["values"]
        assert not call["emb"] and not result["emb"], "tool events are not embedded"
        assert rows[0]["emb"] and rows[3]["emb"], "messages still are"
    finally:
        eng.disable()


def test_tool_events_stay_out_of_excerpts_and_segmentation(engine_factory):
    eng = engine_factory()
    eng.enable()
    try:
        rec = Script([_openai_tool_reply(), OPENAI_CHAT])
        client = _openai(rec)
        history = [{"role": "user", "content": "Where is my order ord_4417?"}]
        with eng.scope(user_id="alice", session_id="loop-2"):
            first = client.chat.completions.create(model="m", messages=list(history))
            history.append(first.choices[0].message.model_dump(exclude_none=True))
            for i in range(60):  # a long tool burst must not count toward the episode's max_turns
                history.append({"role": "tool", "tool_call_id": "call_9", "content": f"poll {i}: in transit"})
            client.chat.completions.create(model="m", messages=list(history))
        s = eng._storage.get_session()
        try:
            n_eps = s.query(Episode).filter(Episode.session_id == "loop-2").count()
        finally:
            s.close()
        assert n_eps == 1
        hits = [i for i, _ in eng._storage.lexical_search("turn", "transit poll", k=20, user_id="alice")]
        kinds = {r.kind for r in eng._storage.get_session().query(Turn).filter(Turn.id.in_(hits or [-1]))}
        assert kinds <= {"message"}, "tool results leaked into excerpt retrieval"
    finally:
        eng.disable()


def test_anthropic_tool_use_loop(engine_factory):
    eng = engine_factory()
    eng.enable()
    try:
        tool_reply = {**ANTHROPIC_MSG, "stop_reason": "tool_use", "content": [
            {"type": "text", "text": "Let me run the tests."},
            {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"cmd": "pytest -q"}}]}
        rec = Script([tool_reply, ANTHROPIC_MSG])
        client = _anthropic(rec)
        msgs = [{"role": "user", "content": "Are the tests green?"}]
        with eng.scope(user_id="bob", session_id="a-loop"):
            client.messages.create(model="claude-test", max_tokens=50, messages=list(msgs))
            msgs.append({"role": "assistant", "content": tool_reply["content"]})
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "is_error": True,
                                                      "content": "FAILED tests/test_api.py::test_login"}]})
            client.messages.create(model="claude-test", max_tokens=50, messages=list(msgs))
        rows = _rows(eng, "a-loop")
        assert [r["kind"] for r in rows] == ["message", "message", "tool_call", "tool_result", "message"]
        assert rows[2]["tool"] == "bash" and rows[3]["err"] is True and rows[3]["tool"] == "bash"
    finally:
        eng.disable()


def test_capture_can_be_disabled(engine_factory):
    eng = engine_factory()
    eng.config.memory.capture.tool_events = False
    eng._episodic = None  # rebuilt with the new setting
    eng._initialized = False
    eng.enable()
    try:
        rec = Script([_openai_tool_reply(), OPENAI_CHAT])
        with eng.scope(user_id="carol", session_id="off-1"):
            _openai(rec).chat.completions.create(model="m", messages=[{"role": "user", "content": "order ord_1?"}])
        assert {r["kind"] for r in _rows(eng, "off-1")} == {"message"}
    finally:
        eng.disable()


@pytest.mark.asyncio
async def test_resent_history_with_tool_events_is_idempotent(engine_factory):
    eng = engine_factory()
    hist = [
        {"role": "user", "content": "deploy payments"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "deploy", "arguments": '{"svc": "payments"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "deployed v42"},
    ]
    await eng.ingest(hist, "Done: v42 is live.", user_id="dan", session_id="d1")
    await eng.ingest(hist + [{"role": "assistant", "content": "Done: v42 is live."},
                             {"role": "user", "content": "thanks"}], None, user_id="dan", session_id="d1")
    rows = _rows(eng, "d1")
    assert [r["kind"] for r in rows] == ["message", "tool_call", "tool_result", "message", "message"]


def test_streamed_tool_calls_are_captured(engine_factory):
    from tests.integration.test_sdk_interception import _sse

    eng = engine_factory()
    eng.enable()
    try:
        oa_chunks = [
            {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [{"index": 0, "delta": {
                "role": "assistant", "tool_calls": [{"index": 0, "id": "call_s1", "type": "function",
                                                     "function": {"name": "get_weather", "arguments": ""}}]},
                "finish_reason": None}]},
            {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": '{"ci'}}]}, "finish_reason": None}]},
            {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Oslo"}'}}]}, "finish_reason": "tool_calls"}]},
        ]
        rec = Recorder(stream_body=_sse(oa_chunks))
        with eng.scope(user_id="gus", session_id="st-1"):
            for _ in _openai(rec).chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "Weather in Oslo?"}], stream=True):
                pass
        rows = _rows(eng, "st-1")
        assert [r["kind"] for r in rows] == ["message", "tool_call"]
        assert rows[1]["tool"] == "get_weather" and rows[1]["call"] == "call_s1" and rows[1]["payload"] == {"city": "Oslo"}

        an_events = [
            {"type": "message_start", "message": {**ANTHROPIC_MSG, "content": [], "stop_reason": None}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Listing files."}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu_s", "name": "bash",
                                                                           "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"cmd": '}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"ls -la"}'}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
             "usage": {"output_tokens": 9}},
            {"type": "message_stop"},
        ]
        rec_a = Recorder(stream_body=_sse(an_events, named=True))
        with eng.scope(user_id="gus", session_id="st-2"):
            stream = _anthropic(rec_a).messages.create(model="claude-test", max_tokens=50, stream=True,
                                                       messages=[{"role": "user", "content": "What files are here?"}])
            for _ in stream:
                pass
        rows = _rows(eng, "st-2")
        assert [r["kind"] for r in rows] == ["message", "message", "tool_call"]
        assert rows[1]["content"] == "Listing files." and rows[2]["tool"] == "bash"
        assert rows[2]["payload"] == {"cmd": "ls -la"}
    finally:
        eng.disable()
