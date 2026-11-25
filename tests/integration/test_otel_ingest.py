"""OpenTelemetry GenAI ingest: spans -> one canonical L0 stream per conversation -> tasks."""

import json

import pytest

from memorable_ai.integrations.otel import conversations

TOOLS = json.dumps([{"type": "function", "name": "lookup_order", "parameters": {"type": "object"}},
                    {"type": "function", "name": "refund", "parameters": {"type": "object"}}])
U = {"role": "user", "parts": [{"type": "text", "content": "Where is order ord_4417?"}]}
CALL = {"role": "assistant", "parts": [{"type": "tool_call", "id": "c1", "name": "lookup_order",
                                        "arguments": {"id": "ord_4417"}}]}
RESP = {"role": "tool", "parts": [{"type": "tool_call_response", "id": "c1", "response": {"status": "shipped"}}]}
ANSWER = {"role": "assistant", "parts": [{"type": "text", "content": "It shipped."}]}


def span(name, t, attrs, status="OK", trace="t1"):
    return {"name": name, "trace_id": trace, "span_id": f"s{t}", "start_time": t, "attributes": attrs,
            "status": {"code": status}}


def agent_run(conv="conv-77"):
    base = {"gen_ai.conversation.id": conv, "gen_ai.agent.name": "support-bot", "user.id": "alice"}
    return [
        span("chat m", 1, {**base, "gen_ai.operation.name": "chat", "gen_ai.tool.definitions": TOOLS,
                           "gen_ai.input.messages": json.dumps([U]), "gen_ai.output.messages": json.dumps([CALL])}),
        span("execute_tool lookup_order", 2, {**base, "gen_ai.operation.name": "execute_tool",
                                              "gen_ai.tool.name": "lookup_order", "gen_ai.tool.call.id": "c1",
                                              "gen_ai.tool.call.arguments": {"id": "ord_4417"},
                                              "gen_ai.tool.call.result": {"status": "shipped"}}),
        span("chat m", 3, {**base, "gen_ai.operation.name": "chat", "gen_ai.tool.definitions": TOOLS,
                           "gen_ai.input.messages": json.dumps([U, CALL, RESP]),
                           "gen_ai.output.messages": json.dumps([ANSWER])}),
    ]


def to_otlp(spans):
    def anyv(v):
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}
        if isinstance(v, dict):
            return {"kvlistValue": {"values": [{"key": k, "value": anyv(x)} for k, x in v.items()]}}
        return {"stringValue": v}

    return {"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [{"spans": [{
        "name": s["name"], "traceId": s["trace_id"], "spanId": s["span_id"], "startTimeUnixNano": str(s["start_time"]),
        "attributes": [{"key": k, "value": anyv(v)} for k, v in s["attributes"].items()],
        "status": {"code": 2 if s["status"]["code"] == "ERROR" else 1}} for s in spans]}]}]}


def test_canonical_stream_has_no_duplicates():
    [c] = conversations(agent_run())
    roles = [(m["role"], bool(m.get("tool_calls"))) for m in c["messages"]]
    assert roles == [("user", False), ("assistant", True), ("tool", False), ("assistant", False)]
    assert c["offered_tools"] == ["lookup_order", "refund"]
    assert (c["session_id"], c["user_id"], c["agent_id"]) == ("conv-77", "alice", "support-bot")


@pytest.mark.asyncio
@pytest.mark.parametrize("encode", [lambda s: s, to_otlp], ids=["plain", "otlp-json"])
async def test_ingest_otel_builds_events_and_task(engine_factory, encode):
    eng = engine_factory()
    [r] = await eng.ingest_otel(encode(agent_run()))
    assert r["session_id"] == "conv-77"
    [t] = eng.list_tasks(user_id="alice", session_id="conv-77")
    assert t["goal"] == "Where is order ord_4417?" and t["used_tools"] == ["lookup_order"]
    assert t["offered_tools"] == ["lookup_order", "refund"] and t["status"] == "closed"
    events = eng.get_task(t["id"], user_id="alice")["events"]
    assert [e["event_kind"] for e in events] == ["message", "tool_call", "tool_result", "message"]
    again = await eng.ingest_otel(encode(agent_run()))  # re-sent export: idempotent
    assert again[0]["new_turns"] == 0


@pytest.mark.asyncio
async def test_tool_spans_alone_with_error_status(engine_factory):
    """Without content capture only execute_tool spans carry data; ERROR status marks the result."""
    eng = engine_factory()
    base = {"gen_ai.conversation.id": "ops-1", "user.id": "bob"}
    spans = [span("execute_tool deploy", 1, {**base, "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "deploy",
                                            "gen_ai.tool.call.id": "d1"}, status="ERROR"),
             span("execute_tool deploy", 2, {**base, "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "deploy",
                                            "gen_ai.tool.call.id": "d2", "error.type": "TimeoutError"})]
    await eng.ingest_otel(spans)
    s = eng._storage.get_session()
    try:
        from memorable_ai.core.models import Turn

        rows = s.query(Turn).filter(Turn.session_id == "ops-1").order_by(Turn.seq).all()
        assert [r.kind for r in rows] == ["tool_call", "tool_result", "tool_call", "tool_result"]
        assert [r.is_error for r in rows if r.kind == "tool_result"] == [True, True]
        assert "TimeoutError" in json.dumps(rows[-1].payload)
    finally:
        s.close()
