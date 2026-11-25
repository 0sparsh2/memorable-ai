"""Procedural P1b: tasks segmented from the event stream, outcomes, explicit tasks."""

import json
from datetime import datetime, timedelta

import pytest

from memorable_ai.core.models import Turn
from memorable_ai.core.tasks import classify_feedback, signal_outcome
from tests.integration.test_sdk_interception import OPENAI_CHAT, _openai
from tests.integration.test_tool_capture import Script, _openai_tool_reply


def call(cid, name, args):
    return {"role": "assistant", "tool_calls": [{"id": cid, "function": {"name": name, "arguments": json.dumps(args)}}]}


def result(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


@pytest.fixture
def eng(engine_factory):
    return engine_factory()


def test_feedback_and_signal_rules():
    assert classify_feedback("thanks, that worked!") == "success"
    assert classify_feedback("thanks but it still fails") == "failure", "negative wins"
    assert classify_feedback("what about Lisbon?") is None
    assert signal_outcome([{"content": "pytest -> 12 passed in 0.5s"}])[0] == "success"
    assert signal_outcome([{"content": "run -> 2 failed, 10 passed"}])[0] == "failure"
    assert signal_outcome([{"content": "x", "is_error": True, "tool_name": "bash"}])[0] == "failure"
    assert signal_outcome([{"content": "deploy -> {\"status\": 201}"}])[0] == "success"
    assert signal_outcome([{"content": "lookup -> shipped via DHL"}]) is None


@pytest.mark.asyncio
async def test_task_segmentation_goal_tools_and_feedback(eng):
    h = [{"role": "user", "content": "Where is order ord_4417?"}, call("c1", "lookup_order", {"id": "ord_4417"}),
         result("c1", "shipped via DHL")]
    await eng.ingest(h, "It shipped with DHL.", user_id="alice", session_id="s1")
    [t] = eng.list_tasks(user_id="alice", session_id="s1")
    assert t["goal"] == "Where is order ord_4417?" and t["used_tools"] == ["lookup_order"]
    assert t["status"] == "closed" and t["close_reason"] == "final_answer" and t["outcome"] == "unknown"
    assert t["tool_calls"] == 1
    await eng.ingest(h + [{"role": "assistant", "content": "It shipped with DHL."},
                          {"role": "user", "content": "thanks, that worked"}], None, user_id="alice", session_id="s1")
    [t] = eng.list_tasks(user_id="alice", session_id="s1")
    assert (t["outcome"], t["outcome_source"], t["outcome_confidence"]) == ("success", "feedback", 0.6)
    detail = eng.get_task(t["id"], user_id="alice")
    assert [e["event_kind"] for e in detail["events"]] == ["message", "tool_call", "tool_result", "message"]
    assert eng.get_task(t["id"], user_id="mallory") == {}, "another user's task leaked"


@pytest.mark.asyncio
async def test_signal_beats_feedback(eng):
    h = [{"role": "user", "content": "Run the test suite"}, call("c1", "bash", {"cmd": "pytest -q"}),
         result("c1", "FAILED tests/test_api.py::test_login - 1 failed, 11 passed")]
    await eng.ingest(h, "One test fails.", user_id="bob", session_id="s2")
    await eng.ingest(h + [{"role": "assistant", "content": "One test fails."},
                          {"role": "user", "content": "great, thanks"}], None, user_id="bob", session_id="s2")
    [t] = eng.list_tasks(user_id="bob", session_id="s2")
    assert (t["outcome"], t["outcome_source"]) == ("failure", "signal"), "0.6 feedback must not override 0.8 signal"
    assert t["outcome_reason"].startswith("tests:"), "the test summary is the signal"
    assert t["error_count"] == 0, "the tool itself ran fine; only the tests failed"


@pytest.mark.asyncio
async def test_child_task_and_idle_gap(eng):
    t0 = datetime(2026, 11, 16, 9, 0)
    first = [{"role": "user", "content": "Deploy payments to staging", "timestamp": t0.isoformat()},
             call("c1", "deploy", {"svc": "payments"}), result("c1", "deployed v41")]
    await eng.ingest(first, None, user_id="cy", session_id="s3", occurred_at=t0)
    second = first + [{"role": "user", "content": "now do the same for auth",
                       "timestamp": (t0 + timedelta(minutes=2)).isoformat()},
                      call("c2", "deploy", {"svc": "auth"}), result("c2", "deployed v7")]
    await eng.ingest(second, None, user_id="cy", session_id="s3", occurred_at=t0 + timedelta(minutes=2))
    tasks = eng.list_tasks(user_id="cy", session_id="s3")
    child, parent = tasks[0], tasks[1]
    assert child["parent_task_id"] == parent["id"] and parent["close_reason"] == "next_user_message"
    later = second + [call("c3", "deploy", {"svc": "billing"})]
    await eng.ingest(later, None, user_id="cy", session_id="s3", occurred_at=t0 + timedelta(hours=2))
    tasks = eng.list_tasks(user_id="cy", session_id="s3")
    assert tasks[1]["id"] == child["id"] and tasks[1]["close_reason"] == "idle"


@pytest.mark.asyncio
async def test_explicit_task_success_failure_and_mark_outcome(eng):
    with eng.scope(user_id="dee", session_id="s4"):
        with eng.task("Rotate the API keys") as t:
            await eng.ingest([{"role": "user", "content": "rotate keys"}, call("c1", "rotate", {"env": "prod"}),
                              result("c1", "rotated 3 keys")], "Rotated.")
            t.success("all keys rotated")
        with pytest.raises(RuntimeError):
            with eng.task("Migrate the database"):
                raise RuntimeError("lock timeout")
    tasks = {x["goal"]: x for x in eng.list_tasks(user_id="dee")}
    ok, bad = tasks["Rotate the API keys"], tasks["Migrate the database"]
    assert (ok["outcome"], ok["outcome_source"], ok["outcome_confidence"]) == ("success", "explicit", 1.0)
    assert ok["explicit"] and ok["used_tools"] == ["rotate"] and ok["status"] == "closed"
    assert bad["outcome"] == "failure" and "lock timeout" in bad["outcome_reason"]
    s = eng._storage.get_session()
    try:
        assert {r.task_id for r in s.query(Turn).filter(Turn.session_id == "s4")} == {ok["id"]}
    finally:
        s.close()
    eng.mark_outcome(False, "rolled back later", task_id=ok["id"])
    assert eng.get_task(ok["id"], user_id="dee")["task"]["outcome"] == "failure"


def test_interceptor_records_offered_tools_and_task(engine_factory):
    eng = engine_factory()
    eng.enable()
    try:
        rec = Script([_openai_tool_reply(), OPENAI_CHAT])
        client = _openai(rec)
        tools = [{"type": "function", "function": {"name": "lookup_order", "parameters": {"type": "object"}}},
                 {"type": "function", "function": {"name": "refund", "parameters": {"type": "object"}}}]
        history = [{"role": "user", "content": "Where is my order ord_4417?"}]
        with eng.scope(user_id="eve", session_id="s5"):
            first = client.chat.completions.create(model="m", messages=list(history), tools=tools)
            history.append(first.choices[0].message.model_dump(exclude_none=True))
            history.append(result("call_9", json.dumps({"status": 200, "state": "shipped"})))
            client.chat.completions.create(model="m", messages=list(history), tools=tools)
        [t] = eng.list_tasks(user_id="eve", session_id="s5")
        assert t["offered_tools"] == ["lookup_order", "refund"] and t["used_tools"] == ["lookup_order"]
        assert t["llm_calls"] == 2 and t["status"] == "closed"
        assert (t["outcome"], t["outcome_source"]) == ("success", "signal"), "HTTP 200 in the last result"
    finally:
        eng.disable()


@pytest.mark.asyncio
async def test_record_trace_from_any_framework(eng):
    t = await eng.record_trace(
        "Fix the failing login test",
        steps=[
            {"tool": "bash", "args": {"cmd": "pytest tests/test_api.py"}, "result": "1 failed", "rationale": "reproduce first"},
            {"tool": "edit", "args": {"path": "api/auth.py", "line": 42}, "result": {"ok": True}},
            {"tool": "bash", "args": {"cmd": "pytest tests/test_api.py"}, "result": "12 passed in 0.4s"},
        ],
        final_answer="Fixed the token expiry check in api/auth.py:42.",
        user_id="fay", agent_id="coder",
    )
    assert t["goal"] == "Fix the failing login test" and t["explicit"]
    assert t["used_tools"] == ["bash", "edit"] and t["tool_calls"] == 3
    assert (t["outcome"], t["outcome_source"]) == ("success", "signal"), "last step: tests passed"
    events = eng.get_task(t["id"], user_id="fay")["events"]
    assert [e["event_kind"] for e in events] == [
        "message", "message", "tool_call", "tool_result", "tool_call", "tool_result", "tool_call", "tool_result",
        "message"]
    failed = await eng.record_trace("Deploy", steps=[{"tool": "deploy", "args": {}, "result": "boom", "is_error": True}],
                                    user_id="fay")
    assert failed["outcome"] == "failure" and failed["error_count"] == 1
    told = await eng.record_trace("Book a table", steps=[{"tool": "book", "args": {"n": 2}, "result": "ok"}],
                                  outcome="success", outcome_reason="user confirmed", user_id="fay")
    assert (told["outcome"], told["outcome_source"], told["outcome_reason"]) == ("success", "explicit", "user confirmed")
