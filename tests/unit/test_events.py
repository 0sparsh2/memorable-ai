"""L0 tool events: normalization per provider, redaction, payload cap, values, error flags, hashes."""

import hashlib
import json

from memorable_ai.core.events import (MESSAGE, TOOL_CALL, TOOL_RESULT, cap_payload, detect_error, event_hash,
                                      extract_values, normalize_events)
from memorable_ai.core.redact import redact, redact_text


def kinds(events):
    return [e["kind"] for e in events]


def test_openai_chat_tool_loop():
    msgs = [
        {"role": "system", "content": "You are a travel agent."},
        {"role": "user", "content": "Find flights SFO to Lisbon on May 3"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "search_flights", "arguments": json.dumps({"from": "SFO", "to": "LIS", "date": "2026-05-03"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"flights": [{"id": "TP238", "price": 612}]})},
        {"role": "assistant", "content": "TP238 at $612 is the cheapest."},
    ]
    ev = normalize_events(msgs)
    assert kinds(ev) == [MESSAGE, TOOL_CALL, TOOL_RESULT, MESSAGE]
    call, res = ev[1], ev[2]
    assert call["tool_name"] == "search_flights" and call["tool_call_id"] == "call_1"
    assert call["payload"] == {"from": "SFO", "to": "LIS", "date": "2026-05-03"}
    assert res["tool_name"] == "search_flights", "result is paired with its call by id"
    assert "TP238" in res["values"] and 612 in res["values"]
    assert res["role"] == "tool" and res["source"] == "tool" and not res["is_error"]
    assert call["content"].startswith("search_flights(")


def test_anthropic_blocks_results_lead_and_calls_follow_text():
    msgs = [
        {"role": "user", "content": "Run the tests"},
        {"role": "assistant", "content": [{"type": "text", "text": "Running them now."},
                                          {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"cmd": "pytest -q"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "is_error": True,
                                      "content": [{"type": "text", "text": "2 failed, 10 passed"}]},
                                     {"type": "text", "text": "fix them"}]},
    ]
    ev = normalize_events(msgs)
    assert kinds(ev) == [MESSAGE, MESSAGE, TOOL_CALL, TOOL_RESULT, MESSAGE]
    assert ev[2]["tool_name"] == "bash" and ev[2]["payload"] == {"cmd": "pytest -q"}
    assert ev[3]["tool_name"] == "bash" and ev[3]["is_error"] is True
    assert ev[4]["content"] == "fix them"


def test_responses_api_items_and_sdk_objects():
    class Obj:
        def __init__(self, d):
            self.d = d

        def model_dump(self, exclude_none=True):
            return self.d

    items = [
        {"role": "user", "content": "weather in Oslo?"},
        Obj({"type": "function_call", "name": "get_weather", "call_id": "fc_1", "arguments": '{"city": "Oslo"}'}),
        {"type": "function_call_output", "call_id": "fc_1", "output": "4C, light rain"},
    ]
    ev = normalize_events(items)
    assert kinds(ev) == [MESSAGE, TOOL_CALL, TOOL_RESULT]
    assert ev[1]["payload"] == {"city": "Oslo"} and ev[2]["tool_name"] == "get_weather"


def test_tool_events_can_be_turned_off():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]}]
    assert kinds(normalize_events(msgs, tool_events=False)) == [MESSAGE]


def test_secrets_are_always_redacted_in_payloads():
    secret_key = "sk-proj-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4"
    msgs = [{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "http", "arguments": json.dumps(
        {"url": "https://admin:hunter2pass@db.internal/x", "headers": {"Authorization": "Bearer abcdefghijklmnopqrstu"},
         "api_key": secret_key, "note": f"use {secret_key} here"})}}]}]
    p = normalize_events(msgs)[0]["payload"]
    blob = json.dumps(p)
    assert secret_key not in blob and "hunter2pass" not in blob and "abcdefghijklmnopqrstu" not in blob
    assert p["api_key"] == "[REDACTED:secret]" and p["headers"]["Authorization"] == "[REDACTED:secret]"
    assert "[REDACTED:url_credentials]" in p["url"] and "[REDACTED:openai_key]" in p["note"]


def test_pii_modes():
    t = "Mail ana.silva@example.com or call +1 415-555-0134; server 10.0.0.12; card 4111 1111 1111 1111"
    assert redact_text(t, "off") == t
    mid = redact_text(t, "emails_phones")
    assert "ana.silva@example.com" not in mid and "415-555-0134" not in mid and "10.0.0.12" in mid
    strict = redact_text(t, "strict")
    assert "10.0.0.12" not in strict and "4111 1111 1111 1111" not in strict
    assert redact({"password": "pw-123456"}, "off") == {"password": "[REDACTED:secret]"}


def test_payload_cap_keeps_head_tail_hash_and_values_from_the_full_payload():
    big = {"rows": ["x" * 50] * 4000, "last_id": "ord_998877"}
    capped = cap_payload(big, max_bytes=4096)
    raw = json.dumps(big, separators=(",", ":")).encode()
    assert capped["truncated"] and capped["length"] == len(raw)
    assert capped["sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(capped["head"]) + len(capped["tail"]) <= 4096
    ev = normalize_events([{"role": "tool", "tool_call_id": "c", "content": json.dumps(big)}], max_payload_bytes=4096)[0]
    assert ev["payload"]["truncated"] and "ord_998877" in ev["values"], "values come from before the cut"
    assert cap_payload({"a": 1}, max_bytes=4096) == {"a": 1}


def test_error_detection_is_conservative():
    assert detect_error("Traceback (most recent call last):\n  File x")
    assert detect_error({"error": {"code": 500}}) and detect_error({"exit_code": 2})
    assert detect_error("exit code: 1") and detect_error("ok", explicit=True)
    assert not detect_error("All 12 tests passed") and not detect_error({"error": None, "result": 3})
    assert not detect_error("the error rate fell to 0.1%")


def test_extract_values_and_hashes():
    vals = extract_values({"path": "src/app/main.py", "n": 42, "flag": True, "nested": [{"id": "usr_123"}]})
    assert vals == ["src/app/main.py", 42, "usr_123"]
    msg = {"role": "user", "content": "hello", "kind": MESSAGE}
    assert event_hash(msg) == hashlib.sha1("user\x00hello".encode()).hexdigest(), "v0.2 message hashes unchanged"
    a = {"role": "tool", "content": "f -> ok", "kind": TOOL_RESULT, "tool_call_id": "c1"}
    b = dict(a, tool_call_id="c2")
    assert event_hash(a) != event_hash(b)
