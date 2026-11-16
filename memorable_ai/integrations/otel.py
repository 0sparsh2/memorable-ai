"""
OpenTelemetry GenAI ingest (procedural P1c; decision D4 "standards first").

Turns spans that follow the GenAI semantic conventions into Memorable L0 events:
  chat / generate_content / text_completion spans
      gen_ai.input.messages, gen_ai.output.messages: ChatMessage {role, parts}
      parts: text {content}, tool_call {id, name, arguments}, tool_call_response {id, response}
      gen_ai.tool.definitions: tools offered to the model
  execute_tool spans
      gen_ai.tool.name, gen_ai.tool.call.id, gen_ai.tool.call.arguments, gen_ai.tool.call.result,
      error.type / span status ERROR
  gen_ai.conversation.id (session), gen_ai.agent.id / gen_ai.agent.name (agent), user.id / enduser.id
Message content is opt-in in the spec (OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT); without it,
only tool spans carry data.

Every chat span repeats the conversation so far, and tool spans repeat its calls, so each
conversation becomes one canonical stream: the last chat span's input + output, with tool
spans filling in calls the messages lack and flagging errors.

Accepts OTLP-JSON (``resourceSpans`` / ``scopeSpans`` / ``spans`` with typed attribute
values) or plain dicts ``{name, trace_id, span_id, start_time, attributes, status}``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

CHAT_OPS = {"chat", "generate_content", "text_completion"}


def _any_value(v: Any) -> Any:
    """OTLP AnyValue -> Python."""
    if not isinstance(v, dict):
        return v
    for k in ("stringValue", "boolValue", "doubleValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        return int(v["intValue"])
    if "arrayValue" in v:
        return [_any_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {kv["key"]: _any_value(kv.get("value")) for kv in v["kvlistValue"].get("values", [])}
    return v


def _attrs(span: Dict[str, Any]) -> Dict[str, Any]:
    a = span.get("attributes") or {}
    if isinstance(a, list):  # OTLP: [{key, value: AnyValue}]
        return {kv["key"]: _any_value(kv.get("value")) for kv in a}
    return dict(a)


def flatten_spans(payload: Any) -> List[Dict[str, Any]]:
    """OTLP-JSON export, a list of plain span dicts, or a single span -> plain span dicts."""
    if isinstance(payload, dict) and "resourceSpans" in payload:
        out = []
        for rs in payload["resourceSpans"]:
            res_attrs = _attrs(rs.get("resource") or {})
            for ss in rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []:
                for sp in ss.get("spans", []):
                    out.append({
                        "name": sp.get("name"), "trace_id": sp.get("traceId"), "span_id": sp.get("spanId"),
                        "parent_span_id": sp.get("parentSpanId"),
                        "start_time": int(sp.get("startTimeUnixNano") or 0),
                        "attributes": {**res_attrs, **_attrs(sp)},
                        "status": {"code": "ERROR" if (sp.get("status") or {}).get("code") in (2, "STATUS_CODE_ERROR")
                                   else "OK"},
                    })
        return out
    if isinstance(payload, dict):
        return [payload]
    return [dict(s, attributes=_attrs(s)) for s in payload or []]


def _json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _to_messages(chat_messages: Any) -> List[Dict[str, Any]]:
    """GenAI ChatMessage list -> OpenAI-style messages that ``normalize_events`` reads."""
    out: List[Dict[str, Any]] = []
    for m in _json(chat_messages) or []:
        if not isinstance(m, dict):
            continue
        role, parts = m.get("role"), m.get("parts") or []
        text = "\n".join(str(p.get("content", "")) for p in parts if isinstance(p, dict) and p.get("type") == "text")
        calls = [p for p in parts if isinstance(p, dict) and p.get("type") == "tool_call"]
        responses = [p for p in parts if isinstance(p, dict) and p.get("type") == "tool_call_response"]
        if role in ("user", "assistant") and (text or calls):
            msg: Dict[str, Any] = {"role": role, "content": text or None}
            if calls:
                msg["tool_calls"] = [{"id": c.get("id"), "type": "function", "function": {
                    "name": c.get("name"), "arguments": json.dumps(c.get("arguments"), default=str)
                    if not isinstance(c.get("arguments"), str) else c.get("arguments")}} for c in calls]
            out.append(msg)
        for r in responses:
            resp = r.get("response")
            out.append({"role": "tool", "tool_call_id": r.get("id"),
                        "content": resp if isinstance(resp, str) else json.dumps(resp, default=str)})
    return out


def _tool_names(definitions: Any) -> List[str]:
    names = []
    for d in _json(definitions) or []:
        if isinstance(d, dict) and d.get("name") and d["name"] not in names:
            names.append(str(d["name"]))
    return names


def conversations(payload: Any) -> List[Dict[str, Any]]:
    """
    Group spans into conversations and build each one's canonical message stream.
    Returns ``[{session_id, user_id, agent_id, messages, offered_tools}]`` in time order.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sp in flatten_spans(payload):
        a = sp["attributes"]
        key = a.get("gen_ai.conversation.id") or f"otel-{sp.get('trace_id') or 'unknown'}"
        groups[str(key)].append(sp)
    out = []
    for key, spans in groups.items():
        spans.sort(key=lambda s: s.get("start_time") or 0)
        chats = [s for s in spans if s["attributes"].get("gen_ai.operation.name") in CHAT_OPS]
        tools = [s for s in spans if s["attributes"].get("gen_ai.operation.name") == "execute_tool"]
        messages: List[Dict[str, Any]] = []
        offered: List[str] = []
        for s in chats:
            offered += [n for n in _tool_names(s["attributes"].get("gen_ai.tool.definitions")) if n not in offered]
        if chats:
            last = chats[-1]["attributes"]
            messages = _to_messages(last.get("gen_ai.input.messages")) + _to_messages(last.get("gen_ai.output.messages"))
        seen_calls = {tc.get("id") for m in messages for tc in m.get("tool_calls") or []}
        seen_results = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
        errors = {}
        for s in tools:
            a = s["attributes"]
            cid = a.get("gen_ai.tool.call.id") or f"{s.get('span_id')}"
            failed = (s.get("status") or {}).get("code") == "ERROR" or bool(a.get("error.type"))
            if failed:
                errors[cid] = a.get("error.type") or "error"
            if cid not in seen_calls:
                args = a.get("gen_ai.tool.call.arguments")
                messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": cid, "type": "function",
                                 "function": {"name": a.get("gen_ai.tool.name"), "arguments": args if isinstance(args, str)
                                              else json.dumps(args or {}, default=str)}}]})
                seen_calls.add(cid)
            if cid not in seen_results:
                res = a.get("gen_ai.tool.call.result")
                if failed:
                    res = {"error": a.get("error.type") or "error", "result": _json(res)}
                messages.append({"role": "tool", "tool_call_id": cid, "name": a.get("gen_ai.tool.name"),
                                 "content": res if isinstance(res, str) else json.dumps(res, default=str)})
                seen_results.add(cid)
        for m in messages:  # span status marks errors the message content may not show
            if m.get("role") == "tool" and m.get("tool_call_id") in errors and '"error"' not in str(m.get("content")):
                m["content"] = json.dumps({"error": errors[m["tool_call_id"]], "result": _json(m.get("content"))},
                                          default=str)
        first = spans[0]["attributes"]
        out.append({
            "session_id": key,
            "user_id": first.get("user.id") or first.get("enduser.id"),
            "agent_id": first.get("gen_ai.agent.id") or first.get("gen_ai.agent.name"),
            "messages": messages,
            "offered_tools": offered,
            "start_time": spans[0].get("start_time"),
        })
    out.sort(key=lambda c: c.get("start_time") or 0)
    return out


def iter_conversations(payload: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for c in conversations(payload):
        yield c["session_id"], c
