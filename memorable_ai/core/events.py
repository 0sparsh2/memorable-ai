"""
L0 conversation events: messages, tool calls and tool results (procedural P1,
docs/procedural_memory_design.md §4.1 and §5).

``normalize_events`` turns any supported request/response shape into one ordered list:
  OpenAI chat         assistant ``tool_calls`` and ``role: "tool"`` results
  Anthropic           ``tool_use`` / ``tool_result`` content blocks
  OpenAI Responses    ``function_call`` / ``function_call_output`` items
  SDK objects         anything with ``model_dump()``
Each tool event carries a redacted payload (secrets always; PII per mode), salient
``values`` extracted *before* any size cut (for value-flow attribution later), an error
flag, and a short rendering in ``content`` for display. Payloads above
``max_payload_bytes`` keep head + tail + sha256 + length (decision D3).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from memorable_ai.core.redact import redact

MESSAGE, TOOL_CALL, TOOL_RESULT = "message", "tool_call", "tool_result"
CONVERSATION_ROLES = {"user", "assistant"}
MAX_PAYLOAD_BYTES = 256 * 1024
PREVIEW_CHARS = 600
MAX_VALUES = 64


def plain(obj: Any) -> Any:
    """SDK objects (pydantic) to plain dicts; everything else unchanged."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(exclude_none=True)
        except Exception:
            pass
    return obj


def message_text(content: Any) -> str:
    """Flatten OpenAI/Anthropic content (string or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            p = plain(p)
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") in ("text", "input_text", "output_text") or ("text" in p and "type" not in p):
                    parts.append(str(p.get("text", "")))
            else:
                text = getattr(p, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(x for x in parts if x)
    return str(content)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if s[:1] in "{[" and s[-1:] in "}]":
            try:
                return json.loads(s)
            except ValueError:
                return value
    return value


def _dumps(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def cap_payload(value: Any, max_bytes: int = MAX_PAYLOAD_BYTES) -> Any:
    """``value`` itself, or head + tail + sha256 + length when its JSON exceeds ``max_bytes``."""
    raw = _dumps(value).encode("utf-8")
    if max_bytes <= 0 or len(raw) <= max_bytes:
        return value
    return {
        "truncated": True,
        "length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "head": raw[: max_bytes * 3 // 4].decode("utf-8", errors="ignore"),
        "tail": raw[-(max_bytes // 4):].decode("utf-8", errors="ignore"),
    }


_VALUE_RE = re.compile(
    r"(?:https?://[^\s\"'<>]+)|(?:[\w./-]+\.(?:py|js|ts|go|rs|java|json|yaml|yml|md|txt|csv|sql|sh)\b)"
    r"|(?:\b[A-Za-z][\w-]*[_-]\w*\d\w*\b)|(?:\b\d{3,}\b)|(?:\"[^\"\n]{3,80}\")"
)


def extract_values(value: Any, limit: int = MAX_VALUES) -> List[Any]:
    """Salient values (ids, paths, URLs, numbers, short strings) for value-flow attribution."""
    seen: Dict[str, None] = {}
    out: List[Any] = []

    def add(v: Any) -> None:
        key = repr(v)
        if key not in seen and len(out) < limit:
            seen[key] = None
            out.append(v)

    def walk(v: Any) -> None:
        if len(out) >= limit:
            return
        v = _maybe_json(v)
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, bool) or v is None:
            return
        elif isinstance(v, (int, float)):
            add(v)
        elif isinstance(v, str):
            s = v.strip()
            if 3 <= len(s) <= 120 and "\n" not in s:
                add(s)
            elif len(s) > 120:
                for m in _VALUE_RE.findall(s[:20000]):
                    add(m.strip('"'))

    walk(value)
    return out


_ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|^\s*(?:error|exception|fatal)\b[:\s]|exit (?:code|status)[:= ]+[1-9]",
    re.I | re.M,
)


def detect_error(result: Any, explicit: Optional[bool] = None) -> bool:
    """Conservative: explicit flags, tracebacks, error lines, non-zero exit codes, {"error": ...}."""
    if explicit:
        return True
    v = _maybe_json(result)
    if isinstance(v, dict):
        if v.get("error") not in (None, "", False, {}, []) or str(v.get("status", "")).lower() in ("error", "failed"):
            return True
        if isinstance(v.get("exit_code"), int) and v["exit_code"] != 0:
            return True
    text = message_text(result) if not isinstance(result, (dict, list)) else _dumps(result)
    return bool(_ERROR_RE.search(text[:20000]))


def _preview(value: Any, limit: int = PREVIEW_CHARS) -> str:
    s = _dumps(value).strip()
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _tool_event(kind: str, name: Optional[str], call_id: Optional[str], data: Any, pii: str, max_bytes: int,
                is_error: Optional[bool] = None) -> Dict[str, Any]:
    data = _maybe_json(plain(data))
    if kind == TOOL_RESULT and isinstance(data, list):  # Anthropic result blocks -> text when possible
        txt = message_text(data)
        data = _maybe_json(txt) if txt else data
    data = redact(data, pii)
    values = extract_values(data)
    err = detect_error(data, is_error) if kind == TOOL_RESULT else False
    name = name or None
    if kind == TOOL_CALL:
        content = f"{name or 'tool'}({_preview(data, 400)})"
    else:
        content = f"{'[error] ' if err else ''}{name or 'tool'} -> {_preview(data)}"
    return {
        "role": "assistant" if kind == TOOL_CALL else "tool",
        "content": content,
        "kind": kind,
        "tool_name": name,
        "tool_call_id": str(call_id) if call_id else None,
        "payload": cap_payload(data, max_bytes),
        "values": values,
        "is_error": err,
        "source": "assistant" if kind == TOOL_CALL else "tool",
    }


def _message(role: str, text: str, m: Dict[str, Any], pii: str) -> Dict[str, Any]:
    item: Dict[str, Any] = {"role": role, "content": redact(text, pii) if pii != "off" else text,
                            "kind": MESSAGE, "source": role}
    ts = m.get("timestamp") or m.get("occurred_at") or (m.get("metadata") or {}).get("timestamp")
    if ts:
        item["timestamp"] = ts
    return item


def normalize_events(
    messages: Sequence[Any],
    pii: str = "off",
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
    tool_events: bool = True,
) -> List[Dict[str, Any]]:
    """One ordered list of message and tool events (see module docstring)."""
    out: List[Dict[str, Any]] = []
    names: Dict[str, str] = {}

    def call(name, call_id, args):
        if call_id and name:
            names[str(call_id)] = name
        out.append(_tool_event(TOOL_CALL, name, call_id, args, pii, max_payload_bytes))

    def result(call_id, output, name=None, is_error=None):
        name = name or (names.get(str(call_id)) if call_id else None)
        out.append(_tool_event(TOOL_RESULT, name, call_id, output, pii, max_payload_bytes, is_error))

    for m in messages or []:
        m = plain(m)
        if isinstance(m, str):
            out.append({"role": "user", "content": m, "kind": MESSAGE, "source": "user"})
            continue
        if not isinstance(m, dict):
            m = {"role": getattr(m, "role", None), "content": getattr(m, "content", None),
                 "tool_calls": getattr(m, "tool_calls", None)}
        typ = m.get("type")
        if typ == "function_call":  # OpenAI Responses
            call(m.get("name"), m.get("call_id") or m.get("id"), m.get("arguments"))
            continue
        if typ == "function_call_output":
            result(m.get("call_id"), m.get("output"))
            continue
        role = m.get("role")
        if role in ("tool", "function"):  # OpenAI chat tool result (and legacy function result)
            result(m.get("tool_call_id"), m.get("content"), name=m.get("name"))
            continue
        if role not in CONVERSATION_ROLES:
            continue
        content = m.get("content")
        blocks = [plain(b) for b in content] if isinstance(content, list) else None
        if blocks is not None and any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
                                      for b in blocks):
            # Anthropic: results answer earlier calls (they lead a user turn); calls follow the text
            text = message_text([b for b in blocks if not (isinstance(b, dict) and
                                                          b.get("type") in ("tool_use", "tool_result"))]).strip()
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    result(b.get("tool_use_id"), b.get("content"), is_error=b.get("is_error"))
            if text:
                out.append(_message(role, text, m, pii))
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    call(b.get("name"), b.get("id"), b.get("input"))
        else:
            text = message_text(content).strip()
            if text:
                out.append(_message(role, text, m, pii))
        for tc in m.get("tool_calls") or []:  # OpenAI chat
            tc = plain(tc)
            fn = plain(tc.get("function")) or {}
            call(fn.get("name"), tc.get("id"), fn.get("arguments"))
        fc = plain(m.get("function_call"))
        if isinstance(fc, dict) and fc.get("name"):  # legacy OpenAI functions
            call(fc.get("name"), None, fc.get("arguments"))
    if not tool_events:
        out = [e for e in out if e["kind"] == MESSAGE]
    return out


def event_hash(e: Dict[str, Any]) -> str:
    """Dedupe key. Messages keep the v0.2 hash (existing databases stay valid)."""
    if e.get("kind", MESSAGE) == MESSAGE:
        return hashlib.sha1(f"{e['role']}\x00{e['content'].strip()}".encode("utf-8")).hexdigest()
    key = f"{e['role']}\x00{e['kind']}\x00{e.get('tool_call_id') or ''}\x00{e['content'].strip()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()
