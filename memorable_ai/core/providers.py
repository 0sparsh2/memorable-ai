"""
Provider adapters: how to read, inject into, and read back from each SDK call shape.

Each adapter knows:
  * where the conversation is (``messages`` / ``input``)
  * where to put memory context *without breaking the API*. Anthropic rejects a
    ``{"role": "system"}`` message, so context goes into the ``system`` parameter;
    the Responses API uses ``instructions``. Context is appended after the caller's
    own system prompt so their prompt-cache prefix is preserved
  * where user/session hints are (``metadata.user_id``, ``user``, ``safety_identifier``,
    ``conversation``)
  * how to read the reply, including streamed deltas
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from memorable_ai.core.extraction import response_text as _response_text


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_given(value: Any) -> bool:
    """False for None and the SDKs' NOT_GIVEN/Omit sentinels."""
    if value is None:
        return False
    return type(value).__name__ not in ("NotGiven", "Omit")


class Adapter:
    name = "base"

    def bind(self, args: tuple, kwargs: Dict[str, Any]) -> Tuple[tuple, Dict[str, Any]]:
        return args, kwargs

    def get_messages(self, kwargs: Dict[str, Any]) -> List[Any]:
        msgs = kwargs.get("messages")
        return list(msgs) if isinstance(msgs, (list, tuple)) else []

    def get_hints(self, kwargs: Dict[str, Any]) -> Dict[str, Optional[str]]:
        md = kwargs.get("metadata")
        md = md if isinstance(md, dict) else {}
        user = md.get("user_id") or kwargs.get("user") or kwargs.get("safety_identifier")
        session = md.get("session_id") or md.get("conversation_id") or md.get("thread_id")
        hints = {
            "user_id": str(user) if _is_given(user) and user else None,
            "session_id": str(session) if session else None,
        }
        if md.get("agent_id"):
            hints["agent_id"] = str(md["agent_id"])
        return hints

    def inject(self, kwargs: Dict[str, Any], context: str, position: str = "system") -> Dict[str, Any]:
        """Add ``context`` to the request: "tail" (latest user message) or "system" (system prompt)."""
        raise NotImplementedError

    def get_tools(self, kwargs: Dict[str, Any]) -> List[str]:
        """Names of the tools offered to the model in this call (OpenAI, Anthropic and Responses shapes)."""
        names: List[str] = []
        for t in list(kwargs.get("tools") or []) + list(kwargs.get("functions") or []):
            fn = _get(t, "function")
            name = _get(fn, "name") if fn is not None else (_get(t, "name") or _get(t, "type"))
            if name and name not in names:
                names.append(str(name))
        return names

    def is_stream(self, kwargs: Dict[str, Any]) -> bool:
        return kwargs.get("stream") is True

    def response_text(self, response: Any) -> str:
        return _response_text(response)

    def response_events(self, response: Any) -> List[Any]:
        """The model's reply as L0 events (text and tool calls), for recording."""
        text = self.response_text(response)
        return [{"role": "assistant", "content": text}] if text else []

    def stream_delta(self, chunk: Any) -> str:
        return ""

    def stream_tools(self, chunk: Any, acc: Dict[Any, Dict[str, Any]]) -> None:
        """Accumulate streamed tool-call fragments into ``acc`` (provider-specific)."""
        return None

    def stream_events(self, text: str, acc: Dict[Any, Dict[str, Any]]) -> List[Any]:
        """A finished stream as L0 events: its text plus the tool calls assembled from fragments."""
        calls = sorted(acc.values(), key=lambda c: c.get("order", 0))
        msg: Dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            msg["tool_calls"] = [{"id": c.get("id"), "type": "function", "function": {
                "name": c.get("name"), "arguments": "".join(c.get("args") or []) or "{}"}} for c in calls]
        return [msg] if (text or calls) else []


def memory_block(context: str) -> str:
    """Retrieved memory, delimited so it reads as reference data next to the user's own words."""
    text = context.strip()
    return text if text.startswith("<memory>") else f"<memory>\n{text}\n</memory>"


def _prepend(content: Any, block: str, part_type: str = "text") -> Any:
    if isinstance(content, str):
        return f"{block}\n\n{content}" if content else block
    if isinstance(content, (list, tuple)):
        return [{"type": part_type, "text": block}] + list(content)
    return block


def _last_user_index(items: Sequence[Any]) -> Optional[int]:
    for i in range(len(items) - 1, -1, -1):
        if isinstance(items[i], dict) and items[i].get("role") == "user":
            return i
    return None


def last_user_text(messages: Sequence[Any]) -> str:
    """Text of the latest user message (the retrieval query); "" if there is none."""
    i = _last_user_index(list(messages or []))
    if i is None:
        return ""
    content = messages[i].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def _inject_tail(items: Sequence[Any], context: str, part_type: str = "text") -> Optional[List[Any]]:
    """
    Put memory at the tail: in front of the latest user message's text. Everything before
    that message stays byte-identical across calls, so the provider's prompt cache keeps
    the whole conversation prefix (it is re-billed in full when memory sits in the system
    prompt and changes every turn). None when there is no user message.
    """
    items = list(items or [])
    i = _last_user_index(items)
    if i is None:
        return None
    msg = dict(items[i])
    msg["content"] = _prepend(msg.get("content"), memory_block(context), part_type)
    items[i] = msg
    return items


class OpenAIChatAdapter(Adapter):
    """openai ``chat.completions.create`` (and OpenAI-compatible shapes)."""

    name = "openai.chat"

    def inject(self, kwargs: Dict[str, Any], context: str, position: str = "system") -> Dict[str, Any]:
        new = dict(kwargs)
        if position == "tail":
            tail = _inject_tail(kwargs.get("messages") or [], context)
            if tail is not None:
                new["messages"] = tail
                return new
        messages = list(kwargs.get("messages") or [])
        if messages and isinstance(messages[0], dict) and messages[0].get("role") in ("system", "developer"):
            first = dict(messages[0])
            content = first.get("content")
            if isinstance(content, str):
                first["content"] = f"{content}\n\n{context}" if content else context
            elif isinstance(content, list):
                first["content"] = list(content) + [{"type": "text", "text": context}]
            else:
                first["content"] = context
            messages[0] = first
        else:
            messages.insert(0, {"role": "system", "content": context})
        new["messages"] = messages
        return new

    def response_events(self, response: Any) -> List[Any]:
        choices = _get(response, "choices") or []
        msg = _get(choices[0], "message") if choices else None
        if msg is None:
            return super().response_events(response)
        return [{"role": "assistant", "content": _get(msg, "content"), "tool_calls": _get(msg, "tool_calls"),
                 "function_call": _get(msg, "function_call")}]

    def stream_tools(self, chunk: Any, acc: Dict[Any, Dict[str, Any]]) -> None:
        choices = _get(chunk, "choices") or []
        delta = _get(choices[0], "delta") if choices else None
        for tc in (_get(delta, "tool_calls") or []) if delta is not None else []:
            slot = acc.setdefault(("openai", _get(tc, "index") or 0), {"order": len(acc), "args": []})
            if _get(tc, "id"):
                slot["id"] = _get(tc, "id")
            fn = _get(tc, "function")
            if fn is not None:
                if _get(fn, "name"):
                    slot["name"] = _get(fn, "name")
                if _get(fn, "arguments"):
                    slot["args"].append(_get(fn, "arguments"))

    def stream_delta(self, chunk: Any) -> str:
        choices = _get(chunk, "choices") or []
        if not choices:
            return ""
        delta = _get(choices[0], "delta")
        return _get(delta, "content") or "" if delta is not None else ""


class LiteLLMAdapter(OpenAIChatAdapter):
    """``litellm.completion`` / ``acompletion``: OpenAI shape, messages may be positional."""

    name = "litellm"

    def bind(self, args: tuple, kwargs: Dict[str, Any]) -> Tuple[tuple, Dict[str, Any]]:
        if "messages" not in kwargs and len(args) == 2 and isinstance(args[1], (list, tuple)):
            kwargs = dict(kwargs)
            kwargs["messages"] = args[1]
            args = args[:1]
        return args, kwargs


class AnthropicAdapter(Adapter):
    """anthropic ``messages.create``: context goes into the latest user turn or ``system`` (never a system-role message)."""

    name = "anthropic.messages"

    def get_hints(self, kwargs: Dict[str, Any]) -> Dict[str, Optional[str]]:
        md = kwargs.get("metadata")
        md = md if isinstance(md, dict) else {}
        user = md.get("user_id")
        hints = {"user_id": str(user) if user else None, "session_id": None}
        if md.get("agent_id"):
            hints["agent_id"] = str(md["agent_id"])
        return hints

    def inject(self, kwargs: Dict[str, Any], context: str, position: str = "system") -> Dict[str, Any]:
        new = dict(kwargs)
        if position == "tail":
            tail = _inject_tail(kwargs.get("messages") or [], context)
            if tail is not None:
                new["messages"] = tail
                return new
        system = kwargs.get("system")
        if isinstance(system, str) and system:
            new["system"] = f"{system}\n\n{context}"
        elif isinstance(system, (list, tuple)) and system:
            new["system"] = list(system) + [{"type": "text", "text": context}]
        else:
            new["system"] = context
        return new

    def response_events(self, response: Any) -> List[Any]:
        content = _get(response, "content")
        if not content:
            return super().response_events(response)
        return [{"role": "assistant", "content": list(content)}]

    def stream_tools(self, chunk: Any, acc: Dict[Any, Dict[str, Any]]) -> None:
        typ = _get(chunk, "type")
        if typ == "content_block_start":
            block = _get(chunk, "content_block")
            if _get(block, "type") == "tool_use":
                acc[("anthropic", _get(chunk, "index"))] = {"order": len(acc), "id": _get(block, "id"),
                                                             "name": _get(block, "name"), "args": []}
        elif typ == "content_block_delta":
            delta = _get(chunk, "delta")
            slot = acc.get(("anthropic", _get(chunk, "index")))
            if slot is not None and _get(delta, "type") == "input_json_delta":
                slot["args"].append(_get(delta, "partial_json") or "")

    def stream_delta(self, chunk: Any) -> str:
        if _get(chunk, "type") != "content_block_delta":
            return ""
        delta = _get(chunk, "delta")
        if _get(delta, "type") == "text_delta":
            return _get(delta, "text") or ""
        return ""


class OpenAIResponsesAdapter(Adapter):
    """openai ``responses.create``: conversation in ``input``, context in ``instructions``."""

    name = "openai.responses"

    def get_messages(self, kwargs: Dict[str, Any]) -> List[Any]:
        inp = kwargs.get("input")
        if isinstance(inp, str):
            return [{"role": "user", "content": inp}]
        out: List[Any] = []
        for item in inp or []:
            role = _get(item, "role")
            typ = _get(item, "type")
            if role in ("user", "assistant", "system", "developer"):
                out.append({"role": role, "content": _get(item, "content")})
            elif typ == "function_call":
                out.append({"type": typ, "name": _get(item, "name"), "call_id": _get(item, "call_id"),
                            "arguments": _get(item, "arguments")})
            elif typ == "function_call_output":
                out.append({"type": typ, "call_id": _get(item, "call_id"), "output": _get(item, "output")})
        return out

    def get_hints(self, kwargs: Dict[str, Any]) -> Dict[str, Optional[str]]:
        hints = super().get_hints(kwargs)
        conv = kwargs.get("conversation")
        if not hints["session_id"] and _is_given(conv) and conv:
            hints["session_id"] = str(_get(conv, "id") or conv) if not isinstance(conv, str) else conv
        return hints

    def inject(self, kwargs: Dict[str, Any], context: str, position: str = "system") -> Dict[str, Any]:
        new = dict(kwargs)
        if position == "tail":
            inp = kwargs.get("input")
            if isinstance(inp, str):
                new["input"] = _prepend(inp, memory_block(context))
                return new
            tail = _inject_tail(inp or [], context, part_type="input_text")
            if tail is not None:
                new["input"] = tail
                return new
        instructions = kwargs.get("instructions")
        if isinstance(instructions, str) and instructions:
            new["instructions"] = f"{instructions}\n\n{context}"
        else:
            new["instructions"] = context
        return new

    def response_text(self, response: Any) -> str:
        text = _get(response, "output_text")
        if isinstance(text, str) and text:
            return text
        return _response_text(response)

    def response_events(self, response: Any) -> List[Any]:
        events: List[Any] = []
        for item in _get(response, "output") or []:
            typ = _get(item, "type")
            if typ == "message":
                events.append({"role": "assistant", "content": _get(item, "content")})
            elif typ == "function_call":
                events.append({"type": typ, "name": _get(item, "name"), "call_id": _get(item, "call_id"),
                               "arguments": _get(item, "arguments")})
        return events or super().response_events(response)

    def stream_tools(self, chunk: Any, acc: Dict[Any, Dict[str, Any]]) -> None:
        if _get(chunk, "type") == "response.output_item.done":
            item = _get(chunk, "item")
            if _get(item, "type") == "function_call":
                acc[("responses", _get(item, "id") or len(acc))] = {
                    "order": len(acc), "id": _get(item, "call_id"), "name": _get(item, "name"),
                    "args": [_get(item, "arguments") or ""]}

    def stream_delta(self, chunk: Any) -> str:
        if _get(chunk, "type") == "response.output_text.delta":
            return _get(chunk, "delta") or ""
        return ""


class SyncStreamProxy:
    """Wraps a sync stream; calls ``on_done(text, events)`` once the stream ends or closes.

    ``events`` carries streamed tool calls (assembled from fragments), or None for text only.
    """

    def __init__(self, stream: Any, adapter: Adapter, on_done: Callable[..., None]):
        self._stream = stream
        self._adapter = adapter
        self._on_done = on_done
        self._parts: List[str] = []
        self._tools: Dict[Any, Dict[str, Any]] = {}
        self._it = None
        self._done = False

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            text = "".join(self._parts)
            try:
                if self._tools:  # tool calls streamed: hand over the assembled events too
                    self._on_done(text, self._adapter.stream_events(text, self._tools))
                else:  # text-only streams keep the one-argument callback
                    self._on_done(text)
            except Exception:
                pass

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self._it = iter(self._stream)
        try:
            chunk = next(self._it)
        except StopIteration:
            self._finish()
            raise
        try:
            self._parts.append(self._adapter.stream_delta(chunk))
            self._adapter.stream_tools(chunk, self._tools)
        except Exception:
            pass
        return chunk

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *exc):
        self._finish()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*exc)
        return False

    def close(self) -> None:
        self._finish()
        if hasattr(self._stream, "close"):
            self._stream.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class AsyncStreamProxy:
    """Async counterpart of :class:`SyncStreamProxy`."""

    def __init__(self, stream: Any, adapter: Adapter, on_done: Callable[..., None]):
        self._stream = stream
        self._adapter = adapter
        self._on_done = on_done
        self._parts: List[str] = []
        self._tools: Dict[Any, Dict[str, Any]] = {}
        self._it = None
        self._done = False

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            text = "".join(self._parts)
            try:
                if self._tools:  # tool calls streamed: hand over the assembled events too
                    self._on_done(text, self._adapter.stream_events(text, self._tools))
                else:  # text-only streams keep the one-argument callback
                    self._on_done(text)
            except Exception:
                pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._it is None:
            self._it = self._stream.__aiter__()
        try:
            chunk = await self._it.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        try:
            self._parts.append(self._adapter.stream_delta(chunk))
            self._adapter.stream_tools(chunk, self._tools)
        except Exception:
            pass
        return chunk

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc):
        self._finish()
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*exc)
        return False

    async def close(self) -> None:
        self._finish()
        closer = getattr(self._stream, "close", None)
        if closer is not None:
            res = closer()
            if hasattr(res, "__await__"):
                await res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)
