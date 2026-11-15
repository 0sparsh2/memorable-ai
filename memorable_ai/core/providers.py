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

from typing import Any, Callable, Dict, List, Optional, Tuple

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
        return {
            "user_id": str(user) if _is_given(user) and user else None,
            "session_id": str(session) if session else None,
        }

    def inject(self, kwargs: Dict[str, Any], context: str) -> Dict[str, Any]:
        raise NotImplementedError

    def is_stream(self, kwargs: Dict[str, Any]) -> bool:
        return kwargs.get("stream") is True

    def response_text(self, response: Any) -> str:
        return _response_text(response)

    def stream_delta(self, chunk: Any) -> str:
        return ""


class OpenAIChatAdapter(Adapter):
    """openai ``chat.completions.create`` (and OpenAI-compatible shapes)."""

    name = "openai.chat"

    def inject(self, kwargs: Dict[str, Any], context: str) -> Dict[str, Any]:
        messages = list(kwargs.get("messages") or [])
        new = dict(kwargs)
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
    """anthropic ``messages.create``: context goes into ``system``, never into messages."""

    name = "anthropic.messages"

    def get_hints(self, kwargs: Dict[str, Any]) -> Dict[str, Optional[str]]:
        md = kwargs.get("metadata")
        md = md if isinstance(md, dict) else {}
        user = md.get("user_id")
        return {"user_id": str(user) if user else None, "session_id": None}

    def inject(self, kwargs: Dict[str, Any], context: str) -> Dict[str, Any]:
        new = dict(kwargs)
        system = kwargs.get("system")
        if isinstance(system, str) and system:
            new["system"] = f"{system}\n\n{context}"
        elif isinstance(system, (list, tuple)) and system:
            new["system"] = list(system) + [{"type": "text", "text": context}]
        else:
            new["system"] = context
        return new

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
            if role in ("user", "assistant", "system", "developer"):
                out.append({"role": role, "content": _get(item, "content")})
        return out

    def get_hints(self, kwargs: Dict[str, Any]) -> Dict[str, Optional[str]]:
        hints = super().get_hints(kwargs)
        conv = kwargs.get("conversation")
        if not hints["session_id"] and _is_given(conv) and conv:
            hints["session_id"] = str(_get(conv, "id") or conv) if not isinstance(conv, str) else conv
        return hints

    def inject(self, kwargs: Dict[str, Any], context: str) -> Dict[str, Any]:
        new = dict(kwargs)
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

    def stream_delta(self, chunk: Any) -> str:
        if _get(chunk, "type") == "response.output_text.delta":
            return _get(chunk, "delta") or ""
        return ""


class SyncStreamProxy:
    """Wraps a sync stream; calls ``on_done(text)`` once the stream ends or closes."""

    def __init__(self, stream: Any, adapter: Adapter, on_done: Callable[[str], None]):
        self._stream = stream
        self._adapter = adapter
        self._on_done = on_done
        self._parts: List[str] = []
        self._it = None
        self._done = False

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            try:
                self._on_done("".join(self._parts))
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

    def __init__(self, stream: Any, adapter: Adapter, on_done: Callable[[str], None]):
        self._stream = stream
        self._adapter = adapter
        self._on_done = on_done
        self._parts: List[str] = []
        self._it = None
        self._done = False

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            try:
                self._on_done("".join(self._parts))
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
