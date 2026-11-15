"""
LLM call interceptor (zero-code integration).

Patches the SDK *resource classes* so every client instance is covered, sync and
async:

    openai.resources.chat.completions.Completions.create      (+ AsyncCompletions)
    openai.resources.responses.Responses.create               (+ AsyncResponses)
    anthropic.resources.messages.Messages.create              (+ AsyncMessages)
    litellm.completion / litellm.acompletion

The previous implementation patched ``OpenAI.chat.completions``, which does not
exist on the class (``chat`` is a cached_property on the instance), so the OpenAI
and Anthropic hooks silently never attached.

Guards:
* calls made by Memorable itself (extraction, summaries) are never intercepted
* a provider call nested inside an intercepted call (litellm -> openai) is not
  intercepted twice
* any memory failure is logged and the user's LLM call proceeds unchanged

Inspired by Memori's interceptor architecture (https://github.com/GibsonAI/Memori).
"""

import functools
import importlib
import inspect
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from memorable_ai.core.providers import (
    Adapter,
    AnthropicAdapter,
    AsyncStreamProxy,
    LiteLLMAdapter,
    OpenAIChatAdapter,
    OpenAIResponsesAdapter,
    SyncStreamProxy,
)
from memorable_ai.core.scope import _suppress, is_suppressed

logger = logging.getLogger(__name__)

# key -> (module, attribute path, adapter class, is_async)
TARGETS: Dict[str, Tuple[str, str, Type[Adapter], bool]] = {
    "openai.chat.completions.create": ("openai.resources.chat.completions", "Completions.create", OpenAIChatAdapter, False),
    "openai.chat.completions.acreate": ("openai.resources.chat.completions", "AsyncCompletions.create", OpenAIChatAdapter, True),
    "openai.responses.create": ("openai.resources.responses", "Responses.create", OpenAIResponsesAdapter, False),
    "openai.responses.acreate": ("openai.resources.responses", "AsyncResponses.create", OpenAIResponsesAdapter, True),
    "anthropic.messages.create": ("anthropic.resources.messages", "Messages.create", AnthropicAdapter, False),
    "anthropic.messages.acreate": ("anthropic.resources.messages", "AsyncMessages.create", AnthropicAdapter, True),
    "litellm.completion": ("litellm", "completion", LiteLLMAdapter, False),
    "litellm.acompletion": ("litellm", "acompletion", LiteLLMAdapter, True),
}


def _resolve(module_name: str, path: str) -> Tuple[Any, str, Callable]:
    module = importlib.import_module(module_name)
    owner: Any = module
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1], getattr(owner, parts[-1])


# Process-wide patch registry. SDK methods are patched once; calls dispatch to the most
# recently enabled interceptor, so several engines (or one that was never disabled)
# cannot stack wrappers or leave a stale engine receiving calls.
_LOCK = threading.RLock()
_ORIGINALS: Dict[str, Callable] = {}
_OWNERS: Dict[str, Tuple[Any, str]] = {}
_STACK: List["LLMInterceptor"] = []


def _current() -> Optional["LLMInterceptor"]:
    with _LOCK:
        for icp in reversed(_STACK):
            if icp._enabled and icp.memory_engine is not None:
                return icp
    return None


def _install(key: str) -> bool:
    if key in _ORIGINALS:
        return True
    module_name, path, adapter_cls, is_async = TARGETS[key]
    try:
        owner, attr, original = _resolve(module_name, path)
    except Exception as e:
        logger.debug("Skipping %s: %s", key, e)
        return False
    original = getattr(original, "__memorable_original__", original)
    adapter = adapter_cls()
    if is_async or inspect.iscoroutinefunction(original):
        wrapper = _make_async(original, adapter)
    else:
        wrapper = _make_sync(original, adapter)
    wrapper.__memorable_wrapped__ = True  # type: ignore[attr-defined]
    wrapper.__memorable_original__ = original  # type: ignore[attr-defined]
    setattr(owner, attr, wrapper)
    _ORIGINALS[key] = original
    _OWNERS[key] = (owner, attr)
    logger.debug("Hooked %s", key)
    return True


def _uninstall_all() -> None:
    for key in list(_ORIGINALS):
        owner, attr = _OWNERS.pop(key)
        try:
            setattr(owner, attr, _ORIGINALS.pop(key))
        except Exception as e:  # pragma: no cover
            logger.debug("Restore failed for %s: %s", key, e)


class LLMInterceptor:
    """Transparently injects memory into LLM calls and records the exchange."""

    def __init__(self, memory_engine: Any, targets: Optional[List[str]] = None):
        self.memory_engine = memory_engine
        self.targets = targets or list(TARGETS)
        self._original_methods: Dict[str, Callable] = {}
        self._enabled = False

    @property
    def hooked(self) -> List[str]:
        return sorted(self._original_methods)

    def enable(self) -> None:
        if self._enabled:
            logger.warning("Interceptor already enabled")
            return
        with _LOCK:
            for key in self.targets:
                _install(key)
            if self in _STACK:
                _STACK.remove(self)
            _STACK.append(self)
            self._original_methods = {k: _ORIGINALS[k] for k in self.targets if k in _ORIGINALS}
            self._enabled = True
        logger.info("LLM interceptor enabled: %s", ", ".join(self.hooked) or "no SDKs found")

    def disable(self) -> None:
        if not self._enabled:
            return
        with _LOCK:
            self._enabled = False
            if self in _STACK:
                _STACK.remove(self)
            if not any(i._enabled for i in _STACK):
                _STACK.clear()
                _uninstall_all()
            self._original_methods = {}
        logger.info("LLM interceptor disabled")

    # Backwards-compatible names
    def _hook_openai(self) -> None:
        for k in ("openai.chat.completions.create", "openai.chat.completions.acreate", "openai.responses.create", "openai.responses.acreate"):
            self._hook(k)

    def _hook_anthropic(self) -> None:
        for k in ("anthropic.messages.create", "anthropic.messages.acreate"):
            self._hook(k)

    def _hook_litellm(self) -> None:
        for k in ("litellm.completion", "litellm.acompletion"):
            self._hook(k)

    def _hook(self, key: str) -> bool:
        with _LOCK:
            ok = _install(key)
            if ok:
                self._original_methods[key] = _ORIGINALS[key]
            return ok

    # Kept for callers that used the old private API directly.
    def _intercept_call_sync(self, original_func: Callable, *args, **kwargs):
        return _make_sync(original_func, LiteLLMAdapter())(*args, **kwargs)


# ---------------------------------------------------------------------- wrappers


def _make_sync(original: Callable, adapter: Adapter) -> Callable:
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        interceptor = None if is_suppressed() else _current()
        if interceptor is None:
            return original(*args, **kwargs)
        engine = interceptor.memory_engine
        args, kwargs = adapter.bind(args, kwargs)
        call = None
        token = _suppress.set(True)
        try:
            try:
                call = engine._prepare_call(adapter, kwargs)
                kwargs = call.kwargs
            except Exception as e:
                logger.warning("Memory retrieval failed; calling LLM without memory: %s", e)
            response = original(*args, **kwargs)
        finally:
            _suppress.reset(token)
        if call is None:
            return response
        if adapter.is_stream(kwargs):
            return SyncStreamProxy(response, adapter, lambda text: engine._record_call(call, text))
        try:
            engine._record_call(call, adapter.response_text(response))
        except Exception as e:
            logger.warning("Failed to record conversation: %s", e)
        return response

    return wrapper


def _make_async(original: Callable, adapter: Adapter) -> Callable:
    @functools.wraps(original)
    async def wrapper(*args, **kwargs):
        interceptor = None if is_suppressed() else _current()
        if interceptor is None:
            return await original(*args, **kwargs)
        engine = interceptor.memory_engine
        args, kwargs = adapter.bind(args, kwargs)
        call = None
        token = _suppress.set(True)
        try:
            try:
                call = await engine._aprepare_call(adapter, kwargs)
                kwargs = call.kwargs
            except Exception as e:
                logger.warning("Memory retrieval failed; calling LLM without memory: %s", e)
            response = await original(*args, **kwargs)
        finally:
            _suppress.reset(token)
        if call is None:
            return response
        if adapter.is_stream(kwargs):
            return AsyncStreamProxy(response, adapter, lambda text: engine._record_call(call, text))
        try:
            await engine._arecord_call(call, adapter.response_text(response))
        except Exception as e:
            logger.warning("Failed to record conversation: %s", e)
        return response

    return wrapper
