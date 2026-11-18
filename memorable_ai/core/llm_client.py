"""
Unified LiteLLM client for Memorable (OpenAI, Gemini, NVIDIA NIM).

NVIDIA NIM uses OpenAI-compatible POST /v1/chat/completions per:
https://docs.api.nvidia.com/nim/reference/llm-apis
"""

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional

from memorable_ai.core.scope import suppressed

logger = logging.getLogger(__name__)

_RATE_LIMIT_MARKERS = ("429", "rate limit", "too many requests", "ratelimit")
_AUTH_MARKERS = (
    "authentication",
    "api key",
    "api_key",
    "401",
    "403",
    "permission denied",
    "invalid x-api-key",
    "llm provider not provided",
    "badrequesterror: llm provider",
)
DISABLED_MODELS = {"", "none", "off", "disabled", "false", "0"}

# Circuit breaker: models that failed with an auth/config error are not retried
# for the rest of the process (avoids one failing network call per conversation turn).
_broken_models: Dict[str, str] = {}
_broken_lock = threading.Lock()


def llm_enabled(model: Optional[str]) -> bool:
    """False when no model is configured, it is explicitly disabled, or it failed auth."""
    if not model or str(model).strip().lower() in DISABLED_MODELS:
        return False
    with _broken_lock:
        return model not in _broken_models


def reset_circuit_breaker() -> None:
    with _broken_lock:
        _broken_models.clear()


def _is_rate_limited(exc: BaseException) -> bool:
    err = str(exc).lower()
    return any(m in err for m in _RATE_LIMIT_MARKERS)


def _is_auth_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "authentication" in name or "permissiondenied" in name:
        return True
    err = str(exc).lower()
    return any(m in err for m in _AUTH_MARKERS)


def _trip(model: str, exc: BaseException) -> None:
    with _broken_lock:
        if model not in _broken_models:
            _broken_models[model] = str(exc)[:200]
            logger.warning(
                "Memorable LLM features disabled for model %s for this process: %s",
                model,
                str(exc)[:200],
            )


async def acompletion(
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 800,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Provider-aware async chat completion.

    Args:
        model: LiteLLM model id (e.g. nvidia_nim/meta/llama-3.3-70b-instruct, gemini/...)
        messages: Chat messages
        max_tokens: Max completion tokens
        api_key: Optional override (provider-specific env used if omitted)
        timeout: Seconds (auto for NIM if omitted)
    """
    import litellm

    from memorable_ai.integrations.nvidia_nim import (
        configure_litellm_env,
        default_timeout_seconds,
        is_nim_model,
        request_defaults,
    )

    if is_nim_model(model):
        configure_litellm_env(api_key=api_key)
        timeout = timeout or default_timeout_seconds(model)

    call_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **kwargs,
    }
    if timeout:
        call_kwargs["timeout"] = timeout
    if is_nim_model(model):
        for k, v in request_defaults(model).items():
            call_kwargs.setdefault(k, v)
    if api_key and not is_nim_model(model):
        call_kwargs["api_key"] = api_key

    if not llm_enabled(model):
        raise RuntimeError(f"LLM disabled or unavailable: {model}")
    try:
        with suppressed():
            return await litellm.acompletion(**call_kwargs)
    except Exception as e:
        if _is_auth_error(e):
            _trip(model, e)
        raise


async def acompletion_with_fallback(
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 800,
    fallback_model: Optional[str] = None,
    max_rate_limit_retries: int = 4,
    **kwargs: Any,
) -> Any:
    """
    Call acompletion; retry on NIM 429 with backoff; on timeout/504 retry fallback model.
    """
    import asyncio

    from memorable_ai.integrations.nvidia_nim import (
        DEFAULT_MEMORY_MODEL,
        is_nim_model,
        to_litellm_model,
    )

    last_err: Optional[BaseException] = None
    for attempt in range(max_rate_limit_retries):
        try:
            return await acompletion(model, messages, max_tokens=max_tokens, **kwargs)
        except Exception as e:
            last_err = e
            if _is_rate_limited(e) and attempt < max_rate_limit_retries - 1:
                wait = min(2.0 * (2**attempt), 30.0)
                logger.warning("LLM rate limited (attempt %s); sleeping %.1fs", attempt + 1, wait)
                await asyncio.sleep(wait)
                continue
            err = str(e).lower()
            retryable = any(
                x in err
                for x in ("timeout", "504", "timed out", "gateway", "apitimeouterror")
            ) or _is_rate_limited(e)
            if not retryable:
                raise
            break

    e = last_err or RuntimeError("LLM call failed")
    fb = fallback_model or to_litellm_model(DEFAULT_MEMORY_MODEL)
    if fb == model:
        raise e
    logger.warning("LLM %s failed (%s); retrying with %s", model, e, fb)
    fb_kwargs = dict(kwargs)
    if kwargs.get("timeout"):
        try:
            fb_kwargs["timeout"] = int(kwargs["timeout"]) + 60
        except (TypeError, ValueError):
            pass
    return await acompletion(fb, messages, max_tokens=max_tokens, **fb_kwargs)


def message_content(response: Any) -> str:
    """Extract assistant text from LiteLLM response."""
    try:
        return (response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError):
        return ""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(raw: str) -> Any:
    """
    Parse JSON from an LLM reply: strips code fences and surrounding prose, and
    falls back to the outermost ``[...]`` or ``{...}`` span.
    """
    if raw is None:
        raise ValueError("empty response")
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in LLM response")
    start = min(starts)
    end_char = "]" if text[start] == "[" else "}"
    end = text.rfind(end_char)
    if end <= start:
        raise ValueError("unterminated JSON in LLM response")
    return json.loads(text[start : end + 1])
