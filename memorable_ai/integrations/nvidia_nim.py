"""
NVIDIA NIM LLM API integration.

Official API: https://docs.api.nvidia.com/nim/reference/llm-apis
- Base URL: https://integrate.api.nvidia.com
- Endpoint: POST /v1/chat/completions (OpenAI-compatible)
- Model id: publisher/model-name (e.g. meta/llama-3.3-70b-instruct)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NIM_API_BASE = "https://integrate.api.nvidia.com/v1"
NIM_CHAT_COMPLETIONS_PATH = "/chat/completions"
LITELLM_PROVIDER_PREFIX = "nvidia_nim/"

# Curated from https://docs.api.nvidia.com/nim/reference/llm-apis (chat completion models)
NIM_LLM_MODELS: Dict[str, str] = {
    # deepseek-ai
    "deepseek-ai/deepseek-v4-flash": "Fast; best default for Memorable extraction",
    "deepseek-ai/deepseek-v4-pro": "High quality; slow — may 504 on long runs",
    # meta
    "meta/llama-3.1-8b-instruct": "Fast, lightweight",
    "meta/llama-3.1-70b-instruct": "Balanced general instruct",
    "meta/llama-3.2-1b-instruct": "Very fast, small",
    "meta/llama-3.2-3b-instruct": "Fast, small",
    "meta/llama-3.3-70b-instruct": "Strong general instruct",
    "meta/llama2-70b": "Legacy chat completion",
    # nvidia
    "nvidia/nemotron-3-nano-30b-a3b": "NVIDIA nano — good speed/quality",
    "nvidia/nemotron-3-super-120b-a12b": "Large NVIDIA model",
    "nvidia/nemotron-mini-4b-instruct": "Very fast NVIDIA instruct",
    "nvidia/nvidia-nemotron-nano-9b-v2": "Compact Nemotron",
    "nvidia/llama-3.1-nemotron-nano-8b-v1": "Nemotron nano 8B",
    "nvidia/llama-3.3-nemotron-super-49b-v1": "High capability Nemotron",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": "Latest Nemotron super",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": "Largest — high latency",
    # mistralai
    "mistralai/mistral-7b-instruct-v0.3": "Fast Mistral 7B",
    "mistralai/mistral-nemotron": "Mistral Nemotron hybrid",
    "mistralai/mixtral-8x7b-instruct": "Mixtral 8x7B",
    "mistralai/mixtral-8x22b-instruct": "Mixtral 8x22B",
    # microsoft
    "microsoft/phi-4-mini-instruct": "Small Microsoft instruct",
    "microsoft/phi-4-mini-flash-reasoning": "Phi flash reasoning",
    # openai (on NIM)
    "openai/gpt-oss-20b": "Open-weight 20B",
    "openai/gpt-oss-120b": "Open-weight 120B",
    # qwen
    "qwen/qwen2.5-coder-32b-instruct": "Code-focused",
    "qwen/qwen3-next-80b-a3b-instruct": "Qwen3 next instruct",
    "qwen/qwq-32b": "Qwen reasoning",
    # moonshotai
    "moonshotai/kimi-k2-instruct": "Kimi K2 instruct",
    # google (on NIM)
    "google/gemma-2-2b-it": "Small Gemma",
    "google/codegemma-7b": "Code Gemma",
}

# Defaults for Memorable memory pipeline (extraction / update / summary)
RECOMMENDED_MEMORY_MODELS: List[str] = [
    "deepseek-ai/deepseek-v4-flash",
    "meta/llama-3.3-70b-instruct",
    "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "meta/llama-3.1-8b-instruct",
    "microsoft/phi-4-mini-instruct",
]

SLOW_MODELS: List[str] = [
    "deepseek-ai/deepseek-v4-pro",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "qwen/qwen3-coder-480b-a35b-instruct",
]

DEFAULT_MEMORY_MODEL = "deepseek-ai/deepseek-v4-flash"


def is_nim_model(model: str) -> bool:
    """True if model id targets NVIDIA NIM (LiteLLM or raw NIM id)."""
    if model.startswith(LITELLM_PROVIDER_PREFIX):
        return True
    return model in NIM_LLM_MODELS or "/" in model and not model.startswith(("gemini/", "gpt-", "claude-"))


def to_nim_model_id(model: str) -> str:
    """Strip LiteLLM prefix to get NIM API model field (publisher/name)."""
    if model.startswith(LITELLM_PROVIDER_PREFIX):
        return model[len(LITELLM_PROVIDER_PREFIX) :]
    return model


def to_litellm_model(model: str) -> str:
    """Ensure LiteLLM provider prefix for NIM chat models."""
    if model.startswith(LITELLM_PROVIDER_PREFIX):
        return model
    if model.startswith("gemini/") or model.startswith("gpt-") or "claude" in model:
        return model
    return f"{LITELLM_PROVIDER_PREFIX}{model}"


def default_timeout_seconds(model: str) -> int:
    """Request timeout — slow catalog models get more time."""
    mid = to_nim_model_id(model)
    if mid in SLOW_MODELS:
        return int(os.getenv("NVIDIA_NIM_TIMEOUT_SLOW", "180"))
    if any(x in mid for x in ("nano", "mini", "1b", "3b", "8b", "flash")):
        return int(os.getenv("NVIDIA_NIM_TIMEOUT_FAST", "120"))
    return int(os.getenv("NVIDIA_NIM_TIMEOUT", "90"))


def configure_litellm_env(api_key: Optional[str] = None, api_base: Optional[str] = None) -> None:
    """
    Set LiteLLM / OpenAI-compatible env vars for NVIDIA NIM.

    See: https://docs.litellm.ai/docs/providers/nvidia_nim
    """
    key = api_key or os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    base = api_base or os.getenv("NVIDIA_NIM_API_BASE") or NIM_API_BASE
    if key:
        os.environ["NVIDIA_NIM_API_KEY"] = key
    if base:
        os.environ["NVIDIA_NIM_API_BASE"] = base.rstrip("/")


def fetch_available_models(api_key: Optional[str] = None) -> List[str]:
    """GET /v1/models from integrate.api.nvidia.com."""
    import urllib.request
    import json

    key = api_key or os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not key:
        return list(NIM_LLM_MODELS.keys())
    url = f"{NIM_API_BASE.replace('/v1', '')}/v1/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return [m["id"] for m in data.get("data", []) if "id" in m]
    except Exception as e:
        logger.warning("Failed to list NIM models: %s", e)
        return list(NIM_LLM_MODELS.keys())


def resolve_memory_model(explicit: Optional[str] = None) -> str:
    """
    Resolve model for Memorable LLM calls (returns LiteLLM id).

    Priority: explicit arg > MEMORABLE_LLM__DEFAULT_MODEL > NVIDIA_MODEL > default flash.
    """
    raw = (
        explicit
        or os.getenv("MEMORABLE_LLM__DEFAULT_MODEL")
        or os.getenv("NVIDIA_MODEL")
        or DEFAULT_MEMORY_MODEL
    )
    if raw.startswith("gemini/"):
        return raw
    return to_litellm_model(raw)


def model_info(model: str) -> Dict[str, Any]:
    """Metadata for a NIM model id."""
    mid = to_nim_model_id(model)
    return {
        "nim_model_id": mid,
        "litellm_model": to_litellm_model(mid),
        "description": NIM_LLM_MODELS.get(mid, "Listed on NVIDIA NIM API"),
        "recommended_for_memory": mid in RECOMMENDED_MEMORY_MODELS,
        "slow": mid in SLOW_MODELS,
        "api_base": NIM_API_BASE,
        "endpoint": NIM_CHAT_COMPLETIONS_PATH,
    }
