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

# Chat models checked against integrate.api.nvidia.com on 2026-11-16 (free tier). NIM retires
# models often: deepseek-v4-flash (the old default), llama-3.1-8b and others now return 410
# Gone, and several listed models return 404. `fetch_available_models()` lists what is live.
NIM_LLM_MODELS: Dict[str, str] = {
    # measured: correct answers, 0.2-6 s per call; reasons by default (chat_template_kwargs.enable_thinking)
    "nvidia/nemotron-3.5-lightning-30b-a3b": "Default: fast MoE (3B active), reasoning on by default",
    # measured: 4-7 s for ~500 output tokens, thinking off
    "nvidia/nemotron-3.5-bolt-e4b": "Fast small model; good for short structured generation",
    # measured: correct, but the free tier queues it (110-500 s per call)
    "deepseek-ai/deepseek-v4.1-flash": "Strong; slow on the free tier",
    # listed on /v1/models on 2026-11-16, not measured
    "nvidia/nemotron-3-ultra-550b-a55b": "Largest Nemotron (listed)",
    "mistralai/mistral-large-4-1t-a52b": "Mistral Large 4 (listed)",
    "z-ai/glm-5.3-flash": "GLM flash (listed; timed out on the free tier)",
    "google/gemma-4-31b-it": "Gemma 4 31B (listed; timed out on the free tier)",
    "openai/gpt-oss-20b": "Open-weight 20B (listed; timed out on the free tier)",
}

# Defaults for Memorable memory pipeline (extraction / update / summary)
RECOMMENDED_MEMORY_MODELS: List[str] = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3.5-bolt-e4b",
    "deepseek-ai/deepseek-v4.1-flash",
]

SLOW_MODELS: List[str] = [
    "deepseek-ai/deepseek-v4.1-flash",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "mistralai/mistral-large-4-1t-a52b",
]

DEFAULT_MEMORY_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


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
