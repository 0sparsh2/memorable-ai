import pytest

from memorable_ai.integrations.nvidia_nim import (
    DEFAULT_MEMORY_MODEL,
    NIM_LLM_MODELS,
    RECOMMENDED_MEMORY_MODELS,
    to_litellm_model,
    to_nim_model_id,
    resolve_memory_model,
    is_nim_model,
    default_timeout_seconds,
)


def test_to_litellm_model():
    assert to_litellm_model("meta/llama-3.3-70b-instruct") == (
        "nvidia_nim/meta/llama-3.3-70b-instruct"
    )
    assert to_litellm_model("nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b").startswith(
        "nvidia_nim/"
    )


def test_to_nim_model_id():
    assert to_nim_model_id("nvidia_nim/meta/llama-3.1-8b-instruct") == (
        "meta/llama-3.1-8b-instruct"
    )


def test_is_nim_model():
    assert is_nim_model("nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b")
    assert not is_nim_model("gemini/gemini-3-flash-preview")


def test_recommended_models_in_catalog():
    for m in RECOMMENDED_MEMORY_MODELS:
        assert "/" in m and m in NIM_LLM_MODELS
    assert DEFAULT_MEMORY_MODEL in RECOMMENDED_MEMORY_MODELS
    for gone in ("deepseek-ai/deepseek-v4-flash", "deepseek-ai/deepseek-v4.1-flash"):
        assert gone not in NIM_LLM_MODELS, "retired on NIM (410 Gone)"


def test_slow_model_longer_timeout(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_TIMEOUT_SLOW", "200")
    assert default_timeout_seconds("z-ai/glm-5.3-flash") == 200
    assert default_timeout_seconds("meta/llama-3.1-8b-instruct") <= 120


def test_resolve_memory_model_defaults_to_live_model(monkeypatch):
    monkeypatch.delenv("MEMORABLE_LLM__DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    assert resolve_memory_model() == f"nvidia_nim/{DEFAULT_MEMORY_MODEL}"
    assert resolve_memory_model("gemini/gemini-3-flash-preview") == "gemini/gemini-3-flash-preview"


@pytest.mark.asyncio
async def test_memory_jobs_turn_thinking_off_on_reasoning_models(monkeypatch):
    """Lightning and Ultra think by default; extraction must not spend max_tokens on thinking."""
    import litellm

    from memorable_ai.core import llm_client

    seen = []

    async def fake(**kw):
        seen.append(kw)
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.delenv("MEMORABLE_LLM__THINKING", raising=False)
    llm_client.reset_circuit_breaker()
    await llm_client.acompletion("nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
                                 [{"role": "user", "content": "x"}], max_tokens=10)
    assert seen[-1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    await llm_client.acompletion("nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b", [{"role": "user", "content": "x"}])
    assert seen[-1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}, "the observer's model"
    await llm_client.acompletion("nvidia_nim/nvidia/nemotron-3.5-bolt-e4b", [{"role": "user", "content": "x"}])
    assert "extra_body" not in seen[-1], "models that answer directly get no extra fields"
    await llm_client.acompletion("nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b", [{"role": "user", "content": "x"}],
                                 extra_body={"custom": 1})
    assert seen[-1]["extra_body"] == {"custom": 1}, "the caller's own fields win"
    monkeypatch.setenv("MEMORABLE_LLM__THINKING", "on")
    await llm_client.acompletion("nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b", [{"role": "user", "content": "x"}])
    assert "extra_body" not in seen[-1]
