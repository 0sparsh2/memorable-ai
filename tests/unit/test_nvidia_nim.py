from memorable_ai.integrations.nvidia_nim import (
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
    assert to_litellm_model("nvidia_nim/deepseek-ai/deepseek-v4-flash").startswith(
        "nvidia_nim/"
    )


def test_to_nim_model_id():
    assert to_nim_model_id("nvidia_nim/meta/llama-3.1-8b-instruct") == (
        "meta/llama-3.1-8b-instruct"
    )


def test_is_nim_model():
    assert is_nim_model("nvidia_nim/deepseek-ai/deepseek-v4-flash")
    assert not is_nim_model("gemini/gemini-3-flash-preview")


def test_recommended_models_in_catalog():
    for m in RECOMMENDED_MEMORY_MODELS:
        assert "/" in m


def test_slow_model_longer_timeout(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_TIMEOUT_SLOW", "200")
    assert default_timeout_seconds("deepseek-ai/deepseek-v4-pro") == 200
    assert default_timeout_seconds("meta/llama-3.1-8b-instruct") <= 120
