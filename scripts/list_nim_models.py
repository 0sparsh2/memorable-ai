#!/usr/bin/env python3
"""List NVIDIA NIM LLM models (catalog + live API)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memorable_ai.integrations.nvidia_nim import (
    NIM_LLM_MODELS,
    RECOMMENDED_MEMORY_MODELS,
    fetch_available_models,
)


def main():
    print("NVIDIA NIM LLM API — https://docs.api.nvidia.com/nim/reference/llm-apis\n")
    print("Recommended for Memorable memory pipeline:")
    for m in RECOMMENDED_MEMORY_MODELS:
        print(f"  * {m} — {NIM_LLM_MODELS.get(m, '')}")

    key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if key:
        live = fetch_available_models(key)
        print(f"\nLive API models ({len(live)} total, showing chat-related):")
        for mid in sorted(live):
            if any(
                mid.startswith(p)
                for p in (
                    "meta/",
                    "nvidia/",
                    "deepseek",
                    "mistral",
                    "microsoft/",
                    "google/",
                    "qwen/",
                    "moonshot",
                    "openai/",
                )
            ):
                mark = " [recommended]" if mid in RECOMMENDED_MEMORY_MODELS else ""
                print(f"  {mid}{mark}")
    else:
        print("\nSet NVIDIA_API_KEY to list live models from GET /v1/models")
        print("\nCurated catalog:")
        for mid, desc in sorted(NIM_LLM_MODELS.items()):
            print(f"  {mid} — {desc}")


if __name__ == "__main__":
    main()
