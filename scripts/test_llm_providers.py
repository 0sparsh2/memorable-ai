#!/usr/bin/env python3
"""Smoke-test Memorable LLM extraction with LiteLLM providers (Gemini / NVIDIA NIM)."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def _litellm_ping(model: str) -> bool:
    import litellm

    resp = await asyncio.wait_for(
        litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
        ),
        timeout=120,
    )
    text = (resp.choices[0].message.content or "").strip()
    print(f"  LiteLLM ping ({model}): {text[:80]}")
    return bool(text)


async def _memorable_roundtrip(model: str, label: str) -> dict:
    from memorable_ai import MemoryEngine
    from memorable_ai.utils.config import MemorableConfig

    os.environ["MEMORABLE_LLM__DEFAULT_MODEL"] = model
    os.environ["MEMORABLE_EXTRACTION__MODE"] = "llm"

    tmp = tempfile.mkdtemp(prefix="memorable_llm_test_")
    db = f"sqlite:///{tmp}/mem.db"

    config = MemorableConfig.from_env()
    engine = MemoryEngine(database=db, mode="auto", config=config)
    engine._initialize_components()

    messages = [
        {
            "role": "user",
            "content": "I am vegetarian, I prefer Python, and I'm building a FastAPI API.",
        }
    ]
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": "I'll remember your preferences."}}
        ]
    }

    await engine._store_conversation(messages, response)
    hits = await engine.search_memories("vegetarian Python FastAPI", limit=5)
    stats = engine.get_stats()

    return {
        "provider": label,
        "model": model,
        "memories_stored": stats.get("namespace_memories", stats.get("total_memories", 0)),
        "search_hits": len(hits),
        "sample": [h.get("content", "")[:120] for h in hits[:3]],
    }


async def main():
    providers = []

    from memorable_ai.integrations.nvidia_nim import (
        configure_litellm_env,
        resolve_memory_model,
        RECOMMENDED_MEMORY_MODELS,
    )

    nvidia_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if nvidia_key:
        configure_litellm_env(api_key=nvidia_key)
        model = resolve_memory_model()
        os.environ["NVIDIA_NIM_API_KEY"] = nvidia_key
        providers.append(("NVIDIA NIM", model))

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        gm = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        model = gm if gm.startswith("gemini/") else f"gemini/{gm}"
        os.environ["GEMINI_API_KEY"] = gemini_key
        providers.append(("Gemini", model))

    if not providers:
        print("Set NVIDIA_API_KEY and/or GEMINI_API_KEY in the environment.")
        sys.exit(1)

    if nvidia_key:
        print(f"NIM recommended models: {', '.join(RECOMMENDED_MEMORY_MODELS[:4])}...")
        print("List all: python scripts/list_nim_models.py\n")

    results = []
    for label, model in providers:
        print(f"\n=== {label} ({model}) ===")
        try:
            await _litellm_ping(model)
            r = await _memorable_roundtrip(model, label)
            results.append(r)
            print(f"  Memories: {r['memories_stored']}, search hits: {r['search_hits']}")
            for s in r["sample"]:
                print(f"    - {s}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"provider": label, "model": model, "error": str(e)})

    print("\n=== Summary ===")
    for r in results:
        if "error" in r:
            print(f"{r['provider']}: ERROR — {r['error']}")
        else:
            print(
                f"{r['provider']}: OK — {r['memories_stored']} memories, "
                f"{r['search_hits']} retrieval hits"
            )

    failed = [r for r in results if r.get("error")]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
