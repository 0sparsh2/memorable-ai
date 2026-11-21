"""LOCOMO smoke benchmark (no dataset download)."""

import pytest

from memorable_ai import MemoryEngine
from benchmarks.locomo.benchmark import LOCOMOBenchmark


@pytest.mark.asyncio
async def test_locomo_smoke_benchmark():
    engine = MemoryEngine(database="sqlite:///:memory:")
    engine.enable()
    bench = LOCOMOBenchmark(engine, use_llm=False)
    results = await bench.run(mode="smoke")
    assert results["mode"] == "smoke"
    assert "overall_score" in results
    assert results["overall_score"] >= 0.0
    engine.disable()
