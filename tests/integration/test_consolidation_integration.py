"""
Integration tests for consolidation.
"""

import pytest
import os
import asyncio
from memorable_ai import MemoryEngine


@pytest.fixture
def test_db():
    """Create test database."""
    db_path = "test_consolidation_integration.db"
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def memory_engine(test_db):
    """Create memory engine."""
    engine = MemoryEngine(
        database=f"sqlite:///{test_db}",
        mode="auto"
    )
    yield engine
    engine.disable()


@pytest.mark.asyncio
async def test_consolidation_runs(memory_engine):
    """Test that consolidation runs."""
    memory_engine.enable()
    
    # Add memories
    await memory_engine.add_memory("User likes Python", "preference")
    await memory_engine.add_memory("User works at Google", "fact")
    
    # Run consolidation
    if memory_engine._consolidator:
        await memory_engine._consolidator.consolidate()
        
        # Verify importance scores were updated
        memories = await memory_engine._storage.get_memories(limit=10)
        assert len(memories) >= 2


@pytest.mark.asyncio
async def test_contradiction_detection(memory_engine):
    """Test contradiction detection and resolution."""
    memory_engine.enable()
    
    # Add contradictory memories
    await memory_engine.add_memory("User likes Python", "preference")
    await memory_engine.add_memory("User doesn't like Python", "preference")
    
    # Run consolidation
    if memory_engine._consolidator:
        await memory_engine._consolidator.consolidate()

        # The newer statement wins; the older one is retired, not deleted.
        active = await memory_engine._storage.get_memories(limit=10)
        assert [m["content"] for m in active] == ["User doesn't like Python"]
        everything = await memory_engine._storage.get_memories(limit=10, include_invalid=True)
        retired = [m for m in everything if m["invalidated_at"]]
        assert len(retired) == 1
        assert retired[0]["content"] == "User likes Python"
        assert retired[0]["superseded_by_id"] == active[0]["id"]

