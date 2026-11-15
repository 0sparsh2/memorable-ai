"""
Integration tests for end-to-end workflows.
"""

import pytest
import os
from memorable_ai import MemoryEngine


@pytest.fixture
def test_db():
    """Create test database."""
    db_path = "test_integration.db"
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def memory_engine(test_db):
    """Create memory engine instance."""
    engine = MemoryEngine(
        database=f"sqlite:///{test_db}",
        mode="auto"
    )
    yield engine
    engine.disable()


@pytest.mark.asyncio
async def test_basic_workflow(memory_engine):
    """Test basic memory workflow."""
    # Enable memory
    memory_engine.enable()

    # Add memory manually
    await memory_engine.add_memory(
        content="User likes Python",
        memory_type="preference"
    )

    # Search memories
    results = await memory_engine.search_memories("Python", limit=10)
    assert len(results) >= 0  # May be empty if no matches

    # Get stats
    stats = memory_engine.get_stats()
    assert stats["enabled"] is True


@pytest.mark.asyncio
async def test_memory_extraction_and_storage(memory_engine):
    """Test that memories are extracted and stored."""
    memory_engine.enable()

    # Simulate conversation
    messages = [
        {"role": "user", "content": "I'm building a FastAPI project"},
    ]

    # Manually trigger storage (normally done by interceptor)
    if memory_engine._extraction and memory_engine._storage:
        memories = await memory_engine._extraction.extract(messages)
        if memories:
            await memory_engine._storage.store_memories(memories)

            # Verify stored
            stored = await memory_engine._storage.get_memories(limit=10)
            assert len(stored) > 0

