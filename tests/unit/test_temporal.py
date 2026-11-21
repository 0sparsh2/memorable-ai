"""
Unit tests for temporal memory.
"""

import pytest
from datetime import datetime, timedelta
from memorable_ai.core.temporal import TemporalMemory
from memorable_ai.core.storage import Storage
import os


@pytest.fixture
def test_db():
    """Create test database."""
    db_path = "test_temporal.db"
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def storage(test_db):
    """Create storage instance."""
    return Storage(connection_string=f"sqlite:///{test_db}")


@pytest.fixture
def temporal(storage):
    """Create temporal memory instance."""
    return TemporalMemory(storage)


@pytest.mark.asyncio
async def test_add_temporal_memory(temporal):
    """Test adding temporal memory."""
    memory = await temporal.add_temporal_memory(
        content="User started project",
        memory_type="fact",
        timestamp=datetime.now()
    )
    assert memory is not None
    assert "timestamp" in memory.get("metadata", {})


@pytest.mark.asyncio
async def test_get_memories_by_time_range(temporal, storage):
    """Test getting memories by time range."""
    now = datetime.now()
    
    # Add memories at different times
    await temporal.add_temporal_memory(
        content="Memory 1",
        timestamp=now - timedelta(days=5)
    )
    await temporal.add_temporal_memory(
        content="Memory 2",
        timestamp=now - timedelta(days=2)
    )
    await temporal.add_temporal_memory(
        content="Memory 3",
        timestamp=now
    )

    # Query time range
    start = now - timedelta(days=3)
    end = now
    memories = await temporal.get_memories_by_time_range(start, end)
    
    assert len(memories) >= 2  # Should get Memory 2 and Memory 3


def test_extract_temporal_relationships(temporal):
    """Test extracting temporal relationships."""
    text = "Before starting the project, I worked at Google. After that, I joined Microsoft."
    relationships = temporal.extract_temporal_relationships(text)
    
    assert "before" in relationships
    assert "after" in relationships


@pytest.mark.asyncio
async def test_check_temporal_coherence(temporal, storage):
    """Test temporal coherence checking."""
    now = datetime.now()
    memories = [
        await temporal.add_temporal_memory(
            content=f"Memory {i}",
            timestamp=now - timedelta(days=5-i)
        )
        for i in range(3)
    ]

    coherence = await temporal.check_temporal_coherence(memories)
    assert "coherent" in coherence
    assert "issues" in coherence

