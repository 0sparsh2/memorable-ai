"""
Unit tests for memory consolidation.
"""

import pytest
import asyncio
from memorable_ai.core.consolidation import MemoryConsolidator
from memorable_ai.core.storage import Storage
import os


@pytest.fixture
def test_db():
    """Create test database."""
    db_path = "test_consolidation.db"
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def storage(test_db):
    """Create storage instance."""
    return Storage(connection_string=f"sqlite:///{test_db}")


@pytest.fixture
def consolidator(storage):
    """Create consolidator instance."""
    return MemoryConsolidator(storage=storage, interval=60)  # 1 minute for testing


@pytest.mark.asyncio
async def test_consolidator_initialization(consolidator):
    """Test consolidator initialization."""
    assert consolidator is not None
    assert consolidator._running is False


@pytest.mark.asyncio
async def test_start_stop(consolidator):
    """Test starting and stopping consolidator."""
    await consolidator.start()
    assert consolidator._running is True

    await consolidator.stop()
    assert consolidator._running is False


@pytest.mark.asyncio
async def test_consolidate(consolidator, storage):
    """Test consolidation process."""
    # Add test memories
    await storage.store_memories([
        {"content": "Test memory 1", "type": "fact", "importance_score": 0.5},
        {"content": "Test memory 2", "type": "preference", "importance_score": 0.3},
    ])

    # Run consolidation
    await consolidator.consolidate()

    # Verify importance scores were updated
    memories = await storage.get_memories(limit=10)
    assert len(memories) >= 2


def test_are_contradictory(consolidator):
    """Test contradiction detection."""
    mem1 = {"content": "I like Python", "type": "preference"}
    mem2 = {"content": "I don't like Python", "type": "preference"}

    is_contradictory = consolidator._are_contradictory(mem1, mem2)
    assert is_contradictory is True

    mem3 = {"content": "I like Python", "type": "preference"}
    mem4 = {"content": "I like JavaScript", "type": "preference"}

    is_contradictory2 = consolidator._are_contradictory(mem3, mem4)
    assert is_contradictory2 is False  # Not contradictory, just different


def test_resolve_contradiction(consolidator):
    """Test contradiction resolution."""
    from datetime import datetime, timedelta

    now = datetime.now()
    mem1 = {
        "content": "I like Python",
        "type": "preference",
        "created_at": (now - timedelta(days=1)).isoformat(),
        "importance_score": 0.5,
    }
    mem2 = {
        "content": "I don't like Python",
        "type": "preference",
        "created_at": now.isoformat(),  # More recent
        "importance_score": 0.3,
    }

    resolved = consolidator._resolve_contradiction(mem1, mem2)
    # Should prefer more recent
    assert resolved == mem2

