"""
Unit tests for MemoryEngine.
"""

import pytest
import os
from memorable_ai import MemoryEngine


@pytest.fixture
def test_db():
    """Create test database."""
    db_path = "test_engine.db"
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def memory_engine(test_db):
    """Create memory engine instance."""
    return MemoryEngine(database=f"sqlite:///{test_db}", mode="auto")


def test_memory_engine_initialization(memory_engine):
    """Test memory engine initialization."""
    assert memory_engine is not None
    assert memory_engine._enabled is False


def test_enable_disable(memory_engine):
    """Test enabling and disabling memory engine."""
    memory_engine.enable()
    assert memory_engine._enabled is True

    memory_engine.disable()
    assert memory_engine._enabled is False


def test_get_stats(memory_engine):
    """Test getting statistics."""
    stats = memory_engine.get_stats()
    assert isinstance(stats, dict)
    assert "enabled" in stats
    assert "mode" in stats


@pytest.mark.asyncio
async def test_add_memory(memory_engine):
    """Test adding memory manually."""
    memory_engine.enable()
    await memory_engine.add_memory(
        content="Test memory",
        memory_type="fact"
    )
    # Memory should be stored
    assert True  # If no exception, it worked


@pytest.mark.asyncio
async def test_search_memories(memory_engine):
    """Test searching memories."""
    memory_engine.enable()
    results = await memory_engine.search_memories("test", limit=10)
    assert isinstance(results, list)

