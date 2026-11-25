"""
Unit tests for storage layer.
"""

import pytest
import os
from memorable_ai.core.storage import Storage, Memory, Conversation


@pytest.fixture
def test_db():
    """Create test database."""
    db_path = "test_memorable.db"
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def storage(test_db):
    """Create storage instance."""
    return Storage(connection_string=f"sqlite:///{test_db}")


def test_storage_initialization(storage):
    """Test storage initialization."""
    assert storage is not None
    assert storage.connection_string is not None


@pytest.mark.asyncio
async def test_store_memories(storage):
    """Test storing memories."""
    memories = [
        {
            "content": "User likes Python",
            "type": "preference",
            "metadata": {"source": "test"},
        },
        {
            "content": "User works at Google",
            "type": "fact",
            "metadata": {},
        },
    ]

    await storage.store_memories(memories)

    # Verify stored
    stored = await storage.get_memories()
    assert len(stored) >= 2


@pytest.mark.asyncio
async def test_search_memories(storage):
    """Test searching memories."""
    # Store test memory
    await storage.store_memories([
        {"content": "Python is a programming language", "type": "fact"}
    ])

    # Search
    results = await storage.search_memories_text("Python", limit=10)
    assert len(results) > 0
    assert "Python" in results[0]["content"]


@pytest.mark.asyncio
async def test_update_importance(storage):
    """Test updating memory importance."""
    # Store memory
    await storage.store_memories([
        {"content": "Test memory", "type": "fact"}
    ])

    memories = await storage.get_memories(limit=1)
    if memories:
        memory_id = memories[0]["id"]
        await storage.update_memory_importance(memory_id, 0.8)

        # Verify update
        updated = await storage.get_memories(limit=1)
        if updated:
            assert updated[0]["importance_score"] == 0.8

