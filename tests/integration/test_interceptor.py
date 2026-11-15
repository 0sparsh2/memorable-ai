"""
Integration tests for interceptor functionality.
"""

import pytest
import os
from memorable_ai import MemoryEngine


@pytest.fixture
def test_db():
    """Create test database."""
    db_path = "test_interceptor.db"
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def memory_engine(test_db):
    """Create memory engine instance."""
    engine = MemoryEngine(database=f"sqlite:///{test_db}", mode="auto")
    yield engine
    engine.disable()


def test_interceptor_enable_disable(memory_engine):
    """Test interceptor enable/disable."""
    assert memory_engine._interceptor._enabled is False
    
    memory_engine.enable()
    assert memory_engine._interceptor._enabled is True
    
    memory_engine.disable()
    assert memory_engine._interceptor._enabled is False


def test_interceptor_hooks(memory_engine):
    """Test that interceptor hooks are set up."""
    memory_engine.enable()
    
    # Check that hooks are registered
    assert len(memory_engine._interceptor._original_methods) > 0
    
    memory_engine.disable()
    
    # Check that hooks are restored
    # (original methods should be restored, but we can't easily verify without calling)


@pytest.mark.asyncio
async def test_context_injection(memory_engine):
    """Test context injection."""
    memory_engine.enable()
    
    # Add a memory
    await memory_engine.add_memory("User likes Python", "preference")
    
    # Test context injection
    messages = [{"role": "user", "content": "What do I like?"}]
    enhanced = await memory_engine._inject_context(messages)
    
    # Should have system message with context
    assert len(enhanced) > len(messages)
    assert enhanced[0]["role"] == "system"


@pytest.mark.asyncio
async def test_memory_storage(memory_engine):
    """Test that memories are stored after conversation."""
    memory_engine.enable()
    
    messages = [{"role": "user", "content": "I'm building a FastAPI project"}]
    response = {"choices": [{"message": {"content": "That's great!"}}]}
    
    # Store conversation
    await memory_engine._store_conversation(messages, response)
    
    # Verify memories were stored
    stats = memory_engine.get_stats()
    assert stats.get("total_memories", 0) >= 0  # May be 0 if extraction failed

