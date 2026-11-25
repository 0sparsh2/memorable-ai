"""
Integration tests for memory modes.
"""

import pytest
import os
from memorable_ai import MemoryEngine


@pytest.fixture
def test_db():
    """Create test database."""
    db_path = "test_modes_integration.db"
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.mark.asyncio
async def test_conscious_mode_integration(test_db):
    """Test conscious mode end-to-end."""
    memory = MemoryEngine(database=f"sqlite:///{test_db}", mode="conscious")
    memory.enable()
    
    # Add memories
    await memory.add_memory("User likes Python", "preference")
    
    # Test retrieval (should cache for session)
    messages = [{"role": "user", "content": "Test"}]
    context1 = await memory._mode_handler.get_context("session1", messages)
    context2 = await memory._mode_handler.get_context("session1", messages)
    
    # Should be same (cached)
    assert context1 == context2


@pytest.mark.asyncio
async def test_auto_mode_integration(test_db):
    """Test auto mode end-to-end."""
    memory = MemoryEngine(database=f"sqlite:///{test_db}", mode="auto")
    memory.enable()
    
    # Add memories
    await memory.add_memory("User works at Google", "fact")
    
    # Test retrieval (should retrieve dynamically)
    messages = [{"role": "user", "content": "Where do I work?"}]
    context = await memory._mode_handler.get_context(messages)
    
    assert isinstance(context, str)


@pytest.mark.asyncio
async def test_hybrid_mode_integration(test_db):
    """Test hybrid mode end-to-end."""
    memory = MemoryEngine(database=f"sqlite:///{test_db}", mode="hybrid")
    memory.enable()
    
    # Add memories
    await memory.add_memory("User prefers Python", "preference")
    
    # Test retrieval (should combine both modes)
    messages = [{"role": "user", "content": "What do I prefer?"}]
    context = await memory._mode_handler.get_context("session1", messages)
    
    assert isinstance(context, str)

