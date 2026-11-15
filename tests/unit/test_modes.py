"""
Unit tests for memory modes.
"""

import pytest
from memorable_ai.modes.conscious import ConsciousMode
from memorable_ai.modes.auto import AutoMode
from memorable_ai.modes.hybrid import HybridMode
from memorable_ai.core.retrieval import HybridRetriever
from memorable_ai.core.storage import Storage
import os


@pytest.fixture
def test_db():
    """Create test database."""
    db_path = "test_modes.db"
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def retriever(test_db):
    """Create retriever for modes."""
    storage = Storage(connection_string=f"sqlite:///{test_db}")
    return HybridRetriever(storage=storage)


@pytest.fixture
def conscious_mode(retriever):
    """Create conscious mode."""
    return ConsciousMode(retriever)


@pytest.fixture
def auto_mode(retriever):
    """Create auto mode."""
    return AutoMode(retriever)


@pytest.fixture
def hybrid_mode(retriever):
    """Create hybrid mode."""
    return HybridMode(retriever)


@pytest.mark.asyncio
async def test_conscious_mode(conscious_mode):
    """Test conscious mode."""
    session_id = "test_session"
    messages = [{"role": "user", "content": "Test query"}]

    context = await conscious_mode.get_context(session_id, messages)
    assert isinstance(context, str)

    # Second call should use cached memories
    context2 = await conscious_mode.get_context(session_id, messages)
    assert context == context2  # Should be same (cached)


@pytest.mark.asyncio
async def test_auto_mode(auto_mode):
    """Test auto mode."""
    messages = [{"role": "user", "content": "Test query"}]

    context = await auto_mode.get_context(messages)
    assert isinstance(context, str)


@pytest.mark.asyncio
async def test_hybrid_mode(hybrid_mode):
    """Test hybrid mode."""
    session_id = "test_session"
    messages = [{"role": "user", "content": "Test query"}]

    context = await hybrid_mode.get_context(session_id, messages)
    assert isinstance(context, str)


def test_conscious_clear_session(conscious_mode):
    """Test clearing session in conscious mode."""
    session_id = "test_session"
    conscious_mode._injected_memories[session_id] = []
    conscious_mode.clear_session(session_id)
    assert session_id not in conscious_mode._injected_memories

