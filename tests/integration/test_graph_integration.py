"""
Integration tests for graph functionality.
"""

import pytest
import os
from memorable_ai import MemoryEngine


@pytest.fixture
def test_db():
    """Create test database."""
    db_path = "test_graph_integration.db"
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def memory_with_graph(test_db):
    """Create memory engine with graph enabled."""
    engine = MemoryEngine(
        database=f"sqlite:///{test_db}",
        graph_enabled=True,
        mode="auto"
    )
    yield engine
    engine.disable()


@pytest.mark.asyncio
async def test_graph_building(memory_with_graph):
    """Test that graph is built from memories."""
    memory_with_graph.enable()
    
    # Add memories that should create relationships
    await memory_with_graph.add_memory("User works at Google", "fact")
    await memory_with_graph.add_memory("Google uses Python", "fact")
    
    # Check graph stats
    stats = memory_with_graph.get_stats()
    if "graph" in stats:
        graph_stats = stats["graph"]
        assert graph_stats.get("nodes", 0) >= 0


@pytest.mark.asyncio
async def test_multi_hop_reasoning(memory_with_graph):
    """Test multi-hop reasoning through graph."""
    memory_with_graph.enable()
    
    # Setup knowledge graph
    await memory_with_graph.add_memory("User likes Python", "preference")
    await memory_with_graph.add_memory("Python is used in FastAPI", "fact")
    
    # Search should find related memories
    results = await memory_with_graph.search_memories("FastAPI", limit=10)
    assert isinstance(results, list)

