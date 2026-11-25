"""
Unit tests for graph builder.
"""

import pytest
from memorable_ai.graph.builder import GraphBuilder


@pytest.fixture
def graph_builder():
    """Create graph builder instance."""
    return GraphBuilder()


def test_graph_initialization(graph_builder):
    """Test graph builder initialization."""
    assert graph_builder is not None
    assert graph_builder.graph is not None


def test_extract_entities(graph_builder):
    """Test entity extraction."""
    text = "I work at Google. I use Python."
    entities = graph_builder._extract_entities(text)
    assert len(entities) > 0
    assert "Google" in entities or "Python" in entities


def test_extract_relationships(graph_builder):
    """Test relationship extraction."""
    text = "I work at Google. Google uses Python."
    entities = ["I", "Google", "Python"]
    relationships = graph_builder._extract_relationships(text, entities)
    assert isinstance(relationships, list)


@pytest.mark.asyncio
async def test_update_graph(graph_builder):
    """Test updating graph with memories."""
    memories = [
        {
            "content": "User works at Google",
            "type": "fact",
            "id": 1
        },
        {
            "content": "Google uses Python",
            "type": "fact",
            "id": 2
        }
    ]

    await graph_builder.update_graph(memories)
    assert graph_builder.graph.number_of_nodes() > 0


@pytest.mark.asyncio
async def test_find_related(graph_builder):
    """Test finding related memories."""
    # Setup graph
    memories = [
        {"content": "User likes Python", "type": "preference", "id": 1},
        {"content": "Python is used in FastAPI", "type": "fact", "id": 2},
    ]
    await graph_builder.update_graph(memories)

    # Find related
    results = await graph_builder.find_related("Python", limit=10)
    assert isinstance(results, list)

