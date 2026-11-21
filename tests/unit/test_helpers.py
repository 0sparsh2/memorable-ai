"""
Unit tests for helper utilities.
"""

import pytest
from datetime import datetime
from memorable_ai.utils.helpers import (
    generate_memory_id,
    format_timestamp,
    parse_timestamp,
    chunk_text,
    calculate_similarity,
    merge_memories,
)


def test_generate_memory_id():
    """Test memory ID generation."""
    id1 = generate_memory_id("Test content", "fact")
    id2 = generate_memory_id("Test content", "fact")
    id3 = generate_memory_id("Different content", "fact")

    # Same content should generate same ID
    assert id1 == id2
    
    # Different content should generate different ID
    assert id1 != id3


def test_format_timestamp():
    """Test timestamp formatting."""
    timestamp = format_timestamp()
    assert isinstance(timestamp, str)
    assert "T" in timestamp or "-" in timestamp

    dt = datetime(2025, 1, 1, 12, 0, 0)
    formatted = format_timestamp(dt)
    assert "2025" in formatted


def test_parse_timestamp():
    """Test timestamp parsing."""
    dt = datetime(2025, 1, 1, 12, 0, 0)
    timestamp_str = dt.isoformat()
    parsed = parse_timestamp(timestamp_str)
    assert parsed.year == 2025
    assert parsed.month == 1


def test_chunk_text():
    """Test text chunking."""
    short_text = "Short text"
    chunks = chunk_text(short_text, max_length=100)
    assert len(chunks) == 1
    assert chunks[0] == short_text

    long_text = " ".join(["word"] * 1000)
    chunks = chunk_text(long_text, max_length=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 103 for chunk in chunks)  # 100 + "..."

    # Empty text
    assert chunk_text("") == [""]


def test_calculate_similarity():
    """Test similarity calculation."""
    # Identical texts
    assert calculate_similarity("Hello world", "Hello world") == 1.0

    # Completely different
    assert calculate_similarity("Hello", "Goodbye") < 0.5

    # Some overlap
    similarity = calculate_similarity("Hello world", "Hello there")
    assert 0 < similarity < 1.0

    # Empty texts
    assert calculate_similarity("", "") == 0.0
    assert calculate_similarity("Hello", "") == 0.0


def test_merge_memories():
    """Test memory merging."""
    memories = [
        {"content": "User likes Python", "type": "preference"},
        {"content": "user likes python", "type": "preference"},  # Duplicate (case)
        {"content": "User likes JavaScript", "type": "preference"},
    ]

    merged = merge_memories(memories)
    assert len(merged) == 2  # Should remove duplicate

    # Test with similar memories
    memories2 = [
        {"content": "I work at Google", "type": "fact", "metadata": {"source": "conv1"}},
        {"content": "I work at Google Inc", "type": "fact", "metadata": {"source": "conv2"}},
    ]

    merged2 = merge_memories(memories2)
    # Should merge similar memories
    assert len(merged2) <= len(memories2)

