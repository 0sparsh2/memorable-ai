"""
Unit tests for validators.
"""

import pytest
from memorable_ai.utils.validators import (
    validate_connection_string,
    validate_memory_type,
    validate_mode,
    sanitize_content,
    validate_messages,
)


def test_validate_connection_string():
    """Test connection string validation."""
    assert validate_connection_string("sqlite:///test.db") is True
    assert validate_connection_string("postgresql://user:pass@localhost/db") is True
    assert validate_connection_string("mysql://user:pass@localhost/db") is True
    assert validate_connection_string("invalid") is False
    assert validate_connection_string("") is False


def test_validate_memory_type():
    """Test memory type validation."""
    assert validate_memory_type("fact") is True
    assert validate_memory_type("preference") is True
    assert validate_memory_type("skill") is True
    assert validate_memory_type("rule") is True
    assert validate_memory_type("context") is True
    assert validate_memory_type("invalid") is False


def test_validate_mode():
    """Test mode validation."""
    assert validate_mode("conscious") is True
    assert validate_mode("auto") is True
    assert validate_mode("hybrid") is True
    assert validate_mode("invalid") is False


def test_sanitize_content():
    """Test content sanitization."""
    # Normal content
    assert sanitize_content("Test content") == "Test content"
    
    # Long content
    long_content = "a" * 20000
    sanitized = sanitize_content(long_content, max_length=100)
    assert len(sanitized) <= 103  # 100 + "..."
    assert sanitized.endswith("...")
    
    # Empty content
    assert sanitize_content("") == ""
    assert sanitize_content("   ") == ""


def test_validate_messages():
    """Test message validation."""
    # Valid messages
    valid_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    assert validate_messages(valid_messages) is True

    # Invalid - not a list
    assert validate_messages("not a list") is False

    # Invalid - missing role
    invalid_messages = [{"content": "Hello"}]
    assert validate_messages(invalid_messages) is False

    # Invalid - missing content
    invalid_messages2 = [{"role": "user"}]
    assert validate_messages(invalid_messages2) is False

    # Invalid - content not string
    invalid_messages3 = [{"role": "user", "content": 123}]
    assert validate_messages(invalid_messages3) is False

