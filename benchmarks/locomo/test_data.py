"""
Minimal LOCOMO smoke fixtures for CI (no download required).
"""

from typing import Any, Dict, List

TEST_CONVERSATIONS: List[Dict[str, Any]] = [
    {
        "sample_id": "smoke_1",
        "messages": [
            {"role": "user", "content": "My name is Alex and I work at Acme Corp."},
            {"role": "assistant", "content": "Nice to meet you, Alex!"},
            {"role": "user", "content": "I moved to Seattle last month."},
        ],
    }
]

_SINGLE_HOP = [
    {
        "question": "What is Alex's name?",
        "expected_answer": "Alex",
        "category": 4,
    },
    {
        "question": "Where does Alex work?",
        "expected_answer": "Acme Corp",
        "category": 4,
    },
]

_MULTI_HOP = [
    {
        "question": "Where does Alex live now?",
        "expected_answer": "Seattle",
        "category": 1,
    },
]

_TEMPORAL = [
    {
        "question": "When did Alex move to Seattle?",
        "expected_answer": "last month",
        "category": 2,
    },
]

_OPEN_DOMAIN = [
    {
        "question": "Tell me about Alex's career.",
        "expected_answer": "Acme",
        "category": 3,
    },
]


def get_single_hop_questions() -> List[Dict[str, Any]]:
    return list(_SINGLE_HOP)


def get_multi_hop_questions() -> List[Dict[str, Any]]:
    return list(_MULTI_HOP)


def get_temporal_questions() -> List[Dict[str, Any]]:
    return list(_TEMPORAL)


def get_open_domain_questions() -> List[Dict[str, Any]]:
    return list(_OPEN_DOMAIN)
