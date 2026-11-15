"""LOCOMO loader and metrics unit tests."""

import pytest

from benchmarks.locomo.loader import (
    CATEGORY_SINGLE_HOP,
    format_turn_text,
    iter_qa,
    parse_session_datetime,
)
from benchmarks.locomo.metrics import score_answer_f1
from benchmarks.locomo import judge as judge_mod


def test_parse_session_datetime():
    dt = parse_session_datetime("1:56 pm on 8 May, 2023")
    assert dt is not None
    assert dt.year == 2023
    assert dt.month == 5
    assert dt.day == 8


def test_format_turn_with_caption():
    text = format_turn_text({"text": "Hi", "blip_caption": "a dog"})
    assert "dog" in text


def test_score_f1_simple():
    assert score_answer_f1("Seattle", "Seattle", CATEGORY_SINGLE_HOP) == 1.0
    assert score_answer_f1("unknown", "Seattle", CATEGORY_SINGLE_HOP) < 0.5


def test_format_memory_context_truncates():
    mems = [
        {"content": "x" * 500, "metadata": {"dia_id": "D1:1"}},
        {"content": "short", "metadata": {}},
    ]
    out = judge_mod.format_memory_context(
        mems, max_chars=2000, max_memories=2, max_line_chars=80
    )
    assert "..." in out
    assert len(out) < 1200


def test_iter_qa_smoke_sample():
    sample = {
        "qa": [
            {"question": "Q?", "answer": "A", "category": 4, "evidence": ["D1:1"]},
            {"question": "Adv", "answer": "x", "category": 5},
        ]
    }
    items = list(iter_qa(sample, categories={4}))
    assert len(items) == 1
    assert items[0].category == 4
