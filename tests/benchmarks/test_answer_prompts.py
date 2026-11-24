"""prereg v7: answer-prompt variants, the per-row prompt choice, the D8 guard pairing and the report's bars."""

import re

import pytest

from benchmarks.qa.e2e_qa import answer_prompt
from benchmarks.qa.prompts import ANSWER_INSTRUCTIONS, ANSWER_PROMPT, ANSWER_PROMPTS
from benchmarks.qa.v7_build import D8_EXPLANATION, d8_rows
from benchmarks.qa.v7_report import gain_verdict, refuses

EVAL_WORDS = re.compile(r"\b(eval\w*|test\w*|judge\w*|benchmark\w*|score\w*|rubric\w*)\b", re.I)


def test_p0_is_the_existing_prompt():
    assert ANSWER_PROMPTS["p0"] == ANSWER_PROMPT


@pytest.mark.parametrize("key", sorted(ANSWER_PROMPTS))
def test_each_prompt_renders_fully_and_carries_no_eval_words(key):
    text = ANSWER_PROMPTS[key].format(memories="M", date_line="D\n", question="Q?")
    assert "{" not in text and "}" not in text
    assert not EVAL_WORDS.search(text)
    head, tail = ANSWER_PROMPTS["p0"].split(ANSWER_INSTRUCTIONS["p0"])
    assert ANSWER_PROMPTS[key] == head + ANSWER_INSTRUCTIONS[key] + tail  # only the instruction differs


def test_a_row_picks_its_prompt_and_defaults_to_p0():
    rec = {"context": "ctx", "question": "q?", "date_line": "Today is 2023-05-30.\n"}
    assert answer_prompt(rec) == ANSWER_PROMPT.format(memories="ctx", date_line=rec["date_line"], question="q?")
    assert "advice, suggestions or an opinion" in answer_prompt({**rec, "prompt": "p1"})
    assert "short phrase" not in answer_prompt({**rec, "prompt": "p2"})


def test_d8_pairs_each_question_with_the_next_haystack_and_skips_preference():
    rows = [{"qid": g, "group": g, "category": c, "context": f"ctx-{g}", "answer": f"gold-{g}", "abstention": False}
            for g, c in (("c", "temporal-reasoning"), ("a", "multi-session"), ("b", "single-session-preference"))]
    d8 = {r["qid"]: r for r in d8_rows(rows)}
    assert set(d8) == {"a", "c"}  # preference left out
    assert d8["a"]["context"] == "ctx-b" and d8["a"]["d8_from"] == "b"  # sorted order: a -> b -> c -> a
    assert d8["c"]["context"] == "ctx-a"
    assert all(r["abstention"] and r["answer"] == D8_EXPLANATION for r in d8.values())


def test_gain_verdict_follows_b15s_form():
    up = {"diff": 0.05, "lo": 0.01, "hi": 0.09}
    flat = {"diff": -0.005, "lo": -0.03, "hi": 0.02}
    drop = {"diff": -0.02, "lo": -0.05, "hi": 0.01}
    assert gain_verdict({"locomo": flat, "lme": up}) == "pass"
    assert gain_verdict({"locomo": drop, "lme": up}) == "miss"
    assert gain_verdict({"locomo": flat, "lme": {"diff": 0.03, "lo": -0.01, "hi": 0.07}}) == "miss"


def test_refusal_pattern():
    assert refuses("I don't have any specific advice about cookies in my memories.")
    assert not refuses("Try adding sea salt; it pairs well with the turbinado sugar you used.")
