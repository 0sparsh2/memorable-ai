"""
LLM answer generation and judging for LOCOMO (Mem0-style pipeline).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_QA_CONTEXT_CHARS = 4500
DEFAULT_QA_MAX_MEMORIES = 10
DEFAULT_QA_LINE_CHARS = 380

ANSWER_SYSTEM = (
    "You answer questions using ONLY the provided memory context. "
    "Reply with a short phrase. If the answer is not in the context, say 'not mentioned'."
)

ANSWER_SYSTEM_TEMPORAL = (
    "You answer temporal questions using ONLY the provided memory context. "
    "Each line may include [session: DATE] — use it to resolve relative phrases "
    "(yesterday, last week) to calendar dates. Reply with a short date or time phrase only. "
    "If the answer is not in the context, say 'not mentioned'."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator. Given a question, gold answer, and model answer, "
    "reply with exactly one word: CORRECT or WRONG. "
    "CORRECT if the model answer conveys the same fact as the gold answer. "
    "For dates, accept equivalent formats (e.g. '7 May 2023' = '7 May, 2023' = 'May 7, 2023'). "
    "For relative dates, CORRECT only if the gold answer is explicitly stated or clearly implied."
)


def _qa_llm_timeout_seconds(model: str) -> Optional[float]:
    """Benchmark QA calls — default higher than extraction so NIM can finish."""
    raw = os.getenv("LOCOMO_QA_LLM_TIMEOUT")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    from memorable_ai.integrations.nvidia_nim import is_nim_model

    if is_nim_model(model):
        return float(os.getenv("LOCOMO_QA_LLM_TIMEOUT_NIM_DEFAULT", "240"))
    return float(os.getenv("LOCOMO_QA_LLM_TIMEOUT_OTHER_DEFAULT", "120"))


def _truncate_turn(text: str, max_line_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_line_chars:
        return text
    return text[: max_line_chars - 3].rstrip() + "..."


def format_memory_context(
    memories: List[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_QA_CONTEXT_CHARS,
    max_memories: int = DEFAULT_QA_MAX_MEMORIES,
    max_line_chars: int = DEFAULT_QA_LINE_CHARS,
    context_format: str = "dated",
) -> str:
    """Pack retrieved memories for the answer LLM — short prompts reduce NIM timeouts."""
    lines: List[str] = []
    total = 0
    for i, m in enumerate(memories[:max_memories], 1):
        meta = m.get("metadata") or {}
        dia = meta.get("dia_id", "")
        parts = []
        if dia:
            parts.append(f"[{dia}]")
        if context_format == "dated":
            session_date = meta.get("session_date")
            if session_date:
                parts.append(f"[session: {session_date}]")
        prefix = " ".join(parts) + " " if parts else ""
        body = _truncate_turn(str(m.get("content", "")), max_line_chars)
        line = f"{i}. {prefix}{body}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines) if lines else "(no memories retrieved)"


def _context_retry_schedule(
    max_chars: int, max_memories: int, max_line_chars: int
) -> List[Tuple[int, int, int]]:
    """Primary prompt then tighter packing if the provider is slow or overloaded."""
    return [
        (max_chars, max_memories, max_line_chars),
        (max(max_chars // 2, 1800), min(max_memories, 6), min(max_line_chars, 280)),
        (max(max_chars // 4, 1200), min(max_memories, 4), min(max_line_chars, 220)),
    ]


async def generate_answer(
    question: str,
    memories: List[dict[str, Any]],
    model: str,
    *,
    timeout: Optional[float] = None,
    context_max_chars: int = DEFAULT_QA_CONTEXT_CHARS,
    context_max_memories: int = DEFAULT_QA_MAX_MEMORIES,
    context_line_max_chars: int = DEFAULT_QA_LINE_CHARS,
    context_format: str = "dated",
    category: Optional[int] = None,
) -> str:
    from memorable_ai.core.llm_client import acompletion_with_fallback, message_content

    base_timeout = timeout if timeout is not None else _qa_llm_timeout_seconds(model)
    schedule = _context_retry_schedule(
        context_max_chars, context_max_memories, context_line_max_chars
    )

    last_err: Optional[BaseException] = None
    for attempt, (mc, mm, mlc) in enumerate(schedule):
        context = format_memory_context(
            memories,
            max_chars=mc,
            max_memories=mm,
            max_line_chars=mlc,
            context_format=context_format,
        )
        system = ANSWER_SYSTEM_TEMPORAL if category == 2 else ANSWER_SYSTEM
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
            },
        ]
        try:
            t = (base_timeout + 30 * attempt) if base_timeout is not None else None
            resp = await acompletion_with_fallback(
                model=model,
                messages=messages,
                max_tokens=128,
                timeout=t,
            )
            text = message_content(resp)
            if text:
                return text
            logger.warning("Answer LLM returned empty (attempt %s)", attempt + 1)
        except Exception as e:
            last_err = e
            err_l = str(e).lower()
            retryable = any(
                x in err_l
                for x in (
                    "timeout",
                    "504",
                    "timed out",
                    "gateway",
                    "apitimeouterror",
                    "429",
                    "rate limit",
                    "too many requests",
                )
            )
            logger.warning(
                "Answer LLM attempt %s failed (chars=%s memories=%s): %s",
                attempt + 1,
                mc,
                mm,
                e,
            )
            if any(x in err_l for x in ("429", "rate limit", "too many requests")):
                import asyncio

                await asyncio.sleep(3.0 * (attempt + 1))
            if not retryable:
                break

    if last_err:
        logger.warning("Answer generation exhausted retries: %s", last_err)
    return ""


async def judge_answer(
    question: str,
    gold: str,
    prediction: str,
    model: str,
    *,
    timeout: Optional[float] = None,
) -> bool:
    from memorable_ai.core.llm_client import acompletion_with_fallback, message_content

    if not (prediction or "").strip():
        return False

    j_timeout = timeout
    if j_timeout is None:
        raw = os.getenv("LOCOMO_JUDGE_LLM_TIMEOUT")
        if raw:
            try:
                j_timeout = float(raw)
            except ValueError:
                j_timeout = None
        if j_timeout is None:
            j_timeout = float(os.getenv("LOCOMO_JUDGE_LLM_TIMEOUT_NIM_DEFAULT", "180"))

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Gold: {gold}\n"
                f"Model: {prediction}\n"
                "Verdict:"
            ),
        },
    ]
    try:
        resp = await acompletion_with_fallback(
            model=model,
            messages=messages,
            max_tokens=16,
            timeout=j_timeout,
        )
        text = message_content(resp).upper()
        return "CORRECT" in text and "WRONG" not in text
    except Exception as e:
        logger.warning("Judge failed: %s", e)
        return False


def retrieval_hit(expected: str, memories: List[dict[str, Any]]) -> bool:
    """Smoke-test style: gold substring in any retrieved memory."""
    expected_l = expected.lower()
    for m in memories:
        content = (m.get("content") or "").lower()
        if expected_l in content:
            return True
        if any(w in content for w in expected_l.split() if len(w) > 3):
            return True
    return False


def collect_dia_ids(memories: List[dict[str, Any]]) -> Set[str]:
    ids: Set[str] = set()
    for m in memories:
        meta = m.get("metadata") or {}
        dia = meta.get("dia_id")
        if dia:
            ids.add(str(dia))
        extra = meta.get("dia_ids")
        if isinstance(extra, (list, tuple, set)):
            ids.update(str(d) for d in extra if d)
        elif isinstance(extra, str) and extra:
            ids.add(extra)
        content = m.get("content") or ""
        for match in re.findall(r"\bD\d+:\d+\b", content):
            ids.add(match)
    return ids
