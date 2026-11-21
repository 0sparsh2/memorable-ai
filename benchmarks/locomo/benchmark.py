"""
LOCOMO Benchmark — snap-research/LOCOMO dataset + smoke fixtures.

Modes:
- smoke: minimal test_data.py (CI, no download)
- full: locomo10.json with retrieval + optional LLM answer/judge
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from benchmarks.locomo.loader import (
    CATEGORY_NAMES,
    SCORED_CATEGORIES,
    iter_qa,
    iter_sessions,
    iter_turns,
    load_locomo,
    parse_session_datetime,
    sample_id,
)
from benchmarks.locomo.metrics import aggregate_by_category, evidence_recall, score_answer_f1
from benchmarks.locomo import judge as judge_mod

logger = logging.getLogger(__name__)


class LOCOMOBenchmark:
    """
    LOCOMO benchmark for long-term conversation memory.

    Categories (official ids): 4 single-hop, 1 multi-hop, 2 temporal, 3 open-domain.
    """

    def __init__(
        self,
        memory_engine: Any,
        *,
        retrieval_limit: int = 12,
        use_llm: bool = True,
        judge_model: Optional[str] = None,
        answer_model: Optional[str] = None,
        qa_llm_timeout: Optional[float] = None,
        qa_context_max_chars: int = int(os.getenv("LOCOMO_QA_CONTEXT_MAX_CHARS", "4500")),
        qa_context_max_memories: int = int(os.getenv("LOCOMO_QA_CONTEXT_MAX_MEMORIES", "10")),
        qa_context_line_max_chars: int = int(os.getenv("LOCOMO_QA_CONTEXT_LINE_MAX_CHARS", "380")),
        ingest_mode: str = "raw",
        context_format: str = "dated",
        save_rows: bool = False,
    ):
        self.memory_engine = memory_engine
        self.retrieval_limit = retrieval_limit
        self.use_llm = use_llm
        self.ingest_mode = ingest_mode
        self.context_format = context_format
        self.save_rows = save_rows
        cfg = getattr(memory_engine, "config", None)
        default_model = cfg.llm.default_model if cfg else "gpt-4o-mini"
        self.answer_model = answer_model or default_model
        self.judge_model = judge_model or default_model
        self.qa_llm_timeout = qa_llm_timeout
        self.qa_context_max_chars = qa_context_max_chars
        self.qa_context_max_memories = qa_context_max_memories
        self.qa_context_line_max_chars = qa_context_line_max_chars

    async def run(self, mode: str = "smoke", **kwargs: Any) -> Dict[str, Any]:
        if mode == "smoke":
            return await self._run_smoke()
        if mode == "full":
            return await self.run_full(**kwargs)
        raise ValueError(f"Unknown mode: {mode}")

    async def _run_smoke(self) -> Dict[str, Any]:
        """Legacy substring retrieval on test_data fixtures."""
        from benchmarks.locomo.test_data import (
            TEST_CONVERSATIONS,
            get_multi_hop_questions,
            get_open_domain_questions,
            get_single_hop_questions,
            get_temporal_questions,
        )

        await self._ingest_smoke_conversations(TEST_CONVERSATIONS)

        results = {
            "single_hop": await self._score_smoke_category(get_single_hop_questions()),
            "multi_hop": await self._score_smoke_category(
                get_multi_hop_questions(), limit=10, combine=True
            ),
            "temporal": await self._score_smoke_category(get_temporal_questions()),
            "open_domain": await self._score_smoke_open(get_open_domain_questions()),
        }
        results["overall_score"] = sum(results.values()) / len(results)
        results["mode"] = "smoke"
        return results

    async def _ingest_smoke_conversations(self, conversations: List[Dict[str, Any]]) -> None:
        for conv in conversations:
            for msg in conv["messages"]:
                if msg["role"] == "user":
                    await self.memory_engine.add_memory(
                        content=msg["content"],
                        memory_type="fact",
                    )

    async def _score_smoke_category(
        self,
        questions: List[Dict[str, Any]],
        limit: int = 5,
        combine: bool = False,
    ) -> float:
        if not questions:
            return 0.0
        correct = 0
        for q in questions:
            results = await self.memory_engine.search_memories(q["question"], limit=limit)
            expected = q["expected_answer"].lower()
            if combine:
                text = " ".join(r.get("content", "") for r in results).lower()
                found = expected in text or any(
                    w in text for w in expected.split() if len(w) > 3
                )
            else:
                found = judge_mod.retrieval_hit(q["expected_answer"], results)
            if found:
                correct += 1
        acc = correct / len(questions)
        logger.info("Smoke category accuracy: %.2f%% (%d/%d)", acc * 100, correct, len(questions))
        return acc

    async def _score_smoke_open(self, questions: List[Dict[str, Any]]) -> float:
        if not questions:
            return 0.0
        correct = 0
        for q in questions:
            results = await self.memory_engine.search_memories(q["question"], limit=5)
            if results and any(
                kw in r.get("content", "").lower()
                for r in results
                for kw in q["question"].lower().split()
                if len(kw) > 4
            ):
                correct += 1
        return correct / len(questions)

    async def run_full(
        self,
        *,
        max_conversations: Optional[int] = None,
        max_questions_per_conv: Optional[int] = None,
        categories: Optional[Set[int]] = None,
        data_path: Optional[Path] = None,
        ingest_mode: Optional[str] = None,
        context_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run on snap-research/LOCOMO locomo10.json.

        Isolates each conversation via a fresh SQLite DB under a temp directory.
        """
        cats = categories or SCORED_CATEGORIES
        samples = load_locomo(data_path) if data_path else load_locomo()
        if max_conversations is not None:
            samples = samples[:max_conversations]

        ingest = ingest_mode or self.ingest_mode
        ctx_fmt = context_format or self.context_format

        all_rows: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            sid = sample_id(sample, idx)
            logger.info(
                "LOCOMO conversation %s (%d/%d) ingest=%s context=%s",
                sid,
                idx + 1,
                len(samples),
                ingest,
                ctx_fmt,
            )
            rows = await self._run_one_conversation(
                sample,
                sid,
                categories=cats,
                max_questions=max_questions_per_conv,
                ingest_mode=ingest,
                context_format=ctx_fmt,
            )
            all_rows.extend(rows)

        by_category = aggregate_by_category(all_rows)
        include_rows = self.save_rows or bool(os.getenv("LOCOMO_INCLUDE_ROWS"))
        return {
            "mode": "full",
            "ingest_mode": ingest,
            "context_format": ctx_fmt,
            "conversations": len(samples),
            "questions_evaluated": len(all_rows),
            "by_category": by_category,
            "overall_f1": by_category.get("overall", {}).get("f1_mean"),
            "overall_judge_accuracy": by_category.get("overall", {}).get("judge_accuracy"),
            "rows": all_rows if include_rows else None,
        }

    async def _run_one_conversation(
        self,
        sample: dict[str, Any],
        sid: str,
        *,
        categories: Set[int],
        max_questions: Optional[int],
        ingest_mode: str,
        context_format: str,
    ) -> List[Dict[str, Any]]:
        from memorable_ai import MemoryEngine

        db_dir = tempfile.mkdtemp(prefix="locomo_")
        db_path = os.path.join(db_dir, f"{sid}.db")
        engine = MemoryEngine(database=f"sqlite:///{db_path}")
        engine.enable()

        if ingest_mode == "oracle":
            pass
        elif ingest_mode == "extracted":
            await self._ingest_extracted_sessions(engine, sample)
        elif ingest_mode == "hybrid":
            await self._ingest_raw_turns(engine, sample)
            await self._ingest_extracted_sessions(engine, sample)
        else:
            await self._ingest_raw_turns(engine, sample)

        qa_list = list(iter_qa(sample, categories=categories))
        if max_questions is not None:
            qa_list = qa_list[:max_questions]

        rows: List[Dict[str, Any]] = []
        for qa in qa_list:
            row = await self._evaluate_question(
                engine,
                qa,
                sample=sample,
                ingest_mode=ingest_mode,
                context_format=context_format,
            )
            rows.append(row)

        engine.disable()
        return rows

    async def _ingest_raw_turns(self, engine: Any, sample: dict[str, Any]) -> None:
        temporal = engine._temporal
        for turn in iter_turns(sample):
            if not turn.text.strip():
                continue
            content = f"{turn.speaker}: {turn.text}"
            meta = {
                "dia_id": turn.dia_id,
                "session": turn.session_key,
                "session_date": turn.session_date,
            }
            ts = parse_session_datetime(turn.session_date)
            if temporal and ts:
                await temporal.add_temporal_memory(
                    content=content,
                    memory_type="fact",
                    timestamp=ts,
                    **meta,
                )
            else:
                await engine.add_memory(content=content, memory_type="fact", **meta)

    async def _store_extracted_memories(
        self, engine: Any, memories: List[Dict[str, Any]], *, session_key: str, date_str: str
    ) -> None:
        """Store extracted facts without Mem0 update LLM (avoids NOOP dropping benchmark facts)."""
        for mem in memories:
            mem.setdefault("metadata", {})
            mem["metadata"]["session_date"] = date_str
            mem["metadata"]["session"] = session_key
            mem["metadata"]["source"] = "extracted"
            content = (mem.get("content") or "").strip()
            if date_str and date_str not in content:
                mem["content"] = f"[session: {date_str}] {content}"
        await engine._storage.store_memories(memories)

    async def _ingest_raw_session(
        self, engine: Any, session_key: str, date_str: str, turns: List[dict[str, Any]]
    ) -> None:
        """Ingest one session's turns (used as fallback when extraction returns nothing)."""
        temporal = engine._temporal
        for turn in turns:
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            dia_id = turn.get("dia_id", "")
            speaker = turn.get("speaker", "")
            cap = turn.get("blip_caption")
            if cap:
                text = f"{text} [image: {cap}]"
            content = f"{speaker}: {text}"
            meta = {
                "dia_id": dia_id,
                "session": session_key,
                "session_date": date_str,
                "source": "raw_fallback",
            }
            ts = parse_session_datetime(date_str)
            if temporal and ts:
                await temporal.add_temporal_memory(
                    content=content,
                    memory_type="fact",
                    timestamp=ts,
                    **meta,
                )
            else:
                await engine.add_memory(content=content, memory_type="fact", **meta)

    async def _ingest_extracted_sessions(self, engine: Any, sample: dict[str, Any]) -> None:
        """Mem0-fair: LLM-extract atomic facts per session with session date anchor."""
        import asyncio
        import os

        session_delay = float(os.getenv("LOCOMO_EXTRACT_SESSION_DELAY", "1.5"))

        if not engine._extraction:
            logger.warning("Extraction not available; falling back to raw turns")
            await self._ingest_raw_turns(engine, sample)
            return

        conv = sample["conversation"]
        for session_key, date_str, turns in iter_sessions(conv):
            lines: List[str] = []
            session_dia_ids: List[str] = []
            for turn in turns:
                text = (turn.get("text") or "").strip()
                dia_id = turn.get("dia_id", "")
                if dia_id:
                    session_dia_ids.append(dia_id)
                if not text:
                    continue
                cap = turn.get("blip_caption")
                if cap:
                    text = f"{text} [image: {cap}]"
                prefix = f"[{dia_id}] " if dia_id else ""
                lines.append(f'{prefix}{turn.get("speaker", "")}: {text}')
            if not lines:
                continue

            session_text = "\n".join(lines)
            header = (
                f"Conversation session {session_key}. Session date: {date_str}.\n"
                "Extract atomic facts as standalone sentences. Resolve relative dates using the session date.\n"
                "Include dia_ids from lines like [D1:3] when a fact comes from that turn.\n\n"
            )
            messages = [{"role": "user", "content": header + session_text}]
            memories: List[Dict[str, Any]] = []
            try:
                memories = await engine._extraction.extract(messages)
                for mem in memories:
                    mem.setdefault("metadata", {})
                    if not mem["metadata"].get("dia_ids"):
                        mem["metadata"]["dia_ids"] = list(session_dia_ids)
                    elif isinstance(mem["metadata"]["dia_ids"], str):
                        mem["metadata"]["dia_ids"] = [mem["metadata"]["dia_ids"]]
                if memories:
                    await self._store_extracted_memories(
                        engine, memories, session_key=session_key, date_str=date_str
                    )
                else:
                    logger.warning(
                        "No facts extracted for session %s; ingesting raw turns", session_key
                    )
                    await self._ingest_raw_session(engine, session_key, date_str, turns)
            except Exception as e:
                logger.warning("Session extraction failed %s: %s", session_key, e)
                await self._ingest_raw_session(engine, session_key, date_str, turns)
            if session_delay > 0:
                await asyncio.sleep(session_delay)

    def _memories_from_evidence(
        self, sample: dict[str, Any], evidence: List[str]
    ) -> List[Dict[str, Any]]:
        """Oracle retrieval: load gold evidence turns by dia_id."""
        wanted = set(evidence or [])
        if not wanted:
            return []
        out: List[Dict[str, Any]] = []
        for turn in iter_turns(sample):
            if turn.dia_id in wanted:
                out.append(
                    {
                        "content": f"{turn.speaker}: {turn.text}",
                        "metadata": {
                            "dia_id": turn.dia_id,
                            "session_date": turn.session_date,
                            "session": turn.session_key,
                        },
                    }
                )
        return out

    async def _evaluate_question(
        self,
        engine: Any,
        qa: Any,
        *,
        sample: Optional[dict[str, Any]] = None,
        ingest_mode: str = "raw",
        context_format: str = "dated",
    ) -> Dict[str, Any]:
        if ingest_mode == "oracle" and sample is not None:
            memories = self._memories_from_evidence(sample, qa.evidence)
        else:
            memories = await engine.search_memories(qa.question, limit=self.retrieval_limit)
        dia_ids = judge_mod.collect_dia_ids(memories)
        recall = evidence_recall(dia_ids, qa.evidence)

        prediction = ""
        f1 = None
        judge_correct = None

        if self.use_llm:
            prediction = await judge_mod.generate_answer(
                qa.question,
                memories,
                self.answer_model,
                timeout=self.qa_llm_timeout,
                context_max_chars=self.qa_context_max_chars,
                context_max_memories=self.qa_context_max_memories,
                context_line_max_chars=self.qa_context_line_max_chars,
                context_format=context_format,
                category=qa.category,
            )
            f1 = score_answer_f1(prediction, qa.answer, qa.category)
            judge_correct = await judge_mod.judge_answer(
                qa.question, qa.answer, prediction, self.judge_model
            )
        else:
            recall = evidence_recall(dia_ids, qa.evidence)
            judge_correct = recall is not None and recall >= 1.0
            if not judge_correct:
                judge_correct = judge_mod.retrieval_hit(qa.answer, memories)
            f1 = 1.0 if judge_correct else 0.0

        return {
            "category": qa.category,
            "category_name": CATEGORY_NAMES.get(qa.category, str(qa.category)),
            "question": qa.question,
            "gold": qa.answer,
            "prediction": prediction,
            "f1": f1,
            "judge_correct": judge_correct,
            "evidence_recall": recall,
        }


async def compare_with_mem0(memorable_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare against Mem0 published baselines (paper / blog; not apples-to-apples)."""
    mem0_baseline = {
        "single_hop": 0.85,
        "multi_hop": 0.72,
        "temporal": 0.68,
        "open_domain": 0.65,
        "overall": 0.725,
    }
    by = memorable_results.get("by_category") or memorable_results
    memorable_scores = {}
    for key in ("single_hop", "multi_hop", "temporal", "open_domain"):
        entry = by.get(key)
        acc = None
        if isinstance(entry, dict):
            acc = entry.get("judge_accuracy") or entry.get("f1_mean")
        elif isinstance(entry, (int, float)):
            acc = entry
        if acc is None:
            top = memorable_results.get(key)
            if isinstance(top, (int, float)):
                acc = top
        memorable_scores[key] = acc

    overall = memorable_results.get("overall_judge_accuracy") or memorable_results.get(
        "overall_score"
    )
    if overall is not None:
        memorable_scores["overall"] = overall

    improvement = {
        k: (memorable_scores.get(k) or 0) - mem0_baseline.get(k, 0)
        for k in mem0_baseline
    }
    return {
        "memorable": memorable_scores,
        "mem0_baseline": mem0_baseline,
        "improvement": improvement,
    }
