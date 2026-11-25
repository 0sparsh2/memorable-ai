#!/usr/bin/env python3
"""
Run LOCOMO ablation experiments and append to ResearchDoc/experiments/.

Each experiment records: hypothesis, config, results JSON, summary line in experiment_log.md.

Example:
  python scripts/run_locomo_experiments.py --questions 10 --experiments dated,extracted,oracle
  python scripts/run_locomo_experiments.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS_DIR = ROOT / "ResearchDoc" / "experiments"
LOG_MD = EXPERIMENTS_DIR / "experiment_log.md"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# id -> (description, ingest_mode, context_format)
EXPERIMENT_REGISTRY = {
    "baseline_raw": {
        "hypothesis": "Reproduce original benchmark: raw turns, no session dates in QA context.",
        "ingest_mode": "raw",
        "context_format": "minimal",
    },
    "dated_context": {
        "hypothesis": "Fix R2: show [session: DATE] in answer context + temporal answer prompt.",
        "ingest_mode": "raw",
        "context_format": "dated",
    },
    "extracted": {
        "hypothesis": "Fix R1: Mem0-fair per-session LLM extraction into semantic facts.",
        "ingest_mode": "extracted",
        "context_format": "dated",
    },
    "hybrid_ingest": {
        "hypothesis": "Raw episodic + extracted semantic memories (dual store).",
        "ingest_mode": "hybrid",
        "context_format": "dated",
    },
    "oracle": {
        "hypothesis": "Upper bound: gold evidence turns only (perfect retrieval).",
        "ingest_mode": "oracle",
        "context_format": "dated",
    },
    "retrieval_only_dated": {
        "hypothesis": "Retrieval diagnostic without answer LLM cost.",
        "ingest_mode": "raw",
        "context_format": "dated",
        "no_llm": True,
    },
}


def _ensure_log_header() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_MD.exists():
        return
    LOG_MD.write_text(
        "# LOCOMO experiment log\n\n"
        "| ID | Date | Ingest | Context | Judge acc | F1 | Temp recall | Notes |\n"
        "|----|------|--------|---------|-----------|-----|-------------|-------|\n",
        encoding="utf-8",
    )


def _append_log_row(
    exp_id: str,
    cfg: dict,
    results: dict,
    notes: str = "",
) -> None:
    _ensure_log_header()
    by = results.get("by_category", {})
    overall = results.get("overall_judge_accuracy")
    f1 = results.get("overall_f1")
    temp = by.get("temporal", {})
    temp_recall = temp.get("evidence_recall_mean")
    if temp_recall is not None:
        temp_recall = f"{temp_recall * 100:.0f}%"
    judge_s = f"{overall * 100:.1f}%" if overall is not None else "n/a"
    f1_s = f"{f1 * 100:.1f}%" if f1 is not None else "n/a"
    row = (
        f"| {exp_id} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} "
        f"| {cfg.get('ingest_mode')} | {cfg.get('context_format')} "
        f"| {judge_s} | {f1_s} | {temp_recall or 'n/a'} | {notes} |\n"
    )
    with LOG_MD.open("a", encoding="utf-8") as f:
        f.write(row)


async def run_one(
    exp_id: str,
    *,
    conversations: int,
    questions: int,
    qa_timeout: float | None,
) -> dict:
    spec = EXPERIMENT_REGISTRY[exp_id]
    from memorable_ai import MemoryEngine
    from benchmarks.locomo.benchmark import LOCOMOBenchmark, compare_with_mem0

    memory = MemoryEngine(database="sqlite:///locomo_experiments.db")
    memory.enable()

    bench = LOCOMOBenchmark(
        memory,
        ingest_mode=spec["ingest_mode"],
        context_format=spec["context_format"],
        use_llm=not spec.get("no_llm", False),
        save_rows=True,
        qa_llm_timeout=qa_timeout,
    )
    results = await bench.run(
        mode="full",
        max_conversations=conversations,
        max_questions_per_conv=questions,
        ingest_mode=spec["ingest_mode"],
        context_format=spec["context_format"],
    )
    comparison = await compare_with_mem0(results)
    memory.disable()

    payload = {
        "experiment_id": exp_id,
        "hypothesis": spec["hypothesis"],
        "config": {
            "ingest_mode": spec["ingest_mode"],
            "context_format": spec["context_format"],
            "no_llm": spec.get("no_llm", False),
            "conversations": conversations,
            "questions": questions,
            "qa_timeout": qa_timeout,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "comparison": comparison,
    }

    out_path = EXPERIMENTS_DIR / f"{exp_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", out_path)

    _append_log_row(exp_id, spec, results)
    return payload


async def main() -> int:
    parser = argparse.ArgumentParser(description="LOCOMO ablation experiments")
    parser.add_argument(
        "--experiments",
        default="retrieval_only_dated,dated_context",
        help="Comma-separated experiment ids",
    )
    parser.add_argument("--conversations", type=int, default=1)
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--qa-timeout", type=float, default=300.0)
    parser.add_argument("--list", action="store_true", help="List experiment ids")
    args = parser.parse_args()

    if args.list:
        for eid, spec in EXPERIMENT_REGISTRY.items():
            print(f"{eid}: {spec['hypothesis']}")
        return 0

    os.environ.setdefault("LOCOMO_QA_LLM_TIMEOUT", str(int(args.qa_timeout)))
    os.environ.setdefault("LOCOMO_JUDGE_LLM_TIMEOUT", "180")

    import asyncio

    inter_exp_delay = float(os.getenv("LOCOMO_EXPERIMENT_DELAY", "5"))

    ids = [x.strip() for x in args.experiments.split(",") if x.strip()]
    for i, eid in enumerate(ids):
        if i > 0 and inter_exp_delay > 0:
            logger.info("Waiting %.1fs before next experiment (rate-limit spacing)", inter_exp_delay)
            await asyncio.sleep(inter_exp_delay)
        if eid not in EXPERIMENT_REGISTRY:
            logger.error("Unknown experiment %s", eid)
            return 1
        spec = EXPERIMENT_REGISTRY[eid]
        logger.info("=== Running experiment: %s ===", eid)
        if spec.get("ingest_mode") in ("extracted", "hybrid"):
            os.environ["MEMORABLE_EXTRACTION__ADD_ONLY"] = "true"
        payload = await run_one(
            eid,
            conversations=args.conversations,
            questions=args.questions,
            qa_timeout=args.qa_timeout,
        )
        r = payload["results"]
        print(
            f"{eid}: judge={r.get('overall_judge_accuracy')} "
            f"f1={r.get('overall_f1')} n={r.get('questions_evaluated')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
