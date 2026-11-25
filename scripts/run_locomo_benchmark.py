#!/usr/bin/env python3
"""
Run LOCOMO benchmark (smoke or full snap-research dataset).

Examples:
  python scripts/run_locomo_benchmark.py --mode smoke
  python scripts/run_locomo_benchmark.py --mode full --conversations 1 --questions 20
  python scripts/run_locomo_benchmark.py --mode full --no-llm  # retrieval-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run LOCOMO benchmark")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--conversations", type=int, default=None)
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument("--retrieval-limit", type=int, default=12)
    parser.add_argument("--qa-timeout", type=float, default=None, help="Answer LLM timeout seconds")
    parser.add_argument(
        "--qa-context-chars",
        type=int,
        default=int(os.getenv("LOCOMO_QA_CONTEXT_MAX_CHARS", "4500")),
    )
    parser.add_argument(
        "--qa-context-memories",
        type=int,
        default=int(os.getenv("LOCOMO_QA_CONTEXT_MAX_MEMORIES", "10")),
    )
    parser.add_argument(
        "--qa-context-line-chars",
        type=int,
        default=int(os.getenv("LOCOMO_QA_CONTEXT_LINE_MAX_CHARS", "380")),
    )
    parser.add_argument("--no-llm", action="store_true", help="Retrieval hit only (no answer/judge LLM)")
    parser.add_argument(
        "--ingest-mode",
        choices=("raw", "extracted", "hybrid", "oracle"),
        default="raw",
    )
    parser.add_argument(
        "--context-format",
        choices=("minimal", "dated"),
        default="dated",
    )
    parser.add_argument("--save-rows", action="store_true", help="Include per-question rows in JSON")
    parser.add_argument("--database", default="sqlite:///locomo_benchmark.db")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON results here")
    args = parser.parse_args()

    from memorable_ai import MemoryEngine
    from benchmarks.locomo.benchmark import LOCOMOBenchmark, compare_with_mem0

    memory = MemoryEngine(database=args.database)
    memory.enable()

    bench = LOCOMOBenchmark(
        memory,
        retrieval_limit=args.retrieval_limit,
        use_llm=not args.no_llm,
        qa_llm_timeout=args.qa_timeout,
        qa_context_max_chars=args.qa_context_chars,
        qa_context_max_memories=args.qa_context_memories,
        qa_context_line_max_chars=args.qa_context_line_chars,
        ingest_mode=args.ingest_mode,
        context_format=args.context_format,
        save_rows=args.save_rows,
    )

    if args.mode == "smoke":
        results = await bench.run(mode="smoke")
    else:
        results = await bench.run(
            mode="full",
            max_conversations=args.conversations,
            max_questions_per_conv=args.questions,
            ingest_mode=args.ingest_mode,
            context_format=args.context_format,
        )

    comparison = await compare_with_mem0(results)
    payload = {"results": results, "comparison": comparison}

    print(json.dumps(payload, indent=2, default=str))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote %s", args.output)

    memory.disable()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
