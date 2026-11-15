"""
LoCoMo evidence-retrieval evaluation (no LLM calls, no API cost).

Ingests each LoCoMo conversation through the real Memorable write pipeline
(turns -> episodes -> facts -> graph), then for every question checks whether
the annotated evidence turns (``dia_id``s) come back in the top-k:

* turn recall@k for each signal (dense, lexical, fused RRF)
* episode recall@k (an episode "hits" when it contains an evidence turn)

This measures the retrieval layer only. It is *not* LoCoMo QA accuracy (that
needs an answer model plus a judge; see ``benchmark.py``). Category 5
(adversarial) questions have no evidence and are skipped.

    python -m benchmarks.locomo.retrieval_eval --convs 2 --k 5 --embedder sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MEMORABLE_LLM__DEFAULT_MODEL", "none")

from benchmarks.locomo.loader import (  # noqa: E402
    CATEGORY_ADVERSARIAL,
    format_turn_text,
    iter_qa,
    iter_sessions,
    load_locomo,
    parse_session_datetime,
    sample_id,
)

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


WEIGHTS = [(1.0, 0.8), (1.0, 1.0), (0.7, 1.0), (0.5, 1.0), (0.3, 1.0)]


async def evaluate(convs: int, k: int, embedder: str, db_dir: Path, turn_context: float = 0.5) -> Dict:
    os.environ["MEMORABLE_LLM__EMBEDDING_MODEL"] = embedder
    from memorable_ai import MemoryEngine
    from memorable_ai.core.models import Episode, Turn
    from memorable_ai.core.scope import Scope
    from memorable_ai.core.vector_index import reciprocal_rank_fusion

    data = load_locomo()
    totals = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # signal -> category -> [hits, n]
    timings = {"ingest_s": 0.0, "query_ms": []}
    stats = {"turns": 0, "episodes": 0, "facts": 0, "questions": 0}

    for ci, sample in enumerate(data[:convs]):
        conv_id = sample_id(sample, ci)
        eng = MemoryEngine(database=f"sqlite:///{db_dir / f'{conv_id}.db'}", write_mode="sync")
        eng._ensure_components()
        if eng._episodic is not None:
            eng._episodic.cfg.turn_context_weight = turn_context
        conv = sample["conversation"]
        speakers = [conv.get("speaker_a"), conv.get("speaker_b")]
        dia_by_turn: Dict[int, str] = {}
        t0 = time.time()
        for session_key, date_str, turns in iter_sessions(conv):
            when = parse_session_datetime(date_str)
            msgs = [
                {
                    "role": "user" if t.get("speaker") == speakers[0] else "assistant",
                    "content": f"{t.get('speaker')}: {format_turn_text(t)}",
                }
                for t in turns
            ]
            await eng.ingest(msgs, None, user_id=conv_id, session_id=session_key, occurred_at=when)
            await eng.end_session(user_id=conv_id, session_id=session_key)
            s = eng._storage.get_session()
            try:
                rows = s.query(Turn.id).filter(Turn.session_id == session_key).order_by(Turn.seq).all()
            finally:
                s.close()
            for (tid,), t in zip(rows, turns):
                dia_by_turn[tid] = t.get("dia_id")
        timings["ingest_s"] += time.time() - t0

        s = eng._storage.get_session()
        try:
            ep_dias = defaultdict(set)
            for tid, eid in s.query(Turn.id, Turn.episode_id).all():
                ep_dias[eid].add(dia_by_turn.get(tid))
            stats["episodes"] += s.query(Episode).count()
        finally:
            s.close()
        stats["turns"] += len(dia_by_turn)
        stats["facts"] += eng.get_stats().get("active_memories", 0)

        scope = Scope(user_id=conv_id)
        r = eng._retrieval
        for qa in iter_qa(sample, categories={1, 2, 3, 4}):
            if qa.category == CATEGORY_ADVERSARIAL or not qa.evidence:
                continue
            stats["questions"] += 1
            gold = set(qa.evidence)
            q0 = time.time()
            qvec = r._encode(qa.question)
            tvec = r._encode(qa.question, kind="turn")  # the turn tier may use its own embedder
            dense = [i for i, _ in eng._storage.vector_search("turn", tvec, k=50, user_id=conv_id)]
            lexical = [i for i, _ in eng._storage.lexical_search("turn", qa.question, k=50, user_id=conv_id)]
            fused = [
                i
                for i, _ in reciprocal_rank_fusion(
                    {"d": dense, "l": lexical}, {"d": r.semantic_weight, "l": r.bm25_weight}, k=r.rrf_k
                )
            ]
            sweep = {
                f"turn/rrf d{wd}:l{wl}": [
                    i for i, _ in reciprocal_rank_fusion({"d": dense, "l": lexical}, {"d": wd, "l": wl})
                ]
                for wd, wl in WEIGHTS
            }
            eps = r._rank_layer("episode", qa.question, qvec, scope, k)
            timings["query_ms"].append((time.time() - q0) * 1000)
            cat = CATEGORY_NAMES.get(qa.category, str(qa.category))
            for name, ids in (("turn/dense", dense), ("turn/lexical", lexical), ("turn/fused", fused), *sweep.items()):
                hit = bool(gold & {dia_by_turn.get(i) for i in ids[:k]})
                totals[name][cat][0] += hit
                totals[name][cat][1] += 1
            ep_hit = any(gold & ep_dias.get(e, set()) for e in eps[:k])
            totals["episode/fused"][cat][0] += ep_hit
            totals["episode/fused"][cat][1] += 1
        eng.close()

    report = {"k": k, "embedder": embedder, "turn_context": turn_context, "conversations": convs, **stats}
    report["ingest_seconds"] = round(timings["ingest_s"], 1)
    q = sorted(timings["query_ms"])
    report["query_ms_p50"] = round(q[len(q) // 2], 1) if q else None
    report["recall"] = {}
    for signal, cats in totals.items():
        hits = sum(h for h, _ in cats.values())
        n = sum(n for _, n in cats.values())
        report["recall"][signal] = {
            "overall": round(hits / n, 3) if n else None,
            **{c: round(h / n_, 3) for c, (h, n_) in sorted(cats.items()) if n_},
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--convs", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--turn-context", type=float, default=0.5, help="contextual turn embedding weight")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        report = asyncio.run(evaluate(args.convs, args.k, args.embedder, Path(d), args.turn_context))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
