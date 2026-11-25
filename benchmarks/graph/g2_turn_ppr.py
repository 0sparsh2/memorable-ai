"""
G2 experiment: does a typed Personalized PageRank walk over the LLM-built graph put evidence turns
into context that A3's score fusion + rerank leaves out? Retrieval only, no LLM calls.

    python -m benchmarks.graph.g2_turn_ppr                       # dev conversations (prereg v5 split)
    python -m benchmarks.graph.g2_turn_ppr --convs conv-26,...   # one confirmation run on held-out ones

It reads the v4.2 LLM ingests (benchmarks/qa/runs/v4/ingest_cache/locomo), copies each to a temp
file, and adds turn -> entity ``mentions`` edges there (whole-word matches of entity names and
aliases in the turn body; the speaker prefix is ignored). Then, per question:

* ``A3``: the QA harness's turn list (score fusion, Ettin-17M rerank of the top 20).
* ``fuse``: PPR turn scores (seeded HippoRAG-2 style: query entities, plus the top first-stage turns
  at a small weight) join the first stage as a z-scored signal; graph-only turns enter the pool;
  the reranker then orders the top 20 as in A3.
* ``slots``: A3's list with the last ``m`` places given to the best PPR turns A3 left out.

Metric: the share of questions with all gold evidence turns in the top k ("all") and with any
("any"), the coverage numbers prereg v4 used. Paired bootstrap against A3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MEMORABLE_LLM__DEFAULT_MODEL", "none")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "benchmarks" / "qa" / "runs" / "v4" / "ingest_cache" / "locomo"
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
BASE_W = {"about": 1.0, "derived_from": 1.0, "mentions": 1.0, "relates": 1.0, "alias_of": 1.0}
PROFILES = {"typed": BASE_W, "no-mentions": {k: v for k, v in BASE_W.items() if k != "mentions"}}


def _z(v: dict) -> dict:
    if not v:
        return {}
    a = np.array(list(v.values()), dtype=float)
    sd = a.std() or 1.0
    return {k: (x - a.mean()) / sd for k, x in v.items()}


def add_turn_mentions(eng, conv: str, speakers) -> int:
    """turn -mentions-> entity for whole-word name/alias matches in the turn body."""
    from memorable_ai.core.models import Entity, EntityAlias, Turn

    s = eng._storage.get_session()
    try:
        names = defaultdict(set)  # norm -> entity ids
        skip = {(n or "").strip().lower() for n in speakers}
        for eid, norm, is_self in s.query(Entity.id, Entity.norm_name, Entity.is_self).filter(Entity.user_id == conv):
            if norm and len(norm) >= 3 and not is_self and norm not in skip:
                names[norm].add(eid)
        for eid, alias in s.query(EntityAlias.entity_id, EntityAlias.alias).filter(EntityAlias.user_id == conv):
            if alias and len(alias) >= 3 and alias not in skip:
                names[alias].add(eid)
        if not names:
            return 0
        pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b")
        edges = []
        for tid, content in s.query(Turn.id, Turn.content).filter(Turn.user_id == conv):
            body = (content or "").split(": ", 1)[-1].lower()
            hit = set()
            for m in pat.finditer(body):
                hit |= names[m.group(1)]
            edges += [eng._storage._edge_dict("turn", tid, "mentions", "entity", e, user_id=conv) for e in sorted(hit)]
    finally:
        s.close()
    eng._storage.add_edges(edges)
    return len(edges)


async def run_conv(conv: str, sample, ks, slots, fuse_w, turn_seed_n, turn_seed_w, out_rows):
    from benchmarks.locomo.loader import iter_qa, iter_sessions
    from benchmarks.qa.e2e_qa import ETTIN17, _engine
    from memorable_ai.core.models import Entity, Turn
    from memorable_ai.core.rerank import load_reranker
    from memorable_ai.core.scope import Scope
    from memorable_ai.core.vector_index import tokenize

    tmp = Path(tempfile.mkdtemp(prefix="g2_"))
    try:
        db = tmp / f"{conv}.db"
        shutil.copy(CACHE / f"{conv}.llm.db", db)
        eng = _engine(db)
        r = eng._retrieval
        r.fusion, r.rerank, r.rerank_model, r.rerank_depth, r.rerank_blend = "score", True, ETTIN17, 20, 0.0
        r._reranker = load_reranker(ETTIN17)
        kg = eng._knowledge_graph
        conv_data = sample["conversation"]
        n_mentions = add_turn_mentions(eng, conv, [conv_data.get("speaker_a"), conv_data.get("speaker_b")])
        kg.engine.invalidate()
        s = eng._storage.get_session()
        try:
            dia = {}
            for key, _, turns in iter_sessions(conv_data):
                rows = s.query(Turn.id).filter(Turn.user_id == conv, Turn.session_id == key).order_by(Turn.seq).all()
                for (tid,), t in zip(rows, turns):
                    dia[tid] = t.get("dia_id")
            self_ids = [i for (i,) in s.query(Entity.id).filter(Entity.user_id == conv, Entity.is_self == 1)]
        finally:
            s.close()
        scope = Scope(user_id=conv)
        kmax = max(ks)
        for qa in iter_qa(sample, categories=set(CATS)):
            if not qa.evidence:
                continue
            gold = set(qa.evidence)
            q = qa.question
            t0 = time.perf_counter()
            base = (await r.recall(q, scope, k_facts=0, k_episodes=0, k_excerpts=kmax))["excerpts"]
            t_base = time.perf_counter() - t0
            tvec = r._encode(q, kind="turn")
            pool_n = max(kmax * 12, r.rerank_depth * 2)
            first = dict(r._rank_layer_scored("turn", q, tvec, scope, kmax * 3, pool=pool_n))
            ents = kg.match_query_entities(q, scope, False)
            top_turns = sorted(first, key=first.get, reverse=True)[:turn_seed_n]
            fz = _z({t: first[t] for t in top_turns})
            seeds = {("entity", e): w for e, w in ents.items() if e not in self_ids}
            for t in top_turns:  # HippoRAG 2: passage seeds at a small weight next to entity seeds
                seeds[("turn", t)] = turn_seed_w * float(np.exp(fz.get(t, 0.0)))
            rec = {"conv": conv, "qid": qa.question, "cat": CATS[qa.category], "n_gold": len(gold),
                   "n_query_entities": len(ents), "variants": {}}
            names = {"A3": [t["id"] for t in base]}
            slot_lists: dict = {}
            t1 = time.perf_counter()
            for prof, w in PROFILES.items():
                res = kg.engine.ppr(eng._storage.namespace, conv, seeds, exclude=[("entity", i) for i in self_ids],
                                    weights=w)
                graph = dict(res.top("turn", pool_n))
                for m in slots:  # per k: A3's first k - m turns, then the best m graph turns A3 left out
                    by_graph = sorted(graph, key=graph.get, reverse=True)
                    for k in ks:
                        keep = names["A3"][: k - m]
                        extra = [t for t in by_graph if t not in names["A3"][:k]][:m]
                        slot_lists.setdefault(f"slots{m}/{prof}", {})[k] = keep + extra
                for wg in fuse_w:
                    zf, zg = _z(first), _z(graph)
                    lo_f, lo_g = min(zf.values(), default=0.0), min(zg.values(), default=0.0)
                    cand = set(first) | set(graph)
                    score = {t: zf.get(t, lo_f) + wg * zg.get(t, lo_g) for t in cand}
                    order = sorted(cand, key=score.get, reverse=True)
                    items = [t for t in eng._storage.hydrate_turns(order, user_id=conv)
                             if len(tokenize(t.get("content", ""))) >= 4]
                    items = r._rerank(q, items, [score[t["id"]] for t in items], depth=r.rerank_depth)
                    names[f"fuse{wg}/{prof}"] = [t["id"] for t in items]
            t_g2 = (time.perf_counter() - t1) / max(1, len(PROFILES))
            per_k = {name: {k: ids[:k] for k in ks} for name, ids in names.items()}
            per_k.update(slot_lists)
            for name, lists in per_k.items():
                rec["variants"][name] = {f"all@{k}": int(gold <= {dia.get(i) for i in lists[k]}) for k in ks}
                rec["variants"][name].update({f"any@{k}": int(bool(gold & {dia.get(i) for i in lists[k]})) for k in ks})
            rec["ms"] = {"A3": round(1000 * t_base, 1), "g2_per_profile": round(1000 * t_g2, 1)}
            out_rows.append(rec)
        eng.close()
        return n_mentions
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def run_conv_product(conv: str, sample, ks, out_rows, plain: bool = False):
    """The shipped path: KnowledgeGraph.link_turn_mentions per episode, then recall() at weight 0 vs 1.
    ``plain``: a fresh LLM-free ingest (the sparse graph every user without an LLM gets) instead of
    the cached LLM-built graph."""
    from benchmarks.locomo.loader import iter_qa, iter_sessions
    from benchmarks.qa.e2e_qa import ETTIN17, _engine, _ingest, locomo_items
    from memorable_ai.core.models import Episode, Turn
    from memorable_ai.core.rerank import load_reranker
    from memorable_ai.core.scope import Scope

    tmp = Path(tempfile.mkdtemp(prefix="g2p_"))
    try:
        db = tmp / f"{conv}.db"
        if plain:
            item = next(it for it in locomo_items() if it["group"] == conv)
            eng = await _ingest(item, db)
        else:
            shutil.copy(CACHE / f"{conv}.llm.db", db)
            eng = _engine(db)
        r = eng._retrieval
        r.fusion, r.rerank, r.rerank_model, r.rerank_depth, r.rerank_blend = "score", True, ETTIN17, 20, 0.0
        r._reranker = load_reranker(ETTIN17)
        kg = eng._knowledge_graph
        scope = Scope(user_id=conv)
        s = eng._storage.get_session()
        try:
            episodes = [i for (i,) in s.query(Episode.id).filter(Episode.user_id == conv).order_by(Episode.id)]
            dia = {}
            for key, _, turns in iter_sessions(sample["conversation"]):
                rows = s.query(Turn.id).filter(Turn.user_id == conv, Turn.session_id == key).order_by(Turn.seq).all()
                for (tid,), t in zip(rows, turns):
                    dia[tid] = t.get("dia_id")
        finally:
            s.close()
        t0 = time.perf_counter()
        n_edges = sum(kg.link_turn_mentions(scope, episode_id=e) for e in episodes)
        link_s = time.perf_counter() - t0
        kg.engine.invalidate()
        for qa in iter_qa(sample, categories=set(CATS)):
            if not qa.evidence:
                continue
            gold = set(qa.evidence)
            rec = {"conv": conv, "qid": qa.question, "cat": CATS[qa.category], "variants": {}, "ms": {}}
            for name, w in (("A3", 0.0), ("G2", 1.0)):
                r.graph_turn_weight = w
                res, ms = {}, []
                for k in ks:  # the pool depends on k, as in the QA harness
                    t = time.perf_counter()
                    res[k] = [x["id"] for x in (await r.recall(qa.question, scope, k_facts=0, k_episodes=0,
                                                               k_excerpts=k))["excerpts"]]
                    ms.append(1000 * (time.perf_counter() - t))
                rec["variants"][name] = {**{f"all@{k}": int(gold <= {dia.get(i) for i in res[k]}) for k in ks},
                                         **{f"any@{k}": int(bool(gold & {dia.get(i) for i in res[k]})) for k in ks}}
                rec["ms"][name] = round(ms[0], 1)
            out_rows.append(rec)
        eng.close()
        return {"edges": n_edges, "link_s": round(link_s, 2), "episodes": len(episodes)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def boot(a: np.ndarray, b: np.ndarray, n: int = 4000) -> tuple:
    d = b.astype(float) - a.astype(float)
    idx = np.random.default_rng(0).integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(rows, ks) -> dict:
    names = list(rows[0]["variants"])
    out = {}
    for metric in [f"{w}@{k}" for k in ks for w in ("all", "any")]:
        base = np.array([r["variants"]["A3"][metric] for r in rows])
        out[metric] = {}
        for nm in names:
            v = np.array([r["variants"][nm][metric] for r in rows])
            d = boot(base, v) if nm != "A3" else (0.0, 0.0, 0.0)
            mh = [r["variants"][nm][metric] for r in rows if r["cat"] == "multi-hop"]
            out[metric][nm] = {"rate": round(float(v.mean()), 3), "diff": round(d[0], 3), "lo": round(d[1], 3),
                               "hi": round(d[2], 3), "multi_hop": round(float(np.mean(mh)), 3) if mh else None}
    return out


def main() -> None:
    from benchmarks.locomo.loader import load_locomo, sample_id
    from benchmarks.qa.e2e_qa import LOCOMO_DEV

    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default=",".join(LOCOMO_DEV))
    ap.add_argument("--k", default="10,20")
    ap.add_argument("--slots", default="2,3")
    ap.add_argument("--fuse", default="0.5,1.0")
    ap.add_argument("--turn-seeds", type=int, default=10)
    ap.add_argument("--turn-seed-weight", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--product", action="store_true", help="measure the shipped code path (recall at weight 0 vs 1)")
    ap.add_argument("--plain", action="store_true", help="with --product: fresh LLM-free ingest, not the LLM graph")
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",")]
    wanted = args.convs.split(",")
    rows, mentions = [], {}
    for ci, sample in enumerate(load_locomo()):
        conv = sample_id(sample, ci)
        if conv in wanted and args.product:
            mentions[conv] = asyncio.run(run_conv_product(conv, sample, ks, rows, plain=args.plain))
        elif conv in wanted:
            mentions[conv] = asyncio.run(run_conv(conv, sample, ks, [int(x) for x in args.slots.split(",")],
                                                  [float(x) for x in args.fuse.split(",")], args.turn_seeds,
                                                  args.turn_seed_weight, rows))
    summary = {"convs": wanted, "questions": len(rows), "turn_mentions_added": mentions,
               "settings": {"mode": ("product-plain" if args.plain else "product") if args.product else "experiment", "turn_seeds": args.turn_seeds,
                            "turn_seed_weight": args.turn_seed_weight, "profiles": PROFILES},
               "ms_p50": {k: round(float(np.median([r["ms"][k] for r in rows])), 1) for k in rows[0]["ms"]},
               "results": summarise(rows, ks)}
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps({**summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
