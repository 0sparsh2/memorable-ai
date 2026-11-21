"""
Observer granularity, without answers: one call per closed episode (prereg v6's A7 log) against one
call per N tokens of queued messages (Mastra's observer batches ~30k). NIM's free tier limits
requests per model, not tokens, so a batched observer would build the same log in a fraction of
the wall time, if it keeps the detail.

Per setting: observer calls, tokens, wall time, observations, log tokens, and two detail proxies
over LoCoMo questions with gold evidence:
- **cited**: every gold evidence turn is cited by some observation (``source_turn_ids``);
- **contains**: for non-temporal questions with a short answer (1-5 content words), one
  observation holds every content word of the answer.

Dev rows only (prereg v5 split: conv-48, conv-49). The per-episode log is read from the v6 run's
observer cache; batched logs are built fresh on Ultra (thinking off), as the v6 observer is.

    python -m benchmarks.qa.observer_batch --conv conv-48 --tokens 8000,12000,20000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MEMORABLE_LLM__DEFAULT_MODEL", "none")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

V6 = ROOT / "benchmarks" / "qa" / "runs" / "v6"
DEV = {"conv-48", "conv-49"}
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
STOP = set("a an the of in on at to for and or with by from is was were be been my his her their our your "
           "it its this that these those as about into over after before".split())


def words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in STOP}


def score(db: Path, sample: dict, conv: str) -> dict:
    """Detail proxies and log size for the observation log stored in ``db``."""
    from benchmarks.locomo.loader import iter_qa, iter_sessions
    from memorable_ai.core.context import estimate_tokens, render_observation_log

    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    dia = {}
    for key, _, turns in iter_sessions(sample["conversation"]):
        rows = c.execute("select id from turns where user_id=? and session_id=? order by seq", (conv, key)).fetchall()
        for (tid,), t in zip(rows, turns):
            dia[tid] = t.get("dia_id")
    obs = []
    for oid, content, src, meta in c.execute(
            "select id, content, source_turn_ids, extra_metadata from memories "
            "where memory_type='observation' and invalidated_at is null"):
        src = json.loads(src) if isinstance(src, str) else (src or [])
        obs.append({"id": oid, "content": content, "src": src, "metadata": json.loads(meta) if meta else {}})
    c.close()
    cited = {dia.get(t) for o in obs for t in o["src"]} - {None}
    obs_words = [words(o["content"]) for o in obs]
    rows = []
    for qa in iter_qa(sample, categories=set(CATS)):
        if not qa.evidence:
            continue
        gold = set(qa.evidence)
        ans = words(qa.answer)
        contains = None
        if CATS[qa.category] != "temporal" and 1 <= len(ans) <= 5:
            contains = int(any(ans <= w for w in obs_words))
        rows.append({"q": qa.question, "cat": CATS[qa.category], "cited_all": int(gold <= cited),
                     "cited_any": int(bool(gold & cited)), "contains": contains})
    log = render_observation_log(obs)
    n = len(rows)
    cont = [r["contains"] for r in rows if r["contains"] is not None]
    return {"observations": len(obs), "log_tokens": estimate_tokens(log), "turns_cited": len(cited),
            "turns": len(dia), "questions": n, "cited_all": round(100 * sum(r["cited_all"] for r in rows) / n, 1),
            "cited_any": round(100 * sum(r["cited_any"] for r in rows) / n, 1),
            "contains": round(100 * sum(cont) / len(cont), 1), "contains_n": len(cont), "rows": rows}


def score_lme(db: Path, item: dict) -> dict:
    """LongMemEval: is every evidence turn (``has_answer``) cited, and does one observation hold the answer?"""
    from memorable_ai.core.context import estimate_tokens, render_observation_log

    x = item["_x"]
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    gold, uid = set(), item["group"]
    for sid, sess in zip(x["haystack_session_ids"], x["haystack_sessions"]):
        msgs = [t for t in sess if t.get("content")]
        rows = c.execute("select id from turns where user_id=? and session_id=? order by seq", (uid, sid)).fetchall()
        gold |= {tid for (tid,), t in zip(rows, msgs) if t.get("has_answer")}
    obs = []
    for oid, content, src, meta in c.execute(
            "select id, content, source_turn_ids, extra_metadata from memories "
            "where memory_type='observation' and invalidated_at is null"):
        obs.append({"id": oid, "content": content, "src": set(json.loads(src) if isinstance(src, str) else (src or [])),
                    "metadata": json.loads(meta) if meta else {}})
    c.close()
    cited = set().union(*(o["src"] for o in obs)) if obs else set()
    ans = words(item["answer"])
    return {"observations": len(obs), "log_tokens": estimate_tokens(render_observation_log(obs)),
            "evidence_turns": len(gold), "cited_all": int(bool(gold) and gold <= cited),
            "cited_any": int(bool(gold & cited)),
            "contains": int(any(ans <= words(o["content"]) for o in obs)) if 1 <= len(ans) <= 5 else None,
            "category": item["category"]}


async def build(conv: str, tokens: int, rpm: float, out_dir: Path) -> dict:
    """Observer-only ingest with the batched observer at ``tokens``; returns cost figures."""
    from benchmarks.qa import e2e_qa as qa

    item = next(it for it in qa.locomo_items() if it["group"] == conv)
    orig = qa._engine

    def batched(*a, **k):
        eng = orig(*a, **k)
        eng._pipeline.observe_batch_tokens = tokens
        return eng

    qa._engine = batched
    label = f"{conv}@{tokens}"
    token = qa._LLM_GROUP.set(label)
    t0 = time.perf_counter()
    try:
        db = out_dir / f"{conv}.obs{tokens}.db"
        eng = await qa._ingest(item, db, observe=True)
        await eng.flush_observations(user_id=conv)
        eng.close()
    finally:
        qa._engine = orig
        qa._LLM_GROUP.reset(token)
    st = dict(qa._LLM_STATS.get(label, {}))
    return {"db": db, "wall_s": round(time.perf_counter() - t0, 1), "llm": st}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--conv", default="conv-48")
    p.add_argument("--tokens", default="8000,12000,20000", help="batch thresholds to build")
    p.add_argument("--rpm", type=float, default=4.0)
    p.add_argument("--out", default=str(ROOT / "benchmarks" / "qa" / "runs" / "obs_batch"))
    p.add_argument("--score-lme", action="store_true", help="score the finished LME-dev logs in runs/v6 and exit")
    a = p.parse_args()
    if a.score_lme:
        from benchmarks.qa.e2e_qa import _safe, lme_items

        items = {_safe(it["group"]): it for it in lme_items("dev")}  # dev rows only
        rows = []
        for done in sorted((V6 / "ingest_cache" / "lme").glob("*.obs.done")):
            g = done.name[: -len(".obs.done")]
            if g in items:
                llm = json.loads(done.read_text()).get("llm", {})
                sessions = sum(1 for x in items[g]["_x"]["haystack_sessions"] if any(t.get("content") for t in x))
                rows.append({"group": g, **score_lme(done.with_suffix(".db"), items[g]),
                             "calls": llm.get("calls"), "sessions": sessions,
                             "out_tokens": llm.get("completion_tokens"), "json_failed": llm.get("json_failed", 0)})
        for r in rows:
            print(r)
        if rows:
            n = len(rows)
            cont = [r["contains"] for r in rows if r["contains"] is not None]
            print(f"\n{n} logs: cited all {sum(r['cited_all'] for r in rows)}/{n}, any {sum(r['cited_any'] for r in rows)}/{n}, "
                  f"contains {sum(cont)}/{len(cont)}; calls per session "
                  f"{sum(r['calls'] for r in rows) / sum(r['sessions'] for r in rows):.2f}; log tokens median "
                  f"{sorted(r['log_tokens'] for r in rows)[n // 2]:,}")
        return
    if a.conv not in DEV:
        raise SystemExit(f"{a.conv} is held out; this experiment runs on {sorted(DEV)} only")
    from benchmarks.qa import e2e_qa as qa

    sample = next(it for it in qa.locomo_items() if it["group"] == a.conv)["_sample"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    base = V6 / "ingest_cache" / "locomo" / f"{a.conv}.obs.db"
    done = base.with_suffix(".done")
    if done.exists():
        tmp = Path(tempfile.mkdtemp(prefix="obsb_"))
        try:
            for f in base.parent.glob(base.name + "*"):
                shutil.copy(f, tmp / f.name)
            results["episode"] = {**score(tmp / base.name, sample, a.conv), "llm": json.loads(done.read_text())["llm"]}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"no finished per-episode log for {a.conv} yet ({done}); batched settings only")
    qa.install_ingest_llm(a.rpm)
    for n in [int(x) for x in a.tokens.split(",") if x]:
        b = asyncio.run(build(a.conv, n, a.rpm, out))
        results[f"batch{n}"] = {**score(b["db"], sample, a.conv), "llm": b["llm"], "wall_s": b["wall_s"]}
        print(n, {k: v for k, v in results[f"batch{n}"].items() if k != "rows"}, flush=True)
    path = out / f"{a.conv}.json"
    json.dump(results, open(path, "w"), indent=1)
    print(f"\n| setting | calls | out tokens | obs | log tokens | cited all | cited any | contains (n) |")
    print("|---|---|---|---|---|---|---|---|")
    for k, r in results.items():
        llm = r.get("llm", {})
        print(f"| {k} | {llm.get('calls')} | {llm.get('completion_tokens')} | {r['observations']} | {r['log_tokens']} | "
              f"{r['cited_all']} | {r['cited_any']} | {r['contains']} ({r['contains_n']}) |")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
