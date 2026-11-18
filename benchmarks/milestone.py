"""
Benchmark comparison at a milestone (owner's rule: "every time we hit a milestone, we compare benchmarks").

    python -m benchmarks.milestone --name "G2 multi-layer retrieval"

1. LoCoMo retrieval eval, no LLM, zero cost (~5-10 min CPU): turn, episode and fact recall@5,
   ingest time, query latency. Paired bootstrap against the previous milestone's per-question hits.
2. End-to-end QA standings (benchmarks/qa/compare.py; interim until the eval completes) beside
   the published systems.
3. Appends a dated entry to docs/benchmark_log.md and keeps the raw run in benchmarks/milestones/.
Interim QA numbers are for reporting only, never for design decisions (prereg held-out rule).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "benchmarks" / "milestones"
LOG = ROOT / "docs" / "benchmark_log.md"
SIGNALS = ("turn/pipeline", "episode/fused", "fact/graph", "fact/pipeline")


def paired(prev: str, cur: str) -> str:
    a = np.array([int(c) for c in prev], dtype=float)
    b = np.array([int(c) for c in cur], dtype=float)
    if len(a) != len(b) or not len(a):
        return "not comparable"
    d = b - a
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    if not d.any():
        return "unchanged"
    return f"{100 * d.mean():+.1f} pts (CI {100 * lo:+.1f}…{100 * hi:+.1f})"


def qa_section() -> list:
    sys.path.insert(0, str(ROOT))
    from benchmarks.qa import compare

    run = ROOT / "benchmarks" / "qa" / "runs" / "main"
    if not (run / "answers_live.jsonl").exists():
        return ["QA: no answer run found."]
    answers, expected = compare.load(run)
    lines = ["| QA arm | LoCoMo J-strict | LoCoMo J-lenient | LongMemEval J | answered / queued |", "|---|---|---|---|---|"]
    for arm in compare.ARM_LABEL:
        loc = list(answers.get(("locomo", arm), {}).values())
        lme = list(answers.get(("lme", arm), {}).values())
        if not loc and not lme:
            continue
        cells = []
        for rows, key in ((loc, "j_strict"), (loc, "j_lenient"), (lme, "j")):
            a = compare.acc(rows, key)
            cells.append(f"{100 * a[0]:.1f} [{100 * a[1]:.0f}–{100 * a[2]:.0f}]" if a else "—")
        n = len(loc) + len(lme)
        q = len(expected.get(("locomo", arm), ())) + len(expected.get(("lme", arm), ()))
        lines.append(f"| {arm} {compare.ARM_LABEL[arm]} | " + " | ".join(cells) + f" | {n} / {q} |")
    pub = json.load(open(ROOT / "benchmarks" / "qa" / "published_results.json"))["entries"]
    best = {b: max((e for e in pub if e["bench"] == b), key=lambda e: e["score"]) for b in ("locomo", "lme")}
    zep = {e["bench"]: e["score"] for e in pub if e["system"].startswith("Zep")}
    lines += ["", f"Published (self-reported, other answer models): best LoCoMo {best['locomo']['score']} "
                  f"({best['locomo']['system']}), best LongMemEval {best['lme']['score']} ({best['lme']['system']}); "
                  f"Zep {zep.get('locomo')} / {zep.get('lme')}. Full table: `python -m benchmarks.qa.compare`."]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="what was reached, e.g. 'G2 multi-layer retrieval'")
    ap.add_argument("--convs", type=int, default=10)
    ap.add_argument("--skip-retrieval", action="store_true")
    args = ap.parse_args()
    STORE.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    slug = re.sub(r"[^a-z0-9]+", "-", args.name.lower()).strip("-")[:40]
    out = STORE / f"{date.today().isoformat()}-{slug}.json"
    lines = ["", f"## {date.today().isoformat()}: {args.name} (commit {commit})", ""]

    if not args.skip_retrieval:
        subprocess.run([sys.executable, "-m", "benchmarks.locomo.retrieval_eval", "--convs", str(args.convs),
                        "--k", "5", "--out", str(out)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
                       env={**__import__("os").environ, "MEMORABLE_LLM__DEFAULT_MODEL": "none"})
        cur = json.load(open(out))
        prev_files = sorted((p for p in STORE.glob("*.json") if p != out), key=lambda p: p.stat().st_mtime)
        prev = json.load(open(prev_files[-1])) if prev_files else None
        lines += [f"Retrieval, LoCoMo ({cur['questions']} questions, {cur['conversations']} conversations, no LLM):", "",
                  "| Signal | recall@5 | vs previous milestone (paired) |", "|---|---|---|"]
        for s in SIGNALS:
            r = cur["recall"].get(s, {}).get("overall")
            d = paired(prev["per_question"][s], cur["per_question"][s]) if prev and s in prev.get("per_question", {}) \
                else "first entry"
            lines.append(f"| {s} | {r} | {d} |")
        lines += ["", f"Ingest {cur['ingest_seconds']} s, query p50 {cur['query_ms_p50']} ms; graph: {cur.get('entities')} "
                      f"entities, {cur.get('relations')} relations"
                  + (f" (previous: {prev['ingest_seconds']} s, {prev['query_ms_p50']} ms)" if prev else "") + ".", ""]
    lines += qa_section()
    if not LOG.exists():
        LOG.write_text("# Benchmark log\n\nOne entry per milestone (`python -m benchmarks.milestone --name ...`). "
                       "Retrieval numbers are LLM-free LoCoMo recall@5 with paired CIs against the previous entry; "
                       "QA numbers come from the end-to-end eval (interim until it completes).\n")
    with open(LOG, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
