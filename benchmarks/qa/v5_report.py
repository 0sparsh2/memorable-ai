"""
prereg v5 report: bars B11-B14, per-cell accuracy with CIs, throughput, drift against the v3 run.

    python -m benchmarks.qa.v5_report --run benchmarks/qa/runs/v5 [--final]

--final writes eval_plan/results-v5.md and prints RESULTS: ok when every bar has a verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.qa.e2e_qa import READERS, _boot  # noqa: E402

MAIN = ROOT / "benchmarks" / "qa" / "runs" / "main"
STRONG = ("ultra", "mistral")
METRIC = {"locomo": "j_strict", "lme": "j"}  # B11's metric per benchmark
EXPECTED = {("locomo", "A0"): 347, ("locomo", "A3"): 347, ("locomo", "D6"): 50,
            ("lme", "A0"): 48, ("lme", "A3"): 150, ("lme", "D6"): 48}
GAP_MIN = 10.0  # B13: a gap longer than this between a reader's answers counts as blocked time


def load(run: Path):
    cells = defaultdict(dict)  # (bench, arm, reader) -> qid -> row
    for line in open(run / "answers_live.jsonl"):
        r = json.loads(line)
        arm, _, reader = r["arm"].partition("@")
        cells[(r["bench"], arm, reader)].setdefault(r["qid"], r)
    return cells


def acc(rows, key):
    vals = [r.get(key) for r in rows if key in r]
    if not vals:
        return None
    m, lo, hi = _boot(np.array([bool(v) for v in vals]))
    return {"acc": m, "lo": lo, "hi": hi, "n": len(vals), "unparsed": sum(v is None for v in vals)}


def paired(a: dict, b: dict, key: str):
    """b - a over questions both cells answered."""
    q = sorted(set(a) & set(b))
    if not q:
        return None
    x = np.array([bool(a[i].get(key)) for i in q])
    y = np.array([bool(b[i].get(key)) for i in q])
    m, lo, hi = _boot(x, y)
    return {"diff": m, "lo": lo, "hi": hi, "n": len(q)}


def throughput(run: Path) -> dict:
    """Answered questions per minute of unblocked time, per reader (B13)."""
    ts = defaultdict(list)
    path = run / "ledger_live.jsonl"
    if not path.exists():
        return {}
    for line in open(path):
        r = json.loads(line)
        if r.get("call") == "answer" and r.get("model"):
            ts[r["model"]].append(r["ts"])
    out = {}
    for name, model in READERS.items():
        t = sorted(ts.get(model, ()))
        if len(t) < 2:
            continue
        gaps = np.diff(t) / 60.0
        active = float(gaps[gaps <= GAP_MIN].sum())
        ms = [json.loads(line)["ms"] for line in open(path)
              if '"answer"' in line and json.loads(line).get("model") == model]
        out[name] = {"answers": len(t), "per_min": round((len(t) - 1) / active, 2) if active else None,
                     "p50_s": round(float(np.percentile(ms, 50)) / 1000, 1),
                     "p95_s": round(float(np.percentile(ms, 95)) / 1000, 1)}
    return out


def drift(cells) -> dict:
    """v5 Lightning vs the v3 run's Lightning answers to the same questions (verdict agreement)."""
    main = defaultdict(dict)
    if not (MAIN / "answers_live.jsonl").exists():
        return {}
    for line in open(MAIN / "answers_live.jsonl"):
        r = json.loads(line)
        if r["arm"] in ("A0", "A3"):
            main[(r["bench"], r["arm"])].setdefault(r["qid"], r)
    out = {}
    for (bench, arm, reader), rows in cells.items():
        if reader != "lightning" or (bench, arm) not in main:
            continue
        key = METRIC[bench]
        both = [q for q in rows if q in main[(bench, arm)] and key in rows[q] and key in main[(bench, arm)][q]]
        if both:
            agree = sum(bool(rows[q][key]) == bool(main[(bench, arm)][q][key]) for q in both)
            out[f"{bench}/{arm}"] = {"agree": round(agree / len(both), 3), "n": len(both)}
    return out


def fmt(a) -> str:
    return f"{100 * a['acc']:.1f} [{100 * a['lo']:.0f}–{100 * a['hi']:.0f}] (n={a['n']})" if a else "—"


def fmtd(d) -> str:
    return f"{100 * d['diff']:+.1f} pts (CI {100 * d['lo']:+.1f}…{100 * d['hi']:+.1f}, n={d['n']})" if d else "—"


def report(run: Path) -> tuple:
    cells = load(run)
    complete = all(len(cells.get((b, a, r), {})) >= n for (b, a), n in EXPECTED.items()
                   for r in (STRONG if a == "D6" else READERS))
    lines = ["## Cells", "", "| Bench | Context | Reader | J-strict | J-lenient | J (LME) | unparsed | fallbacks |",
             "|---|---|---|---|---|---|---|---|"]
    stop = []
    for (bench, arm, reader), rows in sorted(cells.items()):
        rs = list(rows.values())
        cs = {k: acc(rs, k) for k in ("j_strict", "j_lenient", "j")}
        unp = sum(c["unparsed"] for c in cs.values() if c)
        nj = sum(c["n"] for c in cs.values() if c)
        if nj and unp / nj > 0.02:
            stop.append(f"{bench}/{arm}@{reader}: {unp}/{nj} unparsed judge verdicts (> 2%)")
        lines.append(f"| {bench} | {arm} | {reader} | {fmt(cs['j_strict'])} | {fmt(cs['j_lenient'])} | {fmt(cs['j'])} "
                     f"| {unp} | {sum(bool(r.get('answer_fallback')) for r in rs)} |")

    # B11: strong reader - Lightning on A3, both benchmarks
    b11, qualifiers = {}, []
    for r in STRONG:
        d = {b: paired(cells.get((b, "A3", "lightning"), {}), cells.get((b, "A3", r), {}), METRIC[b])
             for b in ("locomo", "lme")}
        b11[r] = d
        if all(d.values()):
            up = [b for b in d if d[b]["diff"] >= 0.03 and d[b]["lo"] > 0]
            other_ok = all(d[b]["diff"] >= -0.01 for b in d if b not in up)
            if up and other_ok:
                qualifiers.append(r)
    tp = throughput(run)
    chosen = None
    if len(qualifiers) == 1:
        chosen = qualifiers[0]
    elif len(qualifiers) == 2:
        x, y = qualifiers
        dd = {b: paired(cells[(b, "A3", x)], cells[(b, "A3", y)], METRIC[b]) for b in ("locomo", "lme")}
        tied = all(v["lo"] <= 0 <= v["hi"] for v in dd.values())
        if tied:
            chosen = max(qualifiers, key=lambda r: (tp.get(r, {}).get("per_min") or 0))
        else:
            chosen = max(qualifiers, key=lambda r: np.mean([acc(list(cells[(b, "A3", r)].values()), METRIC[b])["acc"]
                                                            for b in ("locomo", "lme")]))
    lines += ["", "## Bars", "", "| Bar | Result | Verdict |", "|---|---|---|"]
    for r in STRONG:
        lines.append(f"| B11 {r} − lightning, A3 | LoCoMo-dev J-strict {fmtd(b11[r]['locomo'])}; LME-dev J "
                     f"{fmtd(b11[r]['lme'])} | {'qualifies' if r in qualifiers else ('misses' if complete else 'pending')} |")
    final = chosen or ("lightning" if complete else None)
    lines.append(f"| B11 choice | qualifiers: {', '.join(qualifiers) or 'none'} | "
                 f"{('reader = ' + final) if complete and final else 'pending'} |")
    b12 = acc(list(cells.get(("locomo", "A0", final), {}).values()), "j_lenient") if final else None
    lines.append(f"| B12 A0 LoCoMo-dev J-lenient ({final or '?'}) ≥ 94.0 | {fmt(b12)} | "
                 f"{('pass' if b12['acc'] >= 0.94 else 'miss') if complete and b12 else 'pending'} |")
    t = tp.get(final or "", {})
    lines.append(f"| B13 throughput ({final or '?'}) ≥ 2 answers/min | {t.get('per_min')} /min, p50 {t.get('p50_s')} s | "
                 f"{('pass' if (t.get('per_min') or 0) >= 2 else 'miss') if complete and t else 'pending'} |")
    for r in STRONG:
        lo_ = acc(list(cells.get(("locomo", "D6", r), {}).values()), "j_strict")
        lm_ = acc(list(cells.get(("lme", "D6", r), {}).values()), "j")
        ok = lo_ and lm_ and lo_["acc"] <= 0.20 and lm_["acc"] <= 0.20
        lines.append(f"| B14 D6 no memory, {r} ≤ 20% | LoCoMo {fmt(lo_)}; LME {fmt(lm_)} | "
                     f"{('pass' if ok else 'FAIL') if lo_ and lm_ else 'pending'} |")
        if lo_ and lm_ and not ok:
            stop.append(f"B14 failed for {r}")
    lines += ["", "## Throughput", "", "| Reader | answers | per min (unblocked) | p50 s | p95 s |", "|---|---|---|---|---|"]
    lines += [f"| {r} | {v['answers']} | {v['per_min']} | {v['p50_s']} | {v['p95_s']} |" for r, v in tp.items()]
    dr = drift(cells)
    lines += ["", "## Drift against the v3 run (Lightning, same questions)", ""]
    lines += [f"- {k}: {100 * v['agree']:.0f}% verdict agreement (n={v['n']})" for k, v in dr.items()] or ["- no overlap yet"]
    return lines, complete, stop, final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "benchmarks" / "qa" / "runs" / "v5"))
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    lines, complete, stop, final = report(Path(args.run))
    print("\n".join(lines))
    for s in stop:
        print(f"STOP: {s}")
    if args.final:
        if not complete:
            raise SystemExit("RESULTS: incomplete (cells still missing)")
        out = ROOT / "benchmarks" / "qa" / "eval_plan" / "results-v5.md"
        out.write_text("# Results v5: the answer model\n\n_Takeaways: see the top of this file once written._\n\n"
                       + "\n".join(lines) + "\n")
        print(f"wrote {out}\nRESULTS: ok (reader = {final})")


if __name__ == "__main__":
    main()
