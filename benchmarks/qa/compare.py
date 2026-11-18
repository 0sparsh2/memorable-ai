"""
One comparison across both benchmarks, every arm (prereg v3 and v4.2), the preregistered bars
and published competitor numbers (eval-plan Step 5: results beside the baseline arm).

    python -m benchmarks.qa.compare --run benchmarks/qa/runs/main             # interim, to the run dir
    python -m benchmarks.qa.compare --run benchmarks/qa/runs/main --final     # eval_plan/results.md

Primary judge: J-strict on LoCoMo, the LongMemEval judge (J) on LongMemEval. Bars use paired
bootstrap CIs over the questions both arms answered; a bar with unanswered questions is "pending".
``RESULTS: ok`` is printed only when every bar has a verdict and every baseline row is present.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
SPLIT: Dict[Tuple[str, str], Optional[str]] = {}
PRIMARY = {"locomo": "j_strict", "lme": "j"}
ARM_LABEL = {
    "A0": "full context", "A1": "BM25 only", "A2": "v0.3 (RRF)", "A3": "current (score fusion + Ettin-17M)",
    "A3w": "current, 40 turns", "A4": "current + fine-tuned MiniLM", "A5": "current + 20 LLM facts",
    "A6": "current, 80 turns", "D5": "decoy: another conversation's facts", "A3r": "A3 answered again (drift check)",
}


def boot(d: np.ndarray, n: int = 4000) -> Tuple[float, float, float]:
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load(run: Path):
    answers: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for line in open(run / "answers_live.jsonl"):
        r = json.loads(line)
        answers[(r["bench"], r["arm"])][r["qid"]] = r
    expected: Dict[Tuple[str, str], set] = defaultdict(set)
    for f in sorted(run.glob("contexts_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            expected[(r["bench"], r["arm"])].add(r["qid"])
            SPLIT[(r["bench"], r["qid"])] = r.get("split")
    return answers, expected


def acc(rows: List[Dict[str, Any]], key: str) -> Optional[Tuple[float, float, float, int]]:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    m, lo, hi = boot(np.array([bool(v) for v in vals], dtype=float))
    return m, lo, hi, len(vals)


def paired(answers, expected, bench: str, a: str, b: str, split: Optional[str] = None,
           qids: Optional[set] = None) -> Dict[str, Any]:
    """b − a on the primary judge, over questions both answered (optionally one split / question set)."""
    key = PRIMARY[bench]
    A, B = answers.get((bench, a), {}), answers.get((bench, b), {})
    want = expected.get((bench, a), set()) & expected.get((bench, b), set())
    if qids is not None:
        want &= qids
    if split:
        want = {q for q in want if SPLIT.get((bench, q)) == split}
    both = [q for q in sorted(want) if q in A and q in B and A[q].get(key) is not None and B[q].get(key) is not None]
    out: Dict[str, Any] = {"n": len(both), "expected": len(want), "complete": len(both) == len(want) and len(want) > 0}
    if both:
        d = np.array([float(bool(B[q][key])) - float(bool(A[q][key])) for q in both])
        out["delta"], out["lo"], out["hi"] = (100 * x for x in boot(d))
        out["agree"] = float(np.mean([bool(B[q][key]) == bool(A[q][key]) for q in both]))
    return out


def fmt_d(p: Dict[str, Any]) -> str:
    if not p.get("n"):
        return "no paired answers yet"
    return f"{p['delta']:+.1f} pts (CI {p['lo']:+.1f}…{p['hi']:+.1f}), n={p['n']}/{p['expected']}"


def tokens(answers, bench: str, arm: str) -> Optional[float]:
    rows = list(answers.get((bench, arm), {}).values())
    return float(np.mean([r["in_tokens"] for r in rows])) if rows else None


def bars(answers, expected, run: Path) -> List[Dict[str, Any]]:
    out = []

    def add(bar, metric, p, verdict_fn, note=""):
        status = verdict_fn(p) if p.get("complete") else f"pending ({p.get('n', 0)}/{p.get('expected', 0)})"
        out.append({"bar": bar, "metric": metric, "result": fmt_d(p), "verdict": status, "note": note})

    ok = lambda thr: (lambda p: "pass" if p["delta"] >= thr and p["lo"] > 0 else "fail")  # noqa: E731
    add("B1", "LoCoMo J-strict: A3 − A2 (≥ +3.0, CI > 0)", paired(answers, expected, "locomo", "A2", "A3"), ok(3.0))
    add("B2", "LME held-out J: A3 − A1 (≥ +3.0, CI > 0)", paired(answers, expected, "lme", "A1", "A3", "heldout"), ok(3.0))
    p = paired(answers, expected, "locomo", "A0", "A3")
    t0, t3 = tokens(answers, "locomo", "A0"), tokens(answers, "locomo", "A3")
    share = (t3 / t0) if t0 and t3 else None
    add("B3", "LoCoMo J-strict: A3 ≥ A0 − 5 and A3 tokens ≤ 15% of A0", p,
        lambda p: "pass" if p["delta"] >= -5 and share is not None and share <= 0.15 else "fail",
        f"A3 uses {share:.1%} of A0's input tokens" if share else "")
    p4l = paired(answers, expected, "locomo", "A3", "A4")
    p4m = paired(answers, expected, "lme", "A3", "A4", "heldout")

    def b4(_):
        best = max(p4l["delta"], p4m["delta"])
        worst = min(p4l["delta"], p4m["delta"])
        return "pass" if best >= 1.5 and worst >= -1.0 else "fail"
    both4 = {"complete": p4l.get("complete") and p4m.get("complete"), "n": p4l.get("n", 0) + p4m.get("n", 0),
             "expected": p4l.get("expected", 0) + p4m.get("expected", 0)}
    out.append({"bar": "B4", "metric": "A4 − A3 (≥ +1.5 on one benchmark, no drop > 1.0 on the other)",
                "result": f"LoCoMo {fmt_d(p4l)}; LME held-out {fmt_d(p4m)}",
                "verdict": b4(None) if both4["complete"] else f"pending ({both4['n']}/{both4['expected']})", "note": ""})
    p = paired(answers, expected, "locomo", "A3", "A3w")
    add("B5", "LoCoMo J-strict: A3w − A3 (report; ≥ +2 = context budget matters)", p,
        lambda p: "context budget matters" if p["delta"] >= 2 else "no clear budget effect")
    # prereg v4.2
    p7l = paired(answers, expected, "locomo", "A3", "A5")
    p7m = paired(answers, expected, "lme", "A3", "A5", "heldout")

    def b7(_):
        good = [p for p in (p7l, p7m) if p["delta"] >= 2.0 and p["lo"] > 0]
        return "pass" if good and min(p7l["delta"], p7m["delta"]) >= -1.0 else "fail"
    c7 = p7l.get("complete") and p7m.get("complete")
    out.append({"bar": "B7", "metric": "A5 (LLM facts) − A3 (≥ +2.0 with CI > 0 on one, no drop > 1.0 on the other)",
                "result": f"LoCoMo {fmt_d(p7l)}; LME held-out {fmt_d(p7m)}",
                "verdict": b7(None) if c7 else f"pending ({p7l.get('n', 0) + p7m.get('n', 0)}/"
                                                f"{p7l.get('expected', 0) + p7m.get('expected', 0)})", "note": ""})
    add("B8", "LoCoMo J-strict: A6 (80 turns) − A3w (≥ +1.5 raise default; ≤ 0 knee at 40)",
        paired(answers, expected, "locomo", "A3w", "A6"),
        lambda p: "raise the default window" if p["delta"] >= 1.5 else ("40 is the knee" if p["delta"] <= 0 else "inconclusive"))
    import sqlite3

    calls, sessions = 0, 0
    for done in (run.parent / "v4" / "ingest_cache").glob("*/*.done"):
        calls += json.loads(done.read_text()).get("llm", {}).get("calls", 0)
        con = sqlite3.connect(done.with_suffix(".db"))
        try:
            sessions += con.execute("select count(distinct session_id) from turns").fetchone()[0]
        finally:
            con.close()
    out.append({"bar": "B9", "metric": "A5 extraction ≤ 2.5 LLM calls per session",
                "result": f"{calls} calls" + (f", {calls / sessions:.2f} per session" if sessions else ""),
                "verdict": "see ingest_cache" if not sessions else ("pass" if calls / sessions <= 2.5 else "fail"),
                "note": "37 calls per 19-session conversation in the pilot (1.95)"})
    add("B10", "LoCoMo J-strict: D5 (decoy) − A3 (≤ +1.0)", paired(answers, expected, "locomo", "A3", "D5"),
        lambda p: "pass" if p["delta"] <= 1.0 else "fail")
    for bench in ("locomo", "lme"):
        p = paired(answers, expected, bench, "A3", "A3r")
        out.append({"bar": f"drift ({bench})", "metric": "A3r vs A3 verdict agreement (≥ 90%)",
                    "result": f"{p['agree']:.1%} agree, n={p['n']}/{p['expected']}" if p.get("n") else "no paired answers yet",
                    "verdict": ("pass" if p["agree"] >= 0.9 else "fail: re-answer A3 and A3w")
                    if p.get("complete") else f"pending ({p.get('n', 0)}/{p.get('expected', 0)})", "note": ""})
    return out


def render(answers, expected, bar_rows, published) -> str:
    L = ["# End-to-end QA results: Memorable vs baselines and published systems", ""]
    L += ["_Takeaways are written when every bar has a verdict._", "", "## Arms", ""]
    for bench in ("locomo", "lme"):
        name = "LoCoMo (categories 1–4)" if bench == "locomo" else "LongMemEval-S"
        L += [f"### {name}", "", "| Arm | What | n answered / queued | J-strict | J-lenient | J (LME) | held-out J | mean input tokens |",
              "|---|---|---|---|---|---|---|---|"]
        for arm in ARM_LABEL:
            rows = list(answers.get((bench, arm), {}).values())
            if not rows and not expected.get((bench, arm)):
                continue
            cells = []
            for key, subset in (("j_strict", None), ("j_lenient", None), ("j", None), ("j", "heldout")):
                rs = [r for r in rows if subset is None or r.get("split") == subset]
                a = acc(rs, key)
                cells.append(f"{100 * a[0]:.1f} [{100 * a[1]:.0f}–{100 * a[2]:.0f}]" if a else "—")
            tk = tokens(answers, bench, arm)
            tk_s = f"{tk:,.0f}" if tk else "—"
            L.append(f"| {arm} | {ARM_LABEL[arm]} | {len(rows)} / {len(expected.get((bench, arm), ()))} | "
                     + " | ".join(cells) + f" | {tk_s} |")
        L.append("")
    L += ["## Preregistered bars", "", "| Bar | Metric | Result | Verdict |", "|---|---|---|---|"]
    for b in bar_rows:
        L.append(f"| {b['bar']} | {b['metric']} | {b['result']}{(' — ' + b['note']) if b['note'] else ''} | {b['verdict']} |")
    L += ["", "## Beside published systems (self-reported; protocols differ)", "",
          "| System | LoCoMo | LongMemEval-S | Answer model | Judge | Memory per question |", "|---|---|---|---|---|---|"]
    by_sys: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for e in published["entries"]:
        by_sys[e["system"]][e["bench"]] = e
    def field(d: Dict[str, Any], key: str, default: str) -> str:
        vals = []
        for e in d.values():
            if e.get(key) and e[key] not in vals:
                vals.append(e[key])
        return "; ".join(vals) or default

    for sysname, d in by_sys.items():
        L.append(f"| {sysname} | {d.get('locomo', {}).get('score', '—')} | {d.get('lme', {}).get('score', '—')} | "
                 f"{field(d, 'answer_model', 'not stated')} | {field(d, 'judge', '—')} | "
                 f"{field(d, 'context', 'not stated')} |")
    for arm in ("A0", "A3", "A4", "A5"):
        lo = acc(list(answers.get(("locomo", arm), {}).values()), "j_lenient")
        lm = acc(list(answers.get(("lme", arm), {}).values()), "j")
        if lo or lm:
            tk = tokens(answers, "locomo", arm)
            tk_s = f"~{tk:,.0f} tokens (LoCoMo)" if tk else "—"
            lo_s = f"{100 * lo[0]:.1f}" if lo else "—"
            lm_s = f"{100 * lm[0]:.1f}" if lm else "—"
            L.append(f"| **Memorable {arm}** ({ARM_LABEL[arm]}) | {lo_s} | {lm_s} | Nemotron-3.5-Lightning (3B active) | "
                     f"LoCoMo: Mem0's lenient judge; LME: its own | {tk_s} |")
    L += ["", "Our LoCoMo column uses J-lenient (Mem0's judge) to sit beside published numbers; the bars use J-strict.",
          "LoCoMo results are in-sample for retrieval configuration (prereg v3, held-out exposure)."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(HERE / "runs" / "main"))
    ap.add_argument("--final", action="store_true", help="write eval_plan/results.md and check completeness")
    args = ap.parse_args()
    run = Path(args.run)
    answers, expected = load(run)
    bar_rows = bars(answers, expected, run)
    published = json.load(open(HERE / "published_results.json"))
    text = render(answers, expected, bar_rows, published)
    dest = (HERE / "eval_plan" / "results.md") if args.final else (run / "results_interim.md")
    dest.write_text(text)
    pending = [b["bar"] for b in bar_rows if str(b["verdict"]).startswith(("pending", "see "))]
    print(f"wrote {dest}")
    print("RESULTS: ok" if not pending and args.final else f"RESULTS: incomplete pending={','.join(pending)}")


if __name__ == "__main__":
    main()
