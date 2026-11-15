#!/usr/bin/env python3
"""Print comparison table from ResearchDoc/experiments/*.json."""

import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "ResearchDoc" / "experiments"


def main() -> None:
    rows = []
    for path in sorted(EXP.glob("*.json")):
        d = json.loads(path.read_text())
        r = d.get("results", {})
        rows.append(
            {
                "id": d.get("experiment_id", path.stem),
                "ingest": r.get("ingest_mode"),
                "context": r.get("context_format"),
                "judge": r.get("overall_judge_accuracy"),
                "f1": r.get("overall_f1"),
                "n": r.get("questions_evaluated"),
            }
        )
    print(f"{'ID':<22} {'ingest':<10} {'ctx':<8} {'judge':>8} {'f1':>8} {'n':>4}")
    print("-" * 62)
    for row in rows:
        j = f"{row['judge']*100:.1f}%" if row["judge"] is not None else "n/a"
        f = f"{row['f1']*100:.1f}%" if row["f1"] is not None else "n/a"
        print(
            f"{row['id']:<22} {str(row['ingest']):<10} {str(row['context']):<8} "
            f"{j:>8} {f:>8} {row['n']:>4}"
        )


if __name__ == "__main__":
    main()
