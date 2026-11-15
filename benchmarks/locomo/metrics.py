"""
LOCOMO scoring: token F1 (Snap official) and retrieval evidence recall.
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Optional, Set


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1_tokens(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer_f1(
    prediction: str,
    ground_truth: str,
    category: int,
) -> float:
    """
    Category-aware F1 (aligned with snap-research/locomo task_eval).
    """
    if category == 3:  # open-domain: truncate gold at ;
        ground_truth = ground_truth.split(";")[0].strip()

    if category == 1:  # multi-hop: comma-separated sub-answers
        parts = [p.strip() for p in ground_truth.split(",") if p.strip()]
        if len(parts) > 1:
            scores = [_f1_tokens(prediction, p) for p in parts]
            return sum(scores) / len(scores)

    return _f1_tokens(prediction, ground_truth)


def evidence_recall(
    retrieved_dia_ids: Set[str],
    evidence: List[str],
) -> Optional[float]:
    """Fraction of gold evidence dia_ids found in retrieval."""
    if not evidence:
        return None
    hits = sum(1 for e in evidence if e in retrieved_dia_ids)
    return hits / len(evidence)


def aggregate_by_category(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate F1, judge accuracy, and recall by category name."""
    from benchmarks.locomo.loader import CATEGORY_NAMES

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        name = CATEGORY_NAMES.get(row["category"], f"cat_{row['category']}")
        buckets.setdefault(name, []).append(row)

    out: Dict[str, Any] = {}
    for name, items in buckets.items():
        f1s = [r["f1"] for r in items if r.get("f1") is not None]
        judges = [r["judge_correct"] for r in items if r.get("judge_correct") is not None]
        recalls = [r["evidence_recall"] for r in items if r.get("evidence_recall") is not None]
        out[name] = {
            "count": len(items),
            "f1_mean": sum(f1s) / len(f1s) if f1s else None,
            "judge_accuracy": sum(1 for j in judges if j) / len(judges) if judges else None,
            "evidence_recall_mean": sum(recalls) / len(recalls) if recalls else None,
        }

    all_f1 = [r["f1"] for r in rows if r.get("f1") is not None]
    all_j = [r["judge_correct"] for r in rows if r.get("judge_correct") is not None]
    out["overall"] = {
        "count": len(rows),
        "f1_mean": sum(all_f1) / len(all_f1) if all_f1 else None,
        "judge_accuracy": sum(1 for j in all_j if j) / len(all_j) if all_j else None,
    }
    return out
