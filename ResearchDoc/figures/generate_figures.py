#!/usr/bin/env python3
"""
Generate publication-style figures for ResearchDoc/research_paper.md.

Style aligned with Mem0 (arXiv:2504.19413), LoCoMo (ACL), and Zep temporal-KG papers:
  - grouped bar charts for per-category benchmark accuracy
  - retrieval vs generation decomposition
  - gap/capability heatmaps
  - pipeline schematic

Run from repo root or this directory:
  python ResearchDoc/figures/generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Paper-style defaults (serif, high DPI, colorblind-friendly)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)
COLORS = {
    "memorable": "#2563eb",
    "mem0": "#64748b",
    "recall": "#059669",
    "judge": "#dc2626",
}


def load_json(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def fig1_locomo_judge_by_category() -> Path:
    """Grouped bar: LLM-judge accuracy by LoCoMo category (Memorable vs Mem0 baseline)."""
    measured = load_json("measured_subset.json")
    baselines = load_json("mem0_baselines.json")
    mem = measured["results"]["by_category"]
    base = baselines["category_baselines_llm_judge"]

    categories = ["multi_hop", "temporal", "open_domain"]
    labels = ["Multi-hop", "Temporal", "Open-domain"]
    memorable_vals = [
        mem.get(c, {}).get("judge_accuracy") or measured["comparison"]["memorable"].get(c, 0) or 0
        for c in categories
    ]
    mem0_vals = [base[c] for c in categories]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(x - w / 2, [v * 100 for v in memorable_vals], w, label="Memorable (measured)", color=COLORS["memorable"])
    ax.bar(x + w / 2, [v * 100 for v in mem0_vals], w, label="Mem0 (published target)", color=COLORS["mem0"])
    ax.set_ylabel("LLM-judge accuracy (%)")
    ax.set_xlabel("LoCoMo category")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("Figure 1: LoCoMo subset — judge accuracy by category")
    ax.grid(axis="y", alpha=0.3)
    path = OUT / "fig1_locomo_judge_by_category.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig2_retrieval_vs_judge_gap() -> Path:
    """Dual metric: evidence recall vs judge accuracy (shows retrieval/generation bottleneck)."""
    measured = load_json("measured_subset.json")
    by = measured["results"]["by_category"]
    cats = ["multi_hop", "temporal", "open_domain"]
    labels = ["Multi-hop", "Temporal", "Open-domain"]
    recall = [by[c].get("evidence_recall_mean", 0) * 100 for c in cats]
    judge = [by[c].get("judge_accuracy", 0) * 100 for c in cats]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(x - w / 2, recall, w, label="Evidence recall (retrieval)", color=COLORS["recall"])
    ax.bar(x + w / 2, judge, w, label="LLM-judge (end-to-end)", color=COLORS["judge"])
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("LoCoMo category")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("Figure 2: Retrieval vs answer+judge gap (Memorable subset)")
    ax.grid(axis="y", alpha=0.3)
    path = OUT / "fig2_retrieval_vs_judge.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig3_overall_comparison() -> Path:
    """Overall bar: Memorable measured vs Mem0 baseline."""
    measured = load_json("measured_subset.json")
    baselines = load_json("mem0_baselines.json")
    memorable = measured["results"]["overall_judge_accuracy"] * 100
    mem0 = baselines["category_baselines_llm_judge"]["overall"] * 100

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["Memorable\n(subset n=10)", "Mem0\n(published)"], [memorable, mem0], color=[COLORS["memorable"], COLORS["mem0"]])
    ax.set_ylabel("Overall LLM-judge accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Figure 3: Overall LoCoMo-style accuracy")
    for i, v in enumerate([memorable, mem0]):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    path = OUT / "fig3_overall_accuracy.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig4_f1_by_category() -> Path:
    """Token F1 (Snap-style metric) per category."""
    measured = load_json("measured_subset.json")
    by = measured["results"]["by_category"]
    cats = ["multi_hop", "temporal", "open_domain"]
    labels = ["Multi-hop", "Temporal", "Open-domain"]
    f1 = [by[c].get("f1_mean", 0) * 100 for c in cats]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(labels, f1, color=COLORS["memorable"])
    ax.set_ylabel("Mean token F1 (%)")
    ax.set_title("Figure 4: Answer F1 by category (Memorable subset)")
    ax.set_ylim(0, max(f1 + [20]))
    ax.grid(axis="y", alpha=0.3)
    path = OUT / "fig4_f1_by_category.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig5_capability_heatmap() -> Path:
    """Gap matrix heatmap (competitive landscape)."""
    systems = ["Memorable", "Mem0", "Zep", "Memori", "Supermemory"]
    capabilities = [
        "Interceptor",
        "SQL storage",
        "LLM extract+update",
        "Multi-signal retrieval",
        "Temporal validity",
        "Graph (prod)",
        "Published LoCoMo",
    ]
    # 1 = yes/partial-strong, 0.5 = partial, 0 = no
    scores = np.array(
        [
            [1, 1, 1, 1, 1, 0.5, 0.5],  # Memorable (post-implementation)
            [0, 0.5, 1, 1, 0.5, 1, 1],
            [0, 0.5, 1, 1, 1, 1, 0.5],
            [1, 1, 0.5, 0.5, 0, 0, 0],
            [0, 0, 1, 1, 0.5, 1, 0.5],
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(scores, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(capabilities)), capabilities, rotation=35, ha="right")
    ax.set_yticks(range(len(systems)), systems)
    ax.set_title("Figure 5: Capability matrix (qualitative, May 2026)")
    plt.colorbar(im, ax=ax, label="Maturity", fraction=0.046)
    path = OUT / "fig5_capability_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig6_pipeline_schematic() -> Path:
    """Architecture / pipeline diagram (common in Mem0-style papers)."""
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    boxes = [
        (0.2, "App / LLM\n(unchanged code)"),
        (2.0, "Interceptor\ninject + capture"),
        (3.8, "Extract\nhybrid LLM"),
        (5.4, "Update\nADD/UPD/DEL"),
        (7.0, "Store\nSQL + FTS5"),
        (8.6, "Retrieve\nsemantic+BM25+graph"),
    ]
    for i, (x, label) in enumerate(boxes):
        rect = plt.Rectangle((x, 0.5), 1.5, 1.0, fill=True, facecolor="#e0e7ff", edgecolor="#2563eb")
        ax.add_patch(rect)
        ax.text(x + 0.75, 1.0, label, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.55, 1.0), xytext=(x + 1.65, 1.0), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_title("Figure 6: Memorable memory pipeline")
    path = OUT / "fig6_pipeline_schematic.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig7_ablation_ladder() -> Path:
    """Bar chart of judge accuracy across ablation experiments."""
    exp_dir = ROOT.parent / "experiments"
    labels: list[str] = []
    values: list[float] = []
    order = [
        ("baseline (10% prior)", "measured_subset.json"),
        ("retrieval_only_dated", "retrieval_only_dated.json"),
        ("dated_context", "dated_context.json"),
    ]
    for label, fname in order:
        path = exp_dir / fname
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        acc = d.get("results", {}).get("overall_judge_accuracy")
        if d.get("results", {}).get("ingest_mode") == "raw" and fname == "measured_subset.json":
            acc = 0.1
        if acc is not None:
            labels.append(label.replace("_", "\n"))
            values.append(acc * 100)

    for path in sorted(exp_dir.glob("*.json")):
        if path.name in ("measured_subset.json", "retrieval_only_dated.json", "dated_context.json"):
            continue
        d = json.loads(path.read_text())
        acc = d.get("results", {}).get("overall_judge_accuracy")
        eid = d.get("experiment_id", path.stem)
        if acc is not None and eid not in [o[0] for o in order]:
            labels.append(eid.replace("_", "\n"))
            values.append(acc * 100)

    if not labels:
        return OUT / "fig7_ablation_ladder.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [COLORS["mem0"] if "baseline" in lb or "prior" in lb else COLORS["memorable"] for lb in labels]
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Overall LLM-judge accuracy (%)")
    ax.set_title("Figure 7: Ablation ladder (subset n=10)")
    ax.set_xlim(0, 100)
    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
    path = OUT / "fig7_ablation_ladder.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    paths = [
        fig1_locomo_judge_by_category(),
        fig2_retrieval_vs_judge_gap(),
        fig3_overall_comparison(),
        fig4_f1_by_category(),
        fig5_capability_heatmap(),
        fig6_pipeline_schematic(),
        fig7_ablation_ladder(),
    ]
    print("Wrote:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
