# Figure specifications (Memorable research paper)

Conventions follow recent agent-memory papers (Mem0 arXiv:2504.19413, LoCoMo ACL 2024, Zep temporal KG arXiv:2501.13956).

## Standard figure types in the literature

| Type | Used by | Our figure |
|------|---------|------------|
| Per-category benchmark bars | Mem0, LoCoMo leaderboard | Fig 1, Fig 4 |
| System vs baseline overall | Mem0 blog, papers | Fig 3 |
| Retrieval ablation / recall | LoCoMo RAG scripts, Mem0 paper | Fig 2 |
| Capability / feature matrix | Survey tables, blog posts | Fig 5 |
| End-to-end pipeline schematic | Mem0, Zep, LangMem | Fig 6 |
| Latency / token cost (log scale) | Mem0 paper | *Planned* Fig 7 |
| Ablation (w/o BM25, w/o graph) | Mem0, RAG papers | *Planned* after ablation runs |

## Generated assets

| File | Description | Data source |
|------|-------------|-------------|
| `output/fig1_locomo_judge_by_category.png` | Grouped bars: judge accuracy | `data/measured_subset.json`, `mem0_baselines.json` |
| `output/fig2_retrieval_vs_judge.png` | Evidence recall vs judge | `measured_subset.json` by_category |
| `output/fig3_overall_accuracy.png` | Overall accuracy | overall_judge_accuracy |
| `output/fig4_f1_by_category.png` | Token F1 | f1_mean per category |
| `output/fig5_capability_heatmap.png` | Qualitative matrix | `agent_memory_landscape_2026/report.md` |
| `output/fig6_pipeline_schematic.png` | Architecture | `memorable_ai/` design |

## Regeneration

```bash
python ResearchDoc/figures/generate_figures.py
```

## Planned (next benchmark runs)

- **Fig 7** — Retrieval latency CDF (Memorable vs full-context baseline)
- **Fig 8** — Token savings vs context window size
- **Fig 9** — Full LoCoMo 10-conversation radar chart (after 1540-Q run)
- **Fig 10** — Ablation stacked bars (pattern-only vs hybrid vs +graph)
