# ResearchDoc

Living research paper and figure assets for Memorable (updated as benchmarks and competitive research evolve).

## Contents

| Path | Purpose |
|------|---------|
| [research_paper.md](research_paper.md) | Draft paper (abstract → conclusion) |
| [diagnosis_retrieval_generation_gap.md](diagnosis_retrieval_generation_gap.md) | Multi-angle root-cause analysis (LOCOMO 10% vs 83% recall) |
| [figures/specs.md](figures/specs.md) | Figure list and plotting conventions |
| [figures/generate_figures.py](figures/generate_figures.py) | Regenerate all plots from repo data |
| [figures/data/](figures/data/) | Snapshot inputs (benchmark JSON, baselines) |
| [figures/output/](figures/output/) | Generated PNG/PDF (gitignored) |
| [STATUS.md](STATUS.md) | Last-updated log for agents |

## Regenerate figures

```bash
cd ResearchDoc/figures
python generate_figures.py
```

Requires: `matplotlib` (in project requirements or `pip install matplotlib`).

## Data sources

- `figures/data/measured_subset.json` — LOCOMO subset (Memorable measured)
- `figures/data/mem0_baselines.json` — published Mem0 / paper targets for comparison bars
- `agent_memory_landscape_2026/` — competitive research JSON + report

## Agent workflow

When benchmarks or `agent_memory_landscape_2026/` change:

1. Copy latest `benchmarks/locomo/data/measured_subset.json` → `figures/data/`
2. Run `generate_figures.py`
3. Append results to `research_paper.md` §5 Experiments
4. Update `STATUS.md` timestamp and metrics table
