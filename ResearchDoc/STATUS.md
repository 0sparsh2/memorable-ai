# ResearchDoc status

| Field | Value |
|-------|--------|
| Last updated | 2026-05-17 |
| LOCOMO subset | 1 conv × 10 Q, NVIDIA `deepseek-v4-flash` |
| Best judge (`dated_context`) | **40%** (hybrid/extracted re-runs: 0% judge — NIM judge timeouts) |
| Oracle ceiling | **60%** |
| Retrieval-only (dated) | **80%** |
| Initial baseline (minimal context) | 10% |
| Mean F1 (`dated_context`) | 30.8% |
| Temporal evidence recall | 83.3% |
| Figures generated | yes — `figures/output/fig1`–`fig7` |
| Experiments | `ResearchDoc/experiments/` + `experiment_log.md` |
| Competitive JSON artifacts | 14 validated (`agent_memory_landscape_2026/results/`) |

## Published competitor benchmarks (May 2026 research JSON)

| System | LoCoMo (LLM-judge) | LongMemEval | BEAM (1M / 10M) | Notes |
|--------|-------------------|-------------|-----------------|-------|
| Mem0 (2026 blog) | **91.6%** | **93.4%** | **64.1% / 48.6%** | Open `memory-benchmarks` harness |
| Mem0 paper (2025) | ~26% rel. vs OpenAI | — | — | Mem0^g +2% over base on LoCoMo |
| Zep / Graphiti | strong temporal (Mem0 paper) | **71.2%** | uncertain | DMR **94.8%** claimed |
| Memorable (`dated_context`, subset) | **40%** | not run | not run | 83% temporal evidence recall |
| Supermemory / Letta | uncertain | uncertain | uncertain | API/agent-model positioning |

## Changelog

- **2026-05-17** — Ablation ladder: `dated_context` 40%, `oracle` 60%, `retrieval_only_dated` 80%; fig7; paper §5.3.
- **2026-05-17** — §8 Competitive Landscape (Detailed) in `research_paper.md`; competitor benchmark table in STATUS.
- **2026-05-17** — Initial ResearchDoc; paper draft; figure generator; LOCOMO measured_subset ingested.
