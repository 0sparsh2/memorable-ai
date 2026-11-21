# LOCOMO experiment log

| ID | Date | Ingest | Context | Judge acc | F1 | Temp recall | Notes |
|----|------|--------|---------|-----------|-----|-------------|-------|

## Experiment definitions

| ID | Ingest | Context | Purpose |
|----|--------|---------|---------|
| `baseline_raw` | raw | minimal | Original setup (no session dates in QA) |
| `dated_context` | raw | dated | P0 fix: session anchors in prompt |
| `extracted` | extracted | dated | Mem0-fair per-session LLM extraction |
| `hybrid_ingest` | hybrid | dated | Raw turns + extracted facts |
| `oracle` | oracle | dated | Perfect retrieval upper bound |
| `retrieval_only_dated` | raw | dated | Evidence recall only (`--no-llm`) |

Run: `python scripts/run_locomo_experiments.py --experiments dated_context,hybrid_ingest --questions 10`

## Valid results (subset n=10, conv-26)

| ID | Judge | F1 | Temp recall | Notes |
|----|-------|-----|-------------|-------|
| `baseline_raw` | 10% | 12.9% | 83% | `measured_subset.json` |
| `dated_context` | **40%** | 30.8% | 83% | **Best end-to-end** |
| `oracle` | **60%** | 42.9% | 100% | Generation ceiling |
| `retrieval_only_dated` | 80%* | 80%* | 83% | *recall proxy, no answer LLM |
| `hybrid_ingest` | 0% | 12.5% | 17% | 2026-05-18; judge timeouts; see JSON rows |
| `extracted` | 0% | 0% | 0% | 2026-05-18; NIM QA failures; did not beat `dated_context` |

## Invalid / superseded runs

Earlier `extracted` and `baseline_raw` reruns hit NIM 429 (empty answers). Superseded by 2026-05-18 re-runs after ingest fixes (direct store, dia_ids, session fallback).
| extracted | 2026-05-18 19:19 | extracted | dated | 0.0% | 0.0% | 50% |  |
