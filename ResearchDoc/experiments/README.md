# LOCOMO ablation experiments

## Quick start

```bash
# List experiments
python scripts/run_locomo_experiments.py --list

# Fast diagnostic (no answer LLM, ~15s)
python scripts/run_locomo_experiments.py --experiments retrieval_only_dated --questions 10

# Full ablation suite (~1–3 hours with NIM)
make locomo-experiments
```

## Interpreting results

| Metric | Meaning |
|--------|---------|
| `evidence_recall_mean` | Gold `dia_id` in top-k retrieved memories |
| `judge_accuracy` | LLM judge CORRECT vs gold (end-to-end) |
| `f1_mean` | Token F1 on short answer (Snap-style) |

**Compare experiments:**

- `retrieval_only_dated` high, `dated_context` low → fix answer/judge path
- `oracle` high, `dated_context` low → fix retrieval ranking
- `extracted` > `dated_context` → semantic ingest wins
- `dated_context` > `baseline_raw` → session dates in prompt matter

## Output files

- `experiment_log.md` — table of all runs
- `{experiment_id}.json` — full results + per-question `rows`
