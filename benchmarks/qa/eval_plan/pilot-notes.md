# Pilot notes

## Attempt 1 (prereg v1.1), 2026-11-15
Model: `nvidia/nemotron-3.5-lightning-30b-a3b` on NIM (free). Inputs: dev rows only (LoCoMo conv-26, 3 LME dev haystacks).

**Decoys (B6).** Each row is 10 decoys of that type.

| | LME judge | LoCoMo J-strict | LoCoMo J-lenient (Mem0) |
|---|---|---|---|
| gold (must be CORRECT) | 1.0 | 1.0 | 1.0 |
| shuffled (must be WRONG) | 1.0 | 1.0 | 0.6 |
| "don't know" (must be WRONG) | 1.0 | 1.0 | 1.0 |
| perturbed (must be WRONG) | 1.0 | **0.9** | 0.3 |

- The one perturbed miss was a faulty decoy, not a judge error. The generator had swapped a non-key word ("Likely no, she does not…" → "Toronto no, she does not…"), so the meaning was unchanged and the answer was correct.
- A second faulty construction replaced one item of a ten-item list.
- Fix: the generator now changes only years, numbers or a non-initial proper noun, and skips list answers. This brings it in line with the preregistered definition.

**Answers, 3 per arm per benchmark (30).**
- **4 empty answers, so the pilot stops under the preregistered rule.** All four stopped at exactly 2,000 output tokens, still thinking. Median answer output was 1,012 tokens.
- Fix (prereg-v2): an 8,192-token answer cap.
- Latency: about 6 s per call at the median, including rate-limiter spacing at 60 requests/min.

## Attempt 2 (prereg v2), 2026-11-15
Fresh run directory, same dev contexts.

**Decoys.** All gold answers were judged CORRECT. Every decoy type scored 1.0 on both strict judges, with one exception: LME shuffled scored 0.9. The miss was the judge accepting another question's rubric-style gold text ("7 days ago. 8 days … is also acceptable") for a "4 days" question; the temporal template tolerates off-by-one, but 7/8 vs 4 is not off-by-one. **B6 as written passes**: LME 29/30 decoys judged WRONG (96.7%), LoCoMo J-strict 30/30, and gold 10/10 on both. The code had applied the bar to each decoy type separately (stricter than written), and was corrected in v3.

**Answers (30).** 1 empty: A0 (full context) on a 108k-token LME history, where thinking looped to the 8,192-token cap. Median output 1,132 tokens. Latency p50: answers 5.2 s, judges 2.4–4.2 s at 120 requests/min. Under the rule the pilot stops again; v3 adds a no-thinking fallback for empty answers in every arm.

## Attempt 3 (prereg v3), 2026-11-15
**Decoys (B6), pass.**

| Judge | Decoys judged WRONG | Gold judged CORRECT |
|---|---|---|
| LME | 29/30 (96.7%) | 100% |
| LoCoMo J-strict | 30/30 | 100% |
| LoCoMo J-lenient (Mem0) | 76.7% | 100% |

J-lenient is reported only; it accepted 7 of 30 wrong answers.

**Answers (30).**
- 0 empty and 0 fallbacks.
- Latency, mean per call: answers 6.0 s (sd 4.0), judges 2.5–4.6 s.
- Throughput: 92 calls/min sustained at concurrency 8 and a 120 requests/min ceiling, with no rate-limit errors.
- Pilot correctness (3 items each, not a result): LoCoMo A0 2/3, A1 2/3, A2 3/3, A3 3/3, A3w 2/3; LME all arms 3/3.

```
PILOT_ARM: answer   unit=wall_clock mean=5.95s sd=4.02s n=30
PILOT_ARM: judge    unit=wall_clock mean=2.48-4.64s n=36
PILOT_PROJECTION: per_call≈0.65s (92 calls/min) total_calls=22678 (+5620 A4) projected=4.1h (+1.0h A4) band95=3.1h..6.3h (120..60 rpm) cap=14h verdict=within
```

- Cost: $0 on NIM's free tier.
- The limiter is NIM's queue and the per-call thinking time, not local CPU.
- The variation across calls comes from thinking length: answer output p50 is about 1.1k tokens, with a tail to the cap.

## Held-out exposure log

- 2026-11-18: a code search (`grep` over `benchmarks/`) matched `runs/main/contexts_lme.jsonl` and printed part of one held-out LongMemEval-S row (`0862e8bf_abs`, an abstention question) to the agent's console. Nothing was changed in response: no prompt, threshold, arm or setting of this eval was touched afterwards, and the row is in the answer queue as before. Searches now exclude `benchmarks/qa/runs/`.

## A4 (amendment A3), 2026-11-18
- **Contexts.** LoCoMo built locally, LongMemEval-S on Kaggle (A4 mode; a first launch missed the mounted model and rebuilt v3's arms instead, which led to the job now refusing to run when an input is mounted but not found). In both, A3 rebuilt in the same process matches the stored A3 contexts byte for byte: LoCoMo 1,540/1,540, LongMemEval 500/500. So A4 differs from A3 by the fine-tuned model alone. A4 contexts: none empty; 1,540/1,540 (LoCoMo) and 441/500 (LME) differ from A3.
- **Pilot** (3 items per benchmark, separate run directories): LoCoMo 3/3 answered, no fallback, about 920–1,080 input tokens; LME 3 dev items answered, no fallback, about 3,900–4,500 input tokens. Correctness on 3 items is not a result.
- **Published** to the answer loop as `contexts_locomo_a4.jsonl` (1,540 rows) and `contexts_lme_a4.jsonl` (500 rows).
