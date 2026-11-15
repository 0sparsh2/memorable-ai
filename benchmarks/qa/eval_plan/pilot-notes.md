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

## Attempt 2 (prereg v2)
Results below.
