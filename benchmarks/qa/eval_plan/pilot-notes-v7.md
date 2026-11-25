# Pilot notes, prereg v7 (the answer prompt)

## Dry run, 2026-11-24

`DRY_RUN_RESULT: pass` ([dry-run-v7.json](dry-run-v7.json)):
- The queue of 1,429 rows matches the prereg.
- Every A7·P row's context, question, date line and answer equals v6's A7 row.
- P0 renders exactly as `ANSWER_PROMPT` did, on 497 of 497 A7 rows.
- No prompt carries an evaluation word or an unfilled placeholder.
- No D8 row is paired with its own haystack.
- Stub answers ran end to end through `v7_report`.

## Pilot, 2026-11-24

The prereg was frozen at 12:45 EST, when the pilot started. The v3/v4 main loop was paused just before, at 3,932 answers (every line parses), so that it does not share Lightning with v7. It resumes after v7.

`v7_build --pilot` queued 3 rows per cell, 27 in all (`runs/v7ans_pilot`), answered by Ultra and judged by Lightning with v6's settings.
- **Answers:** 27 of 27. None empty, no thinking text, 0 unparsed verdicts.
  - One answer, D8·P0 (the existing prompt), came back empty and used its preregistered retry without thinking.
  - One call hit a read timeout after its retries. The one-shot `answer` command stops on that, so the last row was answered by a second run. The full run uses `answer-loop`, which resumes after such stops. The prereg treats timeouts as non-fatal.
- **Drift:** A7r agrees with v6's A7 verdicts on 3 of 3 LoCoMo and 3 of 3 LME rows.
- **Cells at n=3 (not for decisions):**
  - A7·P1: LME 3/3, LoCoMo J-strict 2/3.
  - A7·P2: LME 2/3, LoCoMo J-strict 3/3.
  - D8, declined: P0 2/3, P1 3/3, P2 3/3.
  - Median response length: P1 283 and 531 characters, P2 627 and 193 (LME and LoCoMo).
- **Timing:** Ultra took 122 s per answer at p50 (166 s at p95), for a median of 25.3k input and 429 output tokens. That is slower than v5's 31 s, partly because A7's prompts are 10–30 times longer. 27 answers and 36 judge calls took 11 min.

`PILOT_PROJECTION: per_item=1 answer + 1.5 judge calls total_items=1,429 projected=5-10 h band95=4-12 h cap=2 days verdict=within`

The projection's low end is v6's stage 2 rate (4.4 answers/min, with Ultra and Lightning unshared). The high end is the pilot's rate (2.4/min), which is set by latency because only 27 rows were in flight. The limits are Ultra's latency on 25k-token prompts and Lightning's rate limit (about 2,160 judge calls).
