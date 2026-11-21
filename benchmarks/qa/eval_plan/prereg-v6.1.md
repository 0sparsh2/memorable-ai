# Preregistration v6.1 (DRAFT): the batched observer

**Status:** not run (2026-11-19). Its own rule for the batch size picked none: on conv-48, without answers, every batched log fell far short of the per-episode log (below). The rule was fixed in this draft before the measurement ran (commit 232557f).

| conv-48 log | Calls | Observations | Log tokens | Evidence cited (all) | Answer contained (111) | Rule |
|---|---|---|---|---|---|---|
| Per episode (v6's A7) | 46 | 393 | 10.8k | 94.8% | 65.8% | baseline |
| T = 8,000 | 3 | 94 | 3.6k | 57.6% | 36.0% | fails |
| T = 12,000 | 2 | 111 | 4.2k | 74.9% | 48.6% | fails |
| T = 20,000 | 2 | 129 | 4.1k | 74.9% | 45.9% | fails |

No reply reached its output budget (7.8–10.1k output tokens against 21–31k allowed): given a large batch, the observer summarises instead of recording.

**Why.** NIM's free tier limits requests per model (about 4–6 a minute sustained), not tokens. v6's observer makes one call per closed episode, about 55 calls per LongMemEval haystack. Observing the 350 held-out haystacks for a final claim would take about 19k calls, or 53 hours on Ultra. Mastra's observer makes one call per ~30k tokens of unobserved messages; at that size the same work is about 1.4k calls, or 4 hours. The batched observer is built (`extraction.observe_batch_tokens`, commit 88dfa98). v6.1 asks whether it keeps the answers.

## Bars

**lesson:** We learn whether one observer call per batch of messages gives answers as good as one call per episode. You should use the batched observer for the final held-out runs, and as the default with an LLM, only if it clears B19 and B20.

Primary metric and CIs as in v3 and v6: LLM-judge accuracy; 95% paired bootstrap over questions. The answer model is v6's `<READER>` (v5's B11 choice).

| # | Metric | Pass bar | Direction | A miss means |
|---|---|---|---|---|
| B19 | A7b − A7, LoCoMo-dev J-strict and LME-dev J, paired | the CI's lower bound ≥ −3.0 pts on both, and the point estimate ≥ −1.0 on both (non-inferiority) | higher | Batching loses detail the answers need. Final runs observe per episode (and take days), or the batch size is revisited in a new prereg |
| B20 | Observer calls | ≤ 0.5 per session on LME-dev (at least 2× fewer than v6's per-episode observer, ~1.1), ≤ 2% replies unparseable after salvage, no thinking text, ≤ 2% replies cut off at their output budget | lower | Not worth its loss of provenance granularity, or the batch is too big for one reply |

**Reported, no bar:** A7b against A3g; log tokens per question and budget cuts; observer output tokens and wall time; the conv-48 proxies below.

## Arms

| Arm | Identity | Settings |
|---|---|---|
| A7 | v6's arm, reused | per-episode observer log + A3g |
| A7b | `memorable#observation-log-batched` | A3g's turns and episodes, byte for byte, plus the log of a separate observer-only ingest with `observe_batch_tokens = <T>` and a final `flush_observations()`, within 30k tokens, before the excerpts. Everything else as A7: Ultra, temperature 0, thinking off, `extraction.facts=false`, no LLM entity adjudication |

**Output budget.** v6's pilot measured ~1.8 output tokens of JSON per message token (conv-48: 28.5k output for ~16k tokens of conversation), so batching cuts requests, not output. A batch's reply budget is 1,000 + 2.5 × its message tokens, at most 32k, with a timeout that grows with it.

**Batch size `<T>`, chosen before this freeze by a fixed rule on conv-48 only, without answers** (`benchmarks/qa/observer_batch.py`): among T ∈ {8,000, 12,000, 20,000}, the largest whose log, against v6's per-episode log of conv-48, (a) cites all gold evidence for no fewer than 3 points fewer questions, and (b) holds the short answer in one observation for no fewer than 5 points fewer questions. If none qualifies, v6.1 is not run.

**Questions:** LoCoMo-dev (conv-48, conv-49: 347) and LME-dev (150).

## Baseline arm

A7 (v6), the per-episode observer, with the same reader, turns and log budget. A3g is reported beside it.

## Decoys

v6's D7 covers the log format and token count; v6.1 adds none. The scorer and judge are v6's.

## Stop rules

- **Dry run (zero cost, stubbed observer):** A7b without its log equals A3g byte for byte; the batched ingest leaves no episode queued.
- **Pilot:** conv-48 and the 5 LME-dev haystacks of v6's pilot through the batched observer, then 3 answers per cell once `<READER>` is named. Stop if more than 5% of observer calls fail after retries, any reply has thinking text, or a reply is cut off at its 16k-token output budget.
- **Full run:** v6's rules (the first error that is not a 429, a timeout or a dropped connection stops it; more than 2% unparsed judge verdicts in a cell stops it).
- **Caps:** 1 day for the observer ingest, 1 day for answering. $0 (NIM free tier).
- No prompt, budget, model or context change after the pilot starts.

## Held-out exposure

Dev rows only, as v6. The batch size is chosen on conv-48, a dev conversation. Held-out LoCoMo conversations and the LME 350 are untouched.

## Blinding

As v6: neutral headings; arm and reader names never reach the judge.

## Projection (to be confirmed by the pilot)

- **Observer:** LME-dev 150 haystacks × ~6–15 calls (by T) ≈ 900–2,200 calls; LoCoMo-dev ≈ 2–6 calls. Each reply is long (minutes), so the time is set by how many haystacks run in parallel as much as by 6 requests/min: 3–8 h at 4 in parallel (to be measured in the pilot).
- **Answers:** 497, about 1,000 judge calls on Lightning (~4 h), in v6's answer loop.
