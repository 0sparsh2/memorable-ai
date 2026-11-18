# Preregistration v4 (DRAFT): LLM facts and a wider window, end-to-end QA

**Status:** frozen 2026-11-18 when the pilot started; the owner approved piloting both arms. Before that it was a draft for review. It is a new eval beside [prereg-v3.md](prereg-v3.md): v3's arms, bars and running answers are untouched, and A3 / A3w answers from v3 are the baselines here.

**Why.** Every arm in v3 ingests without an LLM, so contexts hold turns and episode summaries but no extracted facts. Every system above us on the leaderboards (Mem0, Hindsight, Mastra, Agent Zero) answers from LLM-extracted memories. Separately, retrieval coverage keeps rising past A3w's 40 turns (below).

## Bars

**lesson:** We learn whether LLM-extracted facts, and a wider excerpt window, turn into correct answers for the same answer model, and at what extraction cost. You should turn LLM extraction on by default for answering only if it clears B7, and set the default window from B8.

Primary metric and CIs as in v3: LLM-judge accuracy; 95% paired bootstrap over questions; "beats" = the CI of the paired difference excludes 0.

| # | Metric | Pass bar | Direction | A miss means |
|---|---|---|---|---|
| B7 | A5 (facts) − A3, LoCoMo J-strict (subset) or LME held-out J (subset) | ≥ +2.0 pts with CI > 0 on one; no drop > 1.0 on the other | higher | LLM facts do not pay off end to end at this answer model. Keep answering LLM-free; facts stay for the graph and profile |
| B8 | A6 (80 turns) − A3w (40 turns), LoCoMo J-strict (subset) | ≥ +1.5 pts → raise the default window; ≤ 0 → 40 is the knee | higher | More context stops helping this answer model past 40 turns |
| B9 | A5 extraction cost | ≤ 2.5 LLM calls per session; report calls, tokens and ingest time per conversation | lower | Extraction is too expensive for the default path; keep it opt-in |
| B10 | Decoy D5 − A3, LoCoMo J-strict | ≤ +1.0 pt | lower | A5's gain would come from the rendering or the extra tokens, not from the facts |

## Arms

Answer model, judges, prompt and memory format are v3's. Retrieval is A3's unless stated. J-lenient (Mem0's judge) also runs on A5, to place it beside published numbers.

| Arm | Identity | Settings | Per item |
|---|---|---|---|
| A5 | `memorable#llm-facts` | A3 plus the top 20 LLM-extracted facts (k_facts=20, reranked like turns), shown dated in the same oldest-first list, each labelled `[fact]`. Ingest with the LLM on for extraction, the update phase and entity resolution only; episode and session summaries stay extractive as in A3, so A5 − A3 is the facts alone. Extraction model: `nvidia/nemotron-3.5-bolt-e4b` (its own per-model quota; answers directly) | $0; ~2 extraction calls per session |
| A6 | `memorable#window-80` | A3 with 80 excerpts (~3.3k turn tokens, ~4.2k input) | $0 |
| D5 | decoy | A5 with the facts of another LoCoMo conversation (fixed derangement), 100 questions | $0 |

`ARM: A5 unit=nim_calls per_item=1 answer + 2 judges; ingest ≈ 2 calls/session source=measured (stub dry run)`
`ARM: A6 unit=nim_calls per_item=1 answer + 1 judge source=measured`
`ARM: D5 unit=nim_calls per_item=1 answer + 1 judge source=measured`

**Questions.** Fixed before any answer, by seeded stratified sampling (seed 20261118):
- LoCoMo: 500 of the 1,540 questions, stratified by category, from all 10 conversations.
- LongMemEval-S: the 99 held-out questions of A0's subset (stratified by type). Extraction over the 500 haystacks would need ~50k calls; the 99 need ~10k.

**k choices, made without answer outcomes.** k_facts=20 is fixed a priori (about 400–600 tokens, keeping A5 under A3w's input). The 80-turn window comes from LoCoMo retrieval coverage only (LoCoMo is in-sample for retrieval configuration, as in v3):

| Window | all evidence in context | any evidence | turn tokens |
|---|---|---|---|
| 10 (A3) | 0.668 | 0.809 | ~500 |
| 20 | 0.721 | 0.859 | ~900 |
| 10 + neighbouring turns | 0.728 | 0.860 | ~1,080 |
| 40 (A3w) | 0.785 | 0.906 | ~1,740 |
| 80 (A6) | 0.840 | 0.940 | ~3,320 |

Neighbour windows were tried as a third option and dropped: no better per token than top-20.

## Baseline arm

- **A3** for A5 and D5, **A3w** for A6: the same questions, answered in v3 by the same model, prompt and judges.
- **Facts-only check:** A5's ingest with the facts left out (internal arm A5x, never answered) must reproduce the stored A3 contexts byte for byte. Dry run: 75/75.
- **Drift check:** 60 subset questions of A3 are answered again inside this run. If fewer than 90% of verdicts agree with v3's, A3 and A3w are re-answered on the whole subset and used instead.

## Decoys

- **D5 (wrong-conversation facts):** gains from facts must vanish when the facts belong to someone else (B10).
- v3's judge decoys (B6) still hold: same judges, same prompts.
- **Extraction sanity, on the pilot conversation:** at least 80% of sessions yield at least one fact, and a hand check of 20 random facts finds at most 2 that misstate their source turn.

## Stop rules

- Pilot: more than 5% of extraction calls still failing after retries; more than 20% of sessions with zero facts; any thinking text in extraction output.
- Quota: the chunked loop of v3 amendment A1. Extraction stops after 7 days of wall clock; answering after 21.
- First error that is not a 429 or a timeout, in either stage: stop and report.
- Extraction failures are never silently dropped: every call retries until it succeeds or the run stops, and the count is reported (the pipeline itself returns no facts on failure).

## Held-out exposure

- LoCoMo: in-sample for retrieval configuration (k=80 chosen on its retrieval coverage), as in v3. No answer outcome informs any choice here: the interim A3w − A3 difference was seen while planning and is not used.
- LongMemEval-S held-out 99: no choice is made on it. The extraction model reads the haystack sessions only, never questions or answers.

## Blinding

As in v3. `[fact]` is a neutral label; arm names never reach the judge; judge calls for all arms are shuffled together.

## Diagnostics

Facts per session, extraction calls and retries per conversation, entity adjudication calls, ingest wall time, input tokens per question, and on LoCoMo the share of A5 contexts holding a fact whose source turn is gold evidence.

## Projection (to be confirmed by the pilot)

- Extraction (Bolt, ~2 calls/min sustained, measured in question generation): LoCoMo 272 sessions → ~550 calls, ~5 h. LongMemEval 99 haystacks × ~50 sessions → ~10k calls, ~3.5 days.
- Answers and judges (Lightning, shared with v3's loop): ~3.9k calls, ~3 days at the loop's current pace, interleaved with v3's backlog.
- Cost $0 on NIM's free tier.

`PREREG: ok` (frozen at the pilot, 2026-11-18)
