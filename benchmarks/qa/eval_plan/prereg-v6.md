# Preregistration v6 (DRAFT): the observation log, end to end

**Status:** draft, 2026-11-19; dry run passed ([dry-run-v6.json](dry-run-v6.json)). It freezes when its pilot starts, after v5 has chosen the answer model. It is a new eval beside v3–v5, whose arms and answers are untouched.

**Why.** Every LongMemEval leader compresses the whole history with an LLM instead of retrieving raw excerpts. Mastra's observational memory reports 94.87% with gpt-5-mini answering (84.23% with gpt-4o); its context averaged ~30k tokens of dated observations ([docs/observation_memory.md](../../../docs/observation_memory.md)). Memorable now has an observer (one LLM call per closed episode) and an observation log in context (commit aee0fd1). v6 measures it on the iteration slice that v5 defined.

## Bars

**lesson:** We learn whether a whole observation log, on top of the current retrieval, turns into correct answers, at what ingest cost, and whether G2's graph walk changes answers at all. You should turn the observer on by default for users with an LLM only if it clears B15 and B16, and keep the G2 walk on only if B18 shows no harm.

Primary metric and CIs as in v3: LLM-judge accuracy; 95% paired bootstrap over questions; "beats" = the CI of the paired difference excludes 0. Judges and judge prompts are v3's. **The answer model is v5's chosen reader** (bar B11's rule), fixed before this pilot and named here when it is known: `<READER>`.

| # | Metric | Pass bar | Direction | A miss means |
|---|---|---|---|---|
| B15 | A7 (log) − A3g, LoCoMo-dev J-strict and LME-dev J, paired | ≥ +2.0 pts with CI > 0 on one; no drop > 1.0 on the other | higher | The log does not pay off over retrieval at this answer model. Keep the observer opt-in |
| B16 | Decoy D7 (another conversation's log) − A3g | ≤ +1.0 pt on both | lower | A7's gain would come from the format or the extra tokens, not from this user's observations |
| B17 | Observer cost | ≤ 2.5 LLM calls per session (v4's budget for fact extraction); ≤ 2% replies unparseable after salvage; no thinking text | lower | Too costly or unreliable for the default write path |
| B18 | A3g − A3 (the G2 walk, LLM-free ingest), both benchmarks | no drop > 1.0 on either (report the gain) | higher | The walk changes answers for the worse on sparse graphs: default it off without an LLM |

**Reported, no bar:**
- A7 against A0 (the full history), with input tokens per question.
- Per category; J-lenient on A3g and A7 (the number beside published scores).
- Log size in tokens per question, and how often the 30k budget cut lines.
- Observer calls, tokens and wall time per conversation.

## Arms

All answered by `<READER>` with v3's prompt and settings, judged by v3's judge.

| Arm | Identity | Settings |
|---|---|---|
| A3 | stored v3 contexts | Score fusion + Ettin-17M, top 10 turns + 3 episodes, no graph walk. Already answered by `<READER>` in v5 on these questions, so v5's answers are reused, with a drift check below |
| A3g | `memorable#current` | A3 with the G2 walk (`graph_turn_weight=1.0`), rebuilt from an LLM-free ingest |
| A7 | `memorable#observation-log` | A3g's turns and episodes, byte for byte, plus the observation log of a separate observer-only ingest (`extraction.facts=false`, `observations=true`, no LLM entity adjudication), within 30k tokens, before the excerpts |
| D7 | decoy | A3g plus the log of another conversation of the same benchmark (fixed derangement; for LoCoMo-dev the other dev conversation) |
| A0 | full history | v5's answers (same reader, same questions) |

**Observer model:** `nvidia/nemotron-3-ultra-550b-a55b` on NIM's free tier, temperature 0. Whether it thinks by default is checked in the dry run: thinking is switched off for the observer if it does.

**Questions:** LoCoMo-dev (conv-48, conv-49: 347) and LME-dev (150), for every arm including D7.

## Baseline arm

A3g is the baseline for A7 (same turns, the log is the only difference) and for D7. A3 is the baseline for A3g.

**Drift check:** 40 LoCoMo-dev and 20 LME-dev A3 questions are answered again by `<READER>` inside this run. If fewer than 90% of verdicts agree with v5's, A3 is re-answered on the whole slice.

## Decoys

- **D7** (B16).
- **Log sanity on the pilot conversation:**
  - At least 80% of episodes yield an observation.
  - A hand check of 20 random observations finds at most 2 that misstate their source turns, and at most 2 dates more than a day off.

## Stop rules

- **Dry run (zero cost, stubbed observer):**
  - A7 without its log must equal A3g byte for byte on every question.
  - A3 rebuilt must equal the stored A3 (already checked: 191/191 on conv-48).
- **Pilot:** one LoCoMo-dev conversation and 5 LME-dev haystacks through the observer, then 3 answers per cell. Stop if more than 5% of observer calls still fail after retries, any reply has thinking text, or the sanity check above fails.
- **Full run:** the first error that is not a 429, a timeout or a dropped connection stops it and is reported. More than 2% unparsed judge verdicts in a cell stops it.
- **Caps:** 3 days for the observer ingest, 4 days for answering. $0 (NIM free tier).
- No prompt, budget, model or context change after the pilot starts (otherwise: prereg v6.1).

## Held-out exposure

Dev rows only: LoCoMo conv-48/49 (the v5 split) and the LME dev 150. The observer reads haystack sessions only, never questions or answers. Held-out LoCoMo conversations and the LME 350 are untouched; they are kept for the final, frozen claim.

## Blinding

As in v3. The log uses neutral headings ("Observation log", "Retrieved excerpts"), and arm and reader names never reach the judge.

## Projection (to be confirmed by the pilot)

- **Observer:** LoCoMo-dev ~94 episodes (94 calls); LME-dev 150 haystacks × ~50 sessions ≈ 7.5–10k calls. At Ultra's 6 requests/min that is ~21–28 h.
- **Local ingest:** 150 LME-dev haystacks twice (LLM-free for A3g, and the observer's own), about 3–5 h of CPU.
- **Answers:** A3g, A7 and D7 (497 each) plus 60 for the drift check: about 1,550 answers, and about 2,600 judge calls on Lightning (~11 h).
