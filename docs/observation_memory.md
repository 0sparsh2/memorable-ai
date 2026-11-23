# Observation log: design note (2026-11-19)

**Status:** built (commit aee0fd1, opt-in) and measured as arm A7 of [prereg v6](../benchmarks/qa/eval_plan/prereg-v6.md) on 2026-11-23: a 30,000-token log added to A3g's turns gains +11.5 LoCoMo-dev J-strict and +6.7 LME-dev J, and a decoy log gains nothing ([results-v6.md](../benchmarks/qa/eval_plan/results-v6.md)). The product's default log budget (4,000 tokens) has not been measured. The batched observer below is built too; it lost detail on its first measurement.

## Why

Every LongMemEval leader compresses history with an LLM rather than retrieving raw turns. The closest published design is Mastra's Observational Memory, which reports 94.87% on LongMemEval-S ([mastra.ai/research/observational-memory](https://mastra.ai/research/observational-memory)). That is 0.7 points under the best published score (Agent Zero, 95.6).

| | Mastra, as published |
|---|---|
| Observer | Runs when unobserved messages pass a token threshold. Turns them into dated observations (event, statement, preference, tool result), each with a priority (🔴 / 🟡 / 🟢) and up to three dates: when observed, the date referred to, and a computed relative date. The raw messages are then dropped. |
| Reflector | Runs when the log passes a second threshold. Merges related items, notes patterns and removes superseded observations. |
| Format | Plain text: two-level bullets grouped under dated headings. |
| Compression | About 6× on LongMemEval (3–6× generally). The answer context averaged about 30k tokens. |
| Retrieval | None. The whole log sits at the start of the context (cache-friendly), and recent raw messages follow it. |
| Models | Observer and reflector: gemini-2.5-flash, at ingest. Judge: gpt-4o. |
| Answer model | **gpt-4o 84.23%, gemini-3-pro 93.27%, gpt-5-mini 94.87%.** The same memory scores 10.6 points higher with a stronger answer model. |

Per type with gpt-5-mini: knowledge-update 96.2, multi-session 87.2, single-session-assistant 94.6, preference 100.0, single-session-user 95.7, temporal 95.5. Multi-session (counting and aggregating across sessions) is the weakest type even for the leader.

Two lessons for us:
1. The answer model is worth about 10 points on its own. This is why prereg v5 (the answer model) comes first.
2. On LongMemEval, a dense log of the whole history beats retrieval of excerpts. Our A3 shows the model about 3k tokens of excerpts; Mastra shows about 30k tokens of observations. Our A0 (the raw 115k-token history) fails for a different reason: a small answer model gets lost in it (Lightning scored 50% on A0).

## Design for Memorable

It fits the layers we already have, without a new store:

- **Observer = per-session LLM summarisation in observation format.** When a session closes, one LLM call turns its turns into observation bullets: a date heading, priorities, and dates referred to. Each observation is a node with `derived_from` edges to its source turns, like every derived memory (the graph rule in `procedural_memory_design.md` §13). Extracted facts (A5) and observations come from the same pass, so ingest makes about one call per session, not two.
- **Reflector = a dreaming job.** It condenses the log when it passes a budget, through validated, logged edits (G4). Superseded observations are invalidated, not deleted.
- **Context assembly (L4).** The log goes in whole while it fits the budget, oldest first. Beyond the budget, the most recent observations stay in whole and older ones are retrieved (the G2 graph walk plus fusion). That hybrid is the third option: Mastra's own design has no answer for a history whose log outgrows the window.
- **Verbatim turns stay available.** A small top-k of retrieved turns (A3) follows the log for exact wording. Observations lose detail at 6× compression, and turns cover it.

## Batched observer (`extraction.observe_batch_tokens`, 2026-11-19)

NIM's free tier limits requests per model (about 4–6 a minute sustained), not tokens. One observer call per closed episode is about 85 calls per LongMemEval haystack, so observing the 350 held-out haystacks would take 60–80 hours on Ultra. Mastra's observer runs once per ~30k tokens of unobserved messages instead, about 4 calls per haystack.

- **Setting:** `observe_batch_tokens = 0` (the default) keeps one call per episode. With N > 0, a closed episode joins a queue (`observe_pending` in its metadata), and once the queued messages reach N tokens one call observes them all. `MemoryEngine.flush_observations(user_id)` observes what is left: after a bulk import, or when a user's conversations are over.
- **Prompt:** messages are grouped under their dates, and each observation is dated by the first message it cites, so a batch that spans weeks still dates every line. The call also sees the latest ~1.5k tokens of the log, so it can say what changed instead of repeating it.
- **Cost:** the timeout grows with the output budget (16k tokens, up to 400 s), because one call now writes what 20 calls wrote.
- **Measured on conv-48 (2026-11-19), without answers: it loses detail.** Given 8–20k tokens of messages at once, Ultra wrote 94–129 observations against 393 one episode at a time, far inside its output budget. The log cited all the gold evidence for 57.6–74.9% of questions (per episode: 94.8%) and held the short answer in one line for 36.0–48.6% (65.8%). Prereg v6.1's rule picked no batch size, so it stays opt-in and off, and v6 and the final runs observe per episode. The numbers are in `prereg-v6.1.md`.

## The eval arm (preregistered as v6)

- **A7 = observation log (whole, within budget) + A3's top 10 turns**, answered by v5's chosen model on the iteration slice (LoCoMo conv-48/49 and LongMemEval dev 150).
- **Baselines:** A3 and A0 on the same answer model. **Decoy:** another haystack's log, as D5 did for facts.
- **Observer model:** Ultra if v5 shows it sustains the rate, otherwise Bolt. One call per session: LongMemEval dev is 150 haystacks × ~50 sessions ≈ 7.5k calls; LoCoMo dev is 2 conversations × ~30 sessions ≈ 60 calls.
- **The bar** follows v4's B7 style: the gain over A3 on one benchmark with CI > 0, and no drop on the other.

## Result (prereg v6, 2026-11-23)

Answered by Ultra and judged by Lightning, on dev rows (LoCoMo conv-48/49, 347 questions; LongMemEval dev, 150).

| Context | LoCoMo J-strict | LoCoMo J-lenient | LongMemEval J | Reader prompt tokens, median (LoCoMo / LME) |
|---|---|---|---|---|
| A3g: retrieved turns | 71.2 [67–76] | 91.1 [88–94] | 75.3 [68–82] | 1.0k / 3.0k |
| **A7: A3g + observation log** | **82.7** [78–87] | **96.0** [94–98] | **82.0** [76–88] | 12.3k / 26.3k |
| D7: A3g + another conversation's log | 71.2 [66–76] | — | 72.0 [65–79] | 10.9k / 26.3k |
| A0: whole conversation | 86.2 [82–90] | 98.3 [97–99] | 56.2 [42–71] (n=48) | 22.3k / 108.5k |

- The log is the largest gain measured so far (B15), and the decoy shows that its content does the work (B16).
- Observer cost: 1.18 calls per session, 0 of 8,536 replies unparseable. That is 8,536 calls and 25.7M tokens for 152 logs, so about 38 h awake on NIM's free tier for the 150 LongMemEval-dev haystacks.
- On LongMemEval, 15 of A7's 27 misses are preference questions. They are the next target.

