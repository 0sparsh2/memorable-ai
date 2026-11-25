# Preregistration v8: the answer model and step-by-step answering

**Status:** frozen 2026-11-24 at 23:55 EST, when its pilot started; dry run passed ([dry-run-v8.json](dry-run-v8.json)). **Probe ([probe-v8.json](probe-v8.json)):** MiniMax answered in 71 s and Step 5 in 4.8 s, so **X = Step 5**. Kimi and Mistral timed out at 600 s on the first probe and answered on the retry (52 s and 100 s), so all four arms run. It is a new eval beside v3–v7, whose arms and answers are untouched. The owner chose the arms on 2026-11-24: Ultra with step-by-step answering, Kimi K3.1 Pro, Mistral Large 4, and one of MiniMax M3.1 Flash or Step 5 if a probe shows it responds.

**Why.** With v7's prompt (P1), A7 scores 88.7 on LME-dev against the published 95.6. Its 17 misses are preference 5, temporal-reasoning 4, multi-session 3, knowledge-update 3 and single-session-assistant 2 ([results-v7.md](results-v7.md)). Apart from preference, these need date arithmetic, counting or the latest value. Ultra already reasons before answering, but briefly: a median of 465 output tokens for an 83-token answer. Two published results point at the reader:
- Mastra's observational memory, with an observation log like A7's, reports 84.23% with GPT-4o answering and 94.87% with gpt-5-mini ([docs/observation_memory.md](../../../docs/observation_memory.md)).
- LongMemEval's paper ships a step-by-step reader prompt as its chain-of-thought option.

v8 keeps A7's contexts and P1 and changes only the answer model, or for Ultra, adds step-by-step answering. It runs on NIM's free tier.

## Bars

**lesson:** We learn whether a different free answer model, or step-by-step answering on Ultra, turns more of the same memory into correct answers, especially on LME, and at what speed. You should answer every later run, including the final held-out runs, with the arm B24 picks.

The metric and CIs are as in v3–v7: LLM-judge accuracy, with a 95% paired bootstrap over questions. Judges and judge prompts are v3's (Lightning). Every reader answers with thinking on (`enable_thinking=true`), as every reader has since v1.1.

| # | Metric | Pass bar | Direction | A miss means |
|---|---|---|---|---|
| B22 | X − U·P1 on A7's contexts, LoCoMo-dev J-strict and LME-dev J, paired, for each arm X | ≥ +2.0 pts with CI > 0 on one; no drop > 1.0 on the other | higher | X does not beat today's reader. Keep Ultra with P1 |
| B23 | Guard: the share of D8 questions X correctly declines, minus U·P1's, paired | no drop > 5.0 pts | higher | X answers from memories that are not about the question. Do not adopt it |
| B24 | Choice | Among arms passing B22 and B23, the highest LME-dev J. If the CI of its paired LME difference with the runner-up includes 0, the one with the lower median answer time; a model marked "preview" loses ties. If none pass, U·P1 | — | — |

**Reported, no bar:**
- Per category, for each arm (temporal-reasoning, multi-session and knowledge-update above all).
- Median and p95 answer time, output tokens, and answers per minute per arm.
- J-lenient on LoCoMo for each arm (the number beside published scores).
- Refusals (`v7_report.REFUSAL`) and median response length per arm.

## Arms

All arms answer v6's A7 contexts byte for byte (as queued by v7), with v7's settings (`effort="low"` with thinking on, 8,192 output tokens, one retry without thinking on an empty reply) and the HTTP timeout raised to 600 s, since the slower readers answer in 1–3 minutes. All are judged by v3's judges.

| Arm | Reader (NIM id) | Prompt | Questions |
|---|---|---|---|
| U·P1 (baseline) | Ultra, `nvidia/nemotron-3-ultra-550b-a55b` | P1 | v7's A7·P1 and D8·P1 answers, reused (drift check below) |
| U·P1s | Ultra | P1s: P1 plus LongMemEval's step-by-step instruction | A7 347 + 150, D8 125 |
| K·P1 | Kimi K3.1 Pro, `moonshotai/kimi-k3.1-pro` | P1 | A7 347 + 150, D8 125 |
| M·P1 | Mistral Large 4, `mistralai/mistral-large-4-1t-a52b` | P1 | A7 347 + 150, D8 125 |
| X·P1 | MiniMax M3.1 Flash (`minimaxai/minimax-m3.1-flash`) or Step 5 (`stepfun-ai/step-5-preview`), chosen by the probe below | P1 | A7 347 + 150, D8 125 |
| U·P1r | Ultra | P1 | drift check: 40 LoCoMo-dev, 20 LME-dev |

**P1s instruction:** "Answer the question using these memories. Work out relative dates ("yesterday", "last week") from the date of the memory they appear in. Answer step by step: first extract all the relevant information from the memories, and then reason over it to get the answer. End with your answer: if the question asks about facts or events, a short phrase or sentence, or that you don't know if the memories do not contain it; if it asks for advice, suggestions or an opinion, a helpful answer that draws on what the memories say about the user: their preferences, past experiences, plans and circumstances." The step-by-step sentence follows LongMemEval's `run_generation.py` ("Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer."). The rest is P1's rules, moved to the end. The memory header, date line and "Question: … Answer:" are as in P1.

**The probe for X (before the pilot, availability and latency only).** MiniMax M3.1 Flash and Step 5 each receive the same A7·P1 prompt: the first LME-dev question in sorted qid order, with a 600 s timeout. X is whichever returns a non-empty answer sooner. If neither answers, the arm is dropped. Kimi K3.1 Pro and Mistral get the same probe. A reader that does not answer is probed once more; if it fails again, its arm is dropped and reported. (Draft amendment, 2026-11-24, after the first probe: Kimi and Mistral timed out at 600 s, while MiniMax answered in 71 s and Step 5 in 4.8 s. The probe measures availability only, so a retry cannot favour any arm's accuracy.) Probe responses are not judged, printed or read for content.

## Baseline arm

U·P1 (Ultra, P1) is the baseline for every arm: same contexts, same prompt (except U·P1s) and same judge. Its D8·P1 answers are the baseline for B23.

**Drift check:** 40 LoCoMo-dev and 20 LME-dev A7 questions (stratified by category, fixed seed) are answered again by Ultra with P1. If fewer than 90% of verdicts agree with v7's A7·P1 answers, U·P1 is re-answered in full before any bar is scored.

## Decoys

- **D8** (B23), exactly as in v7: each non-preference LME-dev question with the next haystack's whole A7 context, judged with LongMemEval's abstention template.
- The judges are v3's, already checked against empty and foreign memory (D5, D6).

## Stop rules

- **Dry run (zero cost, stub reader and judge):**
  - Every queued row's context, question, date line and answer equal v7's A7·P1 (or D8·P1) row, byte for byte.
  - P1s renders with no unfilled `{…}` and carries no evaluation word.
  - Every new reader id is in the harness's reader table.
  - The queue counts match the table above.
- **Probe:** as above, before the pilot.
- **Pilot:** 3 answers per cell (each arm on LoCoMo, LME and D8, plus the drift check). Stop if any answer is empty after its retry, any response carries thinking text, or any verdict is unparsed. A reader whose pilot answers all time out is dropped and reported.
- **Full run:** the first error that is not a 429, a timeout or a dropped connection stops it and is reported. More than 2% unparsed judge verdicts in a cell stops it.
- **Cap:** 3 days of answering. $0 (NIM free tier).
- No prompt, context, model or judge change after the pilot starts (otherwise: prereg v8.1).

## Held-out exposure

Dev rows only: LoCoMo conv-48/49 and LME-dev 150. The arm B24 picks is fixed before any held-out run. Held-out LoCoMo conversations and the LME 350 are untouched.

## Blinding

As in v3. P1s carries no evaluation words. The judge sees the question, the gold answer or explanation, and the response, never the arm, prompt or reader names. Queued rows are shuffled across arms.

## Projection (to be confirmed by the pilot)

- **Answers:** 4 arms × 622 (497 A7 + 125 D8) plus 60 for the drift check: about 2,550, spread over four reader models with separate per-model limits.
- **Judge calls:** about 4,000 on Lightning, which sets the pace: about 8–12 h at v7's judge rate. The v3/v4 loop is trimmed to 195 LME rows (owner, 2026-11-24). It shares Lightning with the pilot, but finishes before the full run.
