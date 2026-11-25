# Preregistration v7: the answer prompt

**Status:** frozen 2026-11-24 at 12:45 EST, when its pilot started; dry run passed ([dry-run-v7.json](dry-run-v7.json)). It is a new eval beside v3–v6, whose arms and answers are untouched.

**Why.** v6's A7 (A3g plus the observation log) misses 15 of 25 LME-dev single-session-preference questions. 10 of those 15 responses refuse ("I don't have any specific advice about chocolate chip cookies in my memories"), even when the context names what the preference rubric asks for. None of the 10 hits refuse ([pilot-notes-v6.md](pilot-notes-v6.md), "After the results"; read after v6 was final). A refusal pattern (`v7_report.REFUSAL`) also covers 4 of A7's 12 other LME misses and 10 of its 60 LoCoMo J-strict misses (1 of its 287 hits). The cause is our `ANSWER_PROMPT`: "Answer in a short phrase or sentence. If the memories do not contain the answer, say that you don't know." That suits factual recall, but asked for advice, the reader concludes that the advice is not in memory. LongMemEval's own reader prompt has neither rule. Its default (`src/generation/run_generation.py`, retrieval without CoT) reads "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history." v7 changes only the reader's prompt. Contexts, reader and judges stay as in v6.

## Bars

**lesson:** We learn whether the answer prompt's brevity and "don't know" rules cost points on requests for advice, and whether a prompt that answers those from the user's remembered preferences gains without inventing answers when memory has none. You should use the prompt B21 picks for every later run, including the final held-out runs.

The metric and CIs are as in v3–v6: LLM-judge accuracy, with a 95% paired bootstrap over questions. Judges and judge prompts are v3's. The reader is Ultra (v5's B11 choice), answering with thinking on, as in v1.1–v6 (corrected 2026-11-24: this line said "thinking off"; answers have always sent `enable_thinking=true`, and only the observer runs with thinking off).

| # | Metric | Pass bar | Direction | A miss means |
|---|---|---|---|---|
| B19 | P − P0 on A7's contexts, LoCoMo-dev J-strict and LME-dev J, paired, for P in {P1, P2} | ≥ +2.0 pts with CI > 0 on one; no drop > 1.0 on the other | higher | This prompt does not pay off. Keep P0 |
| B20 | Guard: the share of D8 questions a prompt correctly declines, P − P0, paired | no drop > 5.0 pts | higher | The prompt makes the reader answer from memories that are not about the question, which risks LME's abstention questions (about 6% of LME) and invented answers in the product. Do not adopt it |
| B21 | Choice | Among prompts passing B19 and B20, the higher LME-dev J. If the CI of their paired LME difference includes 0, P1 (it keeps the "don't know" rule for facts). If none pass, P0 | — | — |

**Reported, no bar:**
- Per category, for each prompt (the preference category above all).
- The share of responses that refuse (`v7_report.REFUSAL` over the first 400 characters), per prompt and cell.
- Response length (median characters) per prompt, and the reader's input and output tokens.
- J-lenient on LoCoMo for each prompt (the number beside published scores).
- D8 per category.

## Arms

All arms use Ultra as the reader, with v6's settings (`effort="low"`, 8,192 output tokens, one retry without thinking on an empty reply), and v3's judges. The prompts differ only in the instruction paragraph. The memory header, the memories, the date line and "Question: … Answer:" are identical across them.

| Arm | Prompt | Contexts | Questions |
|---|---|---|---|
| A7·P0 | P0, today's `ANSWER_PROMPT` | v6's A7, byte for byte | v6's answers, reused (drift check below) |
| A7·P1 | P1, targeted | v6's A7 | LoCoMo-dev 347, LME-dev 150 |
| A7·P2 | P2, rules removed | v6's A7 | LoCoMo-dev 347, LME-dev 150 |
| D8·P0, D8·P1, D8·P2 | each prompt | guard, below | LME-dev, 125 (preference excluded) |
| A7r | P0 | v6's A7 | drift check: 40 LoCoMo-dev, 20 LME-dev |

**P0 instruction (today's):** "Answer the question using these memories. Work out relative dates ("yesterday", "last week") from the date of the memory they appear in. Answer in a short phrase or sentence. If the memories do not contain the answer, say that you don't know."

**P1 instruction (targeted):** "Answer the question using these memories. Work out relative dates ("yesterday", "last week") from the date of the memory they appear in. If the question asks about facts or events, answer in a short phrase or sentence, and if the memories do not contain the answer, say that you don't know. If it asks for advice, suggestions or an opinion, give a helpful answer that draws on what the memories say about the user: their preferences, past experiences, plans and circumstances."

**P2 instruction (rules removed, as LongMemEval's reader):** "Answer the question based on the relevant memories. Work out relative dates ("yesterday", "last week") from the date of the memory they appear in."

The prompt is chosen by a `prompt` field on each queued row (`p0` when absent), so every earlier run renders exactly as before.

## Baseline arm

A7·P0 is the baseline for A7·P1 and A7·P2: same contexts, same reader, same judge, with only the prompt changed. D8·P0 is the baseline for B20.

**Drift check:** 40 LoCoMo-dev and 20 LME-dev A7 questions (stratified by category, fixed seed) are answered again with P0 inside this run. If fewer than 90% of verdicts agree with v6's A7 answers, A7·P0 is re-answered in full before any bar is scored.

## Decoys

- **D8, the false-answer guard (B20).** Each LME-dev question, preference questions excluded, is paired with the whole A7 context of the next LME-dev haystack in sorted order (the same derangement as D7). That context is another user's log plus excerpts retrieved for that user's question. The row keeps its own question and date line, is judged with LongMemEval's abstention template (`abstention=true`), and its explanation reads: "The memories shown come from a different user's conversations and do not contain the information this question asks about." Preference questions are left out because the right reply to an advice request with no relevant memory is a generic answer, which the abstention template cannot grade. **Limitation:** LongMemEval reuses some sessions across haystacks as fillers, so a few D8 rows may be answerable. This is the same for every prompt, and B20 compares prompts against each other.
- The judges are v3's, whose decoys (D5, D6) already show they are not fooled by empty or foreign memory.

## Stop rules

- **Dry run (zero cost, stub reader and judge):**
  - Every queued A7·P row's context, question and date line equal v6's A7 row for that question, byte for byte.
  - Each prompt renders with no unfilled `{…}`, and none contains an evaluation word (eval, test, judge, benchmark, score, rubric).
  - P0 renders exactly as `ANSWER_PROMPT` did before this change, on every A7 row.
  - No D8 row is paired with its own haystack's context.
  - The queue counts match the table above.
- **Pilot:** 3 answers per cell (A7·P1 and A7·P2 on both benchmarks, D8·P0/P1/P2, A7r on both: 27 answers). Stop if any answer is empty after its retry, any response carries thinking text, or any verdict is unparsed.
- **Full run:** the first error that is not a 429, a timeout or a dropped connection stops it and is reported. More than 2% unparsed judge verdicts in a cell stops it.
- **Cap:** 2 days of answering. $0 (NIM free tier).
- No prompt, context, model or judge change after the pilot starts (otherwise: prereg v7.1).

## Held-out exposure

Dev rows only: LoCoMo conv-48/49 and LME-dev 150. LME-dev holds no abstention questions; the ~30 in LME's held-out 350 stay sealed. That is why D8 is built from dev rows. The prompt B21 picks is fixed before any held-out run. The held-out LoCoMo conversations and the LME 350 are untouched.

## Blinding

As in v3. The prompts carry no evaluation words (checked by the dry run). The judge sees the question, the gold answer, rubric or explanation, and the response, never the arm, prompt or reader names. Queued rows are shuffled across arms.

## Projection (to be confirmed by the pilot)

- **Answers:** A7·P1 and A7·P2 (497 each), D8 × 3 (375) and A7r (60): 1,429 answers on Ultra.
- **Judge calls:** about 2,160 on Lightning (LoCoMo rows get J-strict and J-lenient).
- **Time:** v6's stage 2 ran at 4.4 answers/min with Ultra and Lightning unshared, which gives about 5.5 h. The v3/v4 loop pauses while v7 runs, since it would share Lightning (owner's choice, 2026-11-23).
