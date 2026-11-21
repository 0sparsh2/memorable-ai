# Preregistration: end-to-end QA (LoCoMo + LongMemEval-S)

**Version:** v3, 2026-11-15. Supersedes [prereg-v2.md](prereg-v2.md), which froze when pilot attempt 2 started, and [prereg.md](prereg.md) (v1.1). It freezes when pilot attempt 3 starts.

**Amendment A1 (2026-11-15, owner's decision, during the full run).** NIM's free tier throttles bursts: HTTP 429 after about 250 calls in 4.6 min, then 50 calls in 3 min, lifted within about 20 min, with no `Retry-After`. The 14-hour wall-clock cap is replaced by a **chunked run**: `answer-loop` resumes whenever a probe goes through, until every queued item is answered (give-up limit 21 days). Every call records a timestamp and the model id NIM reports, so a silent model change would show. Arms are interleaved, so any drift spreads evenly across them. Measurement (model, prompts, judges, arms, bars) is unchanged.
- **Measured throttle (2026-11-15 evening).** Over 20 min at a 20 requests/min offer, 80 calls succeeded and 94 got 429: about 4 successful calls/min sustained. **The limit is per model.** While Nemotron-Lightning refused 7 of 10 small requests, Nemotron-Bolt on the same account served 10 of 10. The loop now offers 6 requests/min. ETA for the ~22.5k remaining calls is about 4 days, plus about 1 day for A4.

**Amendment A2 (2026-11-16): A4.** By the preregistered rule ([docs/retrieval_round2.md](../../../docs/retrieval_round2.md) §4), fine-tuning round 1 qualified only the semantic model (MiniLM-ft: episode recall +1.3 pts end to end). The turn model, static model and rerankers did not pass. A4 is built after fine-tuning round 2, which re-tests every model under the same bars, so that one A4 arm carries every qualifying model. The A0–A3w answers already running are unaffected.

**Amendment A3 (2026-11-18): the A4 arm, fixed before any A4 answer.** Round 2 ([docs/retrieval_round2.md](../../../docs/retrieval_round2.md) §5) qualified no further model, and round 2's own MiniLM missed its bar (+0.98 vs ≥ +1.0). A4 is therefore:
- **A3 with round 1's fine-tuned MiniLM as the semantic model** (`training/data/ft_v1/models/minilm`, `model.safetensors` sha256 `41f5364bde780ef4…`). Turn model (leaf-ir), segmentation model (potion-32M), reranker (Ettin-17M base), fusion, k and the prompt are A3's.
- The semantic model shapes ingestion: episode and fact vectors, and the sentences picked for extractive episode summaries. So A4 has its own ingest. A3 is rebuilt in the same process and must match the stored A3 contexts byte for byte, so A4 differs from A3 by the model alone. LoCoMo is built locally; LongMemEval-S on Kaggle (the same job as A3's, in A4 mode).
- **Pilot:** 3 items per benchmark (LME dev rows; LoCoMo has no split), in a separate run directory so pilot answers never enter the results, before A4's contexts join the chunked loop. The stop rules, bars (B4: A4 − A3) and judges are unchanged.

**Changes from v2**, from pilot attempt 2 ([pilot-notes.md](pilot-notes.md)):
1. **Empty-answer fallback.** If a thinking answer comes back empty (thinking ran to the 8,192-token cap), the same prompt is asked once more without thinking. This applies to every arm, and fallbacks are counted per arm. Attempt 2 had 1 empty answer in 30: the full-context arm on a 108k-token LME history. Without the fallback that would count against the ceiling arm, which flatters the memory arms.
2. **No change to B6:** the decoy check now computes the bar as written (≥95% of all decoys WRONG, ≥95% of gold CORRECT, per benchmark and judge). Earlier code demanded ≥95% for each decoy type separately, which with 10 decoys per type meant 10/10.

**Changes from v1.1 to v2**, both from pilot attempt 1 ([pilot-notes.md](pilot-notes.md)):
1. **Answer output cap: 2,000 → 8,192 tokens.** 4 of 30 pilot answers came back empty: the model's thinking hit exactly 2,000 tokens before any reply was written, and two more used over 1,900.
2. **The perturbed-decoy generator now follows this document's own definition** ("key entity, number or year changed"). Attempt 1 changed non-key words, e.g. "Likely no, …" → "Toronto no, …", which kept the meaning. It also replaced one item of a ten-item list. The decoy definition, the bars and the judge prompts are unchanged.

Everything else (arms, bars, held-out policy, stop rules, judge prompts) is identical to v1.1.

**Budget unit:** wall-clock hours, on NIM's free tier ($0). A cap of $60 applies only if the paid fallback is ever approved.

## Bars

**lesson:** We learn whether Memorable's retrieval improvements (score fusion, reranker, fine-tuned models) turn into correct answers, and how close memory gets to "paste the whole history" at a fraction of the tokens. You should ship the winning configuration as the default, and quote its number with the protocol beside it.

**Primary metric:** LLM-judge accuracy.
- LoCoMo categories 1–4, 1,540 questions; adversarial category 5 is excluded, as everyone does.
- LongMemEval-S, all 500 questions, including the 30 abstention questions.

**Confidence intervals:** 95% paired bootstrap over questions. "Beats" means the CI of the paired difference excludes 0.

| # | Metric | Pass bar | Direction | A miss means |
|---|---|---|---|---|
| B1 | LoCoMo J-strict: A3 − A2 | ≥ +3.0 pts, CI > 0 | higher | Retrieval gains (+6 to +10 pts recall) do not reach answers. Stop investing in retrieval tuning and look at the answer stage. |
| B2 | LME held-out J: A3 − A1 (BM25) | ≥ +3.0 pts, CI > 0 | higher | Our semantic stack adds nothing over keyword search end to end on LME. |
| B3 | LoCoMo J-strict: A3 vs A0 (full context) | A3 ≥ A0 − 5 pts **and** A3 mean input tokens ≤ 15% of A0's | higher / lower | Full context is competitive at this scale. Position memory on cost and latency, not accuracy. |
| B4 | A4 (fine-tuned) − A3, on LoCoMo J-strict or LME held-out J | ≥ +1.5 pts on one, no drop > 1.0 on the other | higher | The fine-tunes do not pay off end to end. Keep the base models. |
| B5 | A3w (wide context) − A3, LoCoMo J-strict | report only; ≥ +2 pts means "context budget matters" | — | Informs the default `max_excerpts` and token budget. |
| B6 | Judge reliability on decoys (both benchmarks) | ≥ 95% of decoys WRONG **and** ≥ 95% of gold answers CORRECT | higher | Scores are not trustworthy. Stop; fix the judge before any comparison. |

**Reported, no bar:**
- Per-category accuracy.
- Mean input tokens per question.
- p50 retrieval latency.
- J-lenient on LoCoMo (Mem0 2026's unified judge), on arms A0, A3 and A4 only, to place us beside published numbers.
- **C1, optional:** DeepSeek V4.1 Flash as answer model and judge, on the 99-question A0-LME subset, for arms A0 and A1. This places us beside billion-context's third-party LongMemEval-S evaluation, which used that model (full context 83.2%, BM25 56.4%). NIM serves it at about 1.4 calls/min, so it runs in the background (~5 h).

## Arms

All arms share one answer model and one judge: `nvidia/nemotron-3.5-lightning-30b-a3b` on NVIDIA NIM (free tier), temperature 0.
- **Answers** think (`chat_template_kwargs.enable_thinking=true`, up to 2,000 tokens).
- **The judge** does not think.
- **Measured:** about 2–5 s per call with thinking and 0.2 s without, 8 concurrent calls without errors.
- **DeepSeek V4.1 Flash** answered correctly but queued at about 1.4 calls/min, which is why it is only used for C1.
- **Paid fallback:** Gemini 3 Flash, only with the owner's approval and a new projection.

**Ingestion** (memory arms only):
- Memorable's real write pipeline, LLM-free (`MEMORABLE_LLM__DEFAULT_MODEL=none`). The fact layer is therefore empty: this measures turn and episode retrieval.
- Session dates come from the dataset (`occurred_at`).

**Answer prompt:** one neutral prompt for every arm (below). Memories are shown oldest first, each with its date. LME adds the question date ("today is …").

| Arm | Identity | Settings | Per item, estimate (Step 1b) |
|---|---|---|---|
| A0 | `full-context` | Entire history, sessions in order with dates. LoCoMo: all 1,540 questions (prefix cached per conversation). LME: 99 held-out questions stratified by type (100 targeted; per-type rounding), ~115k tokens each. | LoCoMo ≤ $0.0127, LME $0.059 |
| A1 | `memorable#bm25` | Turn layer, lexical only (FTS5 BM25), top 10 turns; no episodes | LoCoMo $0.0028, LME $0.0022 |
| A2 | `memorable#v0.3` | Quality profile as shipped in v0.3: leaf-ir + BM25 with RRF, top 10 turns + 3 episodes | LoCoMo $0.0028, LME $0.0022 |
| A3 | `memorable#current` | Score fusion (leaf 1.0, BM25 0.5) + Ettin-17M rerank (depth 20, 128 tokens), top 10 turns + 3 episodes | LoCoMo $0.0028, LME $0.0022 |
| A3w | `memorable#current-wide` | As A3 with top 40 turns, about 6–8k tokens (the budget Mem0 reports) | LoCoMo $0.0052, LME $0.0074 |
| A4 | `memorable#finetuned` | A3 with each fine-tuned model (turn embedder, semantic embedder, reranker) substituted **only if it passed its retrieval bar** in [docs/retrieval_round2.md](../../../docs/retrieval_round2.md) §0. If none passed, A4 is dropped. Which models qualify is recorded here before the pilot. | as A3 |

**Per-item cost:** $0 on NIM. The price column above is for the Gemini fallback only.

**Industry protocols, not adopted here:** Mem0's 2026 harness feeds up to 200 memories and uses a LoCoMo-specific answer prompt ("all events occurred in 2022–2024"), with GPT-4o or GPT-5. We do not copy that. J-lenient only reproduces their judge, for comparison.

## Baseline arm

There are three baselines:
- **A1 (BM25), the lexical baseline:** proves the semantic stack is needed at all.
- **A2 (v0.3), the previous release:** proves this round's changes are needed.
- **A0 (full context), the ceiling and cost reference:** proves memory is needed at all.

## Decoys

Built from dev rows (the LoCoMo conversations and the LME 150-question tuning sample), never from held-out rows. 40 per benchmark:
- **10 gold answers** given as the model answer. The judge must say CORRECT (positive control).
- **10 shuffled answers** (the gold answer of another question in the same conversation or type). Must be WRONG.
- **10 "I don't know" / "not mentioned"** answers, on answerable questions. Must be WRONG.
- **10 perturbed answers:** the gold answer with its key entity, number or year changed. Must be WRONG (J-strict and LME). J-lenient is expected to accept some date shifts, so it is reported, not barred.

A decoy pass proves a judge or scorer bug, so the run stops.

## Stop rules

- **Budget cap: 14 hours of wall-clock time** for the full run at the measured request rate (or $60 if the paid fallback is approved). The run is resumable, so stopping loses nothing already answered.
- **First quota, rate-limit-after-retries, or 5xx-after-retries error:** stop and report. No silent skips; a skipped item counts as an error.
- **Judge below B6** on the decoys (dry run with the stub judge, then the real judge in the pilot): stop.
- **Pilot:** stop if any arm's pilot items fail to parse, or the projection exceeds the cap.
- No prompt, k, threshold or model choice may change after the pilot starts (otherwise: prereg-v2).

## Held-out exposure

- **LoCoMo (all 10 conversations): NOT held out for retrieval configuration.** Fusion weights, reranker choice, depth and truncation were selected on LoCoMo recall. The split-half check (conversations 0–4 vs 5–9) showed the gains on both halves. Prompts and answer settings have never seen LoCoMo outcomes. **LoCoMo results are in-sample for retrieval config, and the report says so.**
- **LongMemEval-S:**
  - The 150-question stratified sample (`sample_questions(150, seed=13)`) was used for retrieval tuning. It is **dev**: dry run, pilot, decoys.
  - The other 350 questions, including all 30 abstention questions, are **held out**: never read, printed, or used for any choice before the final run. Their count is the only thing computed.
  - B2 and B4 use the held-out slice. The full 500 is reported for comparison with published numbers.
- **Fine-tuned models** were trained only on synthetic questions over ConversationChronicles, MSC, SODA and WildChat-1M. ShareGPT and UltraChat were excluded because LongMemEval builds its filler from them. The questions are written with random names and dates; no LoCoMo or LME text is involved.

## Blinding

- The judge sees only question, gold answer (or rubric) and model answer. Labels are neutral; the arm name never appears.
- Judge calls for all arms are shuffled together.
- The answer model's prompt carries no evaluation words (eval, test, judge, benchmark, score, rubric).
- Memories are labelled as memories from past conversations.

## Diagnostics

Declared before freeze; the stub-mode dry run must produce:
- The item count per arm and benchmark (LoCoMo 1,540; LME 500, of which held-out 350 and dev 150; A0-LME 99).
- The count of A0-LME prompts over 1M tokens (expect 0).
- Truncation % per arm.
- The number of questions with an empty memory context (A1–A4; reported).
