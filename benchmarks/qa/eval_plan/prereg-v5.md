# Preregistration v5: the answer model

**Version:** v5, 2026-11-19. A new eval beside v3 and v4.2, whose arms, bars and running answers are untouched. It freezes when the pilot starts.

**Why.** The owner's goal is to beat the best published scores: LoCoMo 93.6 and LongMemEval-S 95.6 (Agent Zero Memory; Mem0 92.5 / 94.4; `published_results.json`). Every arm so far is answered by Nemotron-3.5-Lightning (3B active).
- The interim v3 run shows Lightning reaching 91.2 on LoCoMo (lenient judge) **with the whole conversation in context**. A reader that tops out below 93.6 with perfect recall cannot reach the target, whatever the memory does. That interim number is the motivation only; no choice below uses it.
- The leaders answer with GPT-4o or GPT-5-class models.
- NIM's free tier now serves two much larger models that answered a probe in seconds (2026-11-18): Nemotron 3 Ultra (55B active of 550B) and Mistral Large 4 (52B active of 1T). GLM-5.3 Flash timed out, Kimi K3.1 Pro, MiniMax M3.1 Flash and Step 5 took 1–3 min per call, and DeepSeek V4.1 Flash is retired (410).

## Bars

**lesson:** We learn which free answer model gets the most out of the same memory, whether any of them can pass 93.6 on LoCoMo with the whole conversation, and how fast it runs. You should answer every later arm (and the final claim) with the winner, and know before building more memory whether the target needs a paid model.

Primary metric and CIs as in v3: LLM-judge accuracy; 95% paired bootstrap over questions; "beats" = the CI of the paired difference excludes 0. Judge, judge prompts and answer prompt are v3's (judge: Lightning, no thinking).

| # | Metric | Pass bar | Direction | A miss means |
|---|---|---|---|---|
| B11 | Reader R − Lightning on A3 (current memory): LoCoMo-dev J-strict and LME-dev J, paired | R qualifies at ≥ +3.0 pts with CI > 0 on one benchmark and no drop > 1.0 on the other. Among qualifiers, the higher mean of the two accuracies wins; if the qualifiers' paired difference has CI ∋ 0 on both benchmarks, the faster one (B13) wins | higher | The answer model is not what limits the memory arms. Keep Lightning for iteration; the target becomes a memory problem only (B12 still decides the final reader) |
| B12 | Chosen reader's A0 (full context), LoCoMo-dev J-lenient | ≥ 94.0 | higher | No free reader passes 93.6 even with the whole conversation. Propose a paid reader for the final claim (owner's approval, new projection) |
| B13 | Chosen reader's sustained throughput | ≥ 2 answered questions/min over the run's unblocked time | higher | Too slow to iterate with: use it for final claims only, iterate on the fastest qualifying reader |
| B14 | Decoy D6 (no memory at all), each strong reader | ≤ 20% on LoCoMo-dev J-strict and on LME-dev J | lower | A reader scores from priors, or the judge accepts generic answers. Stop and investigate before any comparison |

**Reported, no bar:**
- Per category and per reader: accuracy, A3 − A0 gap, J-lenient on A3 (the number that sits beside published scores).
- A0 on LME-dev (48 questions) per reader.
- Answer latency p50/p95, 429s, empty answers and fallbacks, output tokens.
- Drift: v5's Lightning cells against the v3 run's answers to the same questions, where both exist (verdict agreement).
- **C2, judge:** v3's judge decoys (B6) re-run with Ultra as the judge, and Ultra-judge vs Lightning-judge agreement on 200 v5 answers. If Ultra passes B6 and agrees ≥ 95%, a later prereg may move the judge to Ultra (it would free Lightning's quota). Nothing in v5 changes judge.

## Arms

Each cell is (context, reader). Contexts are the stored v3 contexts, byte for byte; no retrieval is re-run. Every reader gets v3's answer prompt, temperature 0, thinking on (`chat_template_kwargs.enable_thinking=true`), up to 8,192 output tokens, and v3's empty-answer fallback (one retry without thinking).

| Reader | Model (NIM free tier) | Probe 2026-11-18 | Per item |
|---|---|---|---|
| lightning (baseline) | `nvidia/nemotron-3.5-lightning-30b-a3b` | 2–5 s | $0; 1 answer + 1–2 judges |
| ultra | `nvidia/nemotron-3-ultra-550b-a55b` | 2.7 s, correct | $0; 1 answer + 1–2 judges |
| mistral | `mistralai/mistral-large-4-1t-a52b` (reasons even when told not to) | 8.3 s, correct | $0; 1 answer + 1–2 judges |

| Context | Questions | Readers |
|---|---|---|
| A3 (current memory: score fusion + Ettin-17M, top 10 turns + 3 episodes) | LoCoMo-dev 347, LME-dev 150 | all three |
| A0 (full history) | LoCoMo-dev 347; LME-dev 48 (stratified by type, seed 20261119; ~115k tokens each) | all three |
| D6 (decoy: "(no memories found)") | 50 LoCoMo-dev + 48 LME-dev, stratified, seed 20261120 | ultra, mistral |

J-lenient (Mem0's judge) runs on every A0 and A3 LoCoMo cell, as in v3.

`ARM: lightning unit=nim_calls per_item=1 answer + 1.7 judges source=measured (v3)`
`ARM: ultra unit=nim_calls per_item=1 answer + 1.7 judges source=probe (n=2)`
`ARM: mistral unit=nim_calls per_item=1 answer + 1.7 judges source=probe (n=2)`

## Baseline arm

Lightning on the same contexts and questions, answered inside this run (same time window as the other readers), is the baseline for B11. A0 is the ceiling reference for B12.

## Decoys

- **D6 (no memory):** a strong reader answering without memories must score low (B14). This catches priors leaking answers and a judge that accepts generic replies.
- **v3's judge decoys (B6)** stand for the Lightning judge. C2 re-runs them with Ultra as the judge.
- Stronger readers write longer answers: more than 2% unparsed judge verdicts in any cell is a stop rule.

## Stop rules

- **Pilot (3 items per cell, the longest A0-LME prompt included for each reader):** any HTTP error other than 429 or a timeout (e.g. a context-length error) stops the pilot. A reader that cannot take the longest A0-LME prompt loses its A0-LME cell before the full run; it is not skipped mid-run. More than 1 empty answer in 3 for a cell also stops it.
- **Full run:** the first non-429, non-timeout error after retries stops and reports; no silent skips. More than 2% unparsed judge verdicts in a cell stops it.
- **Wall-clock cap:** 4 days. **Cost:** $0 (NIM free tier); the paid fallback needs the owner's approval and a new projection.
- No prompt, context, thinking setting or model change after the pilot starts (otherwise: prereg v5.1).
- Lightning's quota is shared with the v3 answer loop. That loop is paused for the full v5 run (it resumes; nothing answered is lost) so v5's judges get the quota. Its own Lightning answers to these questions are kept separately, for the drift check.

## Held-out exposure

- **LongMemEval-S:** dev 150 only (the retrieval-tuning sample). The 350 held-out rows are skipped unread by `v5_build.py`. No v5 choice touches them.
- **LoCoMo, new split from v5 on.** LoCoMo has no official split and retrieval configuration was tuned on all 10 conversations (in-sample, disclosed in v3). For answer-level choices (reader, prompt, context assembly) v5 fixes a split: **dev = conv-48 and conv-49** (347 questions), drawn with `random.Random(20261119).sample(sorted conversation ids, 2)` before any v5 answer. The other 8 conversations (1,193 questions) are held out for answer-level choices from now on. Final LoCoMo claims report the full 1,540 and the 8-conversation held-out score side by side.
- Interim v3 aggregates over all 10 conversations have been seen (A0 lenient 91.2 motivated this eval). No v5 bar or choice uses them.

## Blinding

As v3. The judge sees question, gold answer and response only; reader and arm names never reach it; judge calls for all cells are shuffled together. Every reader gets the same prompt, with no evaluation words.

## Diagnostics

The stub dry run must produce: items per cell (LoCoMo A0/A3 347 per reader, D6 50 per strong reader; LME A3 150 and A0 48 per reader, D6 48 per strong reader), the estimated maximum prompt tokens per cell, and zero unparsed stub verdicts.

## After v5: the iteration slice

The chosen reader on the dev slice (LoCoMo conv-48/49 + LME-dev 150) becomes the fast benchmark for every later memory change (G2, LLM facts, observation log, context budget). One iteration is about 500 answers and 850 judge calls, a matter of hours. Final claims run once, on the full sets, with the configuration frozen.

## Projection (to be confirmed by the pilot)

- Answers: 892 per reader (LoCoMo 694, LME 198), plus 98 D6 answers for each strong reader: 2,872 in all. Each strong reader runs on its own per-model quota.
- Lightning carries its own 892 answers and every judge call (~4,950): about 5,850 calls. At its measured ~4 calls/min sustained this is the long pole: about 24 h, 1.5–2 days with stalls.
- Cost $0.

`PREREG: ok` (2026-11-19; freezes at the pilot)
