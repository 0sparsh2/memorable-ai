# Pilot notes: prereg v4 / v4.1 (LLM facts, 80 turns)

## Dry run (2026-11-18)
Stub LLM, LoCoMo conv-26 and conv-30. All checks pass ([dry-run-v4.json](dry-run-v4.json)): 37 LLM calls per 19-session conversation; A5 without its facts reproduces the stored A3 contexts byte for byte (75/75); reruns resume from the cache.

## Attempt 1 (prereg v4), 2026-11-18: stopped
Real extraction with `nvidia/nemotron-3.5-bolt-e4b`, conv-26 and conv-30 in parallel, 6 requests/min offered.
- 70 calls in 12 min (about 5.8 calls/min): no 429s, no empty replies, no thinking text. Bolt's quota held at this rate.
- Facts kept: 65 (conv-26) and 45 (conv-30).
- **9 of 38 extraction replies failed to parse (24%)**: the stop rule (more than 5% failing) applies, and the run stopped before any answer. Eight failed at 3,200–3,700 characters, where the 1,200-token output cap cuts the JSON array mid-item; one reply was malformed early (character 230). The pipeline drops the whole reply on a parse failure, so those sessions lost their LLM facts.
- Fix and new version: [prereg-v4.1.md](prereg-v4.1.md).

## Attempt 2 (prereg v4.1), 2026-11-18: extraction passes; one finding
- 74 calls (37 per conversation) in 12.6 min. No transport failures after retries, no thinking text, no repaired replies. One reply in 38 failed to parse (2.6%, under the 5% limit).
- Facts kept: 82 and 65 (attempt 1: 65 and 45). 3 of 38 sessions yielded no fact (7.9%, under 20%).
- A5 without its facts reproduces the stored A3 contexts: 75/75. Every A5 context carries 20 facts (about 6,000 characters of memory, vs about 3,250 for A3).
- **Hand check of 20 random facts:** 1 misstates its source (it merges two different hike incidents from sessions 12 and 14), under the limit of 2. Dates are resolved correctly ("ten years ago" → 2013, "last Friday" → 2023-06-23).
- **Finding:** 7 of the 20 say "User" for the speaker the harness feeds in as the user (Caroline, Jon), while LoCoMo's questions name them. Fixed by naming the user ([prereg-v4.2.md](prereg-v4.2.md)); no answers had been produced.

## Attempt 3 (prereg v4.2), 2026-11-18: pass
**Extraction** (LoCoMo conv-26 and conv-30, Bolt, 6 requests/min offered, 2 conversations in parallel):
- 74 calls in 12.6 min, about 5.9 calls/min with no 429s: 0 transport failures after retries, 0 unparseable replies, 0 repaired, 0 thinking text.
- Facts kept: 69 and 63. 4 of 38 sessions yielded none (10.5%, under 20%).
- No fact says "User" (attempt 2: 7 of 20 sampled); 50 of 69 and 53 of 63 name the speaker. The user's own graph node carries the name, and no second entity was created for it.
- A5 without its facts reproduces the stored A3 contexts: 75/75. D5 rows carry the other conversation's 20 facts.
- Hand check of 20 new random facts: at most 2 slightly off (a tense, "is attending" a conference held two days earlier; one unverified detail), under the limit of 2. The update phase sometimes merges statements into long compound facts; two of the 20 repeat the same opening sentence.

**Answers** (3 per arm, this pilot directory only, never part of the results): 9 answered in 210 s. Input tokens: A5 1.7–2.0k, A6 4.4–4.9k, D5 1.9–2.0k (A3: about 1.0k). One A6 answer used the no-thinking fallback (v3 rule). Correctness on 3 items is not a result.

**Projection**
```
PILOT_ARM: extraction unit=nim_calls(bolt) per_conversation=37 (19 sessions) rate=5.9/min measured n=74
PILOT_ARM: answers    unit=nim_calls(lightning) per_item=1 answer + 1-2 judges; main loop pace 2,614 calls/24 h, 940/6 h
PILOT_PROJECTION: extraction LoCoMo ~530 calls ~1.5 h; LME 99 haystacks ~9.9k calls ~28 h band95=23..55 h cap=7 d verdict=within
PILOT_PROJECTION: answers+judges ~3.3k calls (A5 1.7k, A6 1.2k, D5 0.2k, drift check 0.2k) +1.1 d to the shared queue band95=0.9..2.2 d cap=21 d verdict=within
```
- Cost: $0 on NIM's free tier.
- Limiters: Bolt's per-model throttle for extraction; Lightning's for answers, shared with v3's loop (v3 has about 7.5 days of calls left at the current pace, including A4).

## Full run (approved by the owner, 2026-11-18)
**LoCoMo, built and published.** Extraction over all 10 conversations in 90 min: 531 calls for 272 sessions (1.95 per session; bar B9 ≤ 2.5), 4 unparseable replies (1.5%), 3 repaired (1.1%), no thinking text or empty replies. A5 without its facts equals the stored A3 context for all 500 subset questions. Published to the loop as `contexts_locomo_v4.jsonl`: A5 500, A6 500, D5 100, A3r 40.

**LongMemEval-S, running.** The 99 held-out haystacks, about 28 h of extraction. Built on the PyTorch backend: on 3 dev haystacks both PyTorch and ONNX locally reproduce the Kaggle-built A3 contexts byte for byte; the publish step re-checks all 99.

**LongMemEval-S, stopped and resumed (2026-11-20).** Stopped by hand at 02:45 EST after 73 of the 99 held-out haystacks, at 386 of 500 groups walked. The process had grown to an 18 GB footprint, because PyTorch on Apple's MPS leaks across haystacks, and swap had filled the disk (details in `pilot-notes-v6.md`). Nothing finished was lost: 73 `.done` caches, 146 context rows and 73 check rows, all parseable. The three haystacks in flight are re-extracted from scratch. It resumed on CPU (`MEMORABLE_TORCH_DEVICE=cpu`), which gives byte-identical retrieval contexts. The publish step still re-checks all 99 against the Kaggle-built A3.

**Publish check failed on one haystack, then repaired (2026-11-20, 17:35 EST).** `v4_publish` found A5x equal to the stored A3 on 98 of 99. In the one exception, a single extractive episode-summary line differed (structure compared only; no held-out text was read). On that haystack, MPS reproduces the stored A3 and CPU does not: a near-tie the device flips. Its A5, A5x and A6 rows and its cache were removed (backups `*.bak-20261120`), and it was re-extracted on MPS alone (`--groups`, one process, so the MPS leak does not matter). The publish step then re-runs.

**LongMemEval-S, built and published (2026-11-20, 17:57 EST).** All 99 held-out haystacks:
- Extraction: 9,159 calls for 4,742 sessions, 1.93 per session (bar B9 ≤ 2.5).
- Replies: 89 unparseable (1.0%) and 331 repaired (3.6%); no thinking text and no empty replies.
- Time: 94 h of ingest summed over parallel workers.
- Check: A5 without its facts equals the stored A3 context on all 99, after the repair above.
- Published to the loop as `contexts_lme_v4.jsonl`: A5 99, A6 99, A3r 20.
- They are answered after v6's answers, per the owner's order.
