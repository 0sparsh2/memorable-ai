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
