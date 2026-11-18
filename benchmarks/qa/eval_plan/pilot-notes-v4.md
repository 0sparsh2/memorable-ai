# Pilot notes: prereg v4 / v4.1 (LLM facts, 80 turns)

## Dry run (2026-11-18)
Stub LLM, LoCoMo conv-26 and conv-30. All checks pass ([dry-run-v4.json](dry-run-v4.json)): 37 LLM calls per 19-session conversation; A5 without its facts reproduces the stored A3 contexts byte for byte (75/75); reruns resume from the cache.

## Attempt 1 (prereg v4), 2026-11-18: stopped
Real extraction with `nvidia/nemotron-3.5-bolt-e4b`, conv-26 and conv-30 in parallel, 6 requests/min offered.
- 70 calls in 12 min (about 5.8 calls/min): no 429s, no empty replies, no thinking text. Bolt's quota held at this rate.
- Facts kept: 65 (conv-26) and 45 (conv-30).
- **9 of 38 extraction replies failed to parse (24%)**: the stop rule (more than 5% failing) applies, and the run stopped before any answer. Eight failed at 3,200–3,700 characters, where the 1,200-token output cap cuts the JSON array mid-item; one reply was malformed early (character 230). The pipeline drops the whole reply on a parse failure, so those sessions lost their LLM facts.
- Fix and new version: [prereg-v4.1.md](prereg-v4.1.md).
