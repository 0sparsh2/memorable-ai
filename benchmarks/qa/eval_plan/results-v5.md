# Results v5: the answer model

## Takeaways

1. **A larger answer model adds 7–10 points to memory answers on both benchmarks. Use Nemotron 3 Ultra.** Over Lightning on A3 (current memory), Ultra adds +9.2 on LoCoMo-dev J-strict (CI +5.5…+13.0) and +6.7 on LME-dev J (CI +1.3…+12.7). Mistral Large 4 adds +9.8 and +10.0. Both qualify (B11). Mistral's lead is inside the noise on both benchmarks: +0.6 (CI −2.3…+3.5) and +3.3 (CI −2.0…+9.3). So the rule picks the faster reader, Ultra (median 31 s per answer against Mistral's 82 s). v6 and later preregs answer with Ultra.
2. **On LoCoMo, the reader no longer limits us; memory does.** With the whole conversation, Ultra scores 98.3 J-lenient (B12, bar 94). With A3's retrieved turns it scores 89.6, against the published best of 93.6. On LME, A3 with Ultra is 74.7 J against a published 95.6 (corrected 2026-11-23 from 75.2 to match the cell table). The next gains have to come from what memory puts in front of the reader (v6's observation log).
3. **Ultra handles very long contexts badly.** On LME's full ~115k-token history, Ultra scores 56.2 and Mistral 83.3 (+27.1, CI +14.6…+39.6, n=48). Ultra is fine at LoCoMo's ~22k (98.3), and beats Mistral on LoCoMo J-lenient by 3.2 (CI 0.6…6.0). Don't use Ultra for full-context baselines near 100k tokens. If a later arm hands the reader a context much over 30k tokens, test Mistral on it.
4. **Throughput misses B13 for every reader (0.88–0.95 answers/min against a bar of 2).** As noted before any accuracy was seen, the pace is set by the shared Lightning judge's quota, not by the readers. By B13's own rule ("iterate on the fastest qualifying reader") that is still Ultra.

Also recorded:
- **D6:** with no memory, every reader scores 0%, so the judge is not fooled (B14).
- **Drift:** v5's Lightning verdicts agree with v3's on 83–91% of the same questions. This is reported without a bar. B11's Lightning baseline was answered inside this run, so drift does not touch it. It matters only when comparing with v3's numbers.
- **C2 (the Ultra judge) is deferred.** Its purpose was to free Lightning's quota by moving the judge to Ultra. With Ultra now the reader and v6's observer, Ultra's quota is the scarce one, so C2's calls would slow both. It can run in a quiet window, or against a different judge model in a new prereg.

## Cells

| Bench | Context | Reader | J-strict | J-lenient | J (LME) | unparsed | fallbacks |
|---|---|---|---|---|---|---|---|
| lme | A0 | lightning | — | — | 58.3 [44–73] (n=48) | 0 | 19 |
| lme | A0 | mistral | — | — | 83.3 [73–94] (n=48) | 0 | 0 |
| lme | A0 | ultra | — | — | 56.2 [42–71] (n=48) | 0 | 2 |
| lme | A3 | lightning | — | — | 68.0 [60–75] (n=150) | 0 | 2 |
| lme | A3 | mistral | — | — | 78.0 [71–85] (n=150) | 0 | 0 |
| lme | A3 | ultra | — | — | 74.7 [67–81] (n=150) | 0 | 0 |
| lme | D6 | mistral | — | — | 0.0 [0–0] (n=48) | 0 | 0 |
| lme | D6 | ultra | — | — | 0.0 [0–0] (n=48) | 0 | 0 |
| locomo | A0 | lightning | 69.2 [64–74] (n=347) | 90.2 [87–93] (n=347) | — | 0 | 46 |
| locomo | A0 | mistral | 87.9 [84–91] (n=347) | 95.7 [93–98] (n=347) | — | 0 | 0 |
| locomo | A0 | ultra | 86.2 [82–90] (n=347) | 98.3 [97–99] (n=347) | — | 0 | 0 |
| locomo | A3 | lightning | 62.5 [57–68] (n=347) | 83.9 [80–88] (n=347) | — | 0 | 4 |
| locomo | A3 | mistral | 72.3 [67–77] (n=347) | 86.5 [83–90] (n=347) | — | 0 | 0 |
| locomo | A3 | ultra | 71.8 [67–76] (n=347) | 89.6 [86–93] (n=347) | — | 0 | 0 |
| locomo | D6 | mistral | 0.0 [0–0] (n=50) | — | — | 0 | 0 |
| locomo | D6 | ultra | 0.0 [0–0] (n=50) | — | — | 0 | 0 |

## Bars

| Bar | Result | Verdict |
|---|---|---|
| B11 ultra − lightning, A3 | LoCoMo-dev J-strict +9.2 pts (CI +5.5…+13.0, n=347); LME-dev J +6.7 pts (CI +1.3…+12.7, n=150) | qualifies |
| B11 mistral − lightning, A3 | LoCoMo-dev J-strict +9.8 pts (CI +6.3…+13.3, n=347); LME-dev J +10.0 pts (CI +4.0…+16.0, n=150) | qualifies |
| B11 choice | qualifiers: ultra, mistral | reader = ultra |
| B12 A0 LoCoMo-dev J-lenient (ultra) ≥ 94.0 | 98.3 [97–99] (n=347) | pass |
| B13 throughput (ultra) ≥ 2 answers/min | 0.95 /min, p50 31.3 s | miss |
| B14 D6 no memory, ultra ≤ 20% | LoCoMo 0.0 [0–0] (n=50); LME 0.0 [0–0] (n=48) | pass |
| B14 D6 no memory, mistral ≤ 20% | LoCoMo 0.0 [0–0] (n=50); LME 0.0 [0–0] (n=48) | pass |

## Throughput

| Reader | answers | per min (unblocked) | p50 s | p95 s |
|---|---|---|---|---|
| lightning | 895 | 0.89 | 103.6 | 174.2 |
| ultra | 994 | 0.95 | 31.3 | 394.1 |
| mistral | 991 | 0.88 | 82.2 | 591.8 |

## Drift against the v3 run (Lightning, same questions)

- locomo/A3: 88% verdict agreement (n=128)
- locomo/A0: 83% verdict agreement (n=102)
- lme/A3: 91% verdict agreement (n=70)
