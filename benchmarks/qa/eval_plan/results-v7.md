# Results, prereg v7 (the answer prompt)

## Takeaways

1. **From now on, answer with P1, the targeted prompt.** Over today's prompt (P0) on A7's contexts, P1 gains +6.7 on LME-dev J (CI +1.3…+12.7) and +5.5 on LoCoMo-dev J-strict (CI +2.6…+8.9). It costs nothing measurable on the false-answer guard: the share of D8 questions it declines is 96.8 against 97.6 (−0.8, CI −2.4…0.0). Preference questions go from 40 to 80. P1 keeps "short answer, say you don't know" for facts and answers advice requests from what memory says about the user. B19 and B20 pass, and B21 picks P1.
2. **Dropping both rules (P2) also works, but less well on LME and with less caution.** It gains as much on LoCoMo (+6.3, CI +3.2…+9.5). On LME it gains +4.7, with a CI that crosses 0, and it trails P1 by 2.0 (CI −6.7…+2.7). On D8 it declines 2.4 points less often than P0, and its answers are about 1.7 times as long as P1's (428 against 250 characters on LME). B21's tie rule picks P1.
3. **On dev, LoCoMo is past the published best; LME is 6.9 points short.** A7·P1 scores 97.7 J-lenient on LoCoMo-dev (published best 93.6) and 88.2 J-strict. On LME-dev it scores 88.7 against 95.6. Its 17 LME misses are: preference 5, temporal-reasoning 4, multi-session 3, knowledge-update 3, single-session-assistant 2. Apart from preference, these need date arithmetic, counting or the latest value. Ultra already reasons before it answers, though briefly: a median of 465 output tokens for an 83-token answer. So the next prereg tests deeper reasoning (LongMemEval's own step-by-step reader prompt) and stronger answer models, not thinking as such.

Also recorded:
- **Drift:** A7r agrees with v6's A7 verdicts on 92% of 40 LoCoMo and 90% of 20 LME questions. LME is exactly at the 90% bar, so the check passes with no margin.
- **The run:** 1,429 answers in 6.0 h of loop time, about 4 answers/min. It had one read-timeout stop, which `answer-loop` resumed; there were no other errors and 0 unparsed verdicts.
- **P1 still refuses on 9 of 150 LME and 8 of 347 LoCoMo questions** (`REFUSAL` pattern), against 14 and 11 for P0.

Reader: **ultra**. A7·P0 rows are v6's A7 answers (same contexts, reader and judge).

## Cells

| Bench | Arm | J-strict | J-lenient | J (LME) | refuses | median chars | unparsed |
|---|---|---|---|---|---|---|---|
| lme | A7p0 | — | — | 82.0 [76–88] (n=150) | 14 of 150 | 142 | 0 |
| lme | A7p1 | — | — | 88.7 [83–93] (n=150) | 9 of 150 | 250 | 0 |
| lme | A7p2 | — | — | 86.7 [81–92] (n=150) | 4 of 150 | 428 | 0 |
| lme | A7r | — | — | 85.0 [70–100] (n=20) | 3 of 20 | 298 | 0 |
| lme | D8p0 | — | — | 97.6 [94–100] (n=125) | 115 of 125 | 176 | 0 |
| lme | D8p1 | — | — | 96.8 [94–99] (n=125) | 113 of 125 | 318 | 0 |
| lme | D8p2 | — | — | 95.2 [91–98] (n=125) | 86 of 125 | 654 | 0 |
| locomo | A7p0 | 82.7 [78–87] (n=347) | 96.0 [94–98] (n=347) | — | 11 of 347 | 122 | 0 |
| locomo | A7p1 | 88.2 [85–91] (n=347) | 97.7 [96–99] (n=347) | — | 8 of 347 | 228 | 0 |
| locomo | A7p2 | 89.0 [86–92] (n=347) | 98.9 [97–100] (n=347) | — | 3 of 347 | 449 | 0 |
| locomo | A7r | 85.0 [75–95] (n=40) | 100.0 [100–100] (n=40) | — | 1 of 40 | 113 | 0 |

## Bars

| Bar | Result | Verdict |
|---|---|---|
| B19 A7·P1 − A7·P0 | LoCoMo-dev J-strict +5.5 pts (CI +2.6…+8.9, n=347); LME-dev J +6.7 pts (CI +1.3…+12.7, n=150) | pass |
| B19 A7·P2 − A7·P0 | LoCoMo-dev J-strict +6.3 pts (CI +3.2…+9.5, n=347); LME-dev J +4.7 pts (CI -0.7…+10.7, n=150) | pass |
| B20 D8·P1 − D8·P0 (declined) | LME-dev -0.8 pts (CI -2.4…+0.0, n=125) | pass |
| B20 D8·P2 − D8·P0 (declined) | LME-dev -2.4 pts (CI -5.6…+0.0, n=125) | pass |
| B21 choice | P2 − P1 on LME-dev J -2.0 pts (CI -6.7…+2.7, n=150) | prompt = p1 |
| Drift (A7r vs v6's A7) | locomo 92% of 40; lme 90% of 20 | ok |

## Reported, no bar

| Bench | Category | A7·P0 | A7·P1 | A7·P2 |
|---|---|---|---|---|
| locomo | multi-hop | 72.4 [60–84] (n=58) | 84.5 [74–93] (n=58) | 84.5 [74–93] (n=58) |
| locomo | open-domain | 69.6 [52–87] (n=23) | 78.3 [61–91] (n=23) | 78.3 [61–96] (n=23) |
| locomo | single-hop | 85.3 [80–90] (n=191) | 89.5 [85–94] (n=191) | 91.1 [87–95] (n=191) |
| locomo | temporal | 88.0 [80–95] (n=75) | 90.7 [84–97] (n=75) | 90.7 [84–96] (n=75) |
| lme | knowledge-update | 92.0 [80–100] (n=25) | 88.0 [72–100] (n=25) | 88.0 [76–100] (n=25) |
| lme | multi-session | 84.0 [68–96] (n=25) | 88.0 [72–100] (n=25) | 84.0 [68–96] (n=25) |
| lme | single-session-assistant | 96.0 [88–100] (n=25) | 92.0 [80–100] (n=25) | 92.0 [80–100] (n=25) |
| lme | single-session-preference | 40.0 [20–60] (n=25) | 80.0 [64–92] (n=25) | 76.0 [60–92] (n=25) |
| lme | single-session-user | 96.0 [88–100] (n=25) | 100.0 [100–100] (n=25) | 96.0 [88–100] (n=25) |
| lme | temporal-reasoning | 84.0 [68–96] (n=25) | 84.0 [68–96] (n=25) | 84.0 [68–96] (n=25) |

| D8 category (declined) | D8·P0 | D8·P1 | D8·P2 |
|---|---|---|---|
| knowledge-update | 92.0 [80–100] (n=25) | 92.0 [80–100] (n=25) | 88.0 [76–100] (n=25) |
| multi-session | 100.0 [100–100] (n=25) | 96.0 [88–100] (n=25) | 96.0 [88–100] (n=25) |
| single-session-assistant | 100.0 [100–100] (n=25) | 100.0 [100–100] (n=25) | 100.0 [100–100] (n=25) |
| single-session-user | 100.0 [100–100] (n=25) | 100.0 [100–100] (n=25) | 100.0 [100–100] (n=25) |
| temporal-reasoning | 96.0 [88–100] (n=25) | 96.0 [88–100] (n=25) | 92.0 [80–100] (n=25) |
