# Results, prereg v6 (the observation log)

## Takeaways

1. **The observation log is the largest gain so far, and it comes from what the log says. Make it part of the default memory context.** A7 (A3g plus this conversation's log, up to 30k tokens) beats A3g by +11.5 on LoCoMo-dev J-strict (CI +6.9…+16.1) and +6.7 on LME-dev J (CI +0.7…+12.7). B15 passes on both benchmarks. Another conversation's log (D7) adds nothing (0.0 and −3.3, B16), so the gain is not from extra tokens or the format. The observer meets its cost bar (B17): 1.18 calls per session against a budget of 2.5, 0 of 8,536 replies unparseable, and no thinking text.
2. **On LoCoMo-dev, memory now scores above the published best. That is a lead, not a verdict.** A7 scores 96.0 J-lenient (CI 94–98), against the 93.6 Agent Zero Memory reports (self-reported, with a different answer model). On J-strict it scores 82.7, against 86.2 for the whole conversation (A0): A7 − A0 is −3.5 (CI −7.2…0.0), on 55% of the prompt (12.3k tokens against 22.3k). These are the two dev conversations every prereg since v5 has looked at. The claim needs the eight held-out conversations, run with the configuration frozen.
3. **On LME, A7 is 82.0 against the published 95.6. Preference questions cause more than half of the misses.** 15 of A7's 27 LME misses are single-session-preference (40%, n=25). Multi-session and temporal-reasoning are at 84 each, and the other three categories are at 92–96. A7's LME prompts are 26.3k tokens (median; 2 of 150 are over 30k). That is near LoCoMo's 22k, where Ultra does well, so the reader's context length is not the likely cause, though that is untested. The next LME prereg should target preference questions first.
4. **The graph walk neither helps nor hurts (B18).** A3g − A3 is −0.6 on LoCoMo (CI −3.2…+2.3) and +0.7 on LME (CI −4.7…+5.3). It stays as A7's base because it costs nothing measurable, not because it is a gain.

Also recorded:
- **Cap deviation:** the observer ingest took 95 h of wall clock against a 72 h cap. The Mac was asleep for 57.2 h of it, mostly with the lid closed, and awake for 37.9 h, inside the pilot's band (22.5–38.6 h). No setting changed (pilot-notes-v6.md, "Build done").
- **One D7 row was rebuilt on MPS** because its CPU-built excerpts differed from A3g's (1 of 150 LME-dev haystacks). After the repair, every A7 and D7 row equals A3g plus its log.
- **Throughput:** once the observer had finished, Ultra answered and Lightning judged the 994 A7/D7 rows in 3.7 h (4.4 answers/min). Stage 1, sharing Ultra with the observer, ran at 1.4/min.
- **Held-out cost:** at the dev rate, observing the 350 held-out LME haystacks takes about 88 h of awake time on the NIM free tier.

Reader: **ultra** (v5's B11 choice). A3 and A0 rows are v5's answers.

## Cells

| Bench | Context | J-strict | J-lenient | J (LME) | unparsed |
|---|---|---|---|---|---|
| lme | A0 | — | — | 56.2 [42–71] (n=48) | 0 |
| lme | A3 | — | — | 74.7 [67–81] (n=150) | 0 |
| lme | A3g | — | — | 75.3 [68–82] (n=150) | 0 |
| lme | A3r | — | — | 65.0 [45–85] (n=20) | 0 |
| lme | A7 | — | — | 82.0 [76–88] (n=150) | 0 |
| lme | D7 | — | — | 72.0 [65–79] (n=150) | 0 |
| locomo | A0 | 86.2 [82–90] (n=347) | 98.3 [97–99] (n=347) | — | 0 |
| locomo | A3 | 71.8 [67–76] (n=347) | 89.6 [86–93] (n=347) | — | 0 |
| locomo | A3g | 71.2 [67–76] (n=347) | 91.1 [88–94] (n=347) | — | 0 |
| locomo | A3r | 62.5 [48–78] (n=40) | — | — | 0 |
| locomo | A7 | 82.7 [78–87] (n=347) | 96.0 [94–98] (n=347) | — | 0 |
| locomo | D7 | 71.2 [66–76] (n=347) | — | — | 0 |

## Bars

| Bar | Result | Verdict |
|---|---|---|
| B15 A7 − A3g | LoCoMo-dev J-strict +11.5 pts (CI +6.9…+16.1, n=347); LME-dev J +6.7 pts (CI +0.7…+12.7, n=150) | pass |
| B16 D7 − A3g | LoCoMo-dev J-strict +0.0 pts (CI -2.3…+2.3, n=347); LME-dev J -3.3 pts (CI -8.7…+2.0, n=150) | pass |
| B18 A3g − A3 | LoCoMo-dev J-strict -0.6 pts (CI -3.2…+2.3, n=347); LME-dev J +0.7 pts (CI -4.7…+5.3, n=150) | pass |
| B17 observer | 1.18 calls per session; 0 of 8536 unparseable; 0 with thinking | pass |
| Drift (A3r vs v5's A3) | locomo 98% of 40; lme 95% of 20 | ok |

## Reported, no bar

- **A7 − A0, locomo:** -3.5 pts (CI -7.2…+0.0, n=347)
- **A7 − A0, lme:** +22.9 pts (CI +10.4…+35.4, n=48)
- **Context tokens, locomo A3g:** median 825, max 1,081
- **Context tokens, locomo A7:** median 11,531, max 11,894
- **Context tokens, locomo D7:** median 10,210, max 11,878
- **Context tokens, lme A3g:** median 3,169, max 4,820
- **Context tokens, lme A7:** median 28,257, max 33,591
- **Context tokens, lme D7:** median 28,250, max 32,484

| Bench | Category | A3 | A3g | A7 | A0 |
|---|---|---|---|---|---|
| lme | knowledge-update | 80.0 [64–92] (n=25) | 76.0 [60–92] (n=25) | 92.0 [80–100] (n=25) | 75.0 [50–100] (n=8) |
| lme | multi-session | 68.0 [48–84] (n=25) | 68.0 [48–84] (n=25) | 84.0 [68–96] (n=25) | 50.0 [12–88] (n=8) |
| lme | single-session-assistant | 96.0 [88–100] (n=25) | 92.0 [80–100] (n=25) | 96.0 [88–100] (n=25) | 87.5 [62–100] (n=8) |
| lme | single-session-preference | 32.0 [16–52] (n=25) | 48.0 [28–68] (n=25) | 40.0 [20–60] (n=25) | 12.5 [0–38] (n=8) |
| lme | single-session-user | 96.0 [88–100] (n=25) | 96.0 [88–100] (n=25) | 96.0 [88–100] (n=25) | 75.0 [38–100] (n=8) |
| lme | temporal-reasoning | 76.0 [60–92] (n=25) | 72.0 [56–88] (n=25) | 84.0 [68–96] (n=25) | 37.5 [12–75] (n=8) |
| locomo | multi-hop | 53.4 [41–66] (n=58) | 48.3 [34–60] (n=58) | 72.4 [60–84] (n=58) | 74.1 [62–84] (n=58) |
| locomo | open-domain | 69.6 [52–87] (n=23) | 65.2 [43–83] (n=23) | 69.6 [52–87] (n=23) | 73.9 [57–91] (n=23) |
| locomo | single-hop | 76.4 [70–82] (n=191) | 77.0 [71–83] (n=191) | 85.3 [80–90] (n=191) | 90.6 [86–94] (n=191) |
| locomo | temporal | 74.7 [64–84] (n=75) | 76.0 [65–85] (n=75) | 88.0 [80–95] (n=75) | 88.0 [80–95] (n=75) |

- **Log size:** median 25,138 tokens, max 30,677, over 152 logs; the 30,000-token budget cut lines in 1.
- **Observer:** 8,536 calls over 152 logs, 18,240,133 prompt and 7,508,082 output tokens, 2660 retries, 74.1 h of ingest.
