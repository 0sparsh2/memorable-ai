# Pilot notes, prereg v8 (the answer model)

## Dry run and probe, 2026-11-24

- **Dry run:** `DRY_RUN_RESULT: pass` ([dry-run-v8.json](dry-run-v8.json)). The queue of 2,548 rows equals v7's A7·P1 and D8·P1 rows byte for byte, and stub answers ran through `v8_report`.
- **Probe ([probe-v8.json](probe-v8.json)), availability and latency only, on one 24k-token LME-dev prompt:**
  - MiniMax answered in 71 s and Step 5 in 4.8 s, so X = Step 5.
  - Kimi K3.1 Pro and Mistral Large 4 timed out at 600 s on the first try and answered on the retry, in 52 s and 100 s.
  - The retry rule was added to the draft before the freeze. It cannot favour any arm's accuracy, since no response was judged or read.

## Pilot, 2026-11-24/25

The prereg was frozen at 23:52 EST, when the pilot started (`runs/v8ans_pilot`). It queued 3 rows per cell, 42 in all: four arms on LoCoMo, LME and D8, plus the drift check. `answer-loop` ran with a 600 s HTTP timeout, 6 requests/min per reader and probes every 60 s. The trimmed v3/v4 loop (195 LME rows on Lightning) ran at the same time.
- **Answers:** 41 of 42 by 01:53 EST. None empty, no fallbacks, no thinking text, 0 unparsed verdicts. The last row, Mistral on LoCoMo, had a read timeout and a dropped connection (`RemoteProtocolError`) and is being retried by the loop. Mistral answered its other 8 pilot rows, so it is not dropped.
- **Drift:** U·P1r agrees with v7's A7·P1 verdicts on 3 of 3 LoCoMo and 3 of 3 LME rows.
- **Cells at n=3 (not for decisions):** every arm answered its A7 rows correctly, except one LoCoMo J-strict miss for Ultra with step-by-step. On D8 every arm declined 3 of 3, except Mistral (2 of 3).
- **Answer time, p50 (max):** Ultra P1 27 s (42), Ultra P1s 30 s (48), Step 5 37 s (56), Mistral 55 s (114), Kimi 70 s (144). Median output tokens: Ultra 448, Mistral 497, Step 5 236, Kimi 212. Step-by-step answers are about 3 times as long as P1's (711 against 248 characters on LME).
- **Wall time:** 2.0 h for 41 answers. Most of it went to the stalled Mistral row: each try waits up to 600 s before retrying.

`PILOT_PROJECTION: per_item=1 answer + 1.6 judge calls total_items=2,548 projected=11-20 h band95=9-30 h cap=3 days verdict=within`

The judge sets the pace: about 3,980 Lightning calls at v7's rate of about 6 per minute is about 11 h. The four readers run in parallel under their own per-model limits. The high end allows for stalls like the pilot's: Mistral and Kimi are slower and less reliable on the free tier, and each read-timeout stop costs the requests in flight.
