# Pilot notes, prereg v5 (answer model)

**Pilot, 2026-11-19:** 3 items per cell, 16 cells, dev rows only (`runs/v5_pilot`). Each A0-LME cell included the longest prompt.

## Checks

- **Context length:** every reader answered the longest A0-LME prompt (~109k tokens).
- **Answers:** no empty answers. Lightning used the no-thinking fallback twice on A0-LME, as in v3.
- **Judge:** 0 of 60 verdicts unparsed.
- **Errors:** none other than timeouts. Every logged error is a ReadTimeout: 5 attributed to Mistral; the 3 from the first attempt predate the model field in the error log. The run resumed each time, as the stop rules allow.
- **D6 (no memory):** 0 of 11 correct for the strong readers (bar B14 ≤ 20%).

## Per cell (correct / answered; indicative only at n = 3)

| Cell | Lightning | Ultra | Mistral |
|---|---|---|---|
| LoCoMo A3, J-strict | 1/3 | 3/3 | 2/2 |
| LoCoMo A0, J-lenient | 2/3 | 3/3 | 1/1 |
| LME A3 | 3/3 | 3/3 | 3/3 |
| LME A0 | 2/3 | 0/3 | 1/2 |

## Latency (answer calls)

| Reader | n | p50 | p90 |
|---|---|---|---|
| Ultra | 18 | 5.4 s | 17 s |
| Lightning | 12 | 61 s | 85 s |
| Mistral | 14 | 87 s | 145 s; 5 calls past the 240 s timeout |

## Operational change (not a measurement change)

The HTTP client timeout is now configurable (`E2E_QA_HTTP_TIMEOUT`), and the full v5 run uses 420 s, because Mistral often answers after 240 s. Prompt, contexts, thinking settings, models and bars are unchanged.

## Projection

Lightning carries its own answers and every judge call, so it is the long pole. It sustains about 4 calls/min (v3), with a band of 2.5–5.

| Option | Lightning calls | Estimate (band) | Other readers, in parallel |
|---|---|---|---|
| All three readers, as preregistered | 5,846 (892 answers + 4,954 judges) | ~24 h (20–39 h) | Ultra 990 answers, ~3 h at 6 rpm. Mistral 990, ~5.5 h (4–11 h) |
| Without Mistral | 4,162 | ~17 h (14–28 h) | Ultra ~3 h |

The cap is 4 days and the cost is $0. The v3 answer loop is paused for the run.

`PILOT_PROJECTION: per_item=2.0 lightning calls total_items=2872 projected=24h band95=20h..39h cap=96h verdict=within`

## Approval

On 2026-11-19 the owner approved the full run with all three readers, as preregistered. It started at 2026-11-19 11:06 EST (`runs/v5`, answer-loop, concurrency 8, 6 rpm for the judge and 6 rpm per reader, 420 s HTTP timeout). The v3 answer loop was paused at 3,579 answers (complete lines; restart command in `runs/main/answer_loop.cmd.txt`) and resumes when v5 ends.
- **Run restart, 2026-11-19 (operational).** The first chunk stopped on `RemoteProtocolError: Server disconnected` (Mistral). A dropped connection is the same transient class as a timeout, so the harness now retries it the same way. The loop was restarted from disk; no measurement setting changed.
