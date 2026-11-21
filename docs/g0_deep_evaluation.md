# G0 deep evaluation: testing the memory graph foundation, and the onnxruntime / numba / hypothesis study

**Date:** 2026-11-15
**Scope:**
- Step one of the v0.3 roadmap: the unified `edges` table, the CSR `GraphEngine`, tiers and provenance.
- An evaluation of three candidate dependencies, plus others found along the way.

The embedding bottleneck has its own document, [embedding_study.md](embedding_study.md).

## TL;DR

- **Testing found four real defects in G0, and all are fixed.** Three were found by Hypothesis or the new concurrency test; one was found by reading the code under load.

  | # | Defect | Found by | Impact before the fix | Fix |
  |---|---|---|---|---|
  | 1 | Engine synced invalidations by **timestamp**. A concurrent writer can commit an *older* `invalidated_at` after a newer one, and the engine missed it forever | reasoning, then confirmed with a deterministic repro (the 40-example property run had *not* found it) | stale edges stay in ranking, in multi-process deployments | **Commit-ordered write sequence** (`graph_seq`, allocated under the DB write lock) replaces timestamps |
  | 2 | A node whose edges were all invalidated kept **teleport mass** as a seed (score 1.0 in a live engine, 0 in a fresh one) | Hypothesis stateful test (300 × 40 steps) | latent today; in G2, when dense hits seed the walk, an invalidated fact would come back top-ranked | Nodes count as present only with ≥1 live edge in the view |
  | 3 | `graph_seq` / `graph_epoch` were bumped by a Python **read-modify-write**. Two writers could commit the same value, so an engine could skip one's changes | concurrency test (3 writer + 3 reader threads) | the duplicate-key error the test surfaced, plus silent missed syncs | Atomic SQL increment on an integer `memorable_counters` table; rows created at init |
  | 4 | Query-cache attributes missing on `as_embedder()` wrappers, so `encode_query` crashed for custom models | code reading during the embedding study | crash for users passing their own model object | `_init_query_cache()` on every construction path |

- **Memory:** the engine dropped from **342 to 101 bytes per edge**, CSR and view included (org graph ≈ 80 MB instead of ≈ 280 MB). How: typed arrays, packed integer node keys, and binary-searched edge ids.
- **Speed (unchanged or better):**
  - exact PPR: 0.9 ms per user, 8.7 ms per team, 64 ms per org
  - the Numba push fallback at org scale: **2.3 ms**
- **Tooling verdicts:**
  - **hypothesis:** adopted (dev extra). It earned its place by finding defect #2, and its oracle-checking found two of my own test mistakes.
  - **numba:** adopted as optional (`fast` extra). 63× / 15× / 4× faster than the Python push fallback; the default exact path doesn't need it.
  - **onnxruntime:** adopted as optional, auto-used. Covered in embedding_study.md.
- **Other things found and fixed along the way:**
  - **CI never failed on test failures:** `|| true` in the pytest step, and `.gitignore` excluded every test file. Both fixed.
  - **Disk exhaustion:** the property tests leaked about 10,000 temporary SQLite databases (3.3 GB), filling the disk until SQLite threw I/O errors. They now clean up after themselves.

## 1. What was tested

| Test | Kind | Budget | Result |
|---|---|---|---|
| PPR == networkx PageRank on random scoped, masked graphs | Hypothesis differential | 60 in CI, 500 deep | pass |
| Views never contain other users' nodes | Hypothesis property | 60 / 500 | pass |
| Long-lived engine == fresh engine across random writes, backdated invalidations, hard deletes, second-process writers | Hypothesis **stateful** | 150 × 30 in CI; 300 × 40 deep | pass after fixes #1 and #2 |
| Re-sending a growing history stores exactly that history (duplicates allowed) | Hypothesis property | 60 / 500 | pass |
| Out-of-order invalidation; dead-node seed | Deterministic regression tests | — | pass |
| 3 writer + 3 reader threads, then live == fresh | Concurrency | 6 repeated runs | pass after fix #3 |
| Exactness vs networkx, exclusion = induced subgraph, push ≈ exact, tier isolation, delta sync + epoch, provenance, cascade, legacy backfill, agent_id flow, forget_user | Unit | — | pass |
| LoCoMo retrieval regression (10 conversations, 1,536 questions) | End-to-end | — | recall identical to v0.2 |

Deep runs use `MEMORABLE_PROP_EXAMPLES=500 pytest tests/property`. The suite has 136 tests and runs in about 15 s with the hash embedder (also green with real models).

**Lesson about property testing.** The first 40-example stateful run passed even though defect #1 existed. Random exploration didn't produce invalidate → query → *backdated* invalidate → query within that budget. Two practices follow:
- The budget is now configurable and the CI default is higher.
- Every bug found gets a deterministic regression test.

## 2. Memory footprint (team graph: 134k edges, 59k nodes)

| Version | Bytes per edge (incl. CSR and view) | Change |
|---|---|---|
| Python lists of str/int objects | 342 | — |
| Typed arrays (`array('i'/'B'/'f')`) | 248 | kind strings and int objects gone |
| + packed int node keys (`kind << 48 \| id`), binary-searched edge ids (a dict only for out-of-order ids) | **101** | 3.4× smaller in total |

Side effect: matrix builds now read the buffers with `np.frombuffer` instead of converting Python objects.

## 3. numba

**Candidates:**
- the local-push PPR fallback (pure Python)
- BM25 scoring in `LexicalIndex`
- value-flow matching (not built yet)

BM25 was rejected. It is only the fallback when SQLite FTS5 is unavailable, and it is dict-of-dicts code that Numba handles poorly. A scipy term-document matrix is the right tool if it ever matters.

**Push-PPR kernel** ([_push.py](../memorable_ai/graph/_push.py)): JIT with `cache=True`, falling back to pure Python without Numba. Two fixes after the first measurement:
- kernel-ready arrays are cached per view (an org-sized float64 conversion had been happening on every call)
- the queue starts only from the seeds, not from a scan of every node

| Graph | Exact (default) | Numba push ε=1e-5 | Python push | Numba vs Python |
|---|---|---|---|---|
| per-user | 0.9 ms | 1.4 ms | 69.6 ms | 50× |
| team | 8.7 ms | 0.8 ms | 20.2 ms | 25× |
| org | 64 ms | **2.3 ms** | 9.9 ms | 4× |

First call, including JIT and cache: 0.2–0.3 s.

**Verdict:**
- Exact stays the default, because it is quality-first: overlap with exact is 0.86–0.89 for push vs 0.91–0.95 for exact.
- Numba makes the large-graph fallback cheap: 28× faster than exact at org scale.
- An optional dependency is right.

## 4. hypothesis

**Verdict: adopt** (dev extra; runs in CI). It found a real, subtle defect (#2), and reasoning plus a deterministic repro found #1. It also exposed two mistakes in my own oracles:
- A networkx reference built from edges only can't represent an isolated-but-present node.
- A 1e-9 tolerance is below float32 noise; it's now 1e-5.

That second point matters: a property test is only as good as its reference model.

## 5. Other instruments examined

| Instrument | Result |
|---|---|
| ONNX Runtime **CoreML** provider (Apple Neural Engine/GPU) | Ruled out: only 294 of 418 graph nodes are supported, and partitioning overhead makes it 10–30× slower |
| PyTorch **MPS** (Apple GPU) | 1.75× faster on bulk batches, 3× slower on single queries. Useful only for re-index or backfill jobs |
| sentence-transformers' official ONNX extra (`optimum-onnx`) | **Cannot install** with transformers 5.x on Python 3.14. Our ONNX path uses `onnxruntime` + `tokenizers` directly |
| Python 3.14 free-threading | Not tested; the environment is a GIL build. Still a candidate (see instruments_analysis.md) |
| pytest-benchmark, py-spy, scalene | Not needed so far: cProfile plus the purpose-built benchmarks answered every question |

## 6. Reproduce

```bash
pytest tests/                                            # 136 tests, ~15 s
MEMORABLE_PROP_EXAMPLES=500 pytest tests/property        # deep property run
MEMORABLE_LLM__DEFAULT_MODEL=none python benchmarks/graph/ppr_engines.py
```
