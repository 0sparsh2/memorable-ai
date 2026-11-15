# Memorable Benchmarks

## Overview

This document describes benchmark results and methodology for evaluating Memorable's performance.

## Benchmark Suite

### 1. LOCOMO Benchmark

**Purpose**: Evaluate long-term conversation memory capabilities.

**Reference**: "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"
- arXiv:2504.19413 (April 2025)
- https://arxiv.org/abs/2504.19413

**Test Categories**:
- Single-hop questions
- Multi-hop questions
- Temporal questions
- Open-domain questions

**Results**:

| Metric | Status | Notes |
|--------|--------|-------|
| LOCOMO smoke (fixture) | **Measured** | `pytest tests/unit/test_locomo_smoke.py` |
| LOCOMO full dataset loader | **Integrated** | Auto-download `benchmarks/locomo/data/locomo10.json` |
| LOCOMO subset (1 conv × 15 Q, NVIDIA NIM) | **Measured** | `scripts/run_locomo_benchmark.py --mode full --conversations 1 --questions 15` — see `benchmarks/locomo/data/measured_subset.json` |
| Full LOCOMO leaderboard (1540 Q) | **In progress** | Harness ready; full run is API-heavy (~hours) |
| 30%+ vs Mem0 | **Target** | Not yet validated on full benchmark |
| Single-hop / multi-hop / temporal | **Target** | Subset run showed retrieval/judge tuning needed |

**Targets** (not yet validated on full LOCOMO):
- Overall accuracy: >75%
- Single-hop: >85%
- Multi-hop: >70%
- Temporal: >65%
- Open-domain: >60%

**Methodology** (Mem0-comparable pipeline):
1. Download [snap-research/locomo](https://github.com/snap-research/locomo) `locomo10.json`
2. Per conversation: ingest turns with session timestamps (`benchmarks/locomo/loader.py`)
3. Per question: hybrid retrieval → LLM short answer → LLM judge (categories 1–4)
4. Secondary metrics: token F1 (Snap official), evidence `dia_id` recall

**Run commands**:
```bash
# Smoke (CI, no download)
python scripts/run_locomo_benchmark.py --mode smoke

# Subset (tune cost/latency)
python scripts/run_locomo_benchmark.py --mode full --conversations 1 --questions 50

# Retrieval-only (no answer/judge LLM)
python scripts/run_locomo_benchmark.py --mode full --conversations 1 --questions 20 --no-llm
```

**NVIDIA NIM timeouts (answer step)** — If `litellm` raises `APITimeoutError` or answers come back empty, the usual causes are (a) prompt too large and (b) client timeout shorter than server latency under load.

**Mitigations** (implemented in `benchmarks/locomo/judge.py`):
- **Truncate context**: cap total chars (~4500 default), max memories (10), and **per-turn line length** (~380 chars) so multi-turn LOCOMO dialog does not blow up the prompt.
- **Retry with tighter packing**: on timeout / failure, the harness retries with half-sized context (fewer memories, shorter lines).
- **Higher QA timeout**: default answer-call timeout for NIM is **240s** (`LOCOMO_QA_LLM_TIMEOUT_NIM_DEFAULT`); override with `LOCOMO_QA_LLM_TIMEOUT=300` for noisy runs.
- **CLI tuning**: `python scripts/run_locomo_benchmark.py --mode full --qa-timeout 300 --qa-context-chars 3500 --qa-context-memories 8 --qa-context-line-chars 320 ...`

**Measured subset (2026-05-17)** — 1 conversation × 10 questions, `deepseek-ai/deepseek-v4-flash`, `LOCOMO_QA_LLM_TIMEOUT=300`, `LOCOMO_JUDGE_LLM_TIMEOUT=180`:

| Experiment | Ingest | Context | Judge acc | Notes |
|------------|--------|---------|-----------|-------|
| `baseline_raw` | raw | minimal | **10%** | Initial run; `measured_subset.json` |
| `dated_context` | raw | dated | **40%** | Session dates in QA prompt |
| `oracle` | gold `dia_id` | dated | **60%** | Perfect retrieval ceiling |
| `retrieval_only_dated` | raw | dated | **80%** | `--no-llm` evidence recall |

Run ablations: `python scripts/run_locomo_experiments.py --experiments dated_context,oracle --questions 10`. Artifacts: `ResearchDoc/experiments/`, `ResearchDoc/experiment_log.md`.

**Extracted ingest (Mem0-fair)** — many LLM calls per conversation; set `LOCOMO_EXTRACT_SESSION_DELAY=2` (seconds between sessions) to reduce NIM 429s. The client retries 429 with exponential backoff automatically. Extracted facts are stored **directly** (no update-phase LLM) with `dia_ids` from source turns; empty extraction falls back to raw turns for that session.

**Hybrid ingest** — raw turns plus extracted facts (`hybrid_ingest` experiment); expected to match or beat `dated_context` alone.

Earlier run (2026-05-15) showed 0% judge accuracy when judge client timeout was capped at 90s while NIM needed 4+ minutes per call.

**Debugging**: use `--no-llm` to measure evidence `dia_id` recall without answer/judge LLM calls.

### 2. Multi-hop Reasoning Benchmark

**Purpose**: Test graph-based relationship traversal.

**Test Cases**:
- Entity relationship following
- Multi-step reasoning
- Graph traversal accuracy

**Results** (Target):
- Accuracy: >90%
- Average path length: 2-3 hops

**Methodology**:
1. Setup knowledge graph with entities and relationships
2. Ask questions requiring multi-hop reasoning
3. Measure accuracy of graph traversal

### 3. Temporal Coherence Benchmark

**Purpose**: Test temporal memory tracking.

**Reference**: "Highly engaging events reveal semantic and temporal compression in online community discourse"
- PNAS Nexus (March 2025)

**Test Cases**:
- Time-stamped memory retrieval
- Temporal sequence following
- Coherence checking

**Results** (Target):
- Temporal accuracy: >85%
- Coherence score: >90%

**Methodology**:
1. Create time-stamped memories
2. Query memories by time range
3. Check temporal coherence
4. Measure accuracy

## Performance Metrics

### Latency

**Target**: <100ms retrieval latency

**Measurement**:
- Time from query to memory retrieval
- Average across 1000 queries

### Token Efficiency

**Target**: 50%+ token savings vs full-context

**Measurement**:
- Compare tokens used with Memorable vs full-context injection
- Measure on LOCOMO benchmark

### Scalability

**Target**: Handle 10,000+ memories efficiently

**Measurement**:
- Query latency vs memory count
- Storage efficiency

## Comparison with Other Systems

### vs Mem0

**Mem0 Results** (from paper):
- 26% improvement over OpenAI memory
- 91% latency reduction
- 90% token savings

**Memorable Target**:
- 30%+ improvement over Mem0
- <100ms retrieval latency
- 50%+ token savings

### vs Memori

**Comparison Points**:
- Integration ease (both use interceptors)
- Performance (Memorable adds graph support)
- Research validation (Memorable has benchmarks)

### vs Supermemory

**Comparison Points**:
- Graph architecture (both support graphs)
- Performance claims (Supermemory: 25× faster, unverified)
- Research validation (Memorable has benchmarks)

## Running Benchmarks

### LOCOMO Benchmark

```python
from memorable import MemoryEngine
from memorable.benchmarks.locomo import LOCOMOBenchmark

memory = MemoryEngine(database="sqlite:///benchmark.db")
memory.enable()

benchmark = LOCOMOBenchmark(memory)
results = await benchmark.run()
print(results)
```

### Multi-hop Benchmark

```python
from memorable.benchmarks.multihop import MultiHopBenchmark

benchmark = MultiHopBenchmark(memory)
results = await benchmark.run()
print(results)
```

### Temporal Benchmark

```python
from memorable.benchmarks.temporal import TemporalBenchmark

benchmark = TemporalBenchmark(memory)
results = await benchmark.run()
print(results)
```

## Results Publication

Benchmark results will be published in:
- Research paper (planned)
- GitHub repository
- Documentation updates

## References

- Mem0 Paper: https://arxiv.org/abs/2504.19413
- LOCOMO Benchmark: Reference implementation from Mem0
- MemOS Dataset: https://huggingface.co/datasets/MemTensor/MemOS_eval_result

