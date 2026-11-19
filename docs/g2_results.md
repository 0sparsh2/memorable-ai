# G2, part 1: the typed graph walk as a turn signal (2026-11-19)

**We learned that a graph walk over the LLM-built graph puts the complete evidence in front of the answer model for 1.3% more held-out LoCoMo questions, at the same token budget and +7 ms per query. You should keep it on (`retrieval.graph_turn_weight=1.0`, the new default).** It is a modest gain, not the multi-hop fix. Questions whose evidence spans many sessions need aggregated memory (entity cards, the observation log), which comes next.

## What was built

Before G2, Personalized PageRank ranked only facts. Turns, which hold most of the answer evidence, had no graph signal, and the walk could not reach them.

- **Write path:** `turn -mentions-> entity` edges. A turn is linked to every entity whose name or alias it contains as a whole word.
  - A transcript speaker label ("Caroline: …") is not a mention, and neither is the user's own name.
  - Links are made when an episode closes, for its turns, and when a name first appears in the graph, for earlier turns found through the turn FTS index (`KnowledgeGraph.link_turn_mentions`, `backfill_turn_mentions`).
- **Read path:** `KnowledgeGraph.turn_walk`, a PPR over facts, entities and turns.
  - It follows `about`, `derived_from`, `mentions`, `relates` and `alias_of`, all at weight 1.0.
  - Seeds are the query's entities plus the top 10 first-stage turns at weight 0.05, the way HippoRAG 2 seeds passages next to entities.
  - `recall()` z-scores the walk's turn scores and adds them to the first stage (`graph_turn_weight`). Turns found only by the walk join the candidate pool, and the reranker orders the result as before.

## How it was measured

- **Data:** the LLM-built graphs of the v4.2 build (Nemotron Bolt extraction, all 10 LoCoMo conversations), so no new LLM calls.
- **Metric:** the share of questions with *all* gold evidence turns in the top k (and with *any*). Paired bootstrap against A3 (score fusion + Ettin-17M rerank), the configuration the QA eval answers from.
- **Split:**
  - Every choice was made on the dev conversations (conv-48, conv-49; prereg v5).
  - The setting was frozen before one run on the other 8 conversations, and nothing was tuned after it (`benchmarks/graph/g2_turn_ppr.py`).

## Results

| | Questions | all@10 | any@10 | all@20 | any@20 |
|---|---|---|---|---|---|
| Dev, experiment: A3 → G2 | 347 | 63.1 → 65.1 (+2.0, CI 0.0…+4.3) | +2.0 | 69.5 → 73.2 (+3.7, CI +1.2…+6.6) | +2.6 |
| **Held out, frozen setting** | 1,189 | **68.0 → 69.3 (+1.3, CI +0.4…+2.2)** | +0.4 (CI −0.4…+1.3) | 72.9 → 73.9 (+1.0, CI −0.1…+2.1) | +0.3 |
| Dev, shipped code path (`recall()`) | 347 | 62.8 → 64.8 (+2.0, CI 0.0…+4.3) | +1.7 | 69.5 → 72.6 (+3.2, CI +0.6…+5.8) | +2.3 |

- **Per category (held out, all@10):**
  - multi-hop 54 → 59 of 224
  - temporal 190 → 196 of 246
  - single-hop 539 → 543 of 650
  - open-domain 26 → 26
- **Cost:**
  - Query p50 65.6 → 72.9 ms (+7 ms, mostly the entity match).
  - Linking: about 0.2 s per conversation of ~45 episodes (~1,600 edges).
- **Variants tried on dev, all within noise of each other:**
  - Fuse weights 0.5–2.0.
  - Reserved graph slots (2–3 of the top k).
  - An `episode` edge kind (`part_of` at 0.3).
  - The walk without `mentions` edges: +1.7 / +2.3 at k=10 / 20, against +2.0 / +3.7 with them.
  - The simplest consistently positive setting was frozen rather than the best cell.

## Without an LLM

A user without an LLM gets a sparse graph: heuristic entities and episode mentions, about 250 turn mentions per conversation against about 1,600 in the LLM-built graphs. On fresh LLM-free ingests of the dev conversations (shipped code, A3 settings, 347 questions):

| all@10 | any@10 | all@20 | any@20 |
|---|---|---|---|
| 62.8 → 63.4 (+0.6, CI −0.9…+2.3) | +0.3 (CI −1.4…+2.0) | −0.3 (CI −2.3…+1.7) | −1.2 (CI −3.2…+0.9) |

There is no measurable effect either way, for +6.5 ms. The default stays on: the walk helps where the graph has substance and is neutral where it does not.

That run also found a write-path gap. Entities first named in an episode summary (`ingest_episode`) did not backfill earlier turns; only fact-created entities did. A final linking pass still found 263 missing links in the two conversations. Fixed: episode-created entities now backfill too. Afterwards a final pass found 8 missing links in conv-48.

## Not done yet (rest of G2)

- **Entity cards in L4:** the summary and current relations of the query's top entities. That is aggregated memory, which suits the multi-hop questions this walk barely moves, and it shows in answers, not in turn coverage.
- **`as_of` queries:** validity-filtered walks.
- **Typed walks for episodes and facts:** the per-kind lists of the design (§5). Today only facts (old) and turns (new) get a graph list.

The end-to-end check is the QA iteration slice, once prereg v5 has picked the answer model.
