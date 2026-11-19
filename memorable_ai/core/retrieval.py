"""
Hybrid retrieval across memory layers.

Facts (L2) are ranked by fusing three signals with weighted Reciprocal Rank
Fusion (Cormack et al., 2009):

* **dense**: cosine over fact embeddings (shared in-process vector index)
* **lexical**: BM25 (SQLite FTS5, or the in-process index on other databases)
* **graph**: Personalized PageRank over the knowledge graph, seeded by entities
  in the query and by the top dense hits (HippoRAG 2 style). This recovers
  multi-hop facts that share no words with the query.

Episodes (L1) and verbatim turns (L0) are ranked separately, with dense + lexical
RRF, and fill their own slots in the context block. Mixing a 10-word fact with a
200-word excerpt in one ranked list favours whichever signal likes long text.

Every signal is filtered by namespace and user *inside* the search. The old code
filtered nothing on the injection path, so one user's memories could be injected
into another user's prompt.

The old fusion summed raw scores, so unbounded BM25 (2-15) swamped cosine (0-1).
RRF fuses ranks and does not have that problem.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from memorable_ai.core.context import ContextBuilder
from memorable_ai.core.episodic import message_text
from memorable_ai.core.scope import Scope
from memorable_ai.core.vector_index import reciprocal_rank_fusion, score_fusion, tokenize

logger = logging.getLogger(__name__)

_ANAPHORA = re.compile(r"\b(it|that|this|those|these|he|she|they|him|her|them|there|then|one)\b", re.I)
_GENERIC = ("describe me", "tell me about me", "who am i", "what do you know", "about me", "remember about me")


class HybridRetriever:
    """Retrieves facts, episodes and excerpts for a query within a user scope."""

    def __init__(
        self,
        storage: Any,
        embedding_model: Any = None,
        graph: Optional[Any] = None,
        semantic_weight: float = 0.5,
        bm25_weight: float = 1.0,
        entity_weight: float = 0.0,
        graph_weight: float = 0.6,
        knowledge_graph: Optional[Any] = None,
        rrf_k: int = 60,
        min_semantic_score: float = 0.2,
        rerank: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        fusion: str = "score",
        fusion_weights: Optional[Dict[str, float]] = None,
        rerank_depth: int = 20,
        rerank_blend: float = 0.0,
        max_context_tokens: int = 2000,
        max_facts: int = 10,
        max_episodes: int = 3,
        max_excerpts: int = 3,
        include_shared: bool = False,
        profile_facts: int = 3,
        turn_embedding_model: Any = None,
        graph_turn_weight: float = 0.0,
    ):
        from memorable_ai.core.embeddings import as_embedder, get_embedder

        self.storage = storage
        self.embedder = as_embedder(embedding_model) if embedding_model is not None else get_embedder()
        # Queries against the turn layer must use the embedder that indexed it.
        self.turn_embedder = as_embedder(turn_embedding_model) if turn_embedding_model is not None else self.embedder
        # Back-compat: callers test ``retriever.embedding_model`` for truthiness.
        self.embedding_model = self.embedder
        self.embedding_model_name = getattr(self.embedder, "model_name", str(embedding_model))
        from memorable_ai.graph.knowledge_graph import KnowledgeGraph

        if knowledge_graph is None and isinstance(graph, KnowledgeGraph):
            knowledge_graph = graph
        self.graph = graph
        self.knowledge_graph = knowledge_graph
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.entity_weight = entity_weight
        self.graph_weight = graph_weight + max(entity_weight, 0.0)
        self.rrf_k = rrf_k
        self.min_semantic_score = min_semantic_score
        self.rerank = rerank
        self.rerank_model = rerank_model
        self._reranker = None
        # Turn and episode layers: "score" = z-normalised score fusion, "rrf" = rank fusion.
        self.fusion = fusion
        self.fusion_weights = fusion_weights or {"semantic": 1.0, "lexical": 0.5}
        self.rerank_depth = rerank_depth
        self.rerank_blend = rerank_blend
        self.include_shared = include_shared
        self.max_facts = max_facts
        self.max_episodes = max_episodes
        self.max_excerpts = max_excerpts
        self.profile_facts = profile_facts
        self.graph_turn_weight = graph_turn_weight  # G2: typed graph walk as a turn signal (0 = off)
        self.context_builder = ContextBuilder(max_tokens=max_context_tokens)

    # ------------------------------------------------------------------ scope helpers

    def _scope(self, scope: Optional[Scope]) -> Scope:
        return scope or Scope(namespace=getattr(self.storage, "namespace", None))

    def _dense_floor(self) -> float:
        return self.min_semantic_score if getattr(self.embedder, "is_semantic", False) else 0.05

    # ------------------------------------------------------------------ query

    def _extract_query(self, messages: Sequence[Any]) -> str:
        """Last user message; short or anaphoric follow-ups include the previous one."""
        user_msgs: List[str] = []
        for msg in reversed(messages or []):
            if isinstance(msg, str):
                user_msgs.append(msg)
            elif isinstance(msg, dict) and msg.get("role") == "user":
                text = message_text(msg.get("content"))
                if text.strip():
                    user_msgs.append(text.strip())
            if len(user_msgs) == 2:
                break
        if not user_msgs:
            return ""
        query = user_msgs[0]
        if len(user_msgs) > 1 and (len(tokenize(query)) < 4 or _ANAPHORA.search(query)):
            query = f"{user_msgs[1]}\n{query}"
        return query[-2000:]

    def _encode(self, query: str, kind: Optional[str] = None) -> Optional[np.ndarray]:
        if kind == "turn" and self.turn_embedder is not self.embedder:
            if self.turn_embedder is None:
                return None
            try:
                return self.turn_embedder.encode_query(query)
            except Exception as e:
                logger.debug("Turn query embedding failed: %s", e)
                return None
        if self.embedder is None:
            return None
        try:
            encode_query = getattr(self.embedder, "encode_query", None)
            return encode_query(query) if encode_query else self.embedder.encode(query)
        except Exception as e:
            logger.debug("Query embedding failed: %s", e)
            return None

    # ------------------------------------------------------------------ ranking

    def rank_facts(
        self,
        query: str,
        scope: Optional[Scope] = None,
        k: int = 10,
        memory_type: Optional[str] = None,
        qvec: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        scope = self._scope(scope)
        pool = max(k * 3, 20)
        lists: Dict[str, List[int]] = {}
        if qvec is None:
            qvec = self._encode(query)
        dense_ids: List[int] = []
        if qvec is not None and self.semantic_weight > 0:
            floor = self._dense_floor()
            hits = self.storage.vector_search(
                "fact", qvec, k=pool, user_id=scope.user_id, include_shared=self.include_shared
            )
            dense_ids = [i for i, s in hits if s >= floor]
            lists["semantic"] = dense_ids
        if self.bm25_weight > 0:
            lists["lexical"] = [
                i
                for i, _ in self.storage.lexical_search(
                    "fact", query, k=pool, user_id=scope.user_id,
                    include_shared=self.include_shared, memory_type=memory_type,
                )
            ]
        if self.knowledge_graph is not None and self.graph_weight > 0:
            try:
                lists["graph"] = [
                    i
                    for i, _ in self.knowledge_graph.ppr(
                        query, scope, k=pool, seed_memory_ids=dense_ids[:3],
                        include_shared=self.include_shared,
                    )
                ]
            except Exception as e:
                logger.debug("Graph signal failed: %s", e)
        fused = reciprocal_rank_fusion(
            lists,
            {"semantic": self.semantic_weight, "lexical": self.bm25_weight, "graph": self.graph_weight},
            k=self.rrf_k,
        )
        return fused[: max(k * 2, k)]

    def _rank_layer(
        self, kind: str, query: str, qvec: Optional[np.ndarray], scope: Scope, k: int,
        pool: Optional[int] = None,
    ) -> List[int]:
        return [i for i, _ in self._rank_layer_scored(kind, query, qvec, scope, k, pool)]

    def _rank_layer_scored(
        self, kind: str, query: str, qvec: Optional[np.ndarray], scope: Scope, k: int,
        pool: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        pool = pool or max(k * 4, 12)
        dense: Dict[int, float] = {}
        if qvec is not None:
            floor = self._dense_floor()
            dense = {
                i: s
                for i, s in self.storage.vector_search(
                    kind, qvec, k=pool, user_id=scope.user_id, include_shared=self.include_shared
                )
                if s >= floor
            }
        lexical = dict(
            self.storage.lexical_search(
                kind, query, k=pool, user_id=scope.user_id, include_shared=self.include_shared
            )
        )
        # Score fusion trusts the dense margins, so it needs a semantic embedder; the hash
        # fallback's cosines are noise and stay with rank fusion (BM25-weighted).
        embedder = self.turn_embedder if kind == "turn" else self.embedder
        if self.fusion == "score" and getattr(embedder, "is_semantic", False):
            if qvec is not None:
                missing = [i for i in lexical if i not in dense]
                if missing:
                    dense.update(self.storage.vector_scores(kind, qvec, missing))
            fused = score_fusion({"semantic": dense, "lexical": lexical}, self.fusion_weights)
        else:
            lists = {"semantic": list(dense), "lexical": list(lexical)}
            fused = reciprocal_rank_fusion(
                lists, {"semantic": self.semantic_weight, "lexical": self.bm25_weight}, k=self.rrf_k
            )
        return fused

    def _with_graph_turns(
        self, query: str, scope: Scope, ranked: List[Tuple[int, float]], pool: int
    ) -> List[Tuple[int, float]]:
        """
        G2: add the typed graph walk's turn scores to the first stage. Both signals are
        z-scored over their own lists and summed (``graph_turn_weight`` on the walk); a turn
        missing from one list gets that list's minimum. Turns only the walk found join the
        pool, and the reranker orders the result as before (docs/g2_results.md).
        """
        first = dict(ranked)
        graph = dict(self.knowledge_graph.turn_walk(query, scope, first, k=pool, include_shared=self.include_shared))
        if not graph:
            return ranked

        def z(v: Dict[int, float]) -> Dict[int, float]:
            a = np.fromiter(v.values(), dtype=float)
            sd = float(a.std()) or 1.0
            return {i: (x - float(a.mean())) / sd for i, x in v.items()}

        zf, zg = z(first), z(graph)
        lo_f, lo_g = min(zf.values()), min(zg.values())
        fused = {t: zf.get(t, lo_f) + self.graph_turn_weight * zg.get(t, lo_g) for t in set(first) | set(graph)}
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    def _reranker_model(self):
        if self._reranker is None:
            from memorable_ai.core.rerank import load_reranker

            self._reranker = load_reranker(self.rerank_model)
        return self._reranker

    def _rerank(
        self,
        query: str,
        items: List[Dict[str, Any]],
        first_stage: Optional[Sequence[float]] = None,
        depth: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cross-encoder rerank of the top ``depth`` items (the rest keep their order). With
        ``rerank_blend`` > 0 and first-stage scores, the order is z(reranker) +
        blend * z(first stage), which suits rerankers trained off-domain; 0 trusts the
        reranker alone (best for the Ettin family on LoCoMo; docs/retrieval_round2.md).
        """
        if not self.rerank or len(items) < 2:
            return items
        depth = depth or len(items)
        head, tail = items[:depth], items[depth:]
        try:
            scores = np.asarray(
                self._reranker_model().predict([(query, m.get("content", "")) for m in head]), dtype=np.float64
            ).reshape(len(head))
        except Exception as e:
            logger.debug("Rerank unavailable: %s", e)
            self.rerank = False
            return items
        if self.rerank_blend > 0 and first_stage is not None:
            z = lambda v: (v - v.mean()) / (v.std() or 1.0)  # noqa: E731
            scores = z(scores) + self.rerank_blend * z(np.asarray(first_stage[: len(head)], dtype=np.float64))
        order = np.argsort(-scores, kind="stable")
        return [head[i] for i in order] + tail

    # ------------------------------------------------------------------ public API

    async def recall(
        self,
        query: str,
        scope: Optional[Scope] = None,
        k_facts: Optional[int] = None,
        k_episodes: Optional[int] = None,
        k_excerpts: Optional[int] = None,
        memory_type: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve facts, episode summaries and verbatim excerpts for ``query``."""
        scope = self._scope(scope)
        k_facts = self.max_facts if k_facts is None else k_facts
        k_episodes = self.max_episodes if k_episodes is None else k_episodes
        k_excerpts = self.max_excerpts if k_excerpts is None else k_excerpts
        result: Dict[str, List[Dict[str, Any]]] = {"facts": [], "episodes": [], "excerpts": []}
        if not query or not query.strip():
            return result
        qvec = self._encode(query)
        uid = scope.user_id
        if k_facts:
            ranked = self.rank_facts(query, scope, k=k_facts, memory_type=memory_type, qvec=qvec)
            facts = self.storage.hydrate_memories(
                [i for i, _ in ranked], user_id=uid, include_shared=self.include_shared
            )
            score_of = dict(ranked)
            for f in facts:
                f["score"] = score_of.get(f["id"], 0.0)
            if memory_type:
                facts = [f for f in facts if f.get("type") == memory_type]
            result["facts"] = self._rerank(query, facts, [f["score"] for f in facts])[:k_facts]
        if k_episodes:
            ids = self._rank_layer("episode", query, qvec, scope, k_episodes)
            eps = self.storage.hydrate_episodes(ids, user_id=uid, include_shared=self.include_shared)
            result["episodes"] = [e for e in eps if e.get("status") == "closed"][:k_episodes]
        if k_excerpts:
            tvec = qvec if self.turn_embedder is self.embedder else self._encode(query, kind="turn")
            pool = max(k_excerpts * 12, self.rerank_depth * 2)
            ranked = self._rank_layer_scored("turn", query, tvec, scope, k_excerpts * 3, pool=pool)
            if self.graph_turn_weight > 0 and self.knowledge_graph is not None and ranked:
                ranked = self._with_graph_turns(query, scope, ranked, pool)
            first = dict(ranked)
            turns = self.storage.hydrate_turns([i for i, _ in ranked], user_id=uid, include_shared=self.include_shared)
            # The current session is already in the prompt; excerpts come from other sessions.
            candidates = [
                t
                for t in turns
                if (not scope.session_id or t.get("session_id") != scope.session_id)
                and len(tokenize(t.get("content", ""))) >= 4
            ]
            candidates = self._rerank(
                query, candidates, [first.get(t["id"], 0.0) for t in candidates], depth=self.rerank_depth
            )
            # Demote turns a returned fact was extracted from (provenance): usually redundant,
            # but kept as a fallback because a turn can hold more than its extracted facts.
            covered = {tid for f in result["facts"] for tid in (f.get("source_turn_ids") or [])}
            candidates.sort(key=lambda t: t["id"] in covered)
            result["excerpts"] = candidates[:k_excerpts]
        return result

    async def profile(self, scope: Optional[Scope] = None, limit: int = 8) -> List[Dict[str, Any]]:
        """Most important active facts for the user (stable "who is this user" block)."""
        scope = self._scope(scope)
        return await self.storage.get_memories(
            limit=limit, scope_user=scope.user_id, include_shared=self.include_shared
        )

    async def retrieve(
        self, messages: List[Dict[str, Any]], limit: int = 10, scope: Optional[Scope] = None
    ) -> List[Dict[str, Any]]:
        """Facts relevant to the conversation's latest user message (compatibility API)."""
        scope = self._scope(scope)
        query = self._extract_query(messages)
        if not query:
            return await self.profile(scope, limit)
        res = await self.recall(query, scope, k_facts=limit, k_episodes=0, k_excerpts=0)
        facts = res["facts"]
        if not facts and _is_generic(query):
            facts = await self.profile(scope, limit)
        return facts

    async def get_context(
        self,
        messages: Sequence[Any],
        scope: Optional[Scope] = None,
        session_summary: Optional[str] = None,
        profile: Optional[int] = None,
        query: Optional[str] = None,
    ) -> str:
        """
        Formatted, budgeted memory block for the latest user message.

        ``profile`` = number of always-on core facts (most important) to include for a
        user-scoped call, in the spirit of Letta's core memory / Mem0's user profile.
        Defaults to ``profile_facts``; ``True`` means 6.
        """
        scope = self._scope(scope)
        query = query if query is not None else self._extract_query(messages)
        n_profile = self.profile_facts if profile is None else (6 if profile is True else int(profile or 0))
        prof = await self.profile(scope, limit=n_profile) if (n_profile and scope.user_id is not None) else []
        if not query:
            return self.context_builder.build(profile=prof)
        res = await self.recall(query, scope)
        facts = res["facts"]
        if not facts and not prof and _is_generic(query):
            prof = await self.profile(scope, limit=self.max_facts)
        if facts:
            self.storage.touch_memories([f["id"] for f in facts])
        return self.context_builder.build(
            facts=facts,
            episodes=res["episodes"],
            excerpts=res["excerpts"],
            profile=prof,
            session_summary=session_summary,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        user_id: Optional[str] = None,
        canonical_user_id: Optional[str] = None,
        scope: Optional[Scope] = None,
    ) -> List[Dict[str, Any]]:
        """Fact search. ``user_id`` (or ``scope``) selects the user; none = shared pool."""
        if scope is None:
            scope = Scope(
                namespace=getattr(self.storage, "namespace", None),
                user_id=canonical_user_id or user_id,
            )
        res = await self.recall(query, scope, k_facts=limit, k_episodes=0, k_excerpts=0, memory_type=memory_type)
        return res["facts"]

    # ------------------------------------------------------------------ compatibility helpers

    async def _semantic_search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        user_id: Optional[str] = None,
        canonical_user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        qvec = self._encode(query)
        if qvec is None:
            return []
        from memorable_ai.core.storage import ANY_USER

        uid = canonical_user_id or user_id or ANY_USER
        hits = self.storage.vector_search("fact", qvec, k=limit * 2, user_id=uid)
        mems = self.storage.hydrate_memories([i for i, _ in hits], user_id=uid)
        sims = dict(hits)
        out = [{**m, "similarity": sims.get(m["id"], 0.0)} for m in mems]
        if memory_type:
            out = [m for m in out if m.get("type") == memory_type]
        return out[:limit]

    def _combine_and_rank(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        limit: int = 10,
        bm25_results: Optional[List[Dict[str, Any]]] = None,
        entity_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """RRF over pre-ranked result lists (compatibility)."""
        by_id: Dict[Any, Dict[str, Any]] = {}
        lists: Dict[str, List[Any]] = {}
        for name, results in (
            ("semantic", semantic_results),
            ("lexical", (keyword_results or []) + (bm25_results or [])),
            ("graph", (graph_results or []) + (entity_results or [])),
        ):
            ids = []
            for m in results or []:
                mid = m.get("id")
                if mid is None:
                    continue
                by_id.setdefault(mid, m)
                ids.append(mid)
            lists[name] = ids
        fused = reciprocal_rank_fusion(
            lists,
            {"semantic": self.semantic_weight, "lexical": self.bm25_weight, "graph": self.graph_weight},
            k=self.rrf_k,
        )
        return [by_id[i] for i, _ in fused[:limit]]

    def _deduplicate_and_rank(self, results: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        seen, unique = set(), []
        for r in results:
            rid = r.get("id")
            if rid is not None and rid not in seen:
                seen.add(rid)
                unique.append(r)
        return unique[:limit]

    @staticmethod
    def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
        v1, v2 = np.asarray(vec1, dtype=np.float32), np.asarray(vec2, dtype=np.float32)
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 == 0 or n2 == 0 or v1.shape != v2.shape:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))


def _is_generic(query: str) -> bool:
    q = query.lower().strip()
    return len(tokenize(q)) <= 2 or any(g in q for g in _GENERIC)
