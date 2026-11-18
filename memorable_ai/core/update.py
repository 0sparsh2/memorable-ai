"""
Memory update phase: ADD / UPDATE / DELETE / NOOP (Mem0, arXiv:2504.19413).

What changed from the original port:

- **Versioned, not destructive.** UPDATE and DELETE create a new row and
  invalidate the old one (``superseded_by_id``), so memory history is queryable
  and the bi-temporal columns mean something. The old code overwrote content in
  place, without re-embedding, and hard-deleted rows.
- **Scoped candidates.** Similar memories come from the shared vector index within
  the caller's user scope. The old code constructed a new retriever, and loaded a
  SentenceTransformer, for every extracted fact.
- **One LLM call per batch,** with candidate ids remapped to small integers, and
  every returned target validated against the candidates (no acting on
  hallucinated ids).
- **No-LLM conflict detection** via single-valued graph predicates: "User moved to
  Munich" supersedes "User lives in Berlin".
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from memorable_ai.core.scope import Scope
from memorable_ai.core.storage import normalize_text

logger = logging.getLogger(__name__)

# Single-fact prompt kept for compatibility with callers/tests that format it directly.
UPDATE_PROMPT = """You manage a memory knowledge base. Given a new fact and similar existing memories, choose one operation:
- ADD: create new memory
- UPDATE: merge into existing memory id (provide memory_id and merged content)
- DELETE: remove contradicted memory id
- NOOP: no change needed

Respond with JSON only: {{"operation": "ADD|UPDATE|DELETE|NOOP", "memory_id": null or int, "content": "final text if ADD/UPDATE"}}

New fact: {fact}

Similar memories:
{similar}
"""

BATCH_UPDATE_PROMPT = """You maintain a long-term memory store about a user. Decide how each NEW fact changes the store.

Existing memories (ref: text):
{existing}

New facts (index: text):
{new}

For every new fact choose exactly one operation:
- ADD: new information not already stored.
- UPDATE: refines or corrects an existing memory (same subject and attribute, newer or more complete). Give the target ref and the merged content.
- DELETE: the new fact makes an existing memory no longer true (e.g. "no longer", "stopped", "moved"). Give the target ref; the new fact is kept as its replacement.
- NOOP: already stored, or not worth storing.

Return only a JSON array: [{{"index": 0, "operation": "ADD|UPDATE|DELETE|NOOP", "target": null or ref, "content": "final text"}}]
"""


class MemoryUpdater:
    """Applies Mem0-style memory operations after extraction."""

    def __init__(
        self,
        storage: Any,
        embedding_model: Any = None,
        llm_model: Optional[str] = None,
        similar_top_k: int = 10,
        add_only: bool = False,
        duplicate_threshold: float = 0.96,
        candidate_threshold: float = 0.35,
        knowledge_graph: Any = None,
    ):
        from memorable_ai.core.embeddings import as_embedder

        self.storage = storage
        self.embedding_model = embedding_model
        self._embedder = as_embedder(embedding_model)
        self.llm_model = llm_model
        self.similar_top_k = similar_top_k
        self.add_only = add_only
        self.duplicate_threshold = duplicate_threshold
        self.candidate_threshold = candidate_threshold
        self.knowledge_graph = knowledge_graph

    async def apply_facts(
        self, facts: List[Dict[str, Any]], scope: Optional[Scope] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply extracted facts. Returns the facts that changed the store, each with
        ``id`` (the row now holding it), ``op`` and, for updates, ``superseded_id``.
        """
        scope = scope or Scope(namespace=getattr(self.storage, "namespace", None))
        facts = [f for f in facts if (f.get("content") or "").strip()]
        if not facts:
            return []
        for f in facts:
            if scope.user_id is not None and f.get("user_id") is None:
                f["user_id"] = scope.user_id
            if scope.session_id and not f.get("session_id"):
                f["session_id"] = scope.session_id
            if f.get("embedding") is None and self._embedder is not None:
                try:
                    f["embedding"] = self._embedder.encode_list(f["content"])
                except Exception:
                    pass

        if self.add_only:
            return [self._add(f) for f in facts]

        candidates = [self._candidates(f, scope) for f in facts]
        decisions: List[Dict[str, Any]] = []
        from memorable_ai.core.llm_client import llm_enabled

        if llm_enabled(self.llm_model) and any(candidates):
            decisions = await self._decide_batch(facts, candidates)
        if not decisions:
            decisions = [self._decide_heuristic(f, c) for f, c in zip(facts, candidates)]

        applied: List[Dict[str, Any]] = []
        for fact, decision in zip(facts, decisions):
            result = self._execute(decision, fact)
            if result is not None:
                applied.append(result)
        return applied

    # ------------------------------------------------------------------ candidates

    def _candidates(self, fact: Dict[str, Any], scope: Scope) -> List[Tuple[Dict[str, Any], float]]:
        """Similar active memories in the same scope, as (memory, cosine) pairs."""
        scored: Dict[int, float] = {}
        emb = fact.get("embedding")
        if emb is not None:
            for mid, sim in self.storage.vector_search(
                "fact", emb, k=self.similar_top_k, user_id=scope.user_id
            ):
                if sim >= self.candidate_threshold:
                    scored[mid] = sim
        for mid, _ in self.storage.lexical_search(
            "fact", fact["content"], k=5, user_id=scope.user_id
        ):
            scored.setdefault(mid, 0.0)
        if not scored:
            return []
        mems = self.storage.hydrate_memories(list(scored), user_id=scope.user_id)
        out = [(m, scored.get(m["id"], 0.0)) for m in mems]
        out.sort(key=lambda x: x[1], reverse=True)
        return out[: self.similar_top_k]

    # ------------------------------------------------------------------ decisions

    def _decide_heuristic(
        self, fact: Dict[str, Any], candidates: List[Tuple[Dict[str, Any], float]]
    ) -> Dict[str, Any]:
        norm = normalize_text(fact["content"])
        for mem, sim in candidates:
            if normalize_text(mem["content"]) == norm or sim >= self.duplicate_threshold:
                return {"operation": "NOOP", "target_id": mem["id"]}
        conflict = self._functional_conflict(fact, candidates)
        if conflict is not None:
            return {"operation": "UPDATE", "target_id": conflict, "content": fact["content"]}
        return {"operation": "ADD"}

    def _functional_conflict(
        self, fact: Dict[str, Any], candidates: List[Tuple[Dict[str, Any], float]]
    ) -> Optional[int]:
        """Same subject + single-valued predicate + different object => conflict."""
        from memorable_ai.graph.knowledge_graph import KnowledgeGraph, normalize_entity
        from memorable_ai.graph.ontology import default_ontology

        ont = getattr(self.knowledge_graph, "ontology", None) or default_ontology()

        def keys(content: str) -> Dict[Tuple[str, str], str]:
            """(subject, functional predicate) -> object; symmetric ones keyed from both ends."""
            out: Dict[Tuple[str, str], str] = {}
            for s, p, o in KnowledgeGraph.extract(content)[1]:
                r = ont.resolve(p)
                if not ont.is_functional(r.name):
                    continue
                s, o = (o, s) if r.swap else (s, o)
                s, o = normalize_entity(s), normalize_entity(o)
                out[(s, r.name)] = o
                if ont.is_symmetric(r.name):
                    out[(o, r.name)] = s
            return out

        new_keys = keys(fact["content"])
        if not new_keys:
            return None
        for mem, _ in candidates:
            for key, o in keys(mem["content"]).items():
                if key in new_keys and new_keys[key] != o:
                    return mem["id"]
        return None

    async def _decide_batch(
        self,
        facts: List[Dict[str, Any]],
        candidates: List[List[Tuple[Dict[str, Any], float]]],
    ) -> List[Dict[str, Any]]:
        ref_to_id: Dict[int, int] = {}
        id_to_ref: Dict[int, int] = {}
        lines: List[str] = []
        for cands in candidates:
            for mem, _ in cands:
                if mem["id"] not in id_to_ref:
                    ref = len(ref_to_id) + 1
                    ref_to_id[ref] = mem["id"]
                    id_to_ref[mem["id"]] = ref
                    lines.append(f"{ref}: {mem['content']}")
        prompt = BATCH_UPDATE_PROMPT.format(
            existing="\n".join(lines) or "(none)",
            new="\n".join(f"{i}: {f['content']}" for i, f in enumerate(facts)),
        )
        try:
            from memorable_ai.core import llm_client

            resp = await llm_client.acompletion_with_fallback(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200 + 120 * len(facts),
                temperature=0,
            )
            parsed = llm_client.parse_json(llm_client.message_content(resp))
        except Exception as e:
            logger.debug("Update LLM failed, using heuristics: %s", e)
            return []
        if isinstance(parsed, dict):
            parsed = parsed.get("operations") or [parsed]
        by_index: Dict[int, Dict[str, Any]] = {}
        for item in parsed or []:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            op = str(item.get("operation", "ADD")).upper()
            target = item.get("target")
            target_id = None
            if target is not None:
                try:
                    target_id = ref_to_id.get(int(target))
                except (TypeError, ValueError):
                    target_id = None
            if op in ("UPDATE", "DELETE") and target_id is None:
                op = "ADD"  # unknown/hallucinated target: never mutate an arbitrary row
            if op == "NOOP" and target_id is None:
                target_id = None
            by_index[idx] = {
                "operation": op if op in ("ADD", "UPDATE", "DELETE", "NOOP") else "ADD",
                "target_id": target_id,
                "content": item.get("content"),
            }
        decisions = []
        for i, (fact, cands) in enumerate(zip(facts, candidates)):
            decisions.append(by_index.get(i) or self._decide_heuristic(fact, cands))
        return decisions

    async def _decide_operation(self, fact: str, similar: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Single-fact decision (compatibility)."""
        cands = [(m, 0.0) for m in similar]
        decisions = await self._decide_batch([{"content": fact}], [cands]) if self.llm_model else []
        if decisions:
            d = decisions[0]
            return {"operation": d["operation"], "memory_id": d.get("target_id"), "content": d.get("content") or fact}
        return {"operation": "ADD", "memory_id": None, "content": fact}

    # ------------------------------------------------------------------ execution

    def _add(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        mid, op = self.storage.insert_memory(fact)
        return {**fact, "id": mid, "op": "add" if op == "add" else "duplicate"}

    def _execute(self, decision: Dict[str, Any], fact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        op = decision.get("operation", "ADD").upper()
        target = decision.get("target_id")
        if op == "NOOP":
            if target:
                self.storage.touch_memories([target])
            return None
        if op in ("UPDATE", "DELETE") and target:
            new_content = fact["content"]
            if op == "UPDATE" and decision.get("content"):
                new_content = str(decision["content"]).strip() or new_content
            data = dict(fact)
            if new_content != fact["content"]:
                data["content"] = new_content
                if self._embedder is not None:
                    data["embedding"] = self._embedder.encode_list(new_content)
            new_id = self.storage.supersede_memory(int(target), data)
            if new_id is None:
                return self._add(fact)
            if self.knowledge_graph is not None:
                self.knowledge_graph.invalidate_memory(int(target))
            return {**data, "id": new_id, "op": op.lower(), "superseded_id": int(target)}
        content = decision.get("content")
        if content and str(content).strip() and str(content).strip() != fact["content"]:
            fact = dict(fact)
            fact["content"] = str(content).strip()
            if self._embedder is not None:
                fact["embedding"] = self._embedder.encode_list(fact["content"])
        return self._add(fact)
