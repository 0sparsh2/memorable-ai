"""
Knowledge graph over the unified ``edges`` table (docs/memory_graph_design.md).

Nodes are rows addressed as ``(kind, id)``: ``entity`` rows (per namespace and
user), ``fact`` rows (memories), ``episode`` and ``turn`` rows. This module writes:

  * ``relates``  entity -> entity, with ``predicate``, bi-temporal validity, and
    ``evidence`` -> the fact that asserted it
  * ``about``    fact -> entity (role subject / object / mention), so facts are
    graph nodes too (the HippoRAG 2 "passage node" idea)
  * ``mentions`` episode -> entity

Single-valued predicates (lives_in, works_at, ...) invalidate the previous edge
for the same subject, as in Graphiti/Zep. The superseded fact ids are returned so
the write pipeline can retire the stale facts too.

Read side (``ppr``): seed entities matched in the query (n-grams, plus predicate
words on the user's own edges, so "my sister" reaches "Emma"), then run
Personalized PageRank on the in-process sparse-matrix ``GraphEngine`` over the
whole partition, with no hop caps. Facts are ranked by their stationary
probability. The user's own "self" node is a hub, so it is excluded from the walk
exactly.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from memorable_ai.core.models import Edge, Entity
from memorable_ai.core.scope import Scope
from memorable_ai.core.vector_index import STOPWORDS, light_stem
from memorable_ai.graph.engine import GraphEngine

from memorable_ai.core.models import utcnow  # noqa: E402

logger = logging.getLogger(__name__)

SELF_NAMES = {"user", "the user", "i", "me", "myself"}

FUNCTIONAL_PREDICATES = {
    "lives_in",
    "works_at",
    "is_from",
    "named",
    "age",
    "married_to",
    "job_title",
    "has_partner",
    "has_wife",
    "has_husband",
    "has_boyfriend",
    "has_girlfriend",
}

_MONTHS = {m.lower() for m in (
    "January February March April May June July August September October November December "
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday"
).split()}

# Applied to one normalised fact statement. Groups: s (subject), o (object), optional r.
_REL_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"^(?P<s>.+?) (?:lives|resides|is living) in (?P<o>.+)$", re.I), "lives_in", "place"),
    (re.compile(r"^(?P<s>.+?) (?:moved|relocated) to (?P<o>.+)$", re.I), "lives_in", "place"),
    (re.compile(r"^(?P<s>.+?) (?:works|worked|is working) (?:at|for) (?P<o>.+)$", re.I), "works_at", "organization"),
    (re.compile(r"^(?P<s>.+?) (?:is|comes) from (?P<o>.+)$", re.I), "is_from", "place"),
    (re.compile(r"^(?P<s>.+?) (?:went|traveled|travelled|flew) to (?P<o>.+)$", re.I), "visited", "place"),
    (re.compile(r"^(?P<s>.+?) is allergic to (?P<o>.+)$", re.I), "allergic_to", "concept"),
    (re.compile(r"^(?P<s>.+?)'s (?P<r>[a-z]+(?: [a-z]+)?) is named (?P<o>.+)$", re.I), "has_{r}", "person"),
    (re.compile(r"^(?P<s>.+?)'s favou?rite (?P<r>[a-z]+(?: [a-z]+)?) is (?P<o>.+)$", re.I), "favorite_{r}", "concept"),
    (re.compile(r"^(?P<s>.+?)'s name is (?P<o>.+)$", re.I), "named", "person"),
    (re.compile(r"^(?P<s>.+?) is married to (?P<o>.+)$", re.I), "married_to", "person"),
    (re.compile(r"^(?P<s>.+?) (?:likes|loves|enjoys|prefers|adores) (?P<o>.+)$", re.I), "likes", "concept"),
    (re.compile(r"^(?P<s>.+?) (?:dislikes|hates) (?P<o>.+)$", re.I), "dislikes", "concept"),
    (re.compile(r"^(?P<s>.+?) (?:is|are) used (?:in|by|for) (?P<o>.+)$", re.I), "used_in", "concept"),
    (re.compile(r"^(?P<s>.+?) uses (?P<o>.+)$", re.I), "uses", "product"),
    (re.compile(r"^(?P<s>.+?) is (?:building|working on|writing|developing) (?P<o>.+)$", re.I), "working_on", "concept"),
    (re.compile(r"^(?P<s>.+?) is (?:learning|studying|training for|preparing for) (?P<o>.+)$", re.I), "learning", "concept"),
    (re.compile(r"^(?P<s>.+?) (?:is skilled in|has experience with|speaks) (?P<o>.+)$", re.I), "skilled_in", "concept"),
    (re.compile(r"^(?P<s>.+?) (?:studies|teaches|owns|manages|founded|leads) (?P<o>.+)$", re.I), "{verb}", "concept"),
    (re.compile(r"^(?P<s>.+?) has (?:a|an|one|two|three|four|\d+) (?P<o>.+)$", re.I), "has", "concept"),
    (re.compile(r"^(?P<s>.+?) is (?:a|an) (?P<o>.+)$", re.I), "is_a", "concept"),
]
_OBJ_CUT = re.compile(
    r"\s+(?:over|with|because|since|on \d{4}-\d\d-\d\d|in the week of|on the weekend of|in (?:January|"
    r"February|March|April|May|June|July|August|September|October|November|December) \d{4}|in \d{4}|"
    r"and then|but)\b.*$",
    re.I,
)
_CAP_SEQ = re.compile(r"\b([A-Z][\w&'-]*(?:\s+(?:of|the|de|van|von|da)?\s*[A-Z][\w&'-]*)*)")


def normalize_entity(name: str) -> str:
    n = re.sub(r"\s+", " ", (name or "").strip().lower())
    n = re.sub(r"^(?:the|a|an)\s+", "", n)
    n = re.sub(r"'s$", "", n)
    return n.strip(" .,:;!?\"'()[]")


def _stem(word: str) -> str:
    return light_stem(word)


def _clean_obj(o: str) -> str:
    o = _OBJ_CUT.sub("", o.strip())
    return o.strip(" .,:;!?\"'")


class KnowledgeGraph:
    """SQL-backed entity/relation graph with Personalized PageRank retrieval."""

    def __init__(
        self,
        storage: Any,
        functional_predicates: Optional[Iterable[str]] = None,
        engine: Optional[GraphEngine] = None,
    ):
        self.storage = storage
        self.functional = set(functional_predicates or FUNCTIONAL_PREDICATES)
        self.engine = engine or GraphEngine(storage)

    # ------------------------------------------------------------------ extraction

    @staticmethod
    def extract(text: str) -> Tuple[List[Dict[str, str]], List[Tuple[str, str, str]]]:
        """Heuristic entities and (subject, predicate, object) relations from one statement."""
        text = (text or "").strip().rstrip(".")
        text = re.sub(r"^(?:User|Assistant)\s*:\s*", "", text)  # key-point speaker prefix
        entities: Dict[str, Dict[str, str]] = {}
        relations: List[Tuple[str, str, str]] = []
        for pat, pred, otype in _REL_PATTERNS:
            m = pat.match(text)
            if not m:
                continue
            subj = m.group("s").strip()
            obj = _clean_obj(m.group("o"))
            if not subj or not obj or len(subj.split()) > 4 or len(obj.split()) > 6:
                break
            if subj.lower() not in SELF_NAMES and not all(
                w[:1].isupper() or w.lower() in ("of", "the", "de", "van", "von", "da")
                for w in subj.split()
            ):
                break
            if ":" in subj:
                break
            if "{r}" in pred:
                pred = pred.replace("{r}", re.sub(r"\s+", "_", m.group("r").lower()))
            if pred == "{verb}":
                pred = _verb_after(text, subj)
            relations.append((subj, pred, obj))
            if subj.lower() not in SELF_NAMES:
                entities[normalize_entity(subj)] = {"name": subj, "type": "person" if subj[0].isupper() else "concept"}
            ent_type = otype if (obj[:1].isupper() or otype in ("concept", "product")) else "concept"
            entities[normalize_entity(obj)] = {"name": obj, "type": ent_type}
            break
        for m in _CAP_SEQ.finditer(text):
            name = m.group(1).strip()
            norm = normalize_entity(name)
            if (
                not norm
                or norm in SELF_NAMES
                or norm in STOPWORDS
                or norm in _MONTHS
                or (m.start() == 0 and norm in STOPWORDS)
                or re.fullmatch(r"\d+", norm)
            ):
                continue
            entities.setdefault(norm, {"name": name, "type": "concept"})
        return list(entities.values()), relations

    # ------------------------------------------------------------------ writes

    def _get_or_create_entity(
        self, session: Any, scope: Scope, name: str, entity_type: str = "concept", now: Optional[datetime] = None
    ) -> Optional[Entity]:
        norm = normalize_entity(name)
        if not norm:
            return None
        is_self = norm in SELF_NAMES
        if is_self:
            norm, name, entity_type = "user", "User", "person"
        q = session.query(Entity).filter(Entity.norm_name == norm)
        q = q.filter(Entity.namespace == scope.namespace if scope.namespace else Entity.namespace.is_(None))
        q = q.filter(Entity.user_id == scope.user_id if scope.user_id else Entity.user_id.is_(None))
        ent = q.first()
        now = now or utcnow()
        if ent is None:
            ent = Entity(
                namespace=scope.namespace,
                user_id=scope.user_id,
                agent_id=scope.agent_id,
                name=name[:255],
                norm_name=norm[:255],
                entity_type=entity_type or "concept",
                is_self=1 if is_self else 0,
                mention_count=0,
                first_seen=now,
                last_seen=now,
            )
            session.add(ent)
            session.flush()
        else:
            ent.last_seen = now
            if ent.entity_type == "concept" and entity_type not in (None, "", "concept"):
                ent.entity_type = entity_type
        return ent

    def _edge(self, scope: Scope, src: Tuple[str, int], kind: str, dst: Tuple[str, int], **attrs: Any) -> Edge:
        return Edge(
            namespace=self.storage.namespace or scope.namespace,
            user_id=scope.user_id,
            agent_id=scope.agent_id,
            src_kind=src[0], src_id=int(src[1]), kind=kind, dst_kind=dst[0], dst_id=int(dst[1]),
            weight=1.0, confidence=1.0, created_at=attrs.pop("created_at", utcnow()), **attrs,
        )

    def ingest_memory(self, memory: Dict[str, Any], scope: Scope) -> List[int]:
        """
        Link a stored fact into the graph. Returns ids of *other* facts whose
        single-valued relation was superseded by this one (candidates to retire).
        """
        mid = memory.get("id")
        content = memory.get("content") or ""
        if not mid or not content:
            return []
        entities = memory.get("entities")
        relations = memory.get("relations")
        if not entities and not relations:
            entities, relations = self.extract(content)
        else:
            h_ents, h_rels = self.extract(content)
            entities = list(entities or []) or h_ents
            relations = list(relations or []) or h_rels
        scope = Scope(scope.namespace, scope.user_id, scope.session_id,
                      scope.agent_id or memory.get("agent_id"))

        superseded_facts: List[int] = []
        session = self.storage.get_session()
        try:
            now = utcnow()
            valid_from = memory.get("valid_from") if isinstance(memory.get("valid_from"), datetime) else now
            about: Dict[int, str] = {}  # entity id -> role
            by_norm: Dict[str, Entity] = {}

            def link(ent: Entity, role: str) -> None:
                if ent.id not in about:
                    ent.mention_count = (ent.mention_count or 0) + (0 if ent.is_self else 1)
                    about[ent.id] = role
                elif role != "mention" and about[ent.id] == "mention":
                    about[ent.id] = role

            for e in entities:
                name = e.get("name") if isinstance(e, dict) else str(e)
                etype = e.get("type", "concept") if isinstance(e, dict) else "concept"
                ent = self._get_or_create_entity(session, scope, name, etype, now)
                if ent is None:
                    continue
                by_norm[ent.norm_name] = ent
                link(ent, "mention")
            new_edges: List[Edge] = []
            for subj, pred, obj in relations:
                pred = re.sub(r"[^a-z0-9_]+", "_", str(pred).strip().lower()).strip("_")[:100]
                if not pred:
                    continue
                s_ent = by_norm.get(normalize_entity(subj)) or self._get_or_create_entity(session, scope, subj, "concept", now)
                o_ent = by_norm.get(normalize_entity(obj)) or self._get_or_create_entity(session, scope, obj, "concept", now)
                if s_ent is None or o_ent is None or s_ent.id == o_ent.id:
                    continue
                link(s_ent, "subject")
                link(o_ent, "object")
                existing = (
                    session.query(Edge)
                    .filter(
                        Edge.kind == "relates",
                        Edge.src_kind == "entity",
                        Edge.src_id == s_ent.id,
                        Edge.predicate == pred,
                        Edge.invalidated_at.is_(None),
                    )
                    .all()
                )
                if any(r.dst_id == o_ent.id for r in existing):
                    continue  # same edge already asserted
                rel = self._edge(
                    scope, ("entity", s_ent.id), "relates", ("entity", o_ent.id),
                    predicate=pred, valid_from=valid_from, created_at=now,
                    evidence_kind="fact", evidence_id=int(mid),
                )
                session.add(rel)
                session.flush()
                if pred in self.functional:
                    for old in existing:
                        old.invalidated_at = now
                        old.valid_to = valid_from
                        old.superseded_by_id = rel.id
                        if old.evidence_kind == "fact" and old.evidence_id and old.evidence_id != mid:
                            superseded_facts.append(int(old.evidence_id))
            for eid, role in about.items():
                new_edges.append(self._edge(scope, ("fact", int(mid)), "about", ("entity", eid),
                                            role=role, valid_from=valid_from, created_at=now))
            session.add_all(new_edges)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.warning("Knowledge graph ingest failed for memory %s: %s", mid, e)
            return []
        finally:
            session.close()
        return superseded_facts

    def ingest_episode(self, episode_id: int, names: Sequence[str], scope: Scope) -> None:
        """Link an episode to the entities it mentions."""
        if not episode_id or not names:
            return
        session = self.storage.get_session()
        try:
            seen: Set[int] = set()
            for name in names:
                ent = self._get_or_create_entity(session, scope, name)
                if ent is None or ent.is_self or ent.id in seen:
                    continue
                seen.add(ent.id)
                session.add(self._edge(scope, ("episode", int(episode_id)), "mentions", ("entity", ent.id)))
                ent.mention_count = (ent.mention_count or 0) + 1
            session.commit()
        except Exception as e:
            session.rollback()
            logger.debug("Episode graph ingest failed: %s", e)
        finally:
            session.close()

    def invalidate_memory(self, memory_id: int) -> None:
        """Retire the relations asserted by a fact that is no longer valid."""
        try:
            self.storage.invalidate_edges(evidence=("fact", int(memory_id)), kinds=("relates",))
        except Exception as e:
            logger.debug("Relation invalidation failed: %s", e)

    # ------------------------------------------------------------------ reads

    def _scoped(self, q, model, scope: Scope, include_shared: bool = False):
        if scope.namespace:
            q = q.filter(model.namespace == scope.namespace)
        if scope.user_id is None:
            return q.filter(model.user_id.is_(None))
        if include_shared:
            return q.filter((model.user_id == scope.user_id) | (model.user_id.is_(None)))
        return q.filter(model.user_id == scope.user_id)

    def match_query_entities(
        self, query: str, scope: Scope, include_shared: bool = False
    ) -> Dict[int, float]:
        """Entity ids (with seed weights) referenced by the query."""
        words = re.findall(r"[\w'-]+", query or "")
        grams: Set[str] = set()
        for n in (1, 2, 3, 4):
            for i in range(len(words) - n + 1):
                g = normalize_entity(" ".join(words[i : i + n]))
                if g and g not in STOPWORDS and g not in SELF_NAMES and len(g) > 1:
                    grams.add(g)
        seeds: Dict[int, float] = {}
        session = self.storage.get_session()
        try:
            if grams:
                q = session.query(Entity.id, Entity.norm_name, Entity.is_self).filter(
                    Entity.norm_name.in_(list(grams))
                )
                for eid, norm, is_self in self._scoped(q, Entity, scope, include_shared).all():
                    if not is_self:
                        seeds[eid] = max(seeds.get(eid, 0.0), 1.0 + 0.25 * (len(norm.split()) - 1))
            # Predicate words on the user's own edges: "my sister" -> (User)-has_sister->(Emma)
            q_stems = {_stem(w) for w in words if w.lower() not in STOPWORDS}
            if q_stems:
                self_q = self._scoped(session.query(Entity.id).filter(Entity.is_self == 1), Entity, scope, include_shared)
                self_ids = [r[0] for r in self_q.all()]
                if self_ids:
                    rels = (
                        session.query(Edge.predicate, Edge.dst_id)
                        .filter(
                            Edge.kind == "relates",
                            Edge.src_kind == "entity",
                            Edge.src_id.in_(self_ids),
                            Edge.invalidated_at.is_(None),
                        )
                        .all()
                    )
                    for pred, obj_id in rels:
                        pwords = {_stem(w) for w in pred.split("_") if w not in STOPWORDS and w not in ("has", "is")}
                        if pwords & q_stems:
                            seeds[obj_id] = max(seeds.get(obj_id, 0.0), 0.8)
        finally:
            session.close()
        return seeds

    def ppr(
        self,
        query: str,
        scope: Scope,
        k: int = 20,
        seed_memory_ids: Sequence[int] = (),
        include_shared: bool = False,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[int, float]]:
        """Rank facts by Personalized PageRank from query-matched entities."""
        try:
            seeds = self.match_query_entities(query, scope, include_shared)
            session = self.storage.get_session()
            try:
                if seed_memory_ids:
                    for (eid,) in (
                        session.query(Edge.dst_id)
                        .filter(
                            Edge.kind == "about",
                            Edge.src_kind == "fact",
                            Edge.src_id.in_([int(m) for m in seed_memory_ids]),
                            Edge.invalidated_at.is_(None),
                        )
                        .all()
                    ):
                        seeds[eid] = max(seeds.get(eid, 0.0), 0.3)
                if not seeds:
                    return []
                self_ids = [
                    r[0]
                    for r in self._scoped(
                        session.query(Entity.id).filter(Entity.is_self == 1), Entity, scope, include_shared
                    ).all()
                ]
            finally:
                session.close()
            exclude = [("entity", i) for i in self_ids]
            result = self.engine.ppr(
                self.storage.namespace or scope.namespace,
                scope.user_id,
                {("entity", e): w for e, w in seeds.items() if e not in self_ids},
                include_shared=include_shared,
                exclude=exclude,
                weights=weights,
            )
            return result.top("fact", k)
        except Exception as e:
            logger.debug("PPR failed: %s", e)
            return []

    def entity_facts(self, name: str, scope: Scope, limit: int = 20) -> List[Dict[str, Any]]:
        """Valid facts that mention an entity."""
        norm = normalize_entity(name)
        session = self.storage.get_session()
        try:
            ent_q = self._scoped(session.query(Entity.id).filter(Entity.norm_name == norm), Entity, scope)
            ids = [r[0] for r in ent_q.all()]
            if not ids:
                return []
            mem_ids = [
                r[0]
                for r in session.query(Edge.src_id)
                .filter(
                    Edge.kind == "about",
                    Edge.src_kind == "fact",
                    Edge.dst_kind == "entity",
                    Edge.dst_id.in_(ids),
                    Edge.invalidated_at.is_(None),
                )
                .limit(limit * 3)
                .all()
            ]
        finally:
            session.close()
        return self.storage.hydrate_memories(dict.fromkeys(mem_ids))[:limit]

    def relations(self, scope: Scope, include_invalid: bool = False) -> List[Dict[str, Any]]:
        """All edges in scope as dicts (for inspection/export)."""
        session = self.storage.get_session()
        try:
            q = self._scoped(session.query(Edge).filter(Edge.kind == "relates"), Edge, scope)
            if not include_invalid:
                q = q.filter(Edge.invalidated_at.is_(None))
            rels = q.all()
            ent_ids = {r.src_id for r in rels} | {r.dst_id for r in rels}
            names = {
                e.id: e.name for e in session.query(Entity).filter(Entity.id.in_(list(ent_ids))).all()
            } if ent_ids else {}
            return [
                {
                    "id": r.id,
                    "subject": names.get(r.src_id),
                    "predicate": r.predicate,
                    "object": names.get(r.dst_id),
                    "memory_id": r.evidence_id if r.evidence_kind == "fact" else None,
                    "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                    "invalidated_at": r.invalidated_at.isoformat() if r.invalidated_at else None,
                }
                for r in rels
            ]
        finally:
            session.close()

    def to_networkx(self, scope: Scope):
        """Valid entity-relation graph as a networkx MultiDiGraph."""
        import networkx as nx

        g = nx.MultiDiGraph()
        for r in self.relations(scope):
            g.add_edge(r["subject"], r["object"], predicate=r["predicate"], memory_id=r["memory_id"])
        return g

    def stats(self, scope: Optional[Scope] = None) -> Dict[str, int]:
        session = self.storage.get_session()
        try:
            valid = session.query(Edge).filter(Edge.invalidated_at.is_(None))
            eq = session.query(Entity)
            rq = valid.filter(Edge.kind == "relates")
            mq = valid.filter(Edge.kind.in_(("about", "mentions")))
            if scope is not None:
                eq, rq, mq = (self._scoped(eq, Entity, scope), self._scoped(rq, Edge, scope),
                              self._scoped(mq, Edge, scope))
            return {
                "entities": eq.count(),
                "relations": rq.count(),
                "mentions": mq.count(),
                "edges": (self._scoped(valid, Edge, scope) if scope is not None else valid).count(),
            }
        finally:
            session.close()


def _verb_after(text: str, subj: str) -> str:
    rest = text[len(subj):].strip().split()
    return rest[0].lower() if rest else "related_to"
