"""
Persistent knowledge graph in SQL.

Nodes are ``entities`` (per namespace and user). Edges are ``relations``
(subject, predicate, object) with bi-temporal validity and a pointer to the fact
that asserted them. ``mentions`` link entities to the facts and episodes that
mention them, so facts are also graph nodes. This is the "passage node" idea
from HippoRAG 2 (Gutiérrez et al., 2025).

Write side:
  * entities/relations come from the LLM extractor when available, otherwise
    from conservative patterns over normalised facts ("User lives in Berlin")
  * single-valued predicates (lives_in, works_at, ...) invalidate the previous
    edge for the same subject, as in Graphiti/Zep. The superseded fact ids are
    returned so the write pipeline can retire the stale facts too.

Read side (``ppr``): seed entities matched in the query (n-grams, plus
predicate words on the user's own edges, so "my sister" reaches "Emma"),
expand two hops over relations and mentions, and run Personalized PageRank.
Facts are ranked by their stationary probability. This is what makes multi-hop
questions ("where does my sister live?") answerable when no single fact
contains every query term.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from memorable_ai.core.models import Entity, Memory, Mention, Relation
from memorable_ai.core.scope import Scope
from memorable_ai.core.vector_index import STOPWORDS, light_stem

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

    def __init__(self, storage: Any, functional_predicates: Optional[Iterable[str]] = None):
        self.storage = storage
        self.functional = set(functional_predicates or FUNCTIONAL_PREDICATES)

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

        superseded_facts: List[int] = []
        session = self.storage.get_session()
        try:
            now = utcnow()
            linked: Set[int] = set()
            by_norm: Dict[str, Entity] = {}
            for e in entities:
                name = e.get("name") if isinstance(e, dict) else str(e)
                etype = e.get("type", "concept") if isinstance(e, dict) else "concept"
                ent = self._get_or_create_entity(session, scope, name, etype, now)
                if ent is None:
                    continue
                by_norm[ent.norm_name] = ent
                if not ent.is_self and ent.id not in linked:
                    session.add(Mention(entity_id=ent.id, memory_id=mid))
                    ent.mention_count = (ent.mention_count or 0) + 1
                    linked.add(ent.id)
            for subj, pred, obj in relations:
                pred = re.sub(r"[^a-z0-9_]+", "_", str(pred).strip().lower()).strip("_")[:100]
                if not pred:
                    continue
                s_ent = by_norm.get(normalize_entity(subj)) or self._get_or_create_entity(session, scope, subj, "concept", now)
                o_ent = by_norm.get(normalize_entity(obj)) or self._get_or_create_entity(session, scope, obj, "concept", now)
                if s_ent is None or o_ent is None or s_ent.id == o_ent.id:
                    continue
                for ent in (s_ent, o_ent):
                    if not ent.is_self and ent.id not in linked:
                        session.add(Mention(entity_id=ent.id, memory_id=mid))
                        ent.mention_count = (ent.mention_count or 0) + 1
                        linked.add(ent.id)
                existing = (
                    session.query(Relation)
                    .filter(
                        Relation.subject_id == s_ent.id,
                        Relation.predicate == pred,
                        Relation.invalidated_at.is_(None),
                    )
                    .all()
                )
                if any(r.object_id == o_ent.id for r in existing):
                    continue  # same edge already asserted
                rel = Relation(
                    namespace=scope.namespace,
                    user_id=scope.user_id,
                    subject_id=s_ent.id,
                    predicate=pred,
                    object_id=o_ent.id,
                    memory_id=mid,
                    episode_id=memory.get("source_episode_id"),
                    valid_from=now,
                    created_at=now,
                )
                session.add(rel)
                session.flush()
                if pred in self.functional:
                    for old in existing:
                        old.invalidated_at = now
                        old.valid_to = now
                        old.superseded_by_id = rel.id
                        if old.memory_id and old.memory_id != mid:
                            superseded_facts.append(old.memory_id)
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
                session.add(Mention(entity_id=ent.id, episode_id=episode_id))
                ent.mention_count = (ent.mention_count or 0) + 1
            session.commit()
        except Exception as e:
            session.rollback()
            logger.debug("Episode graph ingest failed: %s", e)
        finally:
            session.close()

    def invalidate_memory(self, memory_id: int) -> None:
        """Retire edges asserted by a fact that is no longer valid."""
        session = self.storage.get_session()
        try:
            now = utcnow()
            for rel in (
                session.query(Relation)
                .filter(Relation.memory_id == memory_id, Relation.invalidated_at.is_(None))
                .all()
            ):
                rel.invalidated_at = now
                rel.valid_to = now
            session.commit()
        except Exception as e:
            session.rollback()
            logger.debug("Relation invalidation failed: %s", e)
        finally:
            session.close()

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
                        session.query(Relation.predicate, Relation.object_id)
                        .filter(Relation.subject_id.in_(self_ids), Relation.invalidated_at.is_(None))
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
        alpha: float = 0.85,
        hops: int = 2,
        max_nodes: int = 4000,
    ) -> List[Tuple[int, float]]:
        """Rank facts by Personalized PageRank from query-matched entities."""
        import networkx as nx

        seeds = self.match_query_entities(query, scope, include_shared)
        session = self.storage.get_session()
        try:
            if seed_memory_ids:
                for (eid,) in (
                    session.query(Mention.entity_id).filter(Mention.memory_id.in_(list(seed_memory_ids))).all()
                ):
                    seeds[eid] = max(seeds.get(eid, 0.0), 0.3)
            if not seeds:
                return []
            self_ids = {
                r[0]
                for r in self._scoped(session.query(Entity.id).filter(Entity.is_self == 1), Entity, scope, include_shared).all()
            }
            G = nx.Graph()
            frontier: Set[int] = {e for e in seeds if e not in self_ids}
            visited: Set[int] = set()
            for _ in range(hops):
                if not frontier or G.number_of_nodes() > max_nodes:
                    break
                visited |= frontier
                nxt: Set[int] = set()
                rels = (
                    session.query(Relation.subject_id, Relation.object_id)
                    .filter(
                        Relation.invalidated_at.is_(None),
                        (Relation.subject_id.in_(frontier)) | (Relation.object_id.in_(frontier)),
                    )
                    .all()
                )
                for s, o in rels:
                    if s in self_ids or o in self_ids:
                        continue
                    G.add_edge(f"e{s}", f"e{o}", weight=1.0)
                    nxt.update((s, o))
                ment = (
                    session.query(Mention.entity_id, Mention.memory_id)
                    .join(Memory, Memory.id == Mention.memory_id)
                    .filter(Mention.entity_id.in_(frontier), Memory.invalidated_at.is_(None))
                    .all()
                )
                mem_ids = {m for _, m in ment}
                for e, m in ment:
                    G.add_edge(f"e{e}", f"m{m}", weight=1.0)
                if mem_ids:
                    for e, m in (
                        session.query(Mention.entity_id, Mention.memory_id)
                        .filter(Mention.memory_id.in_(list(mem_ids)))
                        .all()
                    ):
                        if e in self_ids:
                            continue
                        G.add_edge(f"e{e}", f"m{m}", weight=1.0)
                        nxt.add(e)
                frontier = nxt - visited
            personalization = {f"e{e}": w for e, w in seeds.items() if f"e{e}" in G}
            if not personalization:
                return []
            scores = nx.pagerank(G, alpha=alpha, personalization=personalization, weight="weight", max_iter=100)
            ranked = sorted(
                ((int(n[1:]), s) for n, s in scores.items() if n.startswith("m")),
                key=lambda x: x[1],
                reverse=True,
            )
            return ranked[:k]
        except Exception as e:
            logger.debug("PPR failed: %s", e)
            return []
        finally:
            session.close()

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
                for r in session.query(Mention.memory_id)
                .filter(Mention.entity_id.in_(ids), Mention.memory_id.isnot(None))
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
            q = self._scoped(session.query(Relation), Relation, scope)
            if not include_invalid:
                q = q.filter(Relation.invalidated_at.is_(None))
            rels = q.all()
            ent_ids = {r.subject_id for r in rels} | {r.object_id for r in rels}
            names = {
                e.id: e.name for e in session.query(Entity).filter(Entity.id.in_(list(ent_ids))).all()
            } if ent_ids else {}
            return [
                {
                    "id": r.id,
                    "subject": names.get(r.subject_id),
                    "predicate": r.predicate,
                    "object": names.get(r.object_id),
                    "memory_id": r.memory_id,
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
            eq, rq = session.query(Entity), session.query(Relation).filter(Relation.invalidated_at.is_(None))
            if scope is not None:
                eq, rq = self._scoped(eq, Entity, scope), self._scoped(rq, Relation, scope)
            return {
                "entities": eq.count(),
                "relations": rq.count(),
                "mentions": session.query(Mention).count(),
            }
        finally:
            session.close()


def _verb_after(text: str, subj: str) -> str:
    rest = text[len(subj):].strip().split()
    return rest[0].lower() if rest else "related_to"
