"""
Knowledge graph over the unified ``edges`` table (docs/memory_graph_design.md).

Nodes are rows addressed as ``(kind, id)``: ``entity`` rows (per namespace and
user), ``fact`` rows (memories), ``episode`` and ``turn`` rows. This module writes:

  * ``relates``  entity -> entity, with ``predicate``, bi-temporal validity, and
    ``evidence`` -> the fact that asserted it
  * ``about``    fact -> entity (role subject / object / mention), so facts are
    graph nodes too (the HippoRAG 2 "passage node" idea)
  * ``mentions`` episode -> entity

Predicates go through the ontology (``graph/ontology.py``): "moved_to" becomes
``lives_in``, "child_of" becomes ``parent_of`` with the ends swapped, symmetric ones
("sibling_of") are stored once with the user first. The extractor's own word survives
on the edge (``extra.raw``, ``extra.role``), so "sister" is not lost to ``sibling_of``.
Mentions are matched to existing entities by ``graph/resolution.py`` (names, aliases,
nicknames, graph context; an LLM only for leftovers).

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
import math
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from memorable_ai.core.models import Edge, Entity, EntityAlias, Turn
from memorable_ai.core.scope import Scope
from memorable_ai.core.vector_index import STOPWORDS, light_stem
from memorable_ai.graph.engine import GraphEngine
from memorable_ai.graph.ontology import Ontology, default_ontology, raw_role
from memorable_ai.graph.resolution import Adjudicator, EntityResolver, Mention

from memorable_ai.core.models import utcnow  # noqa: E402

logger = logging.getLogger(__name__)

# (subject norm, canonical predicate, object norm, edge attributes)
Rel = Tuple[str, str, str, Dict[str, Any]]

SELF_NAMES = {"user", "the user", "i", "me", "myself"}

# Kept for compatibility; the ontology is the source of truth.
FUNCTIONAL_PREDICATES = default_ontology().functional_predicates()

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


# G2: a transcript's speaker label ("Caroline: ...") is not a mention of Caroline.
_SPEAKER_LABEL = re.compile(r"^\s*[A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,3}:\s+")
# Edge kinds the turn walk follows (docs/memory_graph_design.md §5; measured in docs/g2_results.md).
TURN_WALK_WEIGHTS: Dict[str, float] = {"about": 1.0, "derived_from": 1.0, "mentions": 1.0, "relates": 1.0,
                                       "alias_of": 1.0}
TURN_SEEDS = 10  # top first-stage turns that seed the walk next to the query's entities (HippoRAG 2)
TURN_SEED_WEIGHT = 0.05  # their weight relative to an exact entity match (1.0)
MENTION_MIN_CHARS = 3


def turn_body(content: str) -> str:
    """Lower-cased turn text without a leading speaker label, for mention matching."""
    return _SPEAKER_LABEL.sub("", content or "", count=1).lower()


class KnowledgeGraph:
    """SQL-backed entity/relation graph with Personalized PageRank retrieval."""

    def __init__(
        self,
        storage: Any,
        functional_predicates: Optional[Iterable[str]] = None,
        engine: Optional[GraphEngine] = None,
        ontology: Optional[Ontology] = None,
        adjudicator: Optional[Adjudicator] = None,
    ):
        self.storage = storage
        self.ontology = ontology or default_ontology(functional_predicates)
        self.engine = engine or GraphEngine(storage)
        self.adjudicator = adjudicator  # async LLM tie-breaker for ambiguous mentions

    @property
    def functional(self) -> Set[str]:
        return self.ontology.functional_predicates()

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

    @staticmethod
    def _norm(name: str) -> str:
        n = normalize_entity(name)
        return "user" if n in SELF_NAMES else n

    def _resolver(self, scope: Scope) -> EntityResolver:
        def scoped(q, model):
            q = q.filter(model.namespace == scope.namespace if scope.namespace else model.namespace.is_(None))
            return q.filter(model.user_id == scope.user_id if scope.user_id else model.user_id.is_(None))

        return EntityResolver(scoped)

    def _create_entity(self, session: Any, scope: Scope, name: str, entity_type: str, now: datetime) -> Entity:
        norm = self._norm(name)
        is_self = norm == "user"
        ent = Entity(
            namespace=scope.namespace, user_id=scope.user_id, agent_id=scope.agent_id,
            name="User" if is_self else name[:255], norm_name=norm[:255],
            entity_type="person" if is_self else (entity_type or "concept"),
            is_self=1 if is_self else 0, mention_count=0, first_seen=now, last_seen=now,
        )
        session.add(ent)
        session.flush()
        return ent

    def _self_entity(self, session: Any, scope: Scope, now: datetime, create: bool = True) -> Optional[Entity]:
        q = self._resolver(scope)._scoped(session.query(Entity).filter(Entity.is_self == 1), Entity)
        ent = q.first()
        if ent is None and create:
            ent = self._create_entity(session, scope, "User", "person", now)
        return ent

    def _add_alias(self, session: Any, scope: Scope, ent: Entity, norm: str, source: str) -> None:
        if not norm or norm == ent.norm_name:
            return
        exists = session.query(EntityAlias.id).filter(EntityAlias.entity_id == ent.id, EntityAlias.alias == norm).first()
        if not exists:
            session.add(EntityAlias(namespace=scope.namespace, user_id=scope.user_id, entity_id=ent.id,
                                    alias=norm[:255], source=source))

    def _touch(self, session: Any, scope: Scope, ent: Entity, m: Mention, how: str, now: datetime) -> None:
        """A mention linked to ``ent``: refresh it and keep the surface form as an alias."""
        ent.last_seen = now
        if ent.entity_type in (None, "", "concept") and m.etype not in (None, "", "concept"):
            ent.entity_type = m.etype
        if m.norm == ent.norm_name or ent.is_self:
            return
        if len(m.norm.split()) > len(ent.norm_name.split()) and how in ("partial", "nickname"):
            # "Ana" then "Ana Silva": the fuller name becomes the display name
            old = ent.norm_name
            ent.name, ent.norm_name = m.name[:255], m.norm[:255]
            self._add_alias(session, scope, ent, old, "partial")
        else:
            self._add_alias(session, scope, ent, m.norm, how if how in ("nickname", "partial", "llm") else "surface")

    def _self_aliases(self, session: Any, scope: Scope) -> Set[str]:
        """Names the user goes by in this scope (``set_user_name``), normalised."""
        ent = self._self_entity(session, scope, utcnow(), create=False)
        if ent is None:
            return set()
        return {a for (a,) in session.query(EntityAlias.alias).filter(EntityAlias.entity_id == ent.id).all()}

    def set_user_name(self, scope: Scope, name: str) -> None:
        """
        The user's own name (as Zep keeps a user's first name on the user node): mentions of it
        resolve to the user's entity, and extraction writes facts with the name, not "User".
        """
        norm = normalize_entity(name)
        if not norm or norm in SELF_NAMES:
            return
        session = self.storage.get_session()
        try:
            ent = self._self_entity(session, scope, utcnow())
            ent.name = name.strip()[:255]
            self._add_alias(session, scope, ent, norm, "user")
            session.commit()
        finally:
            session.close()

    def user_name(self, scope: Scope) -> Optional[str]:
        session = self.storage.get_session()
        try:
            ent = self._self_entity(session, scope, utcnow(), create=False)
            return ent.name if ent is not None and ent.name and ent.name != "User" else None
        finally:
            session.close()

    def _get_or_create_entity(
        self, session: Any, scope: Scope, name: str, entity_type: str = "concept", now: Optional[datetime] = None
    ) -> Optional[Entity]:
        """Resolve one name without fact context (episode mentions), creating it if new."""
        norm = self._norm(name)
        if not norm:
            return None
        if norm in self._self_aliases(session, scope):
            norm = "user"
        now = now or utcnow()
        if norm == "user":
            ent = self._self_entity(session, scope, now)
            ent.last_seen = now
            return ent
        m = Mention(name, norm, self.ontology.entity_type(entity_type))
        ids, _ = self._resolver(scope).resolve(session, [m], {}, set())
        ent = session.get(Entity, ids[norm]) if ids.get(norm) else None
        if ent is None:
            return self._create_entity(session, scope, name, m.etype, now)
        self._touch(session, scope, ent, m, self._how(ent, m), now)
        return ent

    def _edge(self, scope: Scope, src: Tuple[str, int], kind: str, dst: Tuple[str, int], **attrs: Any) -> Edge:
        return Edge(
            namespace=self.storage.namespace or scope.namespace,
            user_id=scope.user_id,
            agent_id=scope.agent_id,
            src_kind=src[0], src_id=int(src[1]), kind=kind, dst_kind=dst[0], dst_id=int(dst[1]),
            weight=attrs.pop("weight", 1.0), confidence=attrs.pop("confidence", 1.0),
            created_at=attrs.pop("created_at", utcnow()), **attrs,
        )

    def _mentions(self, memory: Dict[str, Any], self_aliases: Set[str] = frozenset()) -> Tuple[
            Dict[str, Mention], List[Rel], Set[str]]:
        """
        Entities and canonical relations of one fact:
        ({norm: Mention}, [(subject norm, predicate, object norm, edge attrs)], norms named as entities).
        """
        content = memory.get("content") or ""
        entities, relations = memory.get("entities"), memory.get("relations")
        h_ents, h_rels = self.extract(content)
        entities = list(entities or []) or h_ents
        relations = list(relations or []) or h_rels
        ont = self.ontology
        mentions: Dict[str, Mention] = {}

        def mention(name: str, etype: Optional[str]) -> Optional[str]:
            norm = self._norm(name)
            if not norm:
                return None
            if norm in self_aliases:  # the user's own name
                norm = "user"
            if norm != "user":
                t = ont.entity_type(etype)
                cur = mentions.get(norm)
                if cur is None:
                    mentions[norm] = Mention(str(name).strip(), norm, t)
                elif cur.etype == "concept" and t != "concept":
                    cur.etype = t
            return norm

        named: Set[str] = set()
        for e in entities:
            name = e.get("name") if isinstance(e, dict) else str(e)
            n = mention(name, e.get("type", "concept") if isinstance(e, dict) else "concept")
            if n:
                named.add(n)
        rels: List[Rel] = []
        for subj, pred, obj in relations:
            r = ont.resolve(pred)
            if not r.name:
                continue
            if r.swap:
                subj, obj = obj, subj
            p = ont.predicates.get(r.name)
            sn = mention(subj, p.domain[0] if p is not None and len(p.domain) == 1 else None)
            on = mention(obj, p.range[0] if p is not None and len(p.range) == 1 else None)
            if not sn or not on or sn == on:
                continue
            flipped = r.swap
            if ont.is_symmetric(r.name) and on == "user":  # symmetric edges: the user first
                sn, on, flipped = on, sn, not flipped
            word, end = raw_role(r.raw)
            attrs: Dict[str, Any] = {}
            extra: Dict[str, Any] = {}
            if r.raw and r.raw != r.name:
                extra["raw"] = r.raw
            if word:
                extra["role"] = word
                extra["role_end"] = ("src" if end == "subject" else "dst") if not flipped else (
                    "dst" if end == "subject" else "src")
            if not r.known:
                extra["custom"] = True
            st = mentions[sn].etype if sn in mentions else "person"
            ot = mentions[on].etype if on in mentions else "person"
            if not ont.type_ok(r.name, st, ot):
                extra["type_mismatch"] = True
                attrs["confidence"] = 0.6
            if extra:
                attrs["extra"] = extra
            rels.append((sn, r.name, on, attrs))
            if sn in mentions:
                mentions[sn].relations.append((r.name, "subject", on))
            if on in mentions:
                mentions[on].relations.append((r.name, "object", sn))
        return mentions, rels, named

    def _resolve(self, session: Any, scope: Scope, memory: Dict[str, Any], now: datetime, create_self: bool):
        mentions, rels, named = self._mentions(memory, self._self_aliases(session, scope))
        needs_self = any("user" in (sn, on) for sn, _, on, _ in rels) or "user" in named
        self_ent = self._self_entity(session, scope, now, create=create_self) if needs_self or not create_self else None
        resolved: Dict[str, int] = {"user": self_ent.id} if self_ent is not None else {}
        self_ids = {self_ent.id} if self_ent is not None else set()
        ids, uncertain = self._resolver(scope).resolve(session, list(mentions.values()), resolved, self_ids)
        return mentions, rels, named, self_ent, ids, uncertain

    def pending_resolutions(self, memory: Dict[str, Any], scope: Scope) -> List[Dict[str, Any]]:
        """Mentions of a fact the rules could not settle, with candidate summaries (read-only)."""
        session = self.storage.get_session()
        try:
            _, _, _, _, _, uncertain = self._resolve(session, scope, memory, utcnow(), create_self=False)
            return [{"norm": u.mention.norm, "name": u.mention.name, "type": u.mention.etype, "reason": u.reason,
                     "candidates": [{"id": c.entity.id, "name": c.entity.name, "type": c.entity.entity_type,
                                     "summary": c.entity.summary, "how": c.how} for c in u.candidates]}
                    for u in uncertain]
        finally:
            session.rollback()
            session.close()

    async def aingest_memory(self, memory: Dict[str, Any], scope: Scope) -> List[int]:
        """``ingest_memory`` that first asks the adjudicator (LLM) about ambiguous mentions."""
        decisions = None
        if self.adjudicator is not None and memory.get("id") and memory.get("content"):
            try:
                items = self.pending_resolutions(memory, scope)
                if items:
                    decisions = await self.adjudicator(memory["content"], items)
                    allowed = {it["norm"]: {c["id"] for c in it["candidates"]} for it in items}
                    decisions = {k: v for k, v in (decisions or {}).items()
                                 if k in allowed and (v is None or v in allowed[k])}
            except Exception as e:  # noqa: BLE001
                logger.debug("Entity adjudication skipped: %s", e)
                decisions = None
        return self.ingest_memory(memory, scope, decisions=decisions)

    def ingest_memory(self, memory: Dict[str, Any], scope: Scope,
                      decisions: Optional[Dict[str, Optional[int]]] = None) -> List[int]:
        """
        Link a stored fact into the graph. Returns ids of *other* facts whose
        single-valued relation was superseded by this one (candidates to retire).
        ``decisions`` (from ``aingest_memory``) settle ambiguous mentions: entity id or None (new).
        """
        mid = memory.get("id")
        content = memory.get("content") or ""
        if not mid or not content:
            return []
        scope = Scope(scope.namespace, scope.user_id, scope.session_id,
                      scope.agent_id or memory.get("agent_id"))

        superseded_facts: List[int] = []
        session = self.storage.get_session()
        try:
            now = utcnow()
            seq = self.storage.next_graph_seq(session)
            valid_from = memory.get("valid_from") if isinstance(memory.get("valid_from"), datetime) else now
            mentions, rels, named, self_ent, ids, uncertain = self._resolve(session, scope, memory, now, True)
            how_by_norm = {u.mention.norm: "llm" for u in uncertain}
            for norm, eid in (decisions or {}).items():
                ids[norm] = eid
            ents: Dict[str, Entity] = {}
            fresh: Dict[str, Entity] = {}  # names new to the graph: earlier turns may mention them (G2)
            if self_ent is not None:
                self_ent.last_seen = now
                ents["user"] = self_ent
            for norm, m in mentions.items():
                eid = ids.get(norm)
                ent = session.get(Entity, eid) if eid else None
                if ent is None:
                    ent = self._create_entity(session, scope, m.name, m.etype, now)
                    fresh[norm] = ent
                else:
                    how = how_by_norm.get(norm) if decisions and norm in decisions else None
                    how = how or self._how(ent, m)
                    self._touch(session, scope, ent, m, how, now)
                    if how != "exact":
                        fresh[norm] = ent
                ents[norm] = ent

            about: Dict[int, str] = {}  # entity id -> role

            def link(ent: Entity, role: str) -> None:
                if ent.id not in about:
                    ent.mention_count = (ent.mention_count or 0) + (0 if ent.is_self else 1)
                    about[ent.id] = role
                elif role != "mention" and about[ent.id] == "mention":
                    about[ent.id] = role

            for norm in named:
                if norm in ents:
                    link(ents[norm], "mention")
            touched: Set[int] = set()
            for sn, pred, on, attrs in rels:
                s_ent, o_ent = ents.get(sn), ents.get(on)
                if s_ent is None or o_ent is None or s_ent.id == o_ent.id:
                    continue
                link(s_ent, "subject")
                link(o_ent, "object")
                symmetric = self.ontology.is_symmetric(pred)
                q = session.query(Edge).filter(Edge.kind == "relates", Edge.src_kind == "entity",
                                               Edge.predicate == pred, Edge.invalidated_at.is_(None))
                if symmetric:
                    ends = [s_ent.id, o_ent.id]
                    q = q.filter(Edge.src_id.in_(ends) | Edge.dst_id.in_(ends))
                else:
                    q = q.filter(Edge.src_id == s_ent.id)
                existing = q.all()
                pair = {(s_ent.id, o_ent.id)} | ({(o_ent.id, s_ent.id)} if symmetric else set())
                if any((r.src_id, r.dst_id) in pair for r in existing):
                    continue  # same edge already asserted
                rel = self._edge(
                    scope, ("entity", s_ent.id), "relates", ("entity", o_ent.id),
                    predicate=pred, valid_from=valid_from, created_at=now,
                    evidence_kind="fact", evidence_id=int(mid), seq=seq, **attrs,
                )
                session.add(rel)
                session.flush()
                touched |= {s_ent.id, o_ent.id}
                if self.ontology.is_functional(pred):
                    for old in existing:
                        old.seq = seq
                        old.invalidated_at = now
                        old.valid_to = valid_from
                        old.superseded_by_id = rel.id
                        touched |= {old.src_id, old.dst_id}
                        if old.evidence_kind == "fact" and old.evidence_id and old.evidence_id != mid:
                            superseded_facts.append(int(old.evidence_id))
            session.add_all([self._edge(scope, ("fact", int(mid)), "about", ("entity", eid),
                                        role=role, valid_from=valid_from, created_at=now, seq=seq)
                             for eid, role in about.items()])
            self._refresh_summaries(session, touched, now)
            session.commit()
            fresh_ids = {norm: ent.id for norm, ent in fresh.items() if not ent.is_self}
        except Exception as e:
            session.rollback()
            logger.warning("Knowledge graph ingest failed for memory %s: %s", mid, e)
            return []
        finally:
            session.close()
        if fresh_ids:
            self.backfill_turn_mentions(scope, fresh_ids)
        return superseded_facts

    @staticmethod
    def _how(ent: Entity, m: Mention) -> str:
        if ent.norm_name == m.norm:
            return "exact"
        a, b = m.norm.split(), ent.norm_name.split()
        return "partial" if a and b and a[0] == b[0] else "nickname"

    def _refresh_summaries(self, session: Any, entity_ids: Iterable[int], now: datetime, limit: int = 12) -> None:
        """Fact sheet per entity from its valid relations, newest first (used by resolution and cards)."""
        for eid in set(entity_ids):
            ent = session.get(Entity, eid)
            if ent is None or ent.is_self:
                continue
            rows = (session.query(Edge.src_id, Edge.predicate, Edge.dst_id)
                    .filter(Edge.kind == "relates", Edge.invalidated_at.is_(None),
                            (Edge.src_id == eid) | (Edge.dst_id == eid), Edge.src_kind == "entity")
                    .order_by(Edge.id.desc()).limit(limit).all())
            ids = {r[0] for r in rows} | {r[2] for r in rows}
            names = dict(session.query(Entity.id, Entity.name).filter(Entity.id.in_(list(ids))).all()) if ids else {}
            aliases = [a for (a,) in session.query(EntityAlias.alias).filter(EntityAlias.entity_id == eid).limit(5)]
            parts = [f"{names.get(s, '?')} {p.replace('_', ' ')} {names.get(d, '?')}" for s, p, d in rows]
            text_ = (f"also called {', '.join(aliases)}. " if aliases else "") + "; ".join(parts)
            ent.summary = text_[:1000] or None
            ent.summary_updated_at = now

    def find_entity(self, name_or_id: Any, scope: Scope) -> Optional[int]:
        """An entity id in this scope by id, name or alias (merges followed); None if absent."""
        session = self.storage.get_session()
        try:
            if isinstance(name_or_id, int):
                q = self._scoped(session.query(Entity.id).filter(Entity.id == name_or_id), Entity, scope)
                ids = [r[0] for r in q.all()]
            else:
                norm = self._norm(str(name_or_id))
                ids = [r[0] for r in self._scoped(
                    session.query(Entity.id).filter(Entity.norm_name == norm), Entity, scope).all()]
                ids += [r[0] for r in self._scoped(
                    session.query(EntityAlias.entity_id).filter(EntityAlias.alias == norm), EntityAlias, scope).all()]
            canon = sorted(set(self._follow_merges(session, ids).values()))
            return canon[0] if len(canon) == 1 else None
        finally:
            session.close()

    def merge_entities(self, duplicate_id: int, into_id: int, scope: Scope, source: str = "merge") -> bool:
        """
        Merge ``duplicate_id`` into ``into_id``: its names become aliases, its edges move
        (marked ``extra.merged_from``, so the merge can be undone) and an ``alias_of``
        edge records it.
        """
        session = self.storage.get_session()
        try:
            dup, keep = session.get(Entity, int(duplicate_id)), session.get(Entity, int(into_id))
            if dup is None or keep is None or dup.id == keep.id or dup.is_self or keep.is_self:
                return False
            if (dup.namespace, dup.user_id) != (keep.namespace, keep.user_id):
                return False
            now, seq = utcnow(), self.storage.next_graph_seq(session)
            self._add_alias(session, scope, keep, dup.norm_name, source)
            for a in session.query(EntityAlias).filter(EntityAlias.entity_id == dup.id).all():
                if a.alias == keep.norm_name:
                    session.delete(a)
                else:
                    a.entity_id = keep.id
            for e in session.query(Edge).filter(Edge.invalidated_at.is_(None), Edge.kind != "alias_of",
                                                ((Edge.src_kind == "entity") & (Edge.src_id == dup.id))
                                                | ((Edge.dst_kind == "entity") & (Edge.dst_id == dup.id))).all():
                if e.src_kind == "entity" and e.src_id == dup.id:
                    e.src_id = keep.id
                if e.dst_kind == "entity" and e.dst_id == dup.id:
                    e.dst_id = keep.id
                e.extra = {**(e.extra or {}), "merged_from": dup.id}
                e.seq = seq
            keep.mention_count = (keep.mention_count or 0) + (dup.mention_count or 0)
            if keep.entity_type in (None, "", "concept"):
                keep.entity_type = dup.entity_type
            dup.merged_into_id = keep.id
            session.add(self._edge(scope, ("entity", dup.id), "alias_of", ("entity", keep.id),
                                   created_at=now, seq=seq, source=source))
            self._refresh_summaries(session, [keep.id], now)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.warning("Entity merge failed: %s", e)
            return False
        finally:
            session.close()

    def ingest_episode(self, episode_id: int, names: Sequence[str], scope: Scope) -> None:
        """Link an episode to the entities it mentions."""
        if not episode_id or not names:
            return
        session = self.storage.get_session()
        try:
            seq = self.storage.next_graph_seq(session)
            seen: Set[int] = set()
            for name in names:
                ent = self._get_or_create_entity(session, scope, name)
                if ent is None or ent.is_self or ent.id in seen:
                    continue
                seen.add(ent.id)
                session.add(self._edge(scope, ("episode", int(episode_id)), "mentions", ("entity", ent.id), seq=seq))
                ent.mention_count = (ent.mention_count or 0) + 1
            session.commit()
        except Exception as e:
            session.rollback()
            logger.debug("Episode graph ingest failed: %s", e)
        finally:
            session.close()

    # ------------------------------------------------------------------ turn mentions (G2)

    def _mention_names(self, session: Any, scope: Scope) -> Dict[str, Set[int]]:
        """Entity names and aliases of the scope (not the user's own) -> canonical entity ids."""
        skip = self._self_aliases(session, scope) | {"user"}
        raw: List[Tuple[str, int]] = []
        q = session.query(Entity.id, Entity.norm_name, Entity.is_self)
        raw += [(norm, eid) for eid, norm, is_self in self._scoped(q, Entity, scope).all() if not is_self]
        q = session.query(EntityAlias.alias, EntityAlias.entity_id)
        raw += list(self._scoped(q, EntityAlias, scope).all())
        canon = self._follow_merges(session, [eid for _, eid in raw])
        names: Dict[str, Set[int]] = {}
        for norm, eid in raw:
            if norm and len(norm) >= MENTION_MIN_CHARS and norm not in skip and norm not in STOPWORDS:
                names.setdefault(norm, set()).add(canon.get(eid, eid))
        return names

    def _add_turn_mentions(self, session: Any, scope: Scope, turns: Iterable[Tuple[int, str]],
                           names: Dict[str, Set[int]]) -> int:
        """turn -mentions-> entity for whole-word matches, skipping pairs already linked."""
        if not names:
            return 0
        pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b")
        found: Dict[int, Set[int]] = {}
        for tid, content in turns:
            hit: Set[int] = set()
            for m in pat.finditer(turn_body(content)):
                hit |= names.get(m.group(1), set())
            if hit:
                found[int(tid)] = hit
        if not found:
            return 0
        existing = set(
            session.query(Edge.src_id, Edge.dst_id).filter(
                Edge.kind == "mentions", Edge.src_kind == "turn", Edge.dst_kind == "entity",
                Edge.src_id.in_(list(found)), Edge.invalidated_at.is_(None)).all()
        )
        new = [(t, e) for t, es in found.items() for e in sorted(es) if (t, e) not in existing]
        if not new:
            return 0
        seq = self.storage.next_graph_seq(session)
        now = utcnow()
        session.add_all([self._edge(scope, ("turn", t), "mentions", ("entity", e), created_at=now, seq=seq)
                         for t, e in new])
        return len(new)

    def link_turn_mentions(self, scope: Scope, episode_id: Optional[int] = None,
                           turn_ids: Optional[Sequence[int]] = None) -> int:
        """Link an episode's (or the given) turns to the known entities they name. Returns edges added."""
        if episode_id is None and not turn_ids:
            return 0
        session = self.storage.get_session()
        try:
            names = self._mention_names(session, scope)
            q = session.query(Turn.id, Turn.content)
            q = q.filter(Turn.episode_id == int(episode_id)) if episode_id is not None else \
                q.filter(Turn.id.in_([int(t) for t in turn_ids or ()]))
            n = self._add_turn_mentions(session, scope, self._scoped(q, Turn, scope).all(), names)
            session.commit()
            return n
        except Exception as e:
            session.rollback()
            logger.debug("Turn mention linking failed: %s", e)
            return 0
        finally:
            session.close()

    def backfill_turn_mentions(self, scope: Scope, names: Dict[str, int], per_name: int = 200) -> int:
        """A name new to the graph: link the earlier turns that use it (found through the turn index)."""
        session = self.storage.get_session()
        try:
            skip = self._self_aliases(session, scope) | {"user"}
            names = {n: e for n, e in names.items()
                     if n and len(n) >= MENTION_MIN_CHARS and n not in skip and n not in STOPWORDS}
            ids: Set[int] = set()
            for norm in names:
                ids |= {i for i, _ in self.storage.lexical_search("turn", norm, k=per_name, user_id=scope.user_id)}
            if not ids:
                return 0
            turns = self._scoped(session.query(Turn.id, Turn.content).filter(Turn.id.in_(list(ids))), Turn, scope).all()
            n = self._add_turn_mentions(session, scope, turns, {k: {v} for k, v in names.items()})
            session.commit()
            return n
        except Exception as e:
            session.rollback()
            logger.debug("Turn mention backfill failed: %s", e)
            return 0
        finally:
            session.close()

    def turn_walk(self, query: str, scope: Scope, seed_turns: Dict[int, float], k: int = 50,
                  include_shared: bool = False) -> List[Tuple[int, float]]:
        """
        Rank turns by a typed Personalized PageRank (G2). Seeds: the query's entities, plus the top
        first-stage turns at a small weight (HippoRAG 2). The walk follows facts to their source
        turns and entities to the turns that name them.
        """
        try:
            seeds: Dict[Tuple[str, int], float] = {}
            ents = self.match_query_entities(query, scope, include_shared)
            session = self.storage.get_session()
            try:
                self_ids = [r[0] for r in self._scoped(session.query(Entity.id).filter(Entity.is_self == 1),
                                                       Entity, scope, include_shared).all()]
            finally:
                session.close()
            seeds.update({("entity", e): w for e, w in ents.items() if e not in self_ids})
            top = sorted(seed_turns.items(), key=lambda kv: kv[1], reverse=True)[:TURN_SEEDS]
            if top:
                v = [s for _, s in top]
                mean, sd = sum(v) / len(v), (sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** 0.5 or 1.0
                for t, sc in top:
                    seeds[("turn", int(t))] = TURN_SEED_WEIGHT * math.exp((sc - mean) / sd)
            if not seeds:
                return []
            res = self.engine.ppr(self.storage.namespace or scope.namespace, scope.user_id, seeds,
                                  include_shared=include_shared, exclude=[("entity", i) for i in self_ids],
                                  weights=TURN_WALK_WEIGHTS)
            return res.top("turn", k)
        except Exception as e:
            logger.debug("Turn walk failed: %s", e)
            return []

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

    def _follow_merges(self, session: Any, ids: Iterable[int]) -> Dict[int, int]:
        """entity id -> the entity it was merged into (or itself)."""
        out = {i: i for i in ids}
        pending = set(out)
        for _ in range(5):
            if not pending:
                break
            rows = session.query(Entity.id, Entity.merged_into_id).filter(
                Entity.id.in_(list(pending)), Entity.merged_into_id.isnot(None)).all()
            moved = {i: t for i, t in rows}
            pending = set()
            for k, v in out.items():
                if v in moved:
                    out[k] = moved[v]
                    pending.add(moved[v])
        return out

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

        def seed(eid: int, w: float) -> None:
            seeds[eid] = max(seeds.get(eid, 0.0), w)

        session = self.storage.get_session()
        try:
            if grams:
                hits: List[Tuple[int, str, float]] = []
                q = session.query(Entity.id, Entity.norm_name, Entity.is_self).filter(Entity.norm_name.in_(list(grams)))
                hits += [(eid, norm, 1.0) for eid, norm, is_self in self._scoped(q, Entity, scope, include_shared).all()
                         if not is_self]
                q = session.query(EntityAlias.entity_id, EntityAlias.alias).filter(EntityAlias.alias.in_(list(grams)))
                hits += [(eid, a, 0.9) for eid, a in self._scoped(q, EntityAlias, scope, include_shared).all()]
                canon = self._follow_merges(session, [h[0] for h in hits])
                for eid, norm, w in hits:
                    seed(canon[eid], w + 0.25 * (len(norm.split()) - 1))
            # Role and predicate words on the user's own edges, in either direction:
            # "my sister" -> (User)-sibling_of->(Emma); "my mother" -> (Ann)-parent_of->(User)
            q_stems = {_stem(w) for w in words if w.lower() not in STOPWORDS}
            if q_stems:
                self_q = self._scoped(session.query(Entity.id).filter(Entity.is_self == 1), Entity, scope, include_shared)
                self_ids = {r[0] for r in self_q.all()}
                if self_ids:
                    rels = (
                        session.query(Edge.predicate, Edge.src_id, Edge.dst_id, Edge.extra)
                        .filter(
                            Edge.kind == "relates",
                            Edge.src_kind == "entity",
                            Edge.src_id.in_(list(self_ids)) | Edge.dst_id.in_(list(self_ids)),
                            Edge.invalidated_at.is_(None),
                        )
                        .all()
                    )
                    for pred, src, dst, extra in rels:
                        self_is_src = src in self_ids
                        other = dst if self_is_src else src
                        if other in self_ids:
                            continue
                        p = self.ontology.predicates.get(pred)
                        has_roles = p is not None and bool(p.subject_terms or p.object_terms)
                        terms = {_stem(w) for w in self.ontology.query_terms(pred, other_is_object=self_is_src)}
                        if terms & q_stems:
                            seed(other, 0.6 if has_roles else 0.8)
                        ex = extra if isinstance(extra, dict) else {}
                        role = ex.get("role")
                        if role and (ex.get("role_end") == "dst") == self_is_src and _stem(role) in q_stems:
                            seed(other, 0.8)  # the extractor's own word ("sister") beats the generic group
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
            al_q = self._scoped(session.query(EntityAlias.entity_id).filter(EntityAlias.alias == norm), EntityAlias, scope)
            ids = sorted(set(self._follow_merges(session, [r[0] for r in ent_q.all()] + [r[0] for r in al_q.all()])
                             .values()))
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
