"""
Entity resolution (G1, docs/memory_graph_design.md §4): which existing entity a mention
in a new fact refers to.

Deterministic first, the LLM only for what stays ambiguous (the split Graphiti's dedup
helpers use; Mem0 and Zep likewise resolve exact names without a model call):

  candidates   exact normalised name, stored aliases, first-name nickname variants
               ("Bob" ~ "Robert Chen") and partial person names ("Ana" ~ "Ana Silva");
               two full names must share the last name; a near-miss last name ("Sliva")
               is only a "fuzzy" candidate, never linked without the LLM
  link         a unique exact match; else a unique alias / nickname / partial match
  reject       a candidate whose family role contradicts the new fact: if Ana is the
               user's sister, "my daughter Ana" is someone else (kinship roles are
               mutually exclusive between the same two people)
  context      among several candidates, the one already connected to the other
               entities of the same fact (the same relation to the same entity, any
               relation, or facts mentioning both) wins
  uncertain    still ambiguous: one batched LLM call per fact when a model is set;
               otherwise a conservative fallback (the most recent exact match, or a new
               entity: a split is cheap to merge later, a wrong merge corrupts facts)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple

from memorable_ai.core.models import Edge, Entity, EntityAlias

logger = logging.getLogger(__name__)

NICKNAME_GROUPS: Sequence[Sequence[str]] = (
    ("robert", "rob", "bob", "bobby", "robbie"), ("william", "will", "bill", "billy", "liam"),
    ("elizabeth", "liz", "beth", "betty", "eliza", "lizzie", "libby"),
    ("katherine", "catherine", "kathryn", "kate", "katie", "kathy", "cathy", "kat"),
    ("michael", "mike", "mikey", "mick"), ("james", "jim", "jimmy", "jamie"), ("john", "johnny", "jack"),
    ("richard", "rick", "ricky", "rich", "dick"), ("thomas", "tom", "tommy"), ("christopher", "chris"),
    ("christina", "christine", "chris", "tina"), ("alexander", "alex", "sasha", "xander"),
    ("alexandra", "alex", "sasha", "lexi"), ("daniel", "dan", "danny"), ("danielle", "dani"),
    ("matthew", "matt"), ("anthony", "tony"), ("joseph", "joe", "joey"), ("josephine", "jo", "josie"),
    ("samuel", "sam", "sammy"), ("samantha", "sam", "sammy"), ("benjamin", "ben", "benny"),
    ("nicholas", "nick", "nicky"), ("nicole", "nikki"), ("jennifer", "jen", "jenny"),
    ("jessica", "jess", "jessie"), ("margaret", "maggie", "meg", "peggy"), ("rebecca", "becky", "becca"),
    ("patricia", "pat", "patty", "trish"), ("patrick", "pat", "paddy"), ("susan", "sue", "suzy"),
    ("deborah", "deb", "debbie"), ("victoria", "vicky", "tori"), ("andrew", "andy", "drew"),
    ("edward", "ed", "eddie", "ted", "ned"), ("charles", "charlie", "chuck"), ("charlotte", "charlie", "lottie"),
    ("david", "dave", "davey"), ("steven", "stephen", "steve"), ("gregory", "greg"), ("jonathan", "jon"),
    ("timothy", "tim", "timmy"), ("kenneth", "ken", "kenny"), ("ronald", "ron", "ronnie"),
    ("donald", "don", "donnie"), ("joshua", "josh"), ("zachary", "zach", "zack"),
    ("nathan", "nathaniel", "nate"), ("abigail", "abby"), ("isabella", "isabel", "bella", "izzy"),
    ("olivia", "liv", "livvy"), ("gabriel", "gabe"), ("gabriela", "gabriella", "gabi", "gaby"),
    ("francisco", "paco", "pancho"), ("jose", "pepe"), ("manuel", "manny"), ("frederick", "fred", "freddie"),
    ("lawrence", "larry"), ("leonard", "leo", "leon", "len"), ("peter", "pete"), ("philip", "phil"),
    ("raymond", "ray"), ("theodore", "theo", "ted", "teddy"), ("vincent", "vince", "vinny"),
    ("eleanor", "ellie", "nora"), ("madeline", "madeleine", "maddie", "maddy"), ("natalie", "nat"),
    ("pamela", "pam"), ("sandra", "sandy"), ("stephanie", "steph"), ("valentina", "vale", "tina"),
    ("dmitry", "dmitri", "dima"), ("ekaterina", "katya"), ("aleksandr", "sasha"), ("mohammed", "muhammad", "mo"),
)
_NICK: Dict[str, Set[str]] = {}
for _g in NICKNAME_GROUPS:
    for _n in _g:
        _NICK.setdefault(_n, set()).update(_g)

KIN = {"sibling_of", "parent_of", "married_to", "has_partner"}
STRENGTH = {"exact": 3, "alias": 2, "nickname": 1, "partial": 1, "fuzzy": 0}
_SPECIFIC = lambda t: t not in (None, "", "concept")  # noqa: E731


def name_variants(token: str) -> Set[str]:
    return _NICK.get(token, set()) | {token}


def edit_distance(a: str, b: str) -> int:
    """Optimal string alignment distance: edits plus adjacent transpositions ("Silva" / "Sliva" = 1)."""
    prev2: List[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[len(b)]


def is_typo(a: str, b: str) -> bool:
    n = max(len(a), len(b))
    return n >= 4 and a[:1] == b[:1] and edit_distance(a, b) <= (1 if n < 8 else 2)


def looks_like_person(name: str) -> bool:
    toks = (name or "").split()
    return 1 <= len(toks) <= 4 and all(t[:1].isupper() and re.fullmatch(r"[\w'.-]+", t) for t in toks)


def kin_role(pred: str, src: int, dst: int, of: int) -> Optional[str]:
    """Role of entity ``of`` in a kinship edge: sibling / partner / parent / child."""
    if pred == "sibling_of":
        return "sibling"
    if pred in ("married_to", "has_partner"):
        return "partner"
    if pred == "parent_of":
        return "parent" if of == src else "child" if of == dst else None
    return None


@dataclass
class Candidate:
    entity: Entity
    how: str

    @property
    def strength(self) -> int:
        return STRENGTH[self.how]


@dataclass
class Mention:
    name: str
    norm: str
    etype: str = "concept"
    # (canonical predicate, "subject" | "object", other mention's norm)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)


@dataclass
class Uncertain:
    mention: Mention
    candidates: List[Candidate]
    fallback: Optional[int]  # entity id used when nobody decides; None = new entity
    reason: str


class EntityResolver:
    """Read-only: proposes an entity id (or a new entity) for each mention of one fact."""

    def __init__(self, scoped_query: Callable[..., Any]):
        self._scoped = scoped_query  # (query, model) -> query limited to this partition

    # ------------------------------------------------------------------ candidates

    def _canonical(self, session: Any, ent: Entity) -> Entity:
        seen = 0
        while ent is not None and ent.merged_into_id and seen < 5:
            nxt = session.get(Entity, ent.merged_into_id)
            if nxt is None:
                break
            ent, seen = nxt, seen + 1
        return ent

    def candidates(self, session: Any, m: Mention) -> List[Candidate]:
        out: Dict[int, Candidate] = {}

        def add(ent: Optional[Entity], how: str) -> None:
            ent = self._canonical(session, ent) if ent is not None else None
            if ent is None or ent.is_self:
                return
            cur = out.get(ent.id)
            if cur is None or STRENGTH[how] > cur.strength:
                out[ent.id] = Candidate(ent, how)

        for ent in self._scoped(session.query(Entity).filter(Entity.norm_name == m.norm), Entity).all():
            add(ent, "exact")
        alias_ids = [r[0] for r in self._scoped(
            session.query(EntityAlias.entity_id).filter(EntityAlias.alias == m.norm), EntityAlias).all()]
        if alias_ids:
            for ent in session.query(Entity).filter(Entity.id.in_(alias_ids)).all():
                add(ent, "alias")
        if m.etype in ("person", "concept") and looks_like_person(m.name):
            toks = m.norm.split()
            variants = name_variants(toks[0])
            from sqlalchemy import or_

            cond = or_(Entity.norm_name.in_(list(variants)), *[Entity.norm_name.like(f"{v} %") for v in variants])
            q = self._scoped(session.query(Entity).filter(cond), Entity).limit(50)
            for ent in q.all():
                how = self._person_match(toks, ent.norm_name.split(), m.etype, ent.entity_type)
                if how:
                    add(ent, how)
        return list(out.values())

    @staticmethod
    def _person_match(new: List[str], old: List[str], new_type: str, old_type: str) -> Optional[str]:
        if new == old:
            return None  # exact, handled above
        if _SPECIFIC(new_type) and _SPECIFIC(old_type) and new_type != old_type:
            return None
        if old_type not in ("person", "concept", None, ""):
            return None
        same_first = new[0] == old[0]
        if not same_first and old[0] not in name_variants(new[0]):
            return None
        how = "partial" if same_first else "nickname"
        if len(new) > 1 and len(old) > 1:
            if new[-1] == old[-1]:
                return how
            if is_typo(new[-1], old[-1]):
                return "fuzzy"  # "Ana Sliva" vs "Ana Silva": only the LLM may link it
            return None  # two different full names
        return how

    # ------------------------------------------------------------------ signals

    def _kin_conflict(self, session: Any, cand: Entity, m: Mention, resolved: Dict[str, int]) -> bool:
        for pred, end, other_norm in m.relations:
            other = resolved.get(other_norm)
            if pred not in KIN or other is None:
                continue
            new_role = kin_role(pred, 1 if end == "subject" else 2, 2 if end == "subject" else 1, 1)
            rows = session.query(Edge.src_id, Edge.dst_id, Edge.predicate).filter(
                Edge.kind == "relates", Edge.invalidated_at.is_(None), Edge.predicate.in_(list(KIN)),
                ((Edge.src_id == cand.id) & (Edge.dst_id == other)) | ((Edge.src_id == other) & (Edge.dst_id == cand.id)),
            ).all()
            for src, dst, p in rows:
                old_role = kin_role(p, src, dst, cand.id)
                if old_role and new_role and old_role != new_role:
                    return True
        return False

    def _context_score(self, session: Any, cand: Entity, m: Mention, resolved: Dict[str, int],
                       self_ids: Set[int]) -> float:
        score = 0.0
        others = {v for k, v in resolved.items() if k != m.norm}
        for pred, end, other_norm in m.relations:  # the same relation to the same entity
            other = resolved.get(other_norm)
            if other is None:
                continue
            src, dst = (cand.id, other) if end == "subject" else (other, cand.id)
            hit = session.query(Edge.id).filter(
                Edge.kind == "relates", Edge.invalidated_at.is_(None), Edge.predicate == pred,
                ((Edge.src_id == src) & (Edge.dst_id == dst)) | ((Edge.src_id == dst) & (Edge.dst_id == src)),
            ).first()
            if hit:
                score += 3
        plain = [o for o in others if o not in self_ids]
        if plain:
            n = session.query(Edge.id).filter(
                Edge.kind == "relates", Edge.invalidated_at.is_(None),
                ((Edge.src_id == cand.id) & Edge.dst_id.in_(plain)) | ((Edge.dst_id == cand.id) & Edge.src_id.in_(plain)),
            ).count()
            score += 2 * min(n, 3)
            cand_facts = {r[0] for r in session.query(Edge.src_id).filter(
                Edge.kind == "about", Edge.dst_kind == "entity", Edge.dst_id == cand.id,
                Edge.invalidated_at.is_(None)).limit(300).all()}
            if cand_facts:
                co = session.query(Edge.src_id).filter(
                    Edge.kind == "about", Edge.dst_kind == "entity", Edge.dst_id.in_(plain),
                    Edge.src_id.in_(list(cand_facts)), Edge.invalidated_at.is_(None)).distinct().count()
                score += min(co, 3)
        return score

    # ------------------------------------------------------------------ decision

    def resolve(self, session: Any, mentions: Sequence[Mention], resolved: Dict[str, int],
                self_ids: Set[int]) -> Tuple[Dict[str, Optional[int]], List[Uncertain]]:
        """
        ``resolved`` holds already-fixed mentions (the user). Returns entity ids per norm
        (None = create) and the mentions an LLM should look at.
        """
        decided: Dict[str, Optional[int]] = {}
        pending: List[Tuple[Mention, List[Candidate]]] = []
        for m in mentions:  # pass 1: unique strong matches give context for the rest
            cands = self.candidates(session, m)
            top = max((c.strength for c in cands), default=-1)
            tops = [c for c in cands if c.strength == top]
            if not cands:
                decided[m.norm] = None
            elif top >= 1 and len(cands) == 1 or (top == 3 and len(tops) == 1):
                decided[m.norm] = tops[0].entity.id
                resolved[m.norm] = tops[0].entity.id
            else:
                pending.append((m, cands))
        uncertain: List[Uncertain] = []
        # pass 2: kinship conflicts (needs the other ends), then graph context
        by_norm = {x.norm: x for x in mentions}
        for norm, ent_id in list(decided.items()):
            mention = by_norm.get(norm)
            if ent_id is None or mention is None or not mention.relations:
                continue
            ent = session.get(Entity, ent_id)
            if ent is not None and self._kin_conflict(session, ent, mention, resolved):
                decided[norm] = None  # same name, different person
                resolved.pop(norm, None)
        for m, cands in pending:
            cands = [c for c in cands if not self._kin_conflict(session, c.entity, m, resolved)]
            if not cands:
                decided[m.norm] = None
                continue
            strong = [c for c in cands if c.strength > 0]
            scored = sorted(((self._context_score(session, c.entity, m, resolved, self_ids), c) for c in strong),
                            key=lambda x: -x[0])
            if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                decided[m.norm] = scored[0][1].entity.id
                resolved[m.norm] = decided[m.norm]
                continue
            if len(strong) == 1 and len(cands) == 1:
                decided[m.norm] = strong[0].entity.id
                continue
            exact = sorted((c for c in cands if c.how == "exact"),
                           key=lambda c: c.entity.last_seen or c.entity.first_seen, reverse=True)
            fallback = exact[0].entity.id if exact else None
            decided[m.norm] = fallback
            reason = "fuzzy" if not strong else "ambiguous"
            uncertain.append(Uncertain(m, cands, fallback, reason))
        return decided, uncertain


# ---------------------------------------------------------------------- LLM adjudication

RESOLVE_PROMPT = """You maintain a knowledge graph of the people, places and things in a user's life.
A new memory mentions names that may refer to entities already in the graph.

New memory: {text}

{items}

For each mention, answer with the number of the candidate it refers to, or "new" if it is a
different entity from every candidate. Match only when the new memory fits the candidate's
known facts; when in doubt answer "new".
Return JSON only: {{"<mention>": <number or "new">, ...}}"""


def adjudication_prompt(text: str, items: Sequence[Dict[str, Any]]) -> str:
    blocks = []
    for it in items:
        lines = [f'Mention "{it["name"]}" ({it.get("type") or "concept"}):']
        for i, c in enumerate(it["candidates"], 1):
            lines.append(f"  {i}. {c['name']} ({c.get('type') or 'concept'}): {c.get('summary') or 'no known facts'}")
        blocks.append("\n".join(lines))
    return RESOLVE_PROMPT.format(text=text, items="\n\n".join(blocks))


def parse_adjudication(raw: Any, items: Sequence[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    """LLM answer -> {mention norm: entity id or None (new)}; unknown answers are skipped."""
    if not isinstance(raw, dict):
        return {}
    by_name = {str(it["name"]).strip().lower(): it for it in items}
    out: Dict[str, Optional[int]] = {}
    for key, val in raw.items():
        it = by_name.get(str(key).strip().lower())
        if it is None:
            continue
        if isinstance(val, str) and val.strip().lower() == "new":
            out[it["norm"]] = None
            continue
        try:
            idx = int(val)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(it["candidates"]):
            out[it["norm"]] = int(it["candidates"][idx - 1]["id"])
    return out


Adjudicator = Callable[[str, List[Dict[str, Any]]], Awaitable[Dict[str, Optional[int]]]]


def llm_adjudicator(model: str) -> Adjudicator:
    """One small JSON call per fact with uncertain mentions; failures fall back to the rules."""

    async def adjudicate(text: str, items: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
        from memorable_ai.core import llm_client

        if not items or not llm_client.llm_enabled(model):
            return {}
        try:
            resp = await llm_client.acompletion_with_fallback(
                model=model, messages=[{"role": "user", "content": adjudication_prompt(text, items)}],
                max_tokens=60 + 20 * len(items), temperature=0)
            return parse_adjudication(llm_client.parse_json(llm_client.message_content(resp)), items)
        except Exception as e:  # noqa: BLE001 - resolution must never break ingestion
            logger.debug("Entity adjudication failed: %s", e)
            return {}

    return adjudicate
