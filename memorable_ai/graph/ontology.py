"""
Ontology: entity types and a predicate registry for the knowledge graph (G1,
docs/memory_graph_design.md §4).

A predicate entry says whether it is single-valued (``functional``: a new value supersedes
the old, Graphiti-style invalidation), ``symmetric``, its ``inverse`` (stored in this
predicate's direction), its expected subject/object types and its aliases.
``Ontology.resolve`` maps whatever an extractor wrote ("moved_to", "child_of",
"has_wife") onto a canonical predicate, possibly swapping subject and object.
Unknown predicates stay as written and are flagged ``custom`` (dreaming later folds them
into canonical ones). Applications register their own types and predicates, as in
Graphiti custom types and Cognee ontologies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

DEFAULT_TYPES: Dict[str, str] = {
    "person": "a human being, including the user",
    "organization": "a company, team, school or institution",
    "place": "a city, country, address or venue",
    "project": "something being built or worked on",
    "service": "a running software service or system",
    "tool": "software, a library or an instrument used to do work",
    "file": "a file, document or repository path",
    "product": "a product, device or consumer good",
    "event": "a dated happening: trip, meeting, wedding, concert",
    "concept": "anything else: an activity, topic, food, skill",
}


@dataclass(frozen=True)
class Predicate:
    name: str
    functional: bool = False
    symmetric: bool = False
    inverse: Optional[str] = None
    domain: Tuple[str, ...] = ()
    range: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    description: str = ""
    # Words naming each end of the relation in questions: parent_of has subject_terms
    # {mother, father, ...} and object_terms {son, daughter, ...}, so "my mother" only
    # follows edges where the user is the child. Empty: the predicate's own words.
    subject_terms: Tuple[str, ...] = ()
    object_terms: Tuple[str, ...] = ()


P = Predicate
PERSON = ("person",)
DEFAULT_PREDICATES: List[Predicate] = [
    # people
    P("married_to", functional=True, symmetric=True, domain=PERSON, range=PERSON,
      aliases=("spouse_of", "husband_of", "wife_of", "has_wife", "has_husband", "has_spouse", "wed_to"),
      object_terms=("wife", "husband", "spouse", "married")),
    P("has_partner", functional=True, symmetric=True, domain=PERSON, range=PERSON,
      aliases=("partner_of", "dating", "girlfriend_of", "boyfriend_of", "has_girlfriend", "has_boyfriend",
               "in_relationship_with", "engaged_to", "fiance_of", "fiancee_of"),
      object_terms=("partner", "girlfriend", "boyfriend", "fiance", "fiancee", "dating")),
    P("sibling_of", symmetric=True, domain=PERSON, range=PERSON,
      aliases=("brother_of", "sister_of", "has_brother", "has_sister", "has_sibling"),
      object_terms=("sibling", "brother", "sister")),
    P("parent_of", inverse="child_of", domain=PERSON, range=PERSON,
      aliases=("mother_of", "father_of", "has_son", "has_daughter", "has_child", "has_kid"),
      subject_terms=("parent", "mother", "father", "mom", "mum", "dad"),
      object_terms=("child", "children", "son", "daughter", "kid")),
    P("friend_of", symmetric=True, domain=PERSON, range=PERSON, aliases=("friends_with", "has_friend", "best_friend_of"),
      object_terms=("friend", "buddy")),
    P("colleague_of", symmetric=True, domain=PERSON, range=PERSON,
      aliases=("coworker_of", "co_worker_of", "works_with", "teammate_of"),
      object_terms=("colleague", "coworker", "teammate")),
    P("manages", inverse="reports_to", aliases=("manager_of", "supervises", "boss_of"),
      subject_terms=("manager", "boss", "supervisor"), object_terms=("report", "employee")),
    P("mentor_of", inverse="mentee_of", aliases=("mentors", "coaches"),
      subject_terms=("mentor", "coach"), object_terms=("mentee", "student")),
    # attributes
    P("named", functional=True, aliases=("name", "name_is", "called", "is_named")),
    P("age", functional=True, aliases=("is_age", "aged", "age_is", "years_old")),
    P("job_title", functional=True, domain=PERSON,
      aliases=("role", "position", "occupation", "works_as", "profession", "is_job")),
    P("nationality", functional=True, domain=PERSON, aliases=("citizen_of",)),
    # places and organizations
    P("lives_in", functional=True, domain=PERSON, range=("place",),
      aliases=("resides_in", "living_in", "moved_to", "relocated_to", "lives_at", "resident_of")),
    P("is_from", functional=True, domain=PERSON, range=("place",),
      aliases=("from", "born_in", "grew_up_in", "hometown", "comes_from", "native_of")),
    P("works_at", functional=True, domain=PERSON, range=("organization",),
      aliases=("works_for", "employed_by", "employed_at", "job_at", "employee_of")),
    P("studies_at", domain=PERSON, range=("organization",), aliases=("attends", "student_at", "enrolled_at")),
    P("member_of", aliases=("belongs_to", "joined", "is_member_of")),
    P("located_in", functional=True, domain=("place", "organization", "service", "event"), range=("place",),
      aliases=("headquartered_in", "situated_in")),
    P("visited", range=("place", "event"),
      aliases=("went_to", "traveled_to", "travelled_to", "flew_to", "trip_to", "visiting", "toured")),
    P("attended", range=("event",), aliases=("participated_in", "was_at", "took_part_in")),
    # preferences
    P("likes", aliases=("loves", "enjoys", "prefers", "adores", "fan_of", "interested_in", "is_into")),
    P("dislikes", aliases=("hates", "detests", "not_a_fan_of")),
    P("allergic_to", aliases=("is_allergic_to", "allergy_to")),
    # work and skills
    P("working_on", aliases=("building", "developing", "writing", "works_on", "is_working_on")),
    P("uses", aliases=("using", "uses_tool", "relies_on")),
    P("owns", aliases=("has_pet", "bought", "purchased")),
    P("founded", aliases=("started", "co_founded", "established")),
    P("leads", aliases=("heads", "runs", "leads_team")),
    P("skilled_in", aliases=("experienced_in", "expert_in", "proficient_in", "knows", "good_at")),
    P("speaks", aliases=("speaks_language", "fluent_in")),
    P("learning", aliases=("studying", "training_for", "preparing_for", "learns", "is_learning")),
    P("teaches", aliases=("teaching", "instructs")),
    # software
    P("depends_on", aliases=("requires", "needs")),
    P("deployed_to", aliases=("runs_on", "hosted_on", "deployed_on")),
    P("component_of", inverse="has_component", aliases=("module_of", "submodule_of")),
    # generic
    P("is_a", aliases=("is", "type_of", "kind_of", "instance_of")),
    P("has", aliases=("possesses",)),
]

GLUE = {"has", "is", "of", "to", "in", "at", "with", "the", "a", "an", "on", "for", "by"}
_FAMILY_FUNCTIONAL = {"favorite": True}  # favorite_color, favorite_food, ... are single-valued


def snake(pred: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(pred or "").strip().lower()).strip("_")[:100]


@dataclass
class Resolved:
    name: str            # canonical predicate (or the cleaned raw one when custom)
    swap: bool = False   # True: the raw predicate pointed the other way (child_of -> parent_of)
    known: bool = True   # False: custom predicate
    raw: str = ""


@dataclass
class Ontology:
    types: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TYPES))
    predicates: Dict[str, Predicate] = field(default_factory=dict)
    extra_functional: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._alias: Dict[str, str] = {}
        self._inverse: Dict[str, str] = {}
        if not self.predicates:
            for p in DEFAULT_PREDICATES:
                self.register_predicate(p)
        else:
            for p in list(self.predicates.values()):
                self.register_predicate(p)

    # ------------------------------------------------------------------ registry

    def register_type(self, name: str, description: str = "") -> None:
        self.types[snake(name)] = description

    def register_predicate(self, p: Predicate) -> None:
        name = snake(p.name)
        self.predicates[name] = p
        for a in p.aliases:
            self._alias[snake(a)] = name
        if p.inverse:
            self._inverse[snake(p.inverse)] = name

    def entity_type(self, t: Optional[str]) -> str:
        t = snake(t or "")
        return t if t in self.types else "concept"

    # ------------------------------------------------------------------ resolution

    def resolve(self, raw: str) -> Resolved:
        clean = snake(raw)
        if clean in self.predicates:
            return Resolved(clean, raw=clean)
        if clean in self._alias:
            return Resolved(self._alias[clean], raw=clean)
        if clean in self._inverse:
            return Resolved(self._inverse[clean], swap=True, raw=clean)
        # "has_mother" / "has_father": the other person is the parent
        if clean in ("has_mother", "has_father", "has_parent", "son_of", "daughter_of"):
            return Resolved("parent_of", swap=True, raw=clean)
        if clean.startswith("favorite_") or clean.startswith("favourite_"):
            return Resolved("favorite_" + clean.split("_", 1)[1], raw=clean)
        return Resolved(clean, known=False, raw=clean)

    def is_functional(self, name: str) -> bool:
        if name in self.extra_functional:
            return True
        p = self.predicates.get(name)
        if p is not None:
            return p.functional
        family = name.split("_", 1)[0]
        return _FAMILY_FUNCTIONAL.get(family, False) and "_" in name

    def is_symmetric(self, name: str) -> bool:
        p = self.predicates.get(name)
        return bool(p and p.symmetric)

    def type_ok(self, name: str, s_type: Optional[str], o_type: Optional[str]) -> bool:
        """False only when a declared domain/range is contradicted by a specific (non-concept) type."""
        p = self.predicates.get(name)
        if p is None:
            return True
        if p.domain and s_type not in (None, "", "concept") and s_type not in p.domain:
            return False
        if p.range and o_type not in (None, "", "concept") and o_type not in p.range:
            return False
        return True

    def functional_predicates(self) -> Set[str]:
        return {n for n, p in self.predicates.items() if p.functional} | set(self.extra_functional)

    def query_terms(self, name: str, other_is_object: bool = True) -> Set[str]:
        """
        Words a question uses for the *other* end of an edge, from the user's side:
        (User)-parent_of->(Emma) answers "my daughter"; (Ann)-parent_of->(User) answers
        "my mother".
        """
        p = self.predicates.get(name)
        words: Set[str] = set()
        if p is not None and (p.subject_terms or p.object_terms):
            terms = p.object_terms if (other_is_object or p.symmetric) else (p.subject_terms or p.object_terms)
            words |= set(terms)
        else:
            words |= set(name.split("_"))
            if p is not None:
                for a in p.aliases + ((p.inverse,) if p.inverse else ()):
                    words |= set(a.split("_"))
        return {w for w in words if w and w not in GLUE}

    # ------------------------------------------------------------------ prompts

    def prompt_block(self) -> str:
        """Compact vocabulary for extraction prompts."""
        types = ", ".join(self.types)
        preds = []
        for n, p in self.predicates.items():
            flags = ("1" if p.functional else "") + ("~" if p.symmetric else "")
            preds.append(n + (f"[{flags}]" if flags else ""))
        return (f"Entity types: {types}.\n"
                f"Predicates (prefer these; [1] = one value at a time, ~ = symmetric): {', '.join(preds)}, "
                f"favorite_<thing>. Use another snake_case predicate only if none fits.")


def raw_role(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    The role word an extractor's predicate names, and which end it names:
    "has_sister" -> ("sister", "object"); "sister_of" -> ("sister", "subject").
    """
    m = re.fullmatch(r"has_([a-z]+)", raw or "")
    if m and m.group(1) not in GLUE:
        return m.group(1), "object"
    m = re.fullmatch(r"(?:best_)?([a-z]+)_of", raw or "")
    if m and m.group(1) not in GLUE | {"part", "kind", "type", "instance", "member", "component", "module"}:
        return m.group(1), "subject"
    return None, None


def default_ontology(extra_functional: Optional[Iterable[str]] = None) -> Ontology:
    return Ontology(extra_functional=set(extra_functional or ()))
