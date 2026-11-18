"""G1: ontology (predicate registry) and entity resolution for the knowledge graph."""

from datetime import datetime

import pytest

from memorable_ai.core.embeddings import get_embedder
from memorable_ai.core.extraction import MemoryExtractor
from memorable_ai.core.models import Edge, Entity, EntityAlias, Task
from memorable_ai.core.scope import Scope
from memorable_ai.core.storage import Storage
from memorable_ai.graph.knowledge_graph import KnowledgeGraph
from memorable_ai.graph.ontology import Predicate, default_ontology, raw_role
from memorable_ai.graph.resolution import adjudication_prompt, parse_adjudication

U = Scope(user_id="u")


@pytest.fixture
def storage(tmp_path):
    return Storage(f"sqlite:///{tmp_path / 'g1.db'}")


@pytest.fixture
def emb():
    return get_embedder("hash")


@pytest.fixture
def kg(storage):
    return KnowledgeGraph(storage)


def add(storage, emb, kg, content, entities=(), relations=(), scope=U):
    mid, _ = storage.insert_memory({"content": content, "type": "fact", "user_id": scope.user_id,
                                    "embedding": emb.encode_list(content)})
    mem = {"id": mid, "content": content, "user_id": scope.user_id,
           "entities": [{"name": n, "type": t} for n, t in entities], "relations": [tuple(r) for r in relations]}
    return mid, kg.ingest_memory(mem, scope)


def ents(storage, user="u"):
    s = storage.get_session()
    try:
        return {e.id: e for e in s.query(Entity).filter(Entity.user_id == user, Entity.is_self == 0)}
    finally:
        s.close()


def aliases(storage, eid):
    s = storage.get_session()
    try:
        return {a.alias for a in s.query(EntityAlias).filter(EntityAlias.entity_id == eid)}
    finally:
        s.close()


def by_name(storage, name):
    return [e for e in ents(storage).values() if e.name == name]


def seeds_by_name(kg, storage, query):
    names = {e.id: e.name for e in ents(storage).values()}
    return {names.get(i): w for i, w in kg.match_query_entities(query, U).items()}


# ---------------------------------------------------------------- ontology


def test_ontology_canonicalises_aliases_inverses_and_families():
    o = default_ontology()
    assert (o.resolve("moved_to").name, o.resolve("Works For").name) == ("lives_in", "works_at")
    r = o.resolve("child_of")
    assert (r.name, r.swap) == ("parent_of", True)
    assert o.resolve("has_mother").swap and o.resolve("has_mother").name == "parent_of"
    assert o.resolve("favourite_food").name == "favorite_food" and o.is_functional("favorite_food")
    custom = o.resolve("plays_chess_with")
    assert (custom.name, custom.known) == ("plays_chess_with", False)
    assert o.is_functional("married_to") and o.is_symmetric("married_to") and not o.is_functional("likes")
    assert not o.type_ok("works_at", "person", "place") and o.type_ok("works_at", "person", "concept")
    assert o.query_terms("parent_of", other_is_object=True) >= {"daughter", "son"}
    assert o.query_terms("parent_of", other_is_object=False) >= {"mother", "father"}
    assert raw_role("has_sister") == ("sister", "object") and raw_role("sister_of") == ("sister", "subject")
    assert raw_role("member_of") == (None, None)


def test_apps_register_their_own_predicates():
    o = default_ontology(extra_functional={"on_call_for"})
    o.register_type("service", "a running service")
    o.register_predicate(Predicate("owns_service", functional=True, aliases=("service_owner_of",)))
    assert o.resolve("service_owner_of").name == "owns_service" and o.is_functional("owns_service")
    assert o.is_functional("on_call_for")
    assert "owns_service[1]" in o.prompt_block()


# ---------------------------------------------------------------- canonical edges


def test_inverse_and_symmetric_relations_store_one_canonical_edge(storage, emb, kg):
    add(storage, emb, kg, "Emma is User's sister", [("Emma", "person")], [("Emma", "sister_of", "User")])
    add(storage, emb, kg, "User and Emma are siblings", [("Emma", "person")], [("User", "sibling_of", "Emma")])
    add(storage, emb, kg, "Leo is User's son", [("Leo", "person")], [("Leo", "child_of", "User")])
    rels = {(r["subject"], r["predicate"], r["object"]) for r in kg.relations(U)}
    assert rels == {("User", "sibling_of", "Emma"), ("User", "parent_of", "Leo")}
    s = storage.get_session()
    try:
        e = s.query(Edge).filter(Edge.kind == "relates", Edge.predicate == "sibling_of").one()
        assert e.extra == {"raw": "sister_of", "role": "sister", "role_end": "dst"}, "the specific word survives"
    finally:
        s.close()


def test_role_words_follow_edge_direction(storage, emb, kg):
    add(storage, emb, kg, "Leo is User's son", [("Leo", "person")], [("Leo", "child_of", "User")])
    add(storage, emb, kg, "Ann is User's mother", [("Ann", "person")], [("Ann", "mother_of", "User")])
    add(storage, emb, kg, "Emma is User's sister", [("Emma", "person")], [("User", "has_sister", "Emma")])
    assert set(seeds_by_name(kg, storage, "What does my son do?")) == {"Leo"}
    assert set(seeds_by_name(kg, storage, "When is my mother's birthday?")) == {"Ann"}
    sister = seeds_by_name(kg, storage, "Where does my sister live?")
    assert sister == {"Emma": 0.8}, "the extractor's word is a specific match"
    assert seeds_by_name(kg, storage, "Where does my brother live?") == {"Emma": 0.6}, "same group, weaker"


def test_functional_alias_supersedes_and_types_are_checked(storage, emb, kg):
    a, _ = add(storage, emb, kg, "User lives in Berlin", [("Berlin", "place")], [("User", "lives_in", "Berlin")])
    b, retired = add(storage, emb, kg, "User relocated to Munich", [("Munich", "place")],
                     [("User", "relocated_to", "Munich")])
    assert retired == [a]
    assert [(r["predicate"], r["object"]) for r in kg.relations(U)] == [("lives_in", "Munich")]
    add(storage, emb, kg, "User works at Paris", [("Paris", "place")], [("User", "works_at", "Paris")])
    s = storage.get_session()
    try:
        e = s.query(Edge).filter(Edge.predicate == "works_at").one()
        assert e.confidence == 0.6 and e.extra.get("type_mismatch")
    finally:
        s.close()


def test_symmetric_functional_relation_invalidates_from_either_end(storage, emb, kg):
    a, _ = add(storage, emb, kg, "User is married to Alice", [("Alice", "person")], [("Alice", "wife_of", "User")])
    _, retired = add(storage, emb, kg, "User married Beth", [("Beth", "person")], [("User", "married_to", "Beth")])
    assert retired == [a]
    assert [(r["subject"], r["object"]) for r in kg.relations(U)] == [("User", "Beth")]


# ---------------------------------------------------------------- resolution


def test_nickname_links_to_the_full_name_and_becomes_an_alias(storage, emb, kg):
    add(storage, emb, kg, "Robert Chen is User's colleague", [("Robert Chen", "person")],
        [("User", "colleague_of", "Robert Chen")])
    add(storage, emb, kg, "Bob lives in Oslo", [("Bob", "person"), ("Oslo", "place")], [("Bob", "lives_in", "Oslo")])
    [robert] = by_name(storage, "Robert Chen")
    assert aliases(storage, robert.id) == {"bob"} and not by_name(storage, "Bob")
    assert ("Robert Chen", "lives_in", "Oslo") in {(r["subject"], r["predicate"], r["object"]) for r in kg.relations(U)}
    assert "Robert Chen" in seeds_by_name(kg, storage, "Where does Bob live?")
    assert "lives in Oslo" in robert.summary and "colleague of" in robert.summary


def test_first_name_then_full_name_is_one_entity(storage, emb, kg):
    add(storage, emb, kg, "Ana likes hiking", [("Ana", "person")], [("Ana", "likes", "hiking")])
    add(storage, emb, kg, "Ana Silva lives in Lisbon", [("Ana Silva", "person"), ("Lisbon", "place")],
        [("Ana Silva", "lives_in", "Lisbon")])
    [ana] = [e for e in ents(storage).values() if e.entity_type == "person"]
    assert ana.name == "Ana Silva" and aliases(storage, ana.id) == {"ana"}
    assert {r["subject"] for r in kg.relations(U)} == {"Ana Silva"}


def test_different_full_names_never_merge_and_typos_only_go_to_the_llm(storage, emb, kg):
    add(storage, emb, kg, "Ana Silva lives in Lisbon", [("Ana Silva", "person")], [("Ana Silva", "lives_in", "Lisbon")])
    add(storage, emb, kg, "Ana Costa lives in Porto", [("Ana Costa", "person")], [("Ana Costa", "lives_in", "Porto")])
    assert len(by_name(storage, "Ana Silva")) == 1 and len(by_name(storage, "Ana Costa")) == 1
    pending = kg.pending_resolutions({"id": 99, "content": "Ana Sliva called", "entities": [
        {"name": "Ana Sliva", "type": "person"}], "relations": []}, U)
    assert [(p["name"], p["reason"]) for p in pending] == [("Ana Sliva", "fuzzy")]
    add(storage, emb, kg, "Ana Sliva called", [("Ana Sliva", "person")])
    assert by_name(storage, "Ana Sliva"), "no LLM: a near-miss surname stays separate"


def test_ambiguous_first_name_uses_graph_context(storage, emb, kg):
    add(storage, emb, kg, "Ana Silva is Marco's friend", [("Ana Silva", "person"), ("Marco", "person")],
        [("Ana Silva", "friend_of", "Marco")])
    add(storage, emb, kg, "Ana Costa is User's colleague", [("Ana Costa", "person")],
        [("User", "colleague_of", "Ana Costa")])
    add(storage, emb, kg, "Ana and Marco went to Sintra", [("Ana", "person"), ("Marco", "person"), ("Sintra", "place")],
        [("Ana", "visited", "Sintra")])
    rels = {(r["subject"], r["predicate"], r["object"]) for r in kg.relations(U)}
    assert ("Ana Silva", "visited", "Sintra") in rels, "Marco connects the mention to Ana Silva"
    add(storage, emb, kg, "My colleague Ana is ill", [("Ana", "person")], [("User", "colleague_of", "Ana")])
    assert not by_name(storage, "Ana"), "the same relation to the user picks Ana Costa"


def test_ambiguous_without_context_or_llm_creates_a_separate_entity(storage, emb, kg):
    add(storage, emb, kg, "Ana Silva lives in Lisbon", [("Ana Silva", "person")], [("Ana Silva", "lives_in", "Lisbon")])
    add(storage, emb, kg, "Ana Costa lives in Porto", [("Ana Costa", "person")], [("Ana Costa", "lives_in", "Porto")])
    mem = {"id": 99, "content": "Ana likes tea", "entities": [{"name": "Ana", "type": "person"}],
           "relations": [("Ana", "likes", "tea")]}
    [p] = kg.pending_resolutions(mem, U)
    assert p["reason"] == "ambiguous" and {c["name"] for c in p["candidates"]} == {"Ana Silva", "Ana Costa"}
    assert all("lives in" in c["summary"] for c in p["candidates"]), "candidates carry fact sheets for the LLM"
    add(storage, emb, kg, "Ana likes tea", [("Ana", "person")], [("Ana", "likes", "tea")])
    assert len(by_name(storage, "Ana")) == 1


def test_same_name_different_person_by_kinship_conflict(storage, emb, kg):
    """Fixture from the G1 plan: the user's sister Ana is not the user's daughter Ana."""
    add(storage, emb, kg, "User's sister is Ana", [("Ana", "person")], [("User", "has_sister", "Ana")])
    add(storage, emb, kg, "User's daughter is Ana", [("Ana", "person")], [("User", "has_daughter", "Ana")])
    anas = by_name(storage, "Ana")
    assert len(anas) == 2
    add(storage, emb, kg, "User's daughter Ana lives in Porto", [("Ana", "person"), ("Porto", "place")],
        [("User", "has_daughter", "Ana"), ("Ana", "lives_in", "Porto")])
    assert len(by_name(storage, "Ana")) == 2, "the daughter is found again, no third Ana"
    s = storage.get_session()
    try:
        daughter = s.query(Edge.dst_id).filter(Edge.predicate == "parent_of").scalar()
        porto = s.query(Edge.src_id).filter(Edge.predicate == "lives_in").scalar()
    finally:
        s.close()
    assert daughter == porto
    assert set(seeds_by_name(kg, storage, "Where does my daughter live?")) == {"Ana"}


@pytest.mark.asyncio
async def test_llm_adjudicator_settles_ambiguous_mentions(storage, emb):
    calls = []

    async def fake_llm(text, items):
        calls.append((text, items))
        costa = next(c for c in items[0]["candidates"] if c["name"] == "Ana Costa")
        return {items[0]["norm"]: costa["id"], "ghost": 12345}

    kg = KnowledgeGraph(storage, adjudicator=fake_llm)
    for c, r in (("Ana Silva lives in Lisbon", ("Ana Silva", "lives_in", "Lisbon")),
                 ("Ana Costa lives in Porto", ("Ana Costa", "lives_in", "Porto"))):
        mid, _ = storage.insert_memory({"content": c, "type": "fact", "user_id": "u"})
        await kg.aingest_memory({"id": mid, "content": c, "entities": [{"name": r[0], "type": "person"}],
                                 "relations": [r]}, U)
    assert calls == [], "unambiguous facts never reach the LLM"
    mid, _ = storage.insert_memory({"content": "Ana got a cat", "type": "fact", "user_id": "u"})
    await kg.aingest_memory({"id": mid, "content": "Ana got a cat", "entities": [{"name": "Ana", "type": "person"}],
                             "relations": [("Ana", "owns", "cat")]}, U)
    assert len(calls) == 1 and calls[0][0] == "Ana got a cat"
    assert ("Ana Costa", "owns", "cat") in {(r["subject"], r["predicate"], r["object"]) for r in kg.relations(U)}
    [costa] = by_name(storage, "Ana Costa")
    assert "ana" in aliases(storage, costa.id)


def test_adjudication_prompt_and_parsing():
    items = [{"norm": "ana", "name": "Ana", "type": "person",
              "candidates": [{"id": 4, "name": "Ana Silva", "summary": "Ana Silva lives in Lisbon"},
                             {"id": 9, "name": "Ana Costa", "summary": None}]}]
    prompt = adjudication_prompt("Ana got a cat", items)
    assert "1. Ana Silva (concept): Ana Silva lives in Lisbon" in prompt and "2. Ana Costa" in prompt
    assert parse_adjudication({"Ana": 2}, items) == {"ana": 9}
    assert parse_adjudication({"ana": "new"}, items) == {"ana": None}
    assert parse_adjudication({"Ana": 7}, items) == {} and parse_adjudication(["x"], items) == {}


def test_merge_entities_is_reversible_and_feeds_retrieval(storage, emb, kg):
    a, _ = add(storage, emb, kg, "User lives in NYC", [("NYC", "place")], [("User", "lives_in", "NYC")])
    b, _ = add(storage, emb, kg, "New York City has great pizza", [("New York City", "place")],
               [("New York City", "has", "great pizza")])
    [nyc] = by_name(storage, "NYC")
    [full] = by_name(storage, "New York City")
    assert kg.merge_entities(nyc.id, full.id, U)
    assert aliases(storage, full.id) == {"nyc"}
    rels = {(r["subject"], r["predicate"], r["object"]) for r in kg.relations(U)}
    assert ("User", "lives_in", "New York City") in rels
    s = storage.get_session()
    try:
        link = s.query(Edge).filter(Edge.kind == "alias_of").one()
        assert (link.src_id, link.dst_id) == (nyc.id, full.id)
        moved = s.query(Edge).filter(Edge.kind == "relates", Edge.predicate == "lives_in").one()
        assert moved.extra["merged_from"] == nyc.id
    finally:
        s.close()
    assert set(kg.match_query_entities("pizza places in NYC", U)) == {full.id}
    assert {f["id"] for f in kg.entity_facts("NYC", U)} == {a, b}
    add(storage, emb, kg, "User visited NYC in May", [("NYC", "place")], [("User", "visited", "NYC")])
    assert ("User", "visited", "New York City") in {(r["subject"], r["predicate"], r["object"]) for r in kg.relations(U)}
    assert not kg.merge_entities(full.id, full.id, U)


def test_episode_mentions_use_resolution(storage, emb, kg):
    add(storage, emb, kg, "Robert Chen is User's colleague", [("Robert Chen", "person")],
        [("User", "colleague_of", "Robert Chen")])
    kg.ingest_episode(7, ["Bob", "Oslo"], U)
    assert {e.name for e in ents(storage).values()} == {"Robert Chen", "Oslo"}


def test_forget_user_removes_aliases_and_tasks(storage, emb, kg):
    add(storage, emb, kg, "Robert Chen is User's colleague", [("Robert Chen", "person")],
        [("User", "colleague_of", "Robert Chen")])
    add(storage, emb, kg, "Bob lives in Oslo", [("Bob", "person")], [("Bob", "lives_in", "Oslo")])
    s = storage.get_session()
    try:
        s.add(Task(user_id="u", session_id="s", goal_text="x", status="open", started_at=datetime(2026, 11, 18)))
        s.commit()
    finally:
        s.close()
    counts = storage.forget_user("u")
    assert counts["entity_aliases"] == 1 and counts["tasks"] == 1
    s = storage.get_session()
    try:
        assert s.query(EntityAlias).count() == 0 and s.query(Task).count() == 0
    finally:
        s.close()


@pytest.mark.asyncio
async def test_extraction_prompt_carries_the_ontology():
    o = default_ontology()
    o.register_predicate(Predicate("on_call_for", functional=True))
    ext = MemoryExtractor(llm_model="stub", ontology=o)
    seen = []

    async def fake(prompt, max_tokens=800):
        seen.append(prompt)
        return [{"content": "User is on call for payments", "type": "fact",
                 "relations": [["User", "on_call_for", "payments"]]}]

    ext._call_llm_json = fake
    out = await ext._extract_llm([{"role": "user", "content": "I'm on call for payments"}], [], "", datetime(2026, 11, 18))
    assert "on_call_for[1]" in seen[0] and "Entity types: person" in seen[0]
    assert out[0]["relations"] == [("User", "on_call_for", "payments")]


def test_engine_merge_entities_by_name_is_scoped(engine_factory):
    import asyncio

    eng = engine_factory()

    async def go():
        for user in ("alice", "mallory"):
            await eng.add_memory("User lives in NYC", user_id=user, infer=False)
            await eng.add_memory("New York City is loud", user_id=user, infer=False)

    asyncio.run(go())
    assert not eng.merge_entities("NYC", "Nowhere", user_id="alice")
    assert eng.merge_entities("NYC", "New York City", user_id="alice")
    alice = {(r["subject"], r["object"]) for r in eng.graph(user_id="alice")}
    mallory = {(r["subject"], r["object"]) for r in eng.graph(user_id="mallory")}
    assert ("User", "New York City") in alice and ("User", "NYC") in mallory, "only alice's graph changed"


def test_user_name_links_mentions_to_the_user_node(storage, emb, kg):
    kg.set_user_name(U, "Caroline")
    assert kg.user_name(U) == "Caroline"
    add(storage, emb, kg, "Caroline lives in Lisbon", [("Caroline", "person"), ("Lisbon", "place")],
        [("Caroline", "lives_in", "Lisbon")])
    add(storage, emb, kg, "User likes tea", [], [("User", "likes", "tea")])
    rels = {(r["subject"], r["predicate"], r["object"]) for r in kg.relations(U)}
    assert rels == {("Caroline", "lives_in", "Lisbon"), ("Caroline", "likes", "tea")}, "one node for the user"
    assert not by_name(storage, "Caroline"), "no second entity for the user's own name"
    other = Scope(user_id="someone-else")
    assert kg.user_name(other) is None


@pytest.mark.asyncio
async def test_extraction_writes_the_users_name_when_known():
    ext = MemoryExtractor(llm_model="stub")
    seen = []

    async def fake(prompt, max_tokens=800):
        seen.append(prompt)
        return []

    ext._call_llm_json = fake
    turns = [{"role": "user", "content": "I adopted a cat"}]
    await ext._extract_llm(turns, [], "", datetime(2026, 11, 18), "Caroline")
    await ext._extract_llm(turns, [], "", datetime(2026, 11, 18))
    assert 'The user is Caroline: refer to them as "Caroline"' in seen[0]
    assert 'Refer to the user as "User"' in seen[1]


@pytest.mark.asyncio
async def test_pipeline_passes_the_users_name_to_extraction(engine_factory):
    eng = engine_factory()
    eng.set_user_name("Caroline", user_id="u1")
    got = []
    orig = eng._extraction.extract_from_turns

    async def spy(*a, **kw):
        got.append(kw.get("user_name"))
        return await orig(*a, **kw)

    eng._extraction.extract_from_turns = spy
    await eng.ingest([{"role": "user", "content": "I adopted a cat named Miso"}], "Lovely!", user_id="u1", session_id="s")
    await eng.ingest([{"role": "user", "content": "I live in Porto"}], "Nice.", user_id="u2", session_id="s2")
    assert got == ["Caroline", None]
