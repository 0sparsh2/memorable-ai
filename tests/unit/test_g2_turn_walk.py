"""G2: turn -> entity mention edges, and the typed graph walk as a turn-ranking signal."""

import pytest

from memorable_ai.core.models import Edge, Entity, Turn
from memorable_ai.core.scope import Scope
from memorable_ai.graph.knowledge_graph import turn_body

U = Scope(user_id="u")


@pytest.fixture
def eng(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORABLE_LLM__DEFAULT_MODEL", "none")
    monkeypatch.setenv("MEMORABLE_LLM__EMBEDDING_MODEL", "hash")
    monkeypatch.setenv("MEMORABLE_LLM__TURN_EMBEDDING_MODEL", "none")
    from memorable_ai import MemoryEngine

    e = MemoryEngine(database=f"sqlite:///{tmp_path / 'g2.db'}", write_mode="sync")
    e._ensure_components()
    yield e
    e.close()


async def say(eng, texts, session="s1", user="u"):
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(texts)]
    await eng.ingest(msgs, None, user_id=user, session_id=session)
    await eng.end_session(user_id=user, session_id=session)


def fact(eng, content, entities, relations=(), user="u"):
    st = eng._storage
    mid, _ = st.insert_memory({"content": content, "type": "fact", "user_id": user})
    eng._knowledge_graph.ingest_memory({"id": mid, "content": content, "user_id": user,
                                        "entities": [{"name": n, "type": t} for n, t in entities],
                                        "relations": [tuple(r) for r in relations]}, Scope(user_id=user))
    return mid


def mention_pairs(eng, user="u"):
    s = eng._storage.get_session()
    try:
        rows = (s.query(Turn.content, Entity.norm_name)
                .join(Edge, (Edge.src_kind == "turn") & (Edge.src_id == Turn.id))
                .join(Entity, (Edge.dst_kind == "entity") & (Edge.dst_id == Entity.id))
                .filter(Edge.kind == "mentions", Edge.invalidated_at.is_(None), Turn.user_id == user).all())
        return {(c, n) for c, n in rows}
    finally:
        s.close()


def test_turn_body_drops_speaker_label():
    assert turn_body("Caroline: I met Ana in Lisbon") == "i met ana in lisbon"
    assert turn_body("Mary Jane Watson: hi") == "hi"
    assert turn_body("we met at 10:30 today") == "we met at 10:30 today"


@pytest.mark.asyncio
async def test_new_name_backfills_earlier_turns(eng):
    await say(eng, ["We had dinner at the harbour club on Friday night.", "That sounds lovely.",
                    "The food was great and the view even better.", "Glad you enjoyed it."])
    assert not any(n == "harbour club" for _, n in mention_pairs(eng))
    fact(eng, "User likes the harbour club", [("harbour club", "place")])
    assert ("We had dinner at the harbour club on Friday night.", "harbour club") in mention_pairs(eng)


@pytest.mark.asyncio
async def test_entity_from_an_episode_backfills_earlier_turns(eng):
    await say(eng, ["We had dinner at the harbour club on Friday night.", "That sounds lovely.",
                    "The food was great and the view even better.", "Glad you enjoyed it."])
    eng._knowledge_graph.ingest_episode(999, ["harbour club"], U)
    assert ("We had dinner at the harbour club on Friday night.", "harbour club") in mention_pairs(eng)


@pytest.mark.asyncio
async def test_episode_close_links_known_entities(eng):
    fact(eng, "Marguerite is the user's cousin", [("Marguerite", "person"), ("Bob", "person")])
    await say(eng, ["Bob: I saw Marguerite at the station this morning.", "Did she say hello?",
                    "She waved from the platform and then left.", "Nice."])
    pairs = mention_pairs(eng)
    assert ("Bob: I saw Marguerite at the station this morning.", "marguerite") in pairs
    assert not any(n == "bob" for _, n in pairs), "a speaker label is not a mention"


@pytest.mark.asyncio
async def test_no_substring_or_self_mentions(eng):
    eng.set_user_name("Priya", user_id="u")
    fact(eng, "Ana is a friend", [("Ana", "person")])
    await say(eng, ["I ate a banana with Priya's breakfast.", "Healthy choice.",
                    "Ana called me later about the trip.", "What did she say?"])
    pairs = mention_pairs(eng)
    assert ("Ana called me later about the trip.", "ana") in pairs
    assert not any("banana" in c and n == "ana" for c, n in pairs), "no substring matches"
    assert not any(n == "priya" for _, n in pairs), "the user's own name is not a mention"


async def _brother_world(eng):
    await say(eng, [
        "My brother likes to work on old cars in his garage.",
        "Cars are a fun hobby.",
        "Ana and I went hiking last weekend near the lake.",
        "The lake must have been beautiful.",
        "Tomas started a new job at the observatory downtown.",
        "Congratulations to him.",
        "What does a brother do when work gets slow? Probably relax.",
        "Relaxing is important.",
    ])
    fact(eng, "Tomas is Ana's brother", [("Tomas", "person"), ("Ana", "person")], [("Ana", "sibling_of", "Tomas")])


@pytest.mark.asyncio
async def test_turn_walk_reaches_turns_through_the_graph(eng):
    await _brother_world(eng)
    walk = dict(eng._knowledge_graph.turn_walk("What does Ana's brother do?", U, {}, k=20))
    s = eng._storage.get_session()
    try:
        tid = s.query(Turn.id).filter(Turn.content.like("Tomas started%")).scalar()
    finally:
        s.close()
    assert tid in walk, "Ana -relates-> Tomas -mentions-> turn"


@pytest.mark.asyncio
async def test_graph_signal_brings_in_a_turn_with_no_shared_words(eng):
    """The Tomas turn shares no word with the query, so neither dense nor BM25 finds it; the walk does."""
    await _brother_world(eng)
    r = eng._retrieval
    q = "What does Ana's brother do?"
    r.graph_turn_weight = 0.0
    off = [t["content"] for t in (await r.recall(q, U, k_facts=0, k_episodes=0, k_excerpts=4))["excerpts"]]
    r.graph_turn_weight = 1.0
    on = [t["content"] for t in (await r.recall(q, U, k_facts=0, k_episodes=0, k_excerpts=4))["excerpts"]]
    target = "Tomas started a new job at the observatory downtown."
    assert target not in off
    assert target in on


@pytest.mark.asyncio
async def test_weight_zero_is_the_old_ranking(eng):
    await _brother_world(eng)
    r = eng._retrieval
    r.graph_turn_weight = 0.0
    called = []
    eng._knowledge_graph.turn_walk = lambda *a, **k: called.append(1) or []
    await r.recall("What does Ana's brother do?", U, k_facts=0, k_episodes=0, k_excerpts=3)
    assert not called


def test_chat_openers_are_not_entities():
    from memorable_ai.graph.knowledge_graph import KnowledgeGraph

    names = lambda t: {e["name"] for e in KnowledgeGraph.extract(t)[0]}  # noqa: E731
    got = names("Hey Jolene, I'm so proud of you!")
    assert "Jolene" in got and "Hey Jolene" not in got and "I'm" not in got
    assert names("Wow Deb that's amazing") >= {"Deb"} and "Wow Deb" not in names("Wow Deb that's amazing")
    assert not names("Thanks! I've been busy.")
    assert "Great Wall of China" in names("We walked the Great Wall of China")
    assert any("Jolene" in n for n in names("Jolene's partner cooked")), "a possessive is still a name"
