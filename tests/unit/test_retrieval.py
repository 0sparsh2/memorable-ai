import pytest

from memorable_ai.core.bm25 import BM25Index
from memorable_ai.core.entities import entity_overlap_score, extract_entities


def test_bm25_ranking():
    docs = [
        {"id": 1, "content": "vegetarian diet no dairy"},
        {"id": 2, "content": "software engineer at google"},
    ]
    idx = BM25Index()
    idx.build(docs)
    scores = idx.score("vegetarian", top_k=2)
    assert scores[0][0] == 1


def test_entity_extraction():
    ents = extract_entities("Alice works at Google in Paris")
    assert "alice" in ents or "google" in ents


def test_entity_overlap():
    assert entity_overlap_score("Google engineer", "works at Google") > 0
