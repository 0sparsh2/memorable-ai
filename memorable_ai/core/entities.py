"""Simple entity extraction for retrieval signal (Mem0-style entity matching)."""

import re
from typing import List, Set


def extract_entities(text: str) -> List[str]:
    """Extract candidate entities: capitalized phrases, emails, @handles."""
    entities: Set[str] = set()
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        entities.add(match.group(0).lower())
    for match in re.finditer(r"\b[\w.-]+@[\w.-]+\.\w+\b", text):
        entities.add(match.group(0).lower())
    for match in re.finditer(r"@(\w+)", text):
        entities.add(match.group(1).lower())
    # Multi-word proper nouns from all-caps tokens
    for word in re.findall(r"\b[A-Z]{2,}\b", text):
        entities.add(word.lower())
    return list(entities)


def entity_overlap_score(query: str, content: str) -> float:
    q_ents = set(extract_entities(query))
    if not q_ents:
        return 0.0
    c_ents = set(extract_entities(content))
    if not c_ents:
        # Fallback: token overlap on capitalized words in content
        c_ents = set(w.lower() for w in re.findall(r"\b[A-Z]\w+\b", content))
    if not c_ents:
        return 0.0
    overlap = len(q_ents & c_ents) / len(q_ents)
    return float(overlap)
