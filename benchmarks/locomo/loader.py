"""
Load snap-research/LOCOMO dataset (locomo10.json).

Reference: https://github.com/snap-research/locomo
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional, Set
from urllib.request import urlretrieve

LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
DEFAULT_PATH = Path(__file__).parent / "data" / "locomo10.json"

# Official category ids (snap-research/locomo task_eval)
CATEGORY_SINGLE_HOP = 4
CATEGORY_MULTI_HOP = 1
CATEGORY_TEMPORAL = 2
CATEGORY_OPEN_DOMAIN = 3
CATEGORY_ADVERSARIAL = 5
SCORED_CATEGORIES = {
    CATEGORY_SINGLE_HOP,
    CATEGORY_MULTI_HOP,
    CATEGORY_TEMPORAL,
    CATEGORY_OPEN_DOMAIN,
}

CATEGORY_NAMES = {
    CATEGORY_SINGLE_HOP: "single_hop",
    CATEGORY_MULTI_HOP: "multi_hop",
    CATEGORY_TEMPORAL: "temporal",
    CATEGORY_OPEN_DOMAIN: "open_domain",
    CATEGORY_ADVERSARIAL: "adversarial",
}


@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str
    session_key: str
    session_date: str
    blip_caption: Optional[str] = None


@dataclass
class QAItem:
    question: str
    answer: str
    category: int
    evidence: list[str]


def download_dataset(path: Path = DEFAULT_PATH, force: bool = False) -> Path:
    """Download locomo10.json if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if force and path.exists():
        path.unlink()
    if not path.exists():
        urlretrieve(LOCOMO_URL, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 10:
        raise ValueError(f"Expected 10 conversations, got {type(data)} len={len(data)}")
    return path


def load_locomo(path: Path = DEFAULT_PATH) -> list[dict[str, Any]]:
    """Load dataset; downloads on first use."""
    download_dataset(path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_session_datetime(date_str: str) -> Optional[datetime]:
    """Parse e.g. '1:56 pm on 8 May, 2023'."""
    if not date_str:
        return None
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def iter_sessions(conversation: dict[str, Any]) -> Iterator[tuple[str, str, list[dict[str, Any]]]]:
    """Yield (session_key, date_str, turns) in numeric session order."""
    session_keys = sorted(
        (k for k in conversation if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    for key in session_keys:
        date_key = f"{key}_date_time"
        yield key, conversation.get(date_key, ""), conversation[key]


def format_turn_text(turn: dict[str, Any]) -> str:
    text = turn.get("text", "") or ""
    caption = turn.get("blip_caption")
    if caption:
        text = f"{text} [image: {caption}]".strip()
    return text


def iter_turns(sample: dict[str, Any]) -> Iterator[Turn]:
    conv = sample["conversation"]
    for session_key, date_str, turns in iter_sessions(conv):
        for turn in turns:
            yield Turn(
                speaker=turn.get("speaker", ""),
                dia_id=turn.get("dia_id", ""),
                text=format_turn_text(turn),
                session_key=session_key,
                session_date=date_str,
                blip_caption=turn.get("blip_caption"),
            )


def iter_qa(
    sample: dict[str, Any],
    categories: Optional[Set[int]] = None,
) -> Iterator[QAItem]:
    for qa in sample.get("qa", []):
        cat = int(qa["category"])
        if categories is not None and cat not in categories:
            continue
        yield QAItem(
            question=qa["question"],
            answer=str(qa["answer"]),
            category=cat,
            evidence=list(qa.get("evidence", [])),
        )


def sample_id(sample: dict[str, Any], index: int) -> str:
    return str(sample.get("sample_id") or f"conv_{index}")
