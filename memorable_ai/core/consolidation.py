"""
Memory consolidation: the periodic, offline "sleep-time" pass.

Each cycle:
1. **Importance**: recompute from type, access frequency (the ``access_count``
   column; the old code read it from metadata, where it was always 0), recency of
   use, and salience.
2. **Contradictions**: within each user, compare facts that are near neighbours
   in embedding space (not "same first three words"), detect polarity conflicts
   ("likes X" vs "doesn't like X"), and retire the older one with
   ``superseded_by_id``. History is kept.
3. **Jobs**: extra sleep-time work registered by the engine, such as closing and
   summarising idle episodes.

Background reading: Mem0 (arXiv:2504.19413), Letta sleep-time compute (2025),
Anthropic "dreaming" (Nov 2026).
"""

import asyncio
import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from memorable_ai.core.models import utcnow  # noqa: E402

logger = logging.getLogger(__name__)

_NEG = re.compile(
    r"\b(?:don't|do not|doesn't|does not|didn't|did not|never|not|no longer|hates?|dislikes?|stopped|quit|isn't|is not|can't|cannot|won't)\b",
    re.I,
)
_POS = re.compile(r"\b(?:likes?|loves?|prefers?|enjoys?|adores?|wants?|is|uses?|eats?|drinks?)\b", re.I)
_SENTIMENT_WORDS = {
    "like", "likes", "love", "loves", "prefer", "prefers", "enjoy", "enjoys", "adore", "adores",
    "hate", "hates", "dislike", "dislikes", "don't", "do", "not", "doesn't", "does", "never",
    "no", "longer", "stopped", "user", "i", "is", "isn't", "want", "wants", "use", "uses",
}
_TYPE_WEIGHT = {"rule": 1.3, "preference": 1.2, "skill": 1.1, "fact": 1.0, "context": 0.8}


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _SENTIMENT_WORDS and len(w) > 1}


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


class MemoryConsolidator:
    """Background consolidation of the memory store."""

    def __init__(
        self,
        storage: Any,
        interval: int = 21600,
        embedder: Any = None,
        knowledge_graph: Any = None,
        jobs: Optional[Sequence[Callable[[], Awaitable[Any]]]] = None,
        neighbour_threshold: Optional[float] = None,
    ):
        self.storage = storage
        self.interval = interval
        self.embedder = embedder
        self.knowledge_graph = knowledge_graph
        self.jobs = list(jobs or [])
        if neighbour_threshold is None:
            # Hashing embeddings are lexical and score paraphrases lower than neural ones.
            neighbour_threshold = 0.6 if getattr(embedder, "is_semantic", False) else 0.4
        self.neighbour_threshold = neighbour_threshold
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the periodic loop on the current event loop."""
        if self._running:
            logger.warning("Consolidator already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._consolidation_loop())
        logger.info("Memory consolidator started (interval: %ss)", self.interval)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Memory consolidator stopped")

    async def _consolidation_loop(self):
        # Sleep first: consolidating on startup would slow every process launch.
        while self._running:
            try:
                await asyncio.sleep(self.interval)
                if self._running:
                    await self.consolidate()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Consolidation error: %s", e)
                await asyncio.sleep(60)

    async def consolidate(self) -> Dict[str, Any]:
        """One consolidation pass. Returns counts of what changed."""
        report: Dict[str, Any] = {"memories": 0, "contradictions": 0, "jobs": 0}
        try:
            memories = await self.storage.get_memories(limit=100000)
            report["memories"] = len(memories)
            await self._update_importance_scores(memories)
            report["contradictions"] = await self._resolve_contradictions(memories)
        except Exception as e:
            logger.error("Consolidation failed: %s", e)
        for job in self.jobs:
            try:
                result = await job()
                report["jobs"] += len(result) if isinstance(result, (list, tuple)) else 1
            except Exception as e:
                logger.warning("Consolidation job failed: %s", e)
        logger.debug("Consolidation complete: %s", report)
        return report

    async def _update_importance_scores(self, memories: List[Dict[str, Any]]):
        now = utcnow()
        for memory in memories:
            memory_id = memory.get("id")
            if not memory_id:
                continue
            base = float(memory.get("importance_score") or 0.0)
            access = int(memory.get("access_count") or 0)
            access_bonus = min(math.log1p(access) / math.log(11), 1.0)  # saturates at ~10 uses
            last = _parse_dt(memory.get("updated_at")) or _parse_dt(memory.get("created_at"))
            age_days = (now - last).days if last else 180
            recency = math.exp(-age_days / 180.0)  # ~6-month half-life-ish decay
            weight = _TYPE_WEIGHT.get(memory.get("type", "fact"), 1.0)
            new_score = min(1.0, (0.5 * base + 0.3 * access_bonus + 0.2 * recency) * weight)
            if abs(new_score - base) > 1e-3:
                await self.storage.update_memory_importance(memory_id, round(new_score, 4))

    def _neighbours(self, group: List[Dict[str, Any]]) -> List[tuple]:
        """Candidate pairs: embedding neighbours if available, else shared content words."""
        pairs = []
        vecs = {}
        for m in group:
            emb = m.get("embedding")
            if isinstance(emb, list) and emb:
                vecs[m["id"]] = emb
        if len(vecs) == len(group) and len(group) > 1:
            import numpy as np

            mat = np.asarray([vecs[m["id"]] for m in group], dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            sims = mat @ mat.T
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if sims[i, j] >= self.neighbour_threshold:
                        pairs.append((group[i], group[j]))
            return pairs
        index = defaultdict(list)
        for m in group:
            for w in _content_words(m.get("content", "")):
                index[w].append(m)
        seen = set()
        for bucket in index.values():
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    key = tuple(sorted((bucket[i]["id"], bucket[j]["id"])))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((bucket[i], bucket[j]))
        return pairs

    async def _resolve_contradictions(self, memories: List[Dict[str, Any]]) -> int:
        by_user: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for m in memories:
            if m.get("id"):
                by_user[m.get("user_id")].append(m)
        retired = set()
        count = 0
        for group in by_user.values():
            if len(group) < 2 or len(group) > 5000:
                continue
            for mem1, mem2 in self._neighbours(group):
                if mem1["id"] in retired or mem2["id"] in retired:
                    continue
                if not self._are_contradictory(mem1, mem2):
                    continue
                winner = self._resolve_contradiction(mem1, mem2)
                if winner is None:
                    continue
                loser = mem2 if winner is mem1 else mem1
                ok = await self.storage.invalidate_memory(
                    loser["id"], superseded_by_id=winner["id"], reason="contradiction"
                )
                if ok:
                    retired.add(loser["id"])
                    count += 1
                    if self.knowledge_graph is not None:
                        self.knowledge_graph.invalidate_memory(loser["id"])
        return count

    def _are_contradictory(self, mem1: Dict[str, Any], mem2: Dict[str, Any]) -> bool:
        """Opposite polarity about the same object ("likes X" vs "doesn't like X")."""
        c1, c2 = mem1.get("content", ""), mem2.get("content", "")
        neg1, neg2 = bool(_NEG.search(c1)), bool(_NEG.search(c2))
        if neg1 == neg2:
            return False
        pos = _POS.search(c2 if neg1 else c1)
        if not pos:
            return False
        w1, w2 = _content_words(c1), _content_words(c2)
        if not w1 or not w2:
            return False
        # Same object, not merely a shared word ("fan of horror" vs "fan of Star Wars").
        return len(w1 & w2) / len(w1 | w2) >= 0.5

    def _resolve_contradiction(
        self, mem1: Dict[str, Any], mem2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Keep the more recent statement; tie-break on importance."""
        d1, d2 = _parse_dt(mem1.get("created_at")), _parse_dt(mem2.get("created_at"))
        if d1 and d2 and d1 != d2:
            return mem1 if d1 > d2 else mem2
        s1, s2 = mem1.get("importance_score", 0.0) or 0.0, mem2.get("importance_score", 0.0) or 0.0
        if s1 != s2:
            return mem1 if s1 > s2 else mem2
        return None

    async def _remove_outdated(self, memories: List[Dict[str, Any]]):
        """Invalidate very old, never-used, low-importance memories (opt-in)."""
        cutoff = utcnow() - timedelta(days=365)
        for memory in memories:
            created = _parse_dt(memory.get("created_at"))
            if (
                created
                and created < cutoff
                and (memory.get("importance_score") or 0.0) < 0.1
                and not memory.get("access_count")
            ):
                await self.storage.invalidate_memory(memory["id"], reason="outdated")
