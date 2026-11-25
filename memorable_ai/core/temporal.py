"""
Temporal Memory Tracking

Tracks temporal relationships and maintains coherence over time.
Inspired by research on temporal compression in discourse.

Reference: "Highly engaging events reveal semantic and temporal compression 
in online community discourse" - PNAS Nexus (March 2025)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TemporalMemory:
    """
    Tracks temporal relationships between memories.
    
    Maintains:
    - Time-stamped memories
    - Before/after relationships
    - Event sequences
    - Temporal coherence
    """

    def __init__(self, storage: Any):
        """
        Initialize temporal memory tracker.
        
        Args:
            storage: Storage instance
        """
        self.storage = storage

    async def add_temporal_memory(
        self,
        content: str,
        memory_type: str = "fact",
        timestamp: Optional[datetime] = None,
        before: Optional[List[int]] = None,
        after: Optional[List[int]] = None,
        **metadata
    ) -> Dict[str, Any]:
        """
        Add a memory with temporal information.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            timestamp: When this memory occurred (default: now)
            before: IDs of memories that happened before this
            after: IDs of memories that happened after this
            **metadata: Additional metadata
            
        Returns:
            Created memory dictionary
        """
        if timestamp is None:
            timestamp = datetime.now()

        memory = {
            "content": content,
            "type": memory_type,
            "valid_from": timestamp,
            "metadata": {
                **metadata,
                "timestamp": timestamp.isoformat(),
                "before": before or [],
                "after": after or [],
            },
        }

        await self.storage.store_memories([memory])
        return memory

    async def get_temporal_sequence(
        self, start_memory_id: int, direction: str = "forward", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get temporal sequence of memories.
        
        Args:
            start_memory_id: Starting memory ID
            direction: "forward" (after) or "backward" (before)
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memories in temporal order
        """
        memories = await self.storage.get_memories(limit=1000)
        start_memory = next((m for m in memories if m.get("id") == start_memory_id), None)

        if not start_memory:
            return []

        sequence = [start_memory]
        visited = {start_memory_id}

        if direction == "forward":
            # Get memories that come after
            after_ids = start_memory.get("metadata", {}).get("after", [])
            for mem_id in after_ids[:limit]:
                if mem_id not in visited:
                    mem = next((m for m in memories if m.get("id") == mem_id), None)
                    if mem:
                        sequence.append(mem)
                        visited.add(mem_id)
        else:
            # Get memories that come before
            before_ids = start_memory.get("metadata", {}).get("before", [])
            for mem_id in before_ids[:limit]:
                if mem_id not in visited:
                    mem = next((m for m in memories if m.get("id") == mem_id), None)
                    if mem:
                        sequence.insert(0, mem)
                        visited.add(mem_id)

        return sequence

    async def get_memories_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get memories within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            memory_type: Filter by memory type (optional)
            
        Returns:
            List of memories in time range
        """
        memories = await self.storage.get_memories(
            memory_type=memory_type, limit=10000
        )

        result = []
        for memory in memories:
            metadata = memory.get("metadata", {})
            timestamp_str = metadata.get("timestamp")

            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if start_time <= timestamp <= end_time:
                    result.append(memory)
            except Exception as e:
                logger.debug(f"Failed to parse timestamp: {e}")
                continue

        # Sort by timestamp
        result.sort(
            key=lambda m: m.get("metadata", {}).get("timestamp", ""), reverse=True
        )

        return result

    def extract_temporal_relationships(self, text: str) -> Dict[str, Any]:
        """
        Extract temporal relationships from text.
        
        Patterns:
        - "Before...", "After...", "Then...", "Next..."
        - "In 2020...", "Last year...", "Recently..."
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with temporal information
        """
        import re

        relationships = {
            "before": [],
            "after": [],
            "timestamp": None,
        }

        # Extract time references
        time_patterns = [
            r"(?:in|on|at) (\d{4})",  # "in 2020"
            r"(?:last|next) (?:year|month|week)",  # "last year"
            r"recently|yesterday|today|tomorrow",
        ]

        for pattern in time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.group(0):
                    relationships["timestamp"] = match.group(0)

        # Extract sequence markers
        sequence_patterns = [
            (r"before ([^\.]+)", "before"),
            (r"after ([^\.]+)", "after"),
            (r"then ([^\.]+)", "after"),
            (r"next ([^\.]+)", "after"),
        ]

        for pattern, rel_type in sequence_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.group(1):
                    relationships[rel_type].append(match.group(1).strip())

        return relationships

    async def check_temporal_coherence(
        self, memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check temporal coherence of memories.
        
        Detects:
        - Out-of-order sequences
        - Missing temporal links
        - Inconsistent timestamps
        
        Args:
            memories: List of memories to check
            
        Returns:
            Dictionary with coherence analysis
        """
        issues = []
        warnings = []

        # Check for timestamps
        memories_with_time = [
            m for m in memories if m.get("metadata", {}).get("timestamp")
        ]
        memories_without_time = [
            m for m in memories if not m.get("metadata", {}).get("timestamp")
        ]

        if memories_without_time:
            warnings.append(
                f"{len(memories_without_time)} memories missing timestamps"
            )

        # Check for out-of-order sequences
        if len(memories_with_time) > 1:
            timestamps = []
            for mem in memories_with_time:
                try:
                    ts_str = mem.get("metadata", {}).get("timestamp")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        timestamps.append((mem.get("id"), ts))
                except Exception:
                    pass

            # Check if sorted
            if len(timestamps) > 1:
                sorted_timestamps = sorted(timestamps, key=lambda x: x[1])
                if timestamps != sorted_timestamps:
                    issues.append("Memories are out of temporal order")

        return {
            "coherent": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "memories_with_time": len(memories_with_time),
            "memories_without_time": len(memories_without_time),
        }



# ---------------------------------------------------------------------------
# Relative-date anchoring
#
# The May 2026 LOCOMO diagnosis traced most temporal misses to facts stored with
# unresolved relative dates ("yesterday", "last week"). Facts are anchored to the
# session date at write time so they stay true when read months later.
# ---------------------------------------------------------------------------

import calendar as _calendar
import re as _re
from datetime import timedelta as _timedelta

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _week_start(d: datetime) -> datetime:
    return d - _timedelta(days=d.weekday())


def _shift_month(d: datetime, months: int) -> datetime:
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    return d.replace(year=y, month=m, day=min(d.day, _calendar.monthrange(y, m)[1]))


def anchor_relative_dates(text: str, ref: Optional[datetime]) -> "tuple[str, Optional[str]]":
    """
    Rewrite relative date expressions in ``text`` as absolute dates relative to
    ``ref`` (the time the statement was made). Returns ``(new_text, event_date)``
    where ``event_date`` is the ISO date of the first resolved expression.
    """
    if not text or ref is None:
        return text, None
    resolved: List[str] = []

    def day(d: datetime) -> str:
        return d.strftime("%Y-%m-%d")

    def sub(pattern: str, fn) -> None:
        nonlocal text

        def _repl(m):
            out, iso = fn(m)
            if iso:
                resolved.append(iso)
            return out

        text = _re.sub(pattern, _repl, text, flags=_re.IGNORECASE)

    sub(r"\byesterday\b", lambda m: (f"on {day(ref - _timedelta(days=1))}", day(ref - _timedelta(days=1))))
    sub(r"\btomorrow\b", lambda m: (f"on {day(ref + _timedelta(days=1))}", day(ref + _timedelta(days=1))))
    sub(r"\b(?:today|tonight|this (?:morning|afternoon|evening))\b", lambda m: (f"on {day(ref)}", day(ref)))

    def _ago(m):
        n_raw = m.group(1).lower()
        n = int(n_raw) if n_raw.isdigit() else _NUM_WORDS.get(n_raw, 1)
        unit = m.group(2).lower()
        if unit.startswith("day"):
            d = ref - _timedelta(days=n)
            return f"on {day(d)}", day(d)
        if unit.startswith("week"):
            d = _week_start(ref - _timedelta(weeks=n))
            return f"in the week of {day(d)}", day(d)
        if unit.startswith("month"):
            d = _shift_month(ref, -n)
            return f"in {d.strftime('%B %Y')}", d.strftime("%Y-%m-01")
        if unit.startswith("year"):
            return f"in {ref.year - n}", f"{ref.year - n}-01-01"
        return m.group(0), None

    sub(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten) (days?|weeks?|months?|years?) ago\b", _ago)

    def _last_next(m):
        which = m.group(1).lower()
        unit = m.group(2).lower()
        sign = -1 if which == "last" else 1
        if unit == "week":
            d = _week_start(ref + _timedelta(weeks=sign))
            return f"in the week of {day(d)}", day(d)
        if unit == "month":
            d = _shift_month(ref, sign)
            return f"in {d.strftime('%B %Y')}", d.strftime("%Y-%m-01")
        if unit == "year":
            return f"in {ref.year + sign}", f"{ref.year + sign}-01-01"
        if unit == "weekend":
            sat = _week_start(ref) + _timedelta(days=5) + _timedelta(weeks=sign)
            return f"on the weekend of {day(sat)}", day(sat)
        return m.group(0), None

    sub(r"\b(last|next) (week|month|year|weekend)\b", _last_next)

    def _weekday(m):
        which = m.group(1).lower()
        target = list(_calendar.day_name).index(m.group(2).capitalize())
        delta = (ref.weekday() - target) % 7 or 7
        d = ref - _timedelta(days=delta) if which == "last" else ref + _timedelta(days=((target - ref.weekday()) % 7 or 7))
        return f"on {day(d)}", day(d)

    sub(r"\b(last|next) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", _weekday)
    return text, (resolved[0] if resolved else None)
