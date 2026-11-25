"""
Session summaries (the "S" in the Mem0 paper's extraction prompt).

Summaries are keyed by session. Previously ``get_summary`` returned the latest
summary in the whole namespace, so one user's summary leaked into another user's
extraction prompt. The summary is now hierarchical: it is refreshed when an
episode closes (the episode summary is folded into the running session summary),
not on every call. That keeps extraction context fresh and cuts LLM calls.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MERGE_PROMPT = (
    "Merge the running summary of a conversation with a summary of its latest segment. "
    "Keep names, numbers, dates, decisions and user preferences. At most 6 sentences.\n\n"
    "Running summary:\n{previous}\n\nLatest segment:\n{latest}\n\nMerged summary:"
)


class ConversationSummaryStore:
    """Rolling per-session summaries persisted in the ``sessions`` table."""

    def __init__(self, storage: Any, max_messages: int = 10, llm_model: Optional[str] = None, max_chars: int = 1500):
        self.storage = storage
        self.max_messages = max_messages
        self.llm_model = llm_model
        self.max_chars = max_chars
        self._cache: Dict[str, str] = {}

    def _key(self, namespace: Optional[str], session_id: str) -> str:
        return f"{namespace or 'default'}:{session_id}"

    async def get_summary(self, session_id: str = "default") -> str:
        key = self._key(getattr(self.storage, "namespace", None), session_id)
        if key in self._cache:
            return self._cache[key]
        summary = await self._load_from_storage(session_id)
        if summary:
            self._cache[key] = summary
        return summary or ""

    async def fold_episode(self, session_id: str, episode: Dict[str, Any]) -> str:
        """Merge a closed episode's summary into the session summary."""
        latest = " ".join(
            x for x in (episode.get("title"), episode.get("summary")) if x
        ).strip()
        if not latest:
            return await self.get_summary(session_id)
        previous = await self.get_summary(session_id)
        merged = await self._merge(previous, latest)
        await self._save(session_id, merged)
        return merged

    async def refresh(
        self,
        messages: List[Dict[str, Any]],
        session_id: str = "default",
        llm_model: Optional[str] = None,
    ) -> str:
        """Summarise recent messages directly (compatibility path)."""
        text = self._format_messages(messages[-self.max_messages :])
        summary = await self._summarize(text, llm_model or self.llm_model)
        await self._save(session_id, summary)
        return summary

    async def _merge(self, previous: str, latest: str) -> str:
        from memorable_ai.core.llm_client import llm_enabled

        if previous and llm_enabled(self.llm_model):
            try:
                from memorable_ai.core import llm_client

                resp = await llm_client.acompletion(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": MERGE_PROMPT.format(previous=previous, latest=latest)}],
                    max_tokens=350,
                    temperature=0,
                )
                out = llm_client.message_content(resp)
                if out:
                    return out[: self.max_chars]
            except Exception as e:
                logger.debug("Summary merge failed, using concatenation: %s", e)
        merged = f"{previous}\n{latest}".strip() if previous else latest
        if len(merged) > self.max_chars:
            merged = "..." + merged[-(self.max_chars - 3):]
        return merged

    async def _save(self, session_id: str, summary: str) -> None:
        key = self._key(getattr(self.storage, "namespace", None), session_id)
        self._cache[key] = summary
        try:
            if hasattr(self.storage, "store_summary"):
                await self.storage.store_summary(key, summary, session_id=session_id)
        except Exception as e:
            logger.debug("Could not persist summary: %s", e)

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        from memorable_ai.core.episodic import message_text

        lines = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
                content = message_text(msg.get("content"))
                if content:
                    lines.append(f"{msg['role']}: {content}")
        return "\n".join(lines)

    async def _summarize(self, text: str, llm_model: Optional[str]) -> str:
        if not text.strip():
            return ""
        from memorable_ai.core.llm_client import llm_enabled

        if llm_enabled(llm_model):
            try:
                from memorable_ai.core import llm_client

                resp = await llm_client.acompletion(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": "Summarize the conversation in 3-5 sentences. Keep facts, preferences, and names."},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=300,
                )
                return llm_client.message_content(resp)
            except Exception as e:
                logger.debug("LLM summary failed, using truncation: %s", e)
        return text[: self.max_chars]

    async def _load_from_storage(self, session_id: str) -> Optional[str]:
        try:
            if hasattr(self.storage, "get_session_summary"):
                return await self.storage.get_session_summary(session_id)
        except Exception:
            pass
        return None
