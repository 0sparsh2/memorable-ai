"""
Memory extraction: conversation turns -> atomic semantic memories.

Two extractors, used together in ``hybrid`` mode:

- **LLM** (Mem0-style, arXiv:2504.19413). One call per batch of new turns. It gets
  the session date (so relative dates resolve), the session summary and a short
  window of preceding turns (so pronouns resolve). It returns standalone facts
  with entities, (subject, predicate, object) relations and the turns they came
  from, which feed the knowledge graph without a second LLM call.
- **Patterns** (no LLM). Runs on *user* turns only: the old version also ran on the
  assistant's reply, so "I can help with that" became a user skill. Output is a
  full statement ("User lives in Paris"), not a fragment ("Paris").

Relative dates in every fact are anchored to the session date.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from memorable_ai.core.temporal import anchor_relative_dates

from memorable_ai.core.models import utcnow  # noqa: E402

logger = logging.getLogger(__name__)

MEMORY_TYPES = ("fact", "preference", "skill", "rule", "context")
TYPE_IMPORTANCE = {"rule": 0.8, "preference": 0.6, "fact": 0.6, "skill": 0.5, "context": 0.4}
_SALIENT = re.compile(
    r"\b(allerg\w*|medication|diabet\w*|vegan|vegetarian|gluten|pregnan\w*|disabilit\w*|"
    r"birthday|anniversary|password|never|always|name)\b",
    re.IGNORECASE,
)

_VERBS = (
    r"live|lives|living|work|works|working|like|likes|love|loves|enjoy|enjoys|prefer|prefers|hate|"
    r"hates|dislike|dislikes|use|uses|have|has|am|'m|moved|study|studies|play|plays|speak|speaks|"
    r"own|owns|need|needs|want|wants|can|know|built|build|building"
)
_CLAUSE_SPLIT = re.compile(
    r"\s*;\s*|,?\s+(?:and|but)\s+(?=(?:I\b|my\b|also\b|(?:" + _VERBS + r")\b))", re.IGNORECASE
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_OBJ_STOP = re.compile(r"\s+(?:because|since|so that|although|though|which|who|when|if)\b.*$", re.IGNORECASE)

_THIRD_PERSON_STOP = {
    "it", "this", "that", "there", "here", "what", "which", "who", "when", "where", "why", "how",
    "the", "a", "an", "he", "she", "they", "we", "you", "i", "my", "our", "your", "his", "her",
    "its", "their", "yes", "no", "ok", "okay", "please", "thanks", "thank", "hi", "hello", "hey",
    "also", "so", "but", "and", "or", "if", "then", "today", "yesterday", "tomorrow",
}

_RELATIVES = (
    r"sister|brother|wife|husband|partner|girlfriend|boyfriend|mom|mother|dad|father|son|daughter|"
    r"friend|best friend|boss|manager|colleague|coworker|roommate|dog|cat|cousin|uncle|aunt|"
    r"grandmother|grandfather|grandma|grandpa|fiance|fiancee"
)

# (regex, memory_type, template). Templates use match groups; {v} is the verb lemma.
_FIRST_PERSON: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"\b(?i:my name is|i'm called|i am called) ([A-Z][\w'-]+(?: [A-Z][\w'-]+){0,2})"), "fact", "User's name is {1}"),
    (re.compile(r"\b(?:please )?call me ([A-Z][\w'-]+)", re.IGNORECASE), "preference", "User prefers to be called {1}"),
    (re.compile(r"\bI(?:'m| am) (\d{1,3}) years old", re.IGNORECASE), "fact", "User is {1} years old"),
    (re.compile(r"\bI(?:'m| am) allergic to ([^,.;!?]+)", re.IGNORECASE), "fact", "User is allergic to {1}"),
    (re.compile(r"\bI (?:currently |still )?(?:live|am living|reside) in ([^,.;!?]+)", re.IGNORECASE), "fact", "User lives in {1}"),
    (re.compile(r"\bI(?:'ve| have| just)? (?:recently )?moved to ([^,.;!?]+)", re.IGNORECASE), "fact", "User moved to {1}"),
    (re.compile(r"\bI (?:currently |now )?work (?:at|for) ([^,.;!?]+)", re.IGNORECASE), "fact", "User works at {1}"),
    (re.compile(r"\bI(?:'m| am) (?:originally )?from ([^,.;!?]+)", re.IGNORECASE), "fact", "User is from {1}"),
    (re.compile(r"\b(?i:my) ((?i:" + _RELATIVES + r"))(?i:'s name is| is named| is called|,| named| called)?\s+([A-Z][\w'-]+)"), "fact", "User's {1} is named {2}"),
    (re.compile(r"\bI (?:went|traveled|travelled|flew|drove|moved back) to ([^,.;!?]+)", re.IGNORECASE), "fact", "User went to {1}"),
    (re.compile(r"\bmy favou?rite (\w+(?: \w+)?) is ([^,.;!?]+)", re.IGNORECASE), "preference", "User's favorite {1} is {2}"),
    (re.compile(r"\bmy (birthday|email|e-mail|phone number|address|job|role|title|major|goal) is ([^,;!?]+?)(?:[.!?]?$|[,;!?])", re.IGNORECASE), "fact", "User's {1} is {2}"),
    (re.compile(r"\bI (?:really |also |absolutely |just )?(like|love|enjoy|prefer|adore) ([^,.;!?]+)", re.IGNORECASE), "preference", "User {v}s {2}"),
    (re.compile(r"\bI (?:really )?(?:don't|do not) (?:really )?(?:like|enjoy|want) ([^,.;!?]+)", re.IGNORECASE), "preference", "User dislikes {1}"),
    (re.compile(r"\bI (?:really )?(?:hate|dislike|can't stand|cannot stand) ([^,.;!?]+)", re.IGNORECASE), "preference", "User dislikes {1}"),
    (re.compile(r"\bI(?:'m| am) (?:really )?(?:good|skilled|experienced|an expert|proficient) (?:at|in|with) ([^,.;!?]+)", re.IGNORECASE), "skill", "User is skilled in {1}"),
    (re.compile(r"\bI have (?:\d+ years of |years of |a lot of )?experience (?:with|in) ([^,.;!?]+)", re.IGNORECASE), "skill", "User has experience with {1}"),
    (re.compile(r"\bI (?:can|know how to) (?!help|do that|try|see|imagine)([^,.;!?]+)", re.IGNORECASE), "skill", "User can {1}"),
    (re.compile(r"\bI speak ([^,.;!?]+)", re.IGNORECASE), "skill", "User speaks {1}"),
    (re.compile(r"\bI(?:'m| am) (?:currently |now )?(working on|building|writing|learning|planning|preparing for|studying|training for) ([^,.;!?]+)", re.IGNORECASE), "context", "User is {1} {2}"),
    (re.compile(r"\bI(?:'m| am) (?:a |an )?(vegetarian|vegan|pescatarian|gluten[- ]free|lactose[- ]intolerant|"
                r"diabetic|pregnant|married|engaged|divorced|single|widowed|retired|unemployed|self[- ]employed|"
                r"left[- ]handed|colou?r[- ]?blind|deaf|blind|sober|introverted|extroverted)\b", re.IGNORECASE),
     "fact", "User is {1}"),
    (re.compile(r"\bI(?:'m| am) (?:a|an) ([^,.;!?]+)", re.IGNORECASE), "fact", "User is a {1}"),
    (re.compile(r"\bI (?:use|am using) ([^,.;!?]+)", re.IGNORECASE), "fact", "User uses {1}"),
    (re.compile(r"\bI (?:have|own|got) (a|an|one|two|three|four|\d+) ([^,.;!?]+)", re.IGNORECASE), "fact", "User has {1} {2}"),
]

_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(?:please\s+)?(always|never)\s+([^.;!?]{4,})", re.IGNORECASE), "{1} {2}"),
    (re.compile(r"^(?:please\s+)?(?:don't|do not) ever\s+([^.;!?]{4,})", re.IGNORECASE), "Never {1}"),
    (re.compile(r"\b(?:rule|constraint)\s*:\s*([^.;!?]{4,})", re.IGNORECASE), "{1}"),
]

_THIRD_PERSON = re.compile(
    r"^([A-Z][a-z]+(?: [A-Z][a-z]+)?) (lives in|works at|works for|is from|moved to|likes|loves|prefers|"
    r"enjoys|hates|dislikes|is allergic to|is married to|is a|is an|has a|has an|studies|teaches|owns) "
    r"([^,.;!?]+)"
)

_LEMMA = {"like": "like", "love": "love", "enjoy": "enjoy", "prefer": "prefer", "adore": "adore"}


def _clean_object(obj: str) -> str:
    obj = _OBJ_STOP.sub("", obj.strip())
    obj = re.sub(r"\bmy\b", "User's", obj, flags=re.IGNORECASE)
    obj = re.sub(r"\b(?:me|myself)\b", "User", obj, flags=re.IGNORECASE)
    obj = re.sub(r"\s+(?:too|as well|a lot|very much|so much|now|anymore|lately)$", "", obj, flags=re.IGNORECASE)
    return obj.strip(" \"'")


def _split_clauses(sentence: str) -> List[str]:
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(sentence) if p and p.strip()]
    out: List[str] = []
    for i, p in enumerate(parts):
        # "I live in Paris and work at Google" -> "work at Google" inherits the subject.
        if i > 0 and not re.match(r"^(I|my)\b", p, re.IGNORECASE) and re.match(
            r"^(?:also\s+)?(?:" + _VERBS + r")\b", p, re.IGNORECASE
        ):
            p = "I " + re.sub(r"^also\s+", "", p, flags=re.IGNORECASE)
        out.append(p)
    return out


def _turn_text(turn: Dict[str, Any]) -> str:
    content = turn.get("content", "")
    return content if isinstance(content, str) else str(content)


class MemoryExtractor:
    """
    Extracts structured memories from conversations.

    Memory types: fact, preference, skill, rule, context.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        llm_model: Optional[str] = None,
        extraction_mode: str = "hybrid",
        summary_store: Optional[Any] = None,
        context_window: int = 6,
        ontology: Optional[Any] = None,
    ):
        from memorable_ai.core.embeddings import as_embedder
        from memorable_ai.graph.ontology import default_ontology

        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self._embedder = as_embedder(embedding_model)
        self.llm_model = llm_model
        self.extraction_mode = extraction_mode
        self.summary_store = summary_store
        self.context_window = context_window
        self.ontology = ontology or default_ontology()

    # ------------------------------------------------------------------ public API

    async def extract(
        self,
        messages: List[Dict[str, Any]],
        response: Optional[Any] = None,
        session_id: Optional[str] = None,
        session_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Extract from the latest exchange (last user message + response)."""
        turns: List[Dict[str, Any]] = []
        context: List[Dict[str, Any]] = []
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                last_user_idx = i
                break
            if isinstance(msg, str):
                last_user_idx = i
                break
        if last_user_idx is not None:
            m = messages[last_user_idx]
            turns.append(m if isinstance(m, dict) else {"role": "user", "content": m})
            context = [
                x for x in messages[max(0, last_user_idx - self.context_window) : last_user_idx]
                if isinstance(x, dict) and x.get("role") in ("user", "assistant")
            ]
        else:
            turns.extend(
                m if isinstance(m, dict) else {"role": "user", "content": m}
                for m in messages
                if (isinstance(m, dict) and m.get("content")) or isinstance(m, str)
            )
        resp_text = response_text(response)
        if resp_text:
            turns.append({"role": "assistant", "content": resp_text})
        summary = ""
        if self.summary_store and session_id:
            try:
                summary = await self.summary_store.get_summary(session_id)
            except Exception:
                summary = ""
        return await self.extract_from_turns(
            turns, context_turns=context, session_summary=summary, session_date=session_date
        )

    async def extract_from_turns(
        self,
        turns: Sequence[Dict[str, Any]],
        context_turns: Sequence[Dict[str, Any]] = (),
        session_summary: str = "",
        session_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract memories from ``turns`` (dicts with role, content and optional
        ``id``/``occurred_at``). ``context_turns`` are shown to the LLM for
        coreference but nothing is extracted from them.
        """
        turns = [t for t in turns if _turn_text(t).strip()]
        if not turns:
            return []
        ref_date = session_date or _first_time(turns) or utcnow()
        memories: List[Dict[str, Any]] = []

        from memorable_ai.core.llm_client import llm_enabled

        use_llm = self.extraction_mode in ("llm", "hybrid") and (
            llm_enabled(self.llm_model) or self.llm_client is not None
        )
        if use_llm:
            memories.extend(await self._extract_llm(turns, context_turns, session_summary, ref_date))
        if self.extraction_mode in ("pattern", "hybrid") or (self.extraction_mode == "llm" and not memories):
            memories.extend(self._extract_patterns(turns, ref_date, skip_if_llm=bool(memories)))

        memories = self._deduplicate_memories(memories)
        for m in memories:
            anchored, event_date = anchor_relative_dates(m["content"], ref_date)
            m["content"] = anchored
            meta = m.setdefault("metadata", {})
            if event_date and not meta.get("event_date"):
                meta["event_date"] = event_date
            meta.setdefault("stated_at", ref_date.isoformat())
            m.setdefault("valid_from", ref_date)  # true as of when it was said
            m.setdefault("importance_score", _importance(m))
        if self._embedder is not None and memories:
            try:
                vecs = self._embedder.encode([m["content"] for m in memories])
                for m, v in zip(memories, vecs):
                    m["embedding"] = v.tolist()
            except Exception as e:
                logger.debug("Embedding failed: %s", e)
        logger.debug("Extracted %d memories from %d turns", len(memories), len(turns))
        return memories

    # ------------------------------------------------------------------ patterns

    def _extract_patterns(
        self, turns: Sequence[Dict[str, Any]], ref_date: datetime, skip_if_llm: bool = False
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, turn in enumerate(turns):
            if turn.get("role", "user") != "user":
                continue
            for m in self._patterns_from_text(_turn_text(turn)):
                m["metadata"]["turn_index"] = idx
                if turn.get("id") is not None:
                    m["source_turn_ids"] = [turn["id"]]
                out.append(m)
        if skip_if_llm:
            # In hybrid mode the LLM output is primary; keep only high-precision patterns.
            out = [m for m in out if m["type"] in ("rule",) or _SALIENT.search(m["content"])]
        return out

    def _patterns_from_text(self, text: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for sentence in _SENTENCE_SPLIT.split(text or ""):
            sentence = sentence.strip()
            if not sentence or sentence.endswith("?"):
                continue  # questions are not statements about the user
            for clause in _split_clauses(sentence.rstrip(".!")):
                results.extend(self._match_clause(clause))
            for pat, template in _RULES:
                m = pat.search(sentence)
                if m:
                    groups = [g for g in m.groups()]
                    content = template.format(None, *groups).strip()
                    content = content[0].upper() + content[1:]
                    results.append(_mem(content, "rule"))
                    break
            # "My sister Emma lives in Boston" -> third-person statement about Emma.
            third = re.sub(r"^(?i:my) (?i:" + _RELATIVES + r"),? ([A-Z][\w'-]+)", r"\1", sentence)
            tp = _THIRD_PERSON.match(third)
            if tp and tp.group(1).lower() not in _THIRD_PERSON_STOP:
                obj = _clean_object(tp.group(3))
                if len(obj) >= 2:
                    verb = tp.group(2)
                    mtype = "preference" if verb in ("likes", "loves", "prefers", "enjoys", "hates", "dislikes") else "fact"
                    results.append(
                        _mem(f"{tp.group(1)} {verb} {obj}", mtype, person=tp.group(1))
                    )
        return results

    def _match_clause(self, clause: str) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        claimed: List[Tuple[int, int]] = []
        for pat, mtype, template in _FIRST_PERSON:
            for m in pat.finditer(clause):
                span = m.span()
                if any(s <= span[0] < e for s, e in claimed):
                    continue  # a more specific pattern already covered this text
                groups = [None] + [_clean_object(g) if g else "" for g in m.groups()]
                if any(g is not None and len(g) < 2 for g in groups[1:]):
                    continue
                verb = _LEMMA.get((m.group(1) or "").lower(), "like") if "{v}" in template else ""
                try:
                    content = template.format(*groups, v=verb)
                except (IndexError, KeyError):
                    continue
                content = re.sub(r"\s+", " ", content).strip()
                if len(content) > 300:
                    continue
                meta_extra = {}
                if mtype == "preference":
                    meta_extra["sentiment"] = "negative" if "dislikes" in content else "positive"
                found.append(_mem(content, mtype, **meta_extra))
                claimed.append(span)
        return found

    # Compatibility helpers (older callers and tests)
    async def _extract_facts(self, text: str) -> List[Dict[str, Any]]:
        return [m for m in self._patterns_from_text(text) if m["type"] == "fact"]

    async def _extract_preferences(self, text: str) -> List[Dict[str, Any]]:
        return [m for m in self._patterns_from_text(text) if m["type"] == "preference"]

    async def _extract_skills(self, text: str) -> List[Dict[str, Any]]:
        return [m for m in self._patterns_from_text(text) if m["type"] == "skill"]

    async def _extract_rules(self, text: str) -> List[Dict[str, Any]]:
        return [m for m in self._patterns_from_text(text) if m["type"] == "rule"]

    async def _extract_context(self, text: str) -> List[Dict[str, Any]]:
        return [m for m in self._patterns_from_text(text) if m["type"] == "context"]

    def _extract_text(self, messages: List[Dict[str, Any]], response: Optional[Any] = None) -> str:
        parts: List[str] = []
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                parts.append(_turn_text(msg))
                break
            if isinstance(msg, str):
                parts.append(msg)
                break
        r = response_text(response)
        if r:
            parts.append(r)
        return " ".join(parts)

    # ------------------------------------------------------------------ LLM

    async def _extract_llm(
        self,
        turns: Sequence[Dict[str, Any]],
        context_turns: Sequence[Dict[str, Any]],
        session_summary: str,
        ref_date: datetime,
    ) -> List[Dict[str, Any]]:
        lines = []
        for i, t in enumerate(turns):
            lines.append(f"[{i}] {t.get('role', 'user')}: {_turn_text(t)[:2000]}")
        ctx_lines = [
            f"{t.get('role', 'user')}: {_turn_text(t)[:500]}" for t in list(context_turns)[-self.context_window :]
        ]
        prompt = EXTRACTION_PROMPT.format(
            date=ref_date.strftime("%Y-%m-%d (%A)"),
            summary=session_summary or "(none)",
            context="\n".join(ctx_lines) or "(none)",
            turns="\n".join(lines),
            ontology=self.ontology.prompt_block(),
        )
        items = await self._call_llm_json(prompt, max_tokens=1200)
        if isinstance(items, dict):
            items = items.get("memories") or items.get("facts") or [items]
        out: List[Dict[str, Any]] = []
        for it in items or []:
            if not isinstance(it, dict) or not str(it.get("content", "")).strip():
                continue
            mtype = str(it.get("type", "fact")).lower()
            if mtype not in MEMORY_TYPES:
                mtype = "fact"
            meta: Dict[str, Any] = {"source": "llm", "extraction_method": "llm"}
            if it.get("event_date"):
                meta["event_date"] = str(it["event_date"])
            if it.get("dia_ids"):
                meta["dia_ids"] = it["dia_ids"]
            mem: Dict[str, Any] = {"content": str(it["content"]).strip(), "type": mtype, "metadata": meta}
            idxs = [i for i in (it.get("turns") or []) if isinstance(i, int) and 0 <= i < len(turns)]
            turn_ids = [turns[i]["id"] for i in idxs if turns[i].get("id") is not None]
            if turn_ids:
                mem["source_turn_ids"] = turn_ids
            ents = it.get("entities") or []
            mem["entities"] = [
                {"name": str(e.get("name")).strip(), "type": str(e.get("type", "concept")).lower()}
                if isinstance(e, dict)
                else {"name": str(e).strip(), "type": "concept"}
                for e in ents
                if (isinstance(e, dict) and e.get("name")) or isinstance(e, str)
            ]
            rels = []
            for r in it.get("relations") or []:
                if isinstance(r, (list, tuple)) and len(r) == 3 and all(str(x).strip() for x in r):
                    rels.append(tuple(str(x).strip() for x in r))
                elif isinstance(r, dict) and r.get("subject") and r.get("predicate") and r.get("object"):
                    rels.append((str(r["subject"]), str(r["predicate"]), str(r["object"])))
            mem["relations"] = rels
            try:
                imp = float(it.get("importance", 0))
                if imp:
                    mem["importance_score"] = max(0.0, min(1.0, imp / 5.0 if imp > 1 else imp))
            except (TypeError, ValueError):
                pass
            out.append(mem)
        return out

    async def _call_llm_json(self, prompt: str, max_tokens: int = 800) -> Any:
        try:
            if self.llm_model:
                from memorable_ai.core import llm_client

                if not llm_client.llm_enabled(self.llm_model):
                    return []
                resp = await llm_client.acompletion_with_fallback(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                raw = llm_client.message_content(resp)
            elif self.llm_client is not None:
                raw = str(self.llm_client(prompt))
            else:
                return []
            from memorable_ai.core.llm_client import parse_json

            return parse_json(raw)
        except Exception as e:
            logger.warning("LLM extraction failed: %s", str(e)[:200])
            return []

    async def extract_with_llm(
        self,
        text: str,
        memory_types: Optional[List[str]] = None,
        summary_context: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        session_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """LLM extraction over raw text (compatibility entry point)."""
        summary = ""
        if self.summary_store and session_id:
            try:
                summary = await self.summary_store.get_summary(session_id)
            except Exception:
                pass
        return await self._extract_llm(
            [{"role": "user", "content": text}], [], summary, session_date or utcnow()
        )

    # ------------------------------------------------------------------ dedup

    def _deduplicate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop exact duplicates and statements contained in a longer one of the same type."""
        if not memories:
            return []
        ranked = sorted(
            (m for m in memories if (m.get("content") or "").strip()),
            key=lambda m: len(m["content"]),
            reverse=True,
        )
        kept: List[Dict[str, Any]] = []
        seen: List[Tuple[str, str]] = []
        for m in ranked:
            norm = re.sub(r"\s+", " ", m["content"].strip().lower()).rstrip(".")
            mtype = m.get("type", "fact")
            if any(norm == s or (norm in s and t == mtype) for s, t in seen):
                continue
            seen.append((norm, mtype))
            kept.append(m)
        return kept


EXTRACTION_PROMPT = """You extract long-term memories from a conversation for a personal assistant.

Today's date (when these messages were sent): {date}
Session summary so far: {summary}

Earlier messages (context only, do not extract from these):
{context}

New messages (extract from these):
{turns}

Return a JSON array. Each item:
{{"content": "standalone statement", "type": "fact|preference|skill|rule|context",
  "importance": 1-5, "event_date": "YYYY-MM-DD or null",
  "entities": [{{"name": "...", "type": "one of the entity types below"}}],
  "relations": [["subject", "predicate", "object"]],
  "turns": [message indices]}}

{ontology}

Rules:
- One atomic statement per item. Refer to the user as "User" and use full names for other people.
- Resolve pronouns and relative dates ("yesterday", "last week") into explicit names and calendar dates using today's date.
- Extract what the user states about themselves, their life, people they know, plans, preferences and instructions for the assistant ("rule"). Include facts the assistant states only if the user confirmed them.
- Skip greetings, questions, generic knowledge, and anything not useful in a future conversation.
- relations use the predicates above where one fits, e.g. ["User", "lives_in", "Berlin"], ["Emma", "sibling_of", "User"]. Keep the most specific word when it matters ("sister_of" is fine).
- Name each entity the same way every time; for a person you know by full name, use the full name.
- Return [] if there is nothing worth remembering. Output JSON only.
"""


def _mem(content: str, mtype: str, **meta: Any) -> Dict[str, Any]:
    return {"content": content, "type": mtype, "metadata": {"extraction_method": "pattern", **meta}}


def _importance(m: Dict[str, Any]) -> float:
    base = TYPE_IMPORTANCE.get(m.get("type", "fact"), 0.5)
    if _SALIENT.search(m.get("content", "")):
        base += 0.2
    return round(min(base, 1.0), 3)


def _first_time(turns: Sequence[Dict[str, Any]]) -> Optional[datetime]:
    for t in turns:
        ts = t.get("occurred_at")
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                continue
    return None


def response_text(response: Any) -> str:
    """Best-effort assistant text from OpenAI/Anthropic/LiteLLM responses or dicts."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "choices" in response:
            parts = []
            for choice in response.get("choices") or []:
                msg = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(msg, dict) and msg.get("content"):
                    parts.append(str(msg["content"]))
            return "\n".join(parts)
        if "content" in response:
            content = response["content"]
            if isinstance(content, str):
                return content
            return "\n".join(
                str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("text")
            )
        if "output_text" in response:
            return str(response["output_text"])
        return ""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    choices = getattr(response, "choices", None)
    if choices:
        parts = []
        for c in choices:
            msg = getattr(c, "message", None)
            content = getattr(msg, "content", None) if msg is not None else None
            if content:
                parts.append(str(content))
        if parts:
            return "\n".join(parts)
    content = getattr(response, "content", None)
    if isinstance(content, list):
        return "\n".join(str(getattr(b, "text", "")) for b in content if getattr(b, "text", None))
    if isinstance(content, str):
        return content
    return ""
