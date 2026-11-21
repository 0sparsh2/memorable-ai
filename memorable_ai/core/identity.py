"""
Cross-session identity resolution.

Maps session/device aliases to a canonical user id within a namespace. Aliases
are persisted in ``identity_aliases`` (previously they lived only in process
memory and were lost on restart).
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class IdentityResolver:
    """Resolve session/device aliases to a canonical user_id within a namespace."""

    def __init__(self, storage: Any, namespace: Optional[str] = None):
        self.storage = storage
        self.namespace = namespace
        self._aliases: Dict[str, str] = {}
        self._loaded = False

    def _k(self, alias: str) -> str:
        return f"{self.namespace or ''}:{alias}"

    def _load(self) -> None:
        if self._loaded or self.storage is None or not hasattr(self.storage, "get_session"):
            self._loaded = True
            return
        try:
            from memorable_ai.core.models import IdentityAlias

            session = self.storage.get_session()
            try:
                q = session.query(IdentityAlias)
                q = q.filter(
                    IdentityAlias.namespace == self.namespace
                    if self.namespace
                    else IdentityAlias.namespace.is_(None)
                )
                for row in q.all():
                    self._aliases[self._k(row.alias)] = row.canonical_user_id
            finally:
                session.close()
        except Exception as e:
            logger.debug("Could not load identity aliases: %s", e)
        self._loaded = True

    def register_alias(self, alias: str, canonical_user_id: str, confidence: float = 1.0):
        """Link a session or device id to a canonical user id (persisted)."""
        if not alias or not canonical_user_id:
            return
        self._load()
        self._aliases[self._k(alias)] = canonical_user_id
        if self.storage is None or not hasattr(self.storage, "get_session"):
            return
        try:
            from memorable_ai.core.models import IdentityAlias

            session = self.storage.get_session()
            try:
                q = session.query(IdentityAlias).filter(IdentityAlias.alias == alias)
                q = q.filter(
                    IdentityAlias.namespace == self.namespace
                    if self.namespace
                    else IdentityAlias.namespace.is_(None)
                )
                row = q.first()
                if row is None:
                    session.add(
                        IdentityAlias(
                            namespace=self.namespace,
                            alias=alias,
                            canonical_user_id=canonical_user_id,
                            confidence=confidence,
                        )
                    )
                else:
                    row.canonical_user_id = canonical_user_id
                    row.confidence = confidence
                session.commit()
            finally:
                session.close()
        except Exception as e:
            logger.warning("Could not persist identity alias: %s", e)

    def resolve(self, session_or_user_id: Optional[str]) -> Optional[str]:
        """Return the canonical user id for an alias (or the input if unknown)."""
        if session_or_user_id is None:
            return None
        self._load()
        return self._aliases.get(self._k(session_or_user_id), session_or_user_id)

    def merge_memories_query_filter(self, user_id: str) -> Dict[str, Any]:
        canonical = self.resolve(user_id)
        return {"canonical_user_id": canonical, "session_id": user_id}
