"""Optional Neo4j backend for production graph memory."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Neo4jGraphBackend:
    """Thin Neo4j wrapper for entity-relationship storage."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._driver = None
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(connection_string)
            logger.info("Neo4j graph backend connected")
        except ImportError:
            logger.warning("neo4j package not installed; pip install memorable-ai[graph]")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}")

    @property
    def available(self) -> bool:
        return self._driver is not None

    def upsert_entity(self, name: str, entity_type: str = "Entity", memory_id: Optional[int] = None):
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $type, e.memory_id = $mid, e.updated_at = datetime()
                """,
                name=name.lower(),
                type=entity_type,
                mid=memory_id,
            )

    def upsert_relationship(
        self, source: str, rel: str, target: str, valid: bool = True
    ):
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                """
                MERGE (a:Entity {name: $source})
                MERGE (b:Entity {name: $target})
                MERGE (a)-[r:REL {type: $rel}]->(b)
                SET r.valid = $valid, r.updated_at = datetime()
                """,
                source=source.lower(),
                target=target.lower(),
                rel=rel,
                valid=valid,
            )

    def invalidate_relationship(self, source: str, rel: str, target: str):
        self.upsert_relationship(source, rel, target, valid=False)

    def find_related_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self._driver:
            return []
        tokens = [t.lower().strip(".,!?;:") for t in query.split() if len(t) > 2]
        if not tokens:
            return []
        with self._driver.session() as session:
            # Parenthesised: previously `a OR b AND valid` returned invalidated edges,
            # and only the first query token was matched.
            result = session.run(
                """
                MATCH (e:Entity)-[r:REL]->(t:Entity)
                WHERE (any(q IN $tokens WHERE e.name CONTAINS q)
                       OR any(q IN $tokens WHERE t.name CONTAINS q))
                  AND coalesce(r.valid, true) = true
                RETURN e.name AS source, r.type AS rel, t.name AS target
                LIMIT $limit
                """,
                tokens=tokens,
                limit=limit,
            )
            return [dict(record) for record in result]

    def close(self):
        if self._driver:
            self._driver.close()
