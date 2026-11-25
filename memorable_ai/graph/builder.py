"""
Graph Builder

Auto-extracts relationships from conversations and builds knowledge graph.
Inspired by Supermemory's knowledge graph architecture.

Reference: https://github.com/supermemoryai/supermemory
"""

import logging
from typing import Any, Dict, List, Optional
import networkx as nx

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds and maintains a knowledge graph from conversations.
    
    Extracts entities and relationships to enable multi-hop reasoning.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize graph builder.
        
        Args:
            connection_string: Optional graph database connection (Neo4j, etc.)
        """
        self.connection_string = connection_string
        self.graph = nx.MultiDiGraph()
        self._neo4j = None
        if connection_string and "neo4j" in connection_string.lower():
            from memorable_ai.graph.neo4j_backend import Neo4jGraphBackend

            self._neo4j = Neo4jGraphBackend(connection_string)
        logger.info("Graph builder initialized")

    async def update_graph(self, memories: List[Dict[str, Any]]):
        """
        Update graph with new memories.
        
        Extracts entities and relationships from memories.
        
        Args:
            memories: List of extracted memories
        """
        for memory in memories:
            await self._add_memory_to_graph(memory)

    async def _add_memory_to_graph(self, memory: Dict[str, Any]):
        """
        Add a memory to the graph.
        
        Extracts entities and creates relationships.
        
        Args:
            memory: Memory dictionary
        """
        content = memory.get("content", "")
        memory_type = memory.get("type", "fact")
        memory_id = memory.get("id")

        if not content:
            return

        entities = self._extract_entities(content)
        if self._neo4j and self._neo4j.available:
            for entity in entities:
                self._neo4j.upsert_entity(entity, memory_id=memory_id)
        
        # Create nodes for entities
        for entity in entities:
            if not self.graph.has_node(entity):
                self.graph.add_node(entity, type="entity", count=0)
            self.graph.nodes[entity]["count"] += 1

        # Create memory node
        if memory_id:
            memory_node = f"memory_{memory_id}"
            self.graph.add_node(
                memory_node,
                type="memory",
                content=content,
                memory_type=memory_type,
            )

            # Create edges from entities to memory
            for entity in entities:
                self.graph.add_edge(entity, memory_node, relationship="contains")

        # Extract relationships between entities
        relationships = self._extract_relationships(content, entities)
        for rel in relationships:
            source, target, rel_type = rel
            self.graph.add_edge(source, target, relationship=rel_type)
            if self._neo4j and self._neo4j.available:
                self._neo4j.upsert_relationship(source, rel_type, target)

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text.
        
        Simple implementation - can be enhanced with NER models.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entity names
        """
        from memorable_ai.core.entities import extract_entities

        entities = [e.title() for e in extract_entities(text)]
        import re
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", text)
        entities.extend(capitalized)
        
        # Common patterns
        patterns = [
            r"I (?:work at|am at|use) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+) (?:is|are) (?:a|an) ([a-z]+)",
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.extend([g for g in match.groups() if g])
        
        # Deduplicate
        return list(set(entities))

    def _extract_relationships(
        self, text: str, entities: List[str]
    ) -> List[tuple]:
        """
        Extract relationships between entities.
        
        Args:
            text: Text containing relationships
            entities: List of entities
            
        Returns:
            List of (source, target, relationship_type) tuples
        """
        relationships = []
        
        if len(entities) < 2:
            return relationships

        # Simple relationship patterns
        import re
        patterns = [
            (r"(\w+) (?:works at|uses|likes) (\w+)", "related_to"),
            (r"(\w+) (?:is|are) (?:a|an) (\w+)", "is_a"),
        ]

        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source = match.group(1)
                target = match.group(2)
                if source in entities and target in entities:
                    relationships.append((source, target, rel_type))

        return relationships

    async def find_related(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find related memories using graph traversal.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of related memories
        """
        # Extract entities from query
        query_entities = self._extract_entities(query)
        
        if self._neo4j and self._neo4j.available:
            neo_results = self._neo4j.find_related_entities(query, limit=limit)
            if neo_results:
                return [
                    {
                        "content": f"{r.get('source')} {r.get('rel')} {r.get('target')}",
                        "type": "fact",
                        "metadata": {"source": "neo4j"},
                    }
                    for r in neo_results
                ]

        if not query_entities:
            return []

        # Find nodes connected to query entities
        related_nodes = set()
        for entity in query_entities:
            if self.graph.has_node(entity):
                # Get neighbors (1-hop)
                neighbors = list(self.graph.neighbors(entity))
                related_nodes.update(neighbors)
                
                # Get 2-hop neighbors for multi-hop reasoning
                for neighbor in neighbors:
                    two_hop = list(self.graph.neighbors(neighbor))
                    related_nodes.update(two_hop)

        # Extract memory nodes
        memories = []
        for node in related_nodes:
            if self.graph.nodes[node].get("type") == "memory":
                content = self.graph.nodes[node].get("content", "")
                memory_type = self.graph.nodes[node].get("memory_type", "fact")
                memories.append({
                    "content": content,
                    "type": memory_type,
                    "id": node,
                })

        return memories[:limit]

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "entity_nodes": sum(
                1 for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"
            ),
            "memory_nodes": sum(
                1 for n, d in self.graph.nodes(data=True) if d.get("type") == "memory"
            ),
        }

