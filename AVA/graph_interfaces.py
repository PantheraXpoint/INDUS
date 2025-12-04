import numpy as np
import uuid
import copy
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field

# ==========================================
# 1. Core Data Structures
# ==========================================

@dataclass
class Node:
    id: str
    type: str  # 'event' or 'object'
    score: float = 0.0
    embedding: Optional[np.ndarray] = None
    content: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id

@dataclass
class Edge:
    source_id: str
    target_id: str
    type: str
    score: float = 0.0
    metadata: Dict = field(default_factory=dict)

class Subgraph:
    """
    Represents a specific reasoning path or cluster of knowledge.
    """
    def __init__(self, id: str = None):
        self.id = id if id else str(uuid.uuid4())
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        # Prevent strict edge duplicates
        for e in self.edges:
            if e.source_id == edge.source_id and e.target_id == edge.target_id and e.type == edge.type:
                return
        self.edges.append(edge)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, type_name: str) -> List[Node]:
        return [n for n in self.nodes.values() if n.type == type_name]

# ==========================================
# 2. Global Helper Functions
# ==========================================

def get_global_system_counts(subgraphs: List[Subgraph]) -> Dict[str, int]:
    """
    Counts total Events and Objects across ALL active subgraphs.
    Used by the Thermostat to balance the global search state.
    """
    total = {'event': 0, 'object': 0}
    for sg in subgraphs:
        total['event'] += len([n for n in sg.nodes.values() if n.type == 'event'])
        total['object'] += len([n for n in sg.nodes.values() if n.type == 'object'])
    return total

# ==========================================
# 3. Database Interfaces (Mocks/Placeholders)
# ==========================================

class KnowledgeGraphInterface:
    def __init__(self, 
                 object_faiss_db_path: str = "",
                 event_faiss_db_path: str = "",
                 object_sqlite_db_path: str = "",
                 embedding_dim: int = 512):
        # Initialize connections here (Milvus, SQLite, etc.)
        pass

    # --- Vector Search Mocks (Replace with Milvus logic) ---
    def search_events_by_embedding(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Node]:
        # MOCK RETURN
        return [Node(id=f"evt_{i}", type="event", score=0.85, embedding=np.random.rand(512)) for i in range(top_k)]

    def search_objects_by_embedding(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Node]:
        # MOCK RETURN
        return [Node(id=f"obj_{i}", type="object", score=0.82, embedding=np.random.rand(512)) for i in range(top_k)]

    # --- Structural Lookup Mocks (Replace with SQLite/GraphDB logic) ---
    def get_objects_in_event(self, event_id: str) -> List[str]:
        # Returns list of Object IDs contained in Event
        return [f"obj_{i}" for i in range(3)] 

    def get_events_containing_object(self, object_id: str) -> List[str]:
        # Returns list of Event IDs that contain Object
        return [f"evt_{i}" for i in range(2)]

    def get_global_event_count_for_object(self, object_id: str) -> int:
        # Used for Uniqueness Bonus
        return 5 # Example: Object appears in 5 events globally

    def get_relations_involving_object(self, object_id: str) -> List[Dict]:
        # NOTE: This is now unused as relation expansion is context-only.
        # Returns [{'target_id': 'obj_99', 'type': 'is_related_to'}]
        return []

class ContextGraphInterface:
    def __init__(self, db_path: str = "context_memory.db"):
        pass

    def add_context(self, query_embedding: np.ndarray, subgraph: Subgraph):
        print(f"DEBUG: Saved subgraph {subgraph.id} to Context Graph.")

    def search_context(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Subgraph]:
        # MOCK: Return empty list or previous subgraphs
        return []