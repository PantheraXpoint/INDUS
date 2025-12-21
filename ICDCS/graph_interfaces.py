import numpy as np
import uuid
import copy
import sys
import os
import networkx as nx
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# from embeddings.Milvus import MilvusDB
# from embeddings.SQLiteDB import SQLiteDB
# from embeddings.JinaCLIP import JinaCLIP

from AVA.storage import TextNanoVectorDBStorage, NetworkXStorage

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
        
        # Memory for calculating velocity (rate of improvement)
        self.previous_iteration_stats = {
            'avg_score': 0.0,
            'node_count': 0,
            'event_count': 0,
            'object_count': 0,
            'iteration': 0
        }
        self.current_iteration = 0

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
    
    def remove_node(self, node_id: str):
        """Remove a node and all edges connected to it"""
        if node_id in self.nodes:
            # Remove the node
            del self.nodes[node_id]
            
            # Remove all edges connected to this node
            self.edges = [
                edge for edge in self.edges
                if edge.source_id != node_id and edge.target_id != node_id
            ]
    
    def update_iteration_stats(self):
        """
        Update statistics from current iteration.
        Call this at the END of each iteration to capture state.
        """
        if len(self.nodes) > 0:
            avg_score = sum(node.score for node in self.nodes.values()) / len(self.nodes)
        else:
            avg_score = 0.0
        
        self.previous_iteration_stats = {
            'avg_score': avg_score,
            'node_count': len(self.nodes),
            'event_count': len(self.get_nodes_by_type('event')),
            'object_count': len(self.get_nodes_by_type('object')),
            'iteration': self.current_iteration
        }
        self.current_iteration += 1
    
    def get_velocity(self) -> float:
        """
        Calculate velocity: rate of score improvement.
        Positive = finding better nodes, Negative = stagnating.
        """
        if len(self.nodes) == 0:
            return 0.0
        
        # Current average score
        current_avg = sum(node.score for node in self.nodes.values()) / len(self.nodes)
        
        # Previous average score
        previous_avg = self.previous_iteration_stats.get('avg_score', 0.0)
        
        # Velocity = current - previous
        return current_avg - previous_avg

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
                 working_dir: str, 
                 embedding_model,
                 embedding_dim: int = 768):
        """
        Args:
            working_dir: The directory containing AVA 'kg' outputs (json/graphml files).
            embedding_model: JinaCLIP model instance.
        """
        self.working_dir = working_dir
        self.embedding_model = embedding_model
        
        # Create a mock config to satisfy AVA storage classes
        self.global_config = {
            "working_dir": working_dir,
            "embedding_batch_num": 64, # Default param
            "cosine_better_than_threshold": 0.1
        }
        
        print(f"Loading Knowledge Base from: {working_dir}")

        # 1. Initialize Vector DBs (NanoDB)
        # Note: We use TextNanoVectorDBStorage but rely on its parent _query method for embeddings
        self.events_vdb = TextNanoVectorDBStorage(
            namespace="events",
            global_config=self.global_config,
            embedding_model=self.embedding_model,
            embedding_dim=embedding_dim,
            meta_fields={"id", "name", "description", "duration"}
        )
        
        self.entities_vdb = TextNanoVectorDBStorage(
            namespace="entities",
            global_config=self.global_config,
            embedding_model=self.embedding_model,
            embedding_dim=embedding_dim,
            meta_fields={"id", "descriptions", "timestamps", "frame_indices", "durations", "events"}
        )
        
        self.relations_vdb = TextNanoVectorDBStorage(
            namespace="relations",
            global_config=self.global_config,
            embedding_model=self.embedding_model,
            embedding_dim=embedding_dim,
            meta_fields={"id", "entity1", "entity2", "description"}
        )
        
        # 2. Initialize Structure Graph (NetworkX)
        self.graph_storage = NetworkXStorage(
            namespace="event_knowledge_graph",
            global_config=self.global_config
        )
    
    
    # --- Vector Search Methods ---

    def search_events_by_embedding(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Node]:
        """Search events using raw embedding via NanoDB."""
        results = self.events_vdb._query(query_embedding, top_k=top_k)
        return [self._create_node_from_vdb_result(res, 'event') for res in results]

    def search_objects_by_embedding(self, query_embedding: np.ndarray, top_k: int = 30) -> List[Node]:
        """Search objects (entities) using raw embedding via NanoDB."""
        results = self.entities_vdb._query(query_embedding, top_k=top_k)
        return [self._create_node_from_vdb_result(res, 'object') for res in results]

    def search_relations_by_embedding(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Node]:
        """Search relations using raw embedding via NanoDB."""
        results = self.relations_vdb._query(query_embedding, top_k=top_k)
        return [self._create_node_from_vdb_result(res, 'relation') for res in results]

    def search_events_by_description(self, description: str, top_k: int = 20) -> List[Node]:
        query_embedding = self.embedding_model.get_text_features([description])[0]
        return self.search_events_by_embedding(query_embedding, top_k)
    
    def search_objects_by_description(self, description: str, top_k: int = 20) -> List[Node]:
        query_embedding = self.embedding_model.get_text_features([description])[0]
        return self.search_objects_by_embedding(query_embedding, top_k)

    # --- Structural Lookup Methods ---

    def get_objects_in_event(self, event_id: str) -> List[str]:
        """
        Get all entities (objects) connected to this event via NetworkX graph.
        """
        if not self.graph_storage.has_node(event_id):
            return []
        
        # AVA graph structure: Entity --(belong_to)--> Event
        neighbors = self.graph_storage.get_node_edges(event_id)
        object_ids = []
        
        if neighbors:
            for u, v in neighbors:
                other_id = v if u == event_id else u
                # Check NetworkX node data to confirm it is an entity
                node_data = self.graph_storage.get_node(other_id)
                if node_data and node_data.get('type') == 'entity':
                    object_ids.append(other_id)
        return object_ids

    def get_events_containing_object(self, object_id: str) -> List[str]:
        """
        Get all event IDs listed in this object's 'events' metadata.
        """
        data = self.entities_vdb.get_data(object_id)
        if data and 'events' in data:
            return data['events']
        return []

    def get_relations_containing_object(self, object_id: str) -> List[Node]:
        """
        Find all relations where this object is either entity1 or entity2.
        Returns a list of Relation Nodes.
        """
        all_rels = self.relations_vdb.client_storage["data"]
        found_nodes = []
        
        for d in all_rels:
            if d.get('entity1') == object_id or d.get('entity2') == object_id:
                node = self._create_node_from_vdb_result(d, 'relation')
                found_nodes.append(node)
        return found_nodes

    def get_global_event_count_for_object(self, object_id: str) -> int:
        return len(self.get_events_containing_object(object_id))

    # --- Node Fetching ---

    def get_event_node_by_id(self, event_id: str) -> Optional[Node]:
        try:
            data = self.events_vdb.get_data(event_id)
            if not data: return None
            return self._create_node_from_vdb_result(data, 'event')
        except (IndexError, KeyError, TypeError):
            # Event doesn't exist in database
            return None

    def get_object_node_by_id(self, object_id: str) -> Optional[Node]:
        try:
            data = self.entities_vdb.get_data(object_id)
            if not data: return None
            return self._create_node_from_vdb_result(data, 'object')
        except (IndexError, KeyError, TypeError):
            # Object doesn't exist in database
            return None

    # --- Helper ---

    def _create_node_from_vdb_result(self, res: dict, node_type: str) -> Node:
        """Helper to convert NanoDB result dict to ICDCS Node object."""
        # Extract embedding safely
        vector = res.get('__vector__')
        if vector is not None and not isinstance(vector, np.ndarray):
            vector = np.array(vector)
            
        # Handle description mapping based on node type
        content = ""
        if 'description' in res:
            content = res['description']
        elif 'descriptions' in res: # entities have plural descriptions
            content = " ".join(res['descriptions']) if res['descriptions'] else ""

        # Create metadata dict excluding huge vector to save memory
        metadata = {k: v for k, v in res.items() if k != '__vector__'}

        node_id = str(res.get('id', res.get('__id__')))

        return Node(
            id=node_id,
            type=node_type,
            score=float(res.get('__metrics__', 0.0)), # Score from vector query if present
            embedding=vector,
            content=content,
            metadata=metadata
        )

class ContextGraphInterface:
    """
    Interface for cross-query context memory.
    """
    def __init__(self, db_path: str = "context_memory.db", embedding_dim: int = 768):
        # Placeholder initialization
        self.embedding_dim = embedding_dim
        pass

    def add_context(self, keyword_embedding: np.ndarray, subgraph: Subgraph, keywords: str = ""):
        pass

    def search_context(self, keyword_embedding: np.ndarray, top_k: int = 3) -> List[Subgraph]:
        return []

# ==========================================
# 4. Main Testing Function
# ==========================================

def test_knowledge_graph_interface(working_dir: str):
    """
    Test function for the migrated interface.
    """
    print("=" * 80)
    print("KNOWLEDGE GRAPH INTERFACE TEST (NanoDB Backend)")
    print("=" * 80)
    
    # Mock embedding model for testing
    class MockEmbeddingModel:
        def __init__(self, dim=768): self.embedding_dim = dim
        def get_text_features(self, texts): return np.random.rand(len(texts), self.embedding_dim)
    
    try:
        from embeddings.JinaCLIP import JinaCLIP
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
    except ImportError:
        print("⚠️ JinaCLIP not found, using MockEmbeddingModel for test.")
        embedding_model = MockEmbeddingModel()

    # Initialize KG Interface
    print(f"\n🔧 Initializing Knowledge Graph Interface at: {working_dir}")
    try:
        kg = KnowledgeGraphInterface(
            working_dir=working_dir,
            embedding_model=embedding_model
        )
        print("✅ Knowledge Graph Interface initialized")
    except Exception as e:
        print(f"❌ Error initializing KG Interface: {e}")
        return

    # Basic tests
    print("\nTest 1: Search Events")
    try:
        events = kg.search_events_by_description("test query", top_k=2)
        print(f"✅ Found {len(events)} events")
        if events:
            print(f"   First event: {events[0].id} ({events[0].content[:50]}...)")
            
            # Test structure lookup
            print("\nTest 2: Get Objects in Event")
            objs = kg.get_objects_in_event(events[0].id)
            print(f"✅ Found {len(objs)} objects in event {events[0].id}")
            
    except Exception as e:
        print(f"❌ Search failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python graph_interfaces.py <working_dir>")
        print("Example: python graph_interfaces.py datas/AVA100/citytour1/kg")
    else:
        test_knowledge_graph_interface(sys.argv[1])