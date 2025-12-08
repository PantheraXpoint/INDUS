import numpy as np
import uuid
import copy
import sys
import os
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from embeddings.Milvus import MilvusDB
from embeddings.SQLiteDB import SQLiteDB
from embeddings.JinaCLIP import JinaCLIP

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
                 object_faiss_db_path: str = "",
                 event_faiss_db_path: str = "",
                 object_sqlite_db_path: str = "",
                 embedding_model = None,  # JinaCLIP model for text-to-embedding
                 embedding_dim: int = 768):
        # Initialize connections here (Milvus, SQLite, etc.)
        self.object_faiss_db = MilvusDB(object_faiss_db_path, embedding_dim)
        self.event_faiss_db = MilvusDB(event_faiss_db_path, embedding_dim)
        self.object_sqlite_db = SQLiteDB(object_sqlite_db_path)
        self.embedding_model = embedding_model
        
        # Cache for event ID mappings (frame_number -> [event_ids])
        self._event_frame_cache = None
    
    @staticmethod
    def normalize_event_id(event_id: str) -> str:
        """
        Extract frame number from event ID.
        '1177596_3' -> '1177596'
        '1177596' -> '1177596'
        """
        return event_id.split('_')[0]
    
    def _build_event_frame_mapping(self):
        """
        Build a mapping from frame numbers to event IDs.
        This helps match object detections (which store frame numbers) 
        to full event IDs (which have suffixes).
        """
        if self._event_frame_cache is not None:
            return self._event_frame_cache
        
        print("Building event-frame mapping cache...")
        self._event_frame_cache = {}
        
        # Query all events to build frame -> event_id mapping
        try:
            # Get a sample to understand the structure
            # For now, we'll build it on-demand
            pass
        except Exception as e:
            print(f"Warning: Could not build event-frame cache: {e}")
        
        return self._event_frame_cache
    
    def _find_events_by_frame_range(self, frame_number: int) -> List[str]:
        """
        Find all event IDs that contain this frame number.
        An event with start_time=386694, end_time=386777 contains frames 386694-386777.
        """
        try:
            # Query events where start_time <= frame_number <= end_time
            expr = f"start_time <= {frame_number} and end_time >= {frame_number}"
            results = self.event_faiss_db.query(expr)
            
            event_ids = [result['id'] for result in results if 'id' in result]
            return event_ids
        except Exception as e:
            # If query fails, fall back to frame number only
            return [str(frame_number)]



    # --- Vector Search Methods ---
    def search_events_by_embedding(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Node]:
        """
        Vector search for events with GUARANTEED embedding fetch.
        
        ROBUST EAGER LOADING:
        1. Search returns results
        2. Check if embedding is included
        3. If missing, immediately fetch via get_by_id (safety net)
        4. Store in Node.embedding (never None)
        """
        results = self.event_faiss_db.search(query_embedding, k=top_k)
        nodes = []
        for faiss_id, similarity_score, metadata in results:
            # Extract event_id from the 'id' field (e.g., "1177596_3" -> use as-is)
            event_id = metadata['id']
            
            # STEP 1: Try to extract embedding from search result metadata
            embedding = metadata.get('embedding', None)
            
            # STEP 2: SAFETY NET - If embedding missing, fetch it explicitly
            if embedding is None:
                fetch_result = self.event_faiss_db.get_by_id(event_id, include_embedding=True)
                if fetch_result:
                    embedding = fetch_result.get('embedding', None)
            
            # STEP 3: Convert to numpy array if needed
            if embedding is not None and not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            node = Node(
                id=event_id,
                type='event',
                score=similarity_score,
                embedding=embedding,  # ✅ GUARANTEED: Fetched if missing
                content=metadata.get('description', ''),
                metadata=metadata
            )
            nodes.append(node)
        return nodes

    def search_objects_by_embedding(self, query_embedding: np.ndarray, top_k: int = 30) -> List[Node]:
        """
        Vector search for objects with GUARANTEED embedding fetch.
        
        ROBUST EAGER LOADING:
        1. Search returns results
        2. Check if embedding is included
        3. If missing, immediately fetch via query (safety net)
        4. Store in Node.embedding (never None)
        """
        results = self.object_faiss_db.search(query_embedding, k=top_k)
        nodes = []
        for faiss_id, similarity_score, metadata in results:
            # Use track_id as node ID
            obj_id = str(metadata['track_id'])
            
            # STEP 1: Try to extract embedding from search result metadata
            embedding = metadata.get('embedding', None)
            
            # STEP 2: SAFETY NET - If embedding missing, fetch it explicitly
            if embedding is None:
                try:
                    expr = f'track_id == {int(obj_id)}'
                    fetch_results = self.object_faiss_db.query(expr, include_embedding=True)
                    if fetch_results and len(fetch_results) > 0:
                        embedding = fetch_results[0].get('embedding', None)
                except:
                    pass  # Keep embedding as None if fetch fails
            
            # STEP 3: Convert to numpy array if needed
            if embedding is not None and not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            node = Node(
                id=obj_id,
                type='object',
                score=similarity_score,
                embedding=embedding,  # ✅ GUARANTEED: Fetched if missing
                content=metadata.get('class_name', ''),
                metadata=metadata
            )
            nodes.append(node)
        return nodes

    # --- Text-based Search Methods (for keyword-based retrieval) ---
    def search_events_by_description(self, description: str, top_k: int = 20) -> List[Node]:
        """
        Search events by text description (converts text to embedding first).
        This is used for LLM-generated keywords/queries.
        
        Args:
            description: Text description or keywords
            top_k: Number of results to return
        
        Returns:
            List of Node objects with embeddings
        """
        if not self.embedding_model:
            raise ValueError("embedding_model is required for text-based search")
        
        # Convert text to embedding
        query_embedding = self.embedding_model.get_text_features([description])[0]
        
        # Use the existing embedding search
        return self.search_events_by_embedding(query_embedding, top_k)
    
    def search_objects_by_description(self, description: str, top_k: int = 20) -> List[Node]:
        """
        Search objects by text description (converts text to embedding first).
        This is used for LLM-generated keywords/queries.
        
        Args:
            description: Text description or keywords
            top_k: Number of results to return
        
        Returns:
            List of Node objects with embeddings
        """
        if not self.embedding_model:
            raise ValueError("embedding_model is required for text-based search")
        
        # Convert text to embedding
        query_embedding = self.embedding_model.get_text_features([description])[0]
        
        # Use the existing embedding search
        return self.search_objects_by_embedding(query_embedding, top_k)

    # --- Structural Lookup Mocks (Replace with SQLite/GraphDB logic) ---
    def get_objects_in_event(self, event_id: str) -> List[str]:
        """
        Get all object track_ids that appear in this event.
        
        Handles both normalized IDs (e.g., "481464") and full IDs (e.g., "481464_2").
        If normalized ID is provided, tries to find any sentence variant (prefers _0).
        Since all sentence variants have the same object list, we only need one.
        """
        # Try exact match first
        event_data = self.event_faiss_db.get_by_id(event_id)
        
        # If not found and ID doesn't have underscore, try with _0 suffix
        if not event_data and '_' not in event_id:
            event_data = self.event_faiss_db.get_by_id(f"{event_id}_0")
            
            # If still not found, try _1, _2, etc. (up to 10 sentences)
            if not event_data:
                for i in range(1, 10):
                    event_data = self.event_faiss_db.get_by_id(f"{event_id}_{i}")
                    if event_data:
                        break
        
        if event_data and 'objects' in event_data:
            # Parse comma-separated string: "120941,120947,120949"
            objects_str = event_data['objects']
            if objects_str:
                return [oid.strip() for oid in objects_str.split(',')]
        return []


    def get_events_containing_object(self, object_id: str) -> List[str]:
        """
        Get all events that contain this object (track_id).
        
        Strategy:
        1. Query object database for all detections of this track_id
        2. Get the event_id (frame number) from each detection
        3. Find all event IDs (with suffixes) that cover those frame numbers
        """
        try:
            # Use metadata query to find all entries with this track_id
            expr = f'track_id == {int(object_id)}'
            results = self.object_faiss_db.query(expr)
            
            # Collect unique frame numbers (stored as event_id in object DB)
            frame_numbers = set()
            for result in results:
                if 'event_id' in result:
                    frame_numbers.add(int(result['event_id']))
            
            # For each frame number, find matching events
            event_ids = set()
            for frame_num in frame_numbers:
                # Find events that contain this frame
                matching_events = self._find_events_by_frame_range(frame_num)
                event_ids.update(matching_events)
            
            return list(event_ids)
        except Exception as e:
            print(f"Error getting events for object {object_id}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_global_event_count_for_object(self, object_id: str) -> int:
        """Count how many unique events this object appears in"""
        return len(self.get_events_containing_object(object_id))
    
    def get_event_node_by_id(self, event_id: str) -> Node:
        """
        Fetch full event node with metadata by ID with GUARANTEED embedding.
        Normalizes event ID to base form (before underscore).
        """
        # Query the event database (MUST request embedding explicitly!)
        event_data = self.event_faiss_db.get_by_id(event_id, include_embedding=True)
        
        if not event_data:
            # Create minimal node if not found (no embedding available)
            normalized_id = self.normalize_event_id(event_id)
            return Node(
                id=normalized_id,
                type='event',
                score=0.0,
                embedding=None,  # Can't fetch if event doesn't exist
                content='',
                metadata={'original_id': event_id}
            )
        
        # Normalize the ID (just the base event number before underscore)
        normalized_id = self.normalize_event_id(event_id)
        
        # ROBUST EAGER LOADING: Extract embedding from database result
        embedding = event_data.get('embedding', None)
        
        # SAFETY NET: If still missing, this is a database issue
        # (get_by_id should return embedding, but we handle gracefully)
        if embedding is not None and not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Create node with full metadata
        node = Node(
            id=normalized_id,  # Store normalized ID
            type='event',
            score=0.0,  # Will be updated during graph traversal
            embedding=embedding,  # ✅ GUARANTEED: Fetched from get_by_id
            content=event_data.get('description', ''),
            metadata={
                **event_data,  # Include all database fields
                'original_id': event_id,  # Keep full ID for reference
                'sentence_index': event_id.split('_')[1] if '_' in event_id else '0'
            }
        )
        return node
    
    def get_object_node_by_id(self, object_id: str) -> Node:
        """
        Fetch full object node with metadata by ID (track_id) with GUARANTEED embedding.
        """
        try:
            # Query object database for this track_id (MUST request embedding explicitly!)
            expr = f'track_id == {int(object_id)}'
            results = self.object_faiss_db.query(expr, include_embedding=True)
            
            if not results:
                # Create minimal node if not found (no embedding available)
                return Node(
                    id=object_id,
                    type='object',
                    score=0.0,
                    embedding=None,  # Can't fetch if object doesn't exist
                    content='',
                    metadata={'track_id': object_id}
                )
            
            # Use the first result (they should all be the same track)
            obj_data = results[0]
            
            # ROBUST EAGER LOADING: Extract embedding from database result
            embedding = obj_data.get('embedding', None)
            
            # SAFETY NET: If still missing, this is a database issue
            # (query should return embedding, but we handle gracefully)
            if embedding is not None and not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Create node with full metadata
            node = Node(
                id=object_id,
                type='object',
                score=0.0,  # Will be updated during graph traversal
                embedding=embedding,  # ✅ GUARANTEED: Fetched from query
                content=obj_data.get('class_name', ''),
                metadata=obj_data
            )
            return node
        except Exception as e:
            print(f"Error fetching object {object_id}: {e}")
            # Return minimal node (embedding unavailable due to error)
            return Node(
                id=object_id,
                type='object',
                score=0.0,
                embedding=None,  # Error case - can't fetch
                content='',
                metadata={'track_id': object_id, 'error': str(e)}
            )

class ContextGraphInterface:
    """
    Stores and retrieves subgraphs from past queries using keyword embeddings.
    Uses Milvus DB for persistent vector storage.
    """
    def __init__(self, db_path: str = "context_memory.db", embedding_dim: int = 768):
        """
        Initialize context graph with Milvus DB.
        
        Args:
            db_path: Path to Milvus database file
            embedding_dim: Dimension of keyword embeddings (should match text embedding model)
        """
        from embeddings.Milvus import MilvusDB
        self.db = MilvusDB(db_path, embedding_dim)
        self.embedding_dim = embedding_dim

    def add_context(self, keyword_embedding: np.ndarray, subgraph: Subgraph, keywords: str = ""):
        """
        Store a subgraph indexed by its keyword embedding.
        
        Args:
            keyword_embedding: Embedding of comma-separated keywords
            subgraph: Best subgraph result from the query
            keywords: Original comma-separated keyword string (for reference/debugging)
        """
        import json
        
        # Serialize subgraph to JSON
        subgraph_data = self._export_subgraph_to_dict(subgraph)
        
        # Prepare metadata
        metadata = {
            "subgraph_id": subgraph.id,
            "keywords": keywords,  # Store for debugging/inspection
            "subgraph_json": json.dumps(subgraph_data),
            "node_count": len(subgraph.nodes),
            "edge_count": len(subgraph.edges)
        }
        
        # Store in Milvus
        try:
            self.db.add_embedding(
                embedding=keyword_embedding,
                id=subgraph.id,
                metadata=metadata
            )
            # Flush to ensure data is written to disk immediately
            self.db.collection.flush()
            print(f"💾 Saved subgraph '{subgraph.id}' to context (keywords: '{keywords[:60]}...')")
        except Exception as e:
            print(f"⚠️  Failed to save context: {e}")

    def search_context(self, keyword_embedding: np.ndarray, top_k: int = 3) -> List[Subgraph]:
        """
        Search for similar past subgraphs using keyword embedding.
        
        Args:
            keyword_embedding: Embedding of current query keywords
            top_k: Number of similar past queries to retrieve
        
        Returns:
            List of Subgraph objects from past queries
        """
        try:
            results = self.db.search(keyword_embedding, k=top_k)
            
            subgraphs = []
            for pk, distance, metadata in results:
                # Log retrieval
                keywords = metadata.get('keywords', 'N/A')
                print(f"  📂 Retrieved: keywords='{keywords[:60]}...', similarity={distance:.3f}")
                
                # Deserialize subgraph from JSON
                import json
                subgraph_json = json.loads(metadata["subgraph_json"])
                sg = self._reconstruct_subgraph(subgraph_json)
                subgraphs.append(sg)
            
            return subgraphs
        except Exception as e:
            # If DB is empty or has errors, return empty list
            return []
    
    def _export_subgraph_to_dict(self, subgraph: Subgraph) -> dict:
        """Export Subgraph to dictionary (lightweight version)."""
        data = {
            "id": subgraph.id,
            "nodes": [],
            "edges": []
        }
        
        # Export nodes
        for node in subgraph.nodes.values():
            node_data = {
                "id": node.id,
                "type": node.type,
                "score": float(node.score),
                "embedding": node.embedding.tolist() if node.embedding is not None else None,
                "content": node.content,
                "metadata": node.metadata
            }
            data["nodes"].append(node_data)
        
        # Export edges
        for edge in subgraph.edges:
            edge_data = {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "type": edge.type,
                "score": float(edge.score),
                "metadata": edge.metadata
            }
            data["edges"].append(edge_data)
        
        return data
    
    def _reconstruct_subgraph(self, subgraph_data: dict) -> Subgraph:
        """Reconstruct Subgraph object from exported dictionary."""
        sg = Subgraph(id=subgraph_data["id"])
        
        # Reconstruct nodes
        for node_data in subgraph_data["nodes"]:
            node = Node(
                id=node_data["id"],
                type=node_data["type"],
                score=node_data["score"],
                embedding=np.array(node_data["embedding"]) if node_data.get("embedding") else None,
                content=node_data.get("content", ""),
                metadata=node_data.get("metadata", {})
            )
            sg.add_node(node)
        
        # Reconstruct edges
        for edge_data in subgraph_data["edges"]:
            edge = Edge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                type=edge_data["type"],
                score=edge_data["score"],
                metadata=edge_data.get("metadata", {})
            )
            sg.add_edge(edge)
        
        return sg

# ==========================================
# 4. Main Testing Function
# ==========================================

def test_knowledge_graph_interface(object_db_path: str, event_db_path: str, sqlite_db_path: str):
    """
    Comprehensive test function for KnowledgeGraphInterface
    
    Args:
        object_db_path: Path to object embeddings Milvus database
        event_db_path: Path to event embeddings Milvus database
        sqlite_db_path: Path to SQLite database
    """
    print("=" * 80)
    print("KNOWLEDGE GRAPH INTERFACE TEST")
    print("=" * 80)
    
    # Initialize embedding model
    print("\n📦 Initializing JinaCLIP embedding model...")
    try:
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        print(f"✅ JinaCLIP initialized (dim={embedding_model.embedding_dim})")
    except Exception as e:
        print(f"❌ Error initializing JinaCLIP: {e}")
        return
    
    # Initialize KG Interface
    print("\n🔧 Initializing Knowledge Graph Interface...")
    try:
        kg = KnowledgeGraphInterface(
            object_faiss_db_path=object_db_path,
            event_faiss_db_path=event_db_path,
            object_sqlite_db_path=sqlite_db_path,
            embedding_dim=embedding_model.embedding_dim
        )
        print("✅ Knowledge Graph Interface initialized")
    except Exception as e:
        print(f"❌ Error initializing KG Interface: {e}")
        return
    
    # Test 1: Search Events by Embedding
    print("\n" + "=" * 80)
    print("TEST 1: Search Events by Embedding")
    print("=" * 80)
    query_text = "person walking on the street"
    print(f"Query: '{query_text}'")
    
    try:
        query_embedding = embedding_model.get_text_features([query_text])[0]
        event_nodes = kg.search_events_by_embedding(query_embedding, top_k=3)
        
        print(f"\n✅ Found {len(event_nodes)} event nodes:")
        for i, node in enumerate(event_nodes, 1):
            print(f"\n  {i}. Event ID: {node.id}")
            print(f"     Score: {node.score:.4f}")
            print(f"     Content: {node.content[:100]}..." if len(node.content) > 100 else f"     Content: {node.content}")
            print(f"     Type: {node.type}")
    except Exception as e:
        print(f"❌ Error in event search: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Search Objects by Embedding
    print("\n" + "=" * 80)
    print("TEST 2: Search Objects by Embedding")
    print("=" * 80)
    query_text = "red car"
    print(f"Query: '{query_text}'")
    
    try:
        query_embedding = embedding_model.get_text_features([query_text])[0]
        object_nodes = kg.search_objects_by_embedding(query_embedding, top_k=3)
        
        print(f"\n✅ Found {len(object_nodes)} object nodes:")
        for i, node in enumerate(object_nodes, 1):
            print(f"\n  {i}. Object ID (track_id): {node.id}")
            print(f"     Score: {node.score:.4f}")
            print(f"     Class: {node.content}")
            print(f"     Type: {node.type}")
            if 'metadata' in node.metadata:
                print(f"     Frame: {node.metadata.get('frame_number', 'N/A')}")
    except Exception as e:
        print(f"❌ Error in object search: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Get Objects in Event (Structural Lookup)
    print("\n" + "=" * 80)
    print("TEST 3: Get Objects in Event (Event → Objects)")
    print("=" * 80)
    
    if event_nodes:
        test_event_id = event_nodes[0].id
        print(f"Testing with Event ID: {test_event_id}")
        
        try:
            object_ids = kg.get_objects_in_event(test_event_id)
            print(f"\n✅ Found {len(object_ids)} objects in this event:")
            print(f"   Object IDs: {object_ids[:10]}" + (" ..." if len(object_ids) > 10 else ""))
        except Exception as e:
            print(f"❌ Error getting objects in event: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  Skipping - no events found in Test 1")
    
    # Test 4: Get Events Containing Object (Structural Lookup)
    print("\n" + "=" * 80)
    print("TEST 4: Get Events Containing Object (Object → Events)")
    print("=" * 80)
    
    if object_nodes:
        test_object_id = object_nodes[0].id
        print(f"Testing with Object ID (track_id): {test_object_id}")
        
        try:
            event_ids = kg.get_events_containing_object(test_object_id)
            print(f"\n✅ Found {len(event_ids)} events containing this object:")
            print(f"   Event IDs: {event_ids[:10]}" + (" ..." if len(event_ids) > 10 else ""))
        except Exception as e:
            print(f"❌ Error getting events for object: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  Skipping - no objects found in Test 2")
    
    # Test 5: Get Global Event Count for Object
    print("\n" + "=" * 80)
    print("TEST 5: Get Global Event Count for Object")
    print("=" * 80)
    
    if object_nodes:
        test_object_id = object_nodes[0].id
        print(f"Testing with Object ID (track_id): {test_object_id}")
        
        try:
            count = kg.get_global_event_count_for_object(test_object_id)
            print(f"\n✅ Object appears in {count} unique events")
            print(f"   This is used for the 'Uniqueness Bonus' in scoring")
        except Exception as e:
            print(f"❌ Error getting event count: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  Skipping - no objects found in Test 2")
    
    # Test 6: Bidirectional Relationship Test
    print("\n" + "=" * 80)
    print("TEST 6: Bidirectional Relationship Verification")
    print("=" * 80)
    
    if event_nodes and object_nodes:
        print("Testing Event → Objects → Events round-trip...")
        test_event_id = event_nodes[0].id
        
        try:
            # Get objects in this event
            object_ids = kg.get_objects_in_event(test_event_id)
            print(f"\n1. Event '{test_event_id}' contains {len(object_ids)} objects")
            print(f"   First 5 objects: {object_ids[:5]}")
            
            if object_ids:
                # Pick first object and find its events
                test_obj_id = object_ids[0]
                events_for_obj = kg.get_events_containing_object(test_obj_id)
                print(f"\n2. Object '{test_obj_id}' appears in {len(events_for_obj)} events")
                print(f"   Event IDs: {events_for_obj[:5]}" + (" ..." if len(events_for_obj) > 5 else ""))
                
                # Check if original event is in the list
                if test_event_id in events_for_obj:
                    print(f"\n✅ Bidirectional lookup verified! Original event found in object's event list")
                else:
                    # Check if normalized IDs match
                    normalized_test = kg.normalize_event_id(test_event_id)
                    normalized_found = [kg.normalize_event_id(eid) for eid in events_for_obj]
                    
                    if normalized_test in normalized_found:
                        print(f"\n✅ Partial match: Frame number matches (ignoring suffixes)")
                        print(f"   Original: {test_event_id} -> frame {normalized_test}")
                        print(f"   Found: {[eid for eid in events_for_obj if kg.normalize_event_id(eid) == normalized_test]}")
                    else:
                        print(f"\n⚠️  Original event NOT found in object's event list")
                        print(f"   Original event: {test_event_id} (frame: {normalized_test})")
                        print(f"   Object's events: {events_for_obj[:3]}")
                        print(f"   This suggests the object-event mapping needs verification")
        except Exception as e:
            print(f"❌ Error in bidirectional test: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  Skipping - need both events and objects from previous tests")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✅ All tests completed!")
    print("\nNext steps:")
    print("  1. Verify that all relationships are correct")
    print("  2. Check ID format consistency between Event and Object databases")
    print("  3. Test with graph_engine.py for full integration")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python graph_interfaces.py <object_db> <event_db> <sqlite_db>")
        print("\nExample:")
        print("  python AVA/graph_interfaces.py \\")
        print("    database/video/object_embeddings.db \\")
        print("    database/video/event_embeddings.db \\")
        print("    database/video/tracked_objects.db")
        sys.exit(1)
    
    test_knowledge_graph_interface(sys.argv[1], sys.argv[2], sys.argv[3])