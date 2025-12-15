import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional

class FAISSDB:
    """
    FAISS database for storing and searching embeddings
    """
    
    def __init__(self, db_path: str, embedding_dim: int = 512):
        """
        Initialize FAISS database
        
        Args:
            db_path: Path to save/load the FAISS index
            embedding_dim: Dimension of embeddings
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.index = None
        self.id_to_metadata = {}  # Maps FAISS ID to metadata
        self.metadata_to_id = {}  # Maps track_id to FAISS ID
        self.next_id = 0
        
        # Load existing database if it exists
        self._load_database()
    
    def _load_database(self):
        """Load existing FAISS database and metadata"""
        if os.path.exists(self.db_path):
            # Load FAISS index
            self.index = faiss.read_index(self.db_path)
            
            # Load metadata
            metadata_path = self.db_path + ".metadata"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.id_to_metadata = data['id_to_metadata']
                    self.metadata_to_id = data['metadata_to_id']
                    self.next_id = data['next_id']
        else:
            # Create new index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
    
    def _save_database(self):
        """Save FAISS index and metadata"""
        # Save FAISS index
        faiss.write_index(self.index, self.db_path)
        
        # Save metadata
        metadata_path = self.db_path + ".metadata"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'id_to_metadata': self.id_to_metadata,
                'metadata_to_id': self.metadata_to_id,
                'next_id': self.next_id
            }, f)
    
    def add_embedding(self, embedding: np.ndarray, id: str, metadata: dict) -> int:
        """
        Add embedding to the database
        
        Args:
            embedding: Embedding vector
            track_id: Track ID from object tracking
            metadata: Additional metadata (frame_number, bbox, etc.)
            
        Returns:
            FAISS ID assigned to this embedding
        """
        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.reshape(1, -1).astype(np.float32)
        
        # Add to FAISS index
        faiss_id = self.next_id
        self.index.add(embedding)
        
        # Store metadata
        self.id_to_metadata[faiss_id] = {
            'id': id,
            'embedding': embedding.flatten(),
            **metadata
        }
        self.metadata_to_id[id] = faiss_id
        
        self.next_id += 1
        
        # Save database
        self._save_database()
        
        return faiss_id
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float, dict]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (faiss_id, similarity_score, metadata) tuples
        """
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.id_to_metadata:
                results.append((idx, float(score), self.id_to_metadata[idx]))
        
        return results
    
    def get_by_id(self, id: int) -> Optional[dict]:
        """Get metadata by id"""
        if id in self.metadata_to_id:
            faiss_id = self.metadata_to_id[id]
            return self.id_to_metadata[faiss_id]
        return None
    
    def get_all_ids(self) -> List[int]:
        """Get all ids in the database"""
        return list(self.metadata_to_id.keys())
    
    def remove_by_id(self, id: int) -> bool:
        """Remove embedding by id"""
        if id in self.metadata_to_id:
            faiss_id = self.metadata_to_id[id]
            del self.id_to_metadata[faiss_id]
            del self.metadata_to_id[id]
            self._save_database()
            return True
        return False
