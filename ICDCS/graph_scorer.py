import numpy as np
import math
from typing import Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

class GraphScorer:
    def __init__(self, 
                 base_decay: float = 0.8, 
                 alpha: float = None,
                 beta: float = None):
        self.base_decay = base_decay

        # NEW: Dual-component weights
        self.alpha = alpha  # Structure importance
        self.beta = beta    # Query relevance importance
        
        # Policy: EQUAL TRUST for all operations
        self.trust_map = {
            'init': 1.0,
            
            # Structure
            'event_to_object': 0.9, 
            'object_to_event': 0.9,
            
            # Vector / Inference
            'vector_object': 0.9,   
            'vector_event': 0.9,    

            # Relation (Structure)
            'structure_object': 0.9, 
            'relation': 0.9,
            
            # Post-Processing Links (NEW)
            # This replaces the hardcoded '0.8' in finalize_subgraph
            'mutual_object_link': 0.8  
        }

    def compute_similarity(self, vec_a, vec_b) -> float:
        if vec_a is None or vec_b is None: return 0.0
        if len(vec_a.shape) == 1: vec_a = vec_a.reshape(1, -1)
        if len(vec_b.shape) == 1: vec_b = vec_b.reshape(1, -1)
        return float(cosine_similarity(vec_a, vec_b)[0][0])

    def calculate_energy_transfer(self, 
                              source_score: float, 
                              op_type: str, 
                              current_iteration: int = 1,
                              hub_size: int = 1,
                              global_uniqueness: int = 1,
                              node_embedding: Optional[np.ndarray] = None,
                              parent_embedding: Optional[np.ndarray] = None,
                              query_embedding: Optional[np.ndarray] = None) -> float:
        """
        Calculates how much 'Heat' flows from Parent -> Child based on heuristics.
        
        Args:
            current_iteration: Current iteration number (for time-based damping)
        """
        # 1. Base Energy with Uniform Trust
        base_trust = self.trust_map.get(op_type, 0.9)
        
        energy = source_score * self.base_decay * base_trust
        
        # 2. Structural Penalties/Bonuses
        
        # A. Hub Penalty (Logarithmic) - For Event->Object
        if (op_type == 'event_object' or op_type == 'structure_object') and hub_size > 1:
            energy /= math.log(hub_size + 1)
            
        # B. Uniqueness Bonus (Inverse Log) - For Object->Event
        if op_type == 'object_event':
            safe_uniq = max(1, global_uniqueness)
            # Gentle boost for rare items, capped at 1.5x
            uniqueness_factor = 1.0 / (math.log(safe_uniq + 1) * 0.5)
            energy *= min(1.5, uniqueness_factor) 

        # ========================================================================
        # 3. M4: DUAL-COMPONENT SCORING (REPLACES OLD VECTOR LOGIC)
        # ========================================================================
        # Apply to ALL operations (not just vector) if embeddings are available
        if self.alpha is not None and self.beta is not None and node_embedding is not None:
            # Component 1: Structural similarity (parent-child relationship)
            if parent_embedding is not None:
                structural_sim = self.compute_similarity(node_embedding, parent_embedding)
            else:
                structural_sim = 1.0  # No parent = direct seed connection, full trust
            
            # Component 2: Query relevance (how well node matches query)
            if query_embedding is not None:
                query_sim = self.compute_similarity(node_embedding, query_embedding)
            else:
                query_sim = 0.5  # Default if no query embedding available
            
            # Combine with learned weights (alpha + beta should sum to 1.0)
            # alpha = importance of structural relationship
            # beta = importance of query relevance
            relevance_score = self.alpha * structural_sim + self.beta * query_sim
            
            # Apply relevance to energy
            energy *= relevance_score
            
            # Optional debug print (remove after testing)
            # print(f"  [M4] struct={structural_sim:.3f}, query={query_sim:.3f}, "
            #       f"relevance={relevance_score:.3f}, op={op_type}")

        # 4. Time-Based Damping (Score Convergence Prevention)
        # Formula: energy / ln(iteration + 2)
        # Effect: Iter 1 → ÷1.1 (10% damping), Iter 10 → ÷2.5 (60% damping)
        # This forces score convergence and prevents saturation
        damping_divisor = math.log(current_iteration + 2)
        energy = energy / damping_divisor

        return energy

    def accumulate_score(self, current_score: float, incoming_energy: float) -> float:
        """
        Probability Disjunction: S_new = S_old + (Energy * (1 - S_old)).
        Ensures score saturates at 1.0 instead of exploding.
        """
        return current_score + (incoming_energy * (1.0 - current_score))