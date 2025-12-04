import numpy as np
import math
from typing import Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

class GraphScorer:
    def __init__(self, 
                 base_decay: float = 0.8, 
                 target_event_obj_ratio: float = 0.33):
        self.base_decay = base_decay
        self.target_ratio = target_event_obj_ratio
        
        # Policy: Trust factors for different operations
        self.trust_map = {
            'init': 1.0,
            'event_to_object': 0.85,    # Structure (Hub Penalty applies)
            'object_to_event': 0.75,    # Structure (Uniqueness applies)
            'vector_object': 0.5,       # Inference (Risky - Compass Rule)
            'vector_event': 0.6,        # Inference (Bridge)
            'relation': 0.95,           # Fact (Explicit KG relation)
            
            # NEW CONTEXT-BASED RELATIONS: High trust as they come from previous reasoned subgraphs
            'context_relation': 0.90,       # Object->Object from Context Graph
            'context_event_to_event': 0.80  # Event->Event from Context Graph
        }

    def compute_similarity(self, vec_a, vec_b) -> float:
        if vec_a is None or vec_b is None: return 0.0
        if len(vec_a.shape) == 1: vec_a = vec_a.reshape(1, -1)
        if len(vec_b.shape) == 1: vec_b = vec_b.reshape(1, -1)
        return float(cosine_similarity(vec_a, vec_b)[0][0])

    def get_balance_multiplier(self, node_type: str, current_counts: Dict[str, int]) -> float:
        """
        The Thermostat: Maintains health of graph structure.
        """
        n_evt = max(1, current_counts.get('event', 0))
        n_obj = max(1, current_counts.get('object', 0))
        current_ratio = n_evt / n_obj
        
        if node_type == 'event':
            # Boost events if they are scarce (Ratio < 0.33)
            return 1.2 if current_ratio < self.target_ratio else 0.9
        elif node_type == 'object':
            # Suppress objects if they are abundant (Ratio > 0.33)
            return 0.8 if current_ratio < self.target_ratio else 1.0
        return 1.0

    def calculate_energy_transfer(self, 
                                  source_score: float, 
                                  op_type: str, 
                                  hub_size: int = 1,
                                  global_uniqueness: int = 1,
                                  node_embedding: Optional[np.ndarray] = None,
                                  parent_embedding: Optional[np.ndarray] = None,
                                  query_embedding: Optional[np.ndarray] = None) -> float:
        """
        Calculates how much 'Heat' flows from Parent -> Child based on heuristics.
        """
        # 1. Base Energy
        trust = self.trust_map.get(op_type, 0.5)
        energy = source_score * self.base_decay * trust

        # 2. Structural Penalties/Bonuses
        
        # A. Hub Penalty (Logarithmic) - For Event->Object
        if op_type == 'event_to_object' and hub_size > 1:
            energy /= math.log(hub_size + 1)
            
        # B. Uniqueness Bonus (Inverse Log) - For Object->Event
        if op_type == 'object_to_event':
            safe_uniq = max(1, global_uniqueness)
            # Gentle boost for rare items, capped at 1.5x
            uniqueness_factor = 1.0 / (math.log(safe_uniq + 1) * 0.5)
            energy *= min(1.5, uniqueness_factor) 

        # 3. Vector Logic (Compass Rule + Plot Twist Override)
        # Only applied for Vector Operations
        if 'vector' in op_type and query_embedding is not None and node_embedding is not None:
            global_sim = self.compute_similarity(node_embedding, query_embedding)
            
            if parent_embedding is not None:
                local_sim = self.compute_similarity(node_embedding, parent_embedding)
                
                # Plot Twist Logic: Trust Strong Local links (>0.85) over Global Context
                if local_sim > 0.85:
                    relevance = (0.8 * local_sim) + (0.2 * global_sim)
                else:
                    relevance = (0.2 * local_sim) + (0.8 * global_sim)
            else:
                relevance = global_sim
                
            energy *= relevance

        return energy

    def accumulate_score(self, current_score: float, incoming_energy: float) -> float:
        """
        Probability Disjunction: S_new = S_old + (Energy * (1 - S_old)).
        Ensures score saturates at 1.0 instead of exploding.
        """
        return current_score + (incoming_energy * (1.0 - current_score))