import numpy as np
import math
import json
import hashlib
import sys
import os
from typing import List, Dict, Set, Optional, Tuple, Deque, Any
from collections import deque, defaultdict
import networkx as nx
from graph_interfaces import Node, Edge, Subgraph, KnowledgeGraphInterface, ContextGraphInterface, get_global_system_counts
from graph_scorer import GraphScorer
from AVA.prompt import PROMPTS

# Import evaluation functions from time_ref.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from time_ref import time_to_seconds, percentage_overlap, overlap_reference_helper
except ImportError:
    # Fallback: define locally if import fails
    def time_to_seconds(time_str: str):
        if len(time_str.split(":")) == 2:
            minutes, seconds = time_str.split(":")
            return int(minutes) * 60 + int(seconds)
        elif len(time_str.split(":")) == 3:
            hours, minutes, seconds = time_str.split(":")
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        else:
            raise ValueError(f"Invalid time string: {time_str}")
    
    def percentage_overlap(time_list: List[Tuple[int, int]], time_ref: Tuple[int, int]) -> float:
        ref_start, ref_end = time_ref
        if ref_end < ref_start:
            return 0.0
        if ref_start == ref_end:
            point = ref_start
            for start, end in time_list:
                if end <= start:
                    continue
                if start <= point < end:
                    return 1.0
            return 0.0
        ref_length = ref_end - ref_start
        total_covered = 0
        for start, end in time_list:
            if end <= start:
                continue
            s = max(start, ref_start)
            e = min(end, ref_end)
            if e > s:
                total_covered += (e - s)
        return total_covered / ref_length if ref_length > 0 else 0.0
    
    def overlap_reference_helper(time_ref: str, time_list: List[Tuple[int, int]]):
        if time_ref == "N/A" or time_ref == "" or time_ref == "None" or time_ref == "None-None":
            return None
        if "-" in time_ref:
            start_time, end_time = time_ref.split("-")
            if start_time in ["", "None"]:
                start_time = end_time
            if end_time in ["", "None"]:
                end_time = start_time
            start_time_seconds = time_to_seconds(start_time)
            end_time_seconds = time_to_seconds(end_time)
        elif "," in time_ref:
            points_overlap = []
            points = time_ref.split(",")
            for point in points:
                point = point.strip()
                points_overlap.append(percentage_overlap(time_list, (time_to_seconds(point), time_to_seconds(point))))
            return sum(points_overlap) / len(points_overlap)
        else:
            start_time_seconds = time_to_seconds(time_ref)
            end_time_seconds = start_time_seconds
        return percentage_overlap(time_list, (start_time_seconds, end_time_seconds))

class GraphEngine:
    def __init__(self, 
                 knowledge_graph: KnowledgeGraphInterface, 
                 context_graph: Optional[ContextGraphInterface] = None, 
                 scorer: Optional[GraphScorer] = None,
                 llm = None,
                 constrained_propagation: bool = False,
                 top_k_events: int = 5,
                 top_k_objects: int = 5,
                 adaptive_threshold: bool = False,
                 threshold_percentile: int = 80,
                 prize_based_seeds: bool = False,
                 top_k_protected: int = 5):  
        self.kg = knowledge_graph
        self.context_graph = context_graph
        self.scorer = scorer
        self.llm = llm

        # M4: Initialize query embedding storage
        self.query_embedding = None

        # M1: Store constrained propagation settings
        self.constrained_propagation = constrained_propagation
        self.top_k_events = top_k_events
        self.top_k_objects = top_k_objects

        # M3: Store adaptive threshold settings
        self.adaptive_threshold = adaptive_threshold
        self.threshold_percentile = threshold_percentile

        # M7: Store prize-based seeds settings
        self.prize_based_seeds = prize_based_seeds
        self.top_k_protected = top_k_protected
        
        # === PER-QUERY STATE (reset each search) ===
        self._reset_query_state()
        
        # === PER-VIDEO CACHES (persist across queries) ===
        # KG Query Cache
        self._kg_query_cache = {
            'objects_in_event': {},      # event_id → List[str] (object IDs)
            'events_with_object': {},    # object_id → List[str] (event IDs)
            'event_count_per_object': {}, # object_id → int
            'relations_for_object': {}   # object_id -> List[Node] (Relation Nodes) - UPDATED
        }
        
        # Vector Search Cache (hybrid: node-based + embedding-based)
        self._vector_search_cache = {
            'event_to_event_by_node': {},     # event_id → List[Node]
            'object_to_object_by_node': {},   # object_id → List[Node]
            'event_by_embedding': {},         # hash(embedding) → List[Node]
            'object_by_embedding': {},        # hash(embedding) → List[Node]
        }
        
        # === CACHE STATISTICS ===
        self._cache_stats = {
            'kg_queries': {
                'objects_in_event': {'hits': 0, 'misses': 0},
                'events_with_object': {'hits': 0, 'misses': 0},
                'event_count_per_object': {'hits': 0, 'misses': 0},
                'relations_for_object': {'hits': 0, 'misses': 0} # UPDATED
            },
            'vector_searches': {
                'event_to_event': {'hits': 0, 'misses': 0},
                'object_to_object': {'hits': 0, 'misses': 0},
                'event_by_embedding': {'hits': 0, 'misses': 0},
                'object_by_embedding': {'hits': 0, 'misses': 0}
            }
        }

        # Pruning Configuration
        self.pruning_config = {
            # STRICT BUDGET: 20 NODES
            'max_total_nodes': 20,
            
            # Dynamic Ratio: We no longer hard-cap events/objects.
            # They compete based on score.
            'max_event_nodes': 20, 
            'max_object_nodes': 20,
            
            # EDGE SAFETY: High limit to prevent "Prune Nodes -> Keep Edges -> Trigger Again" loop
            'max_edges': 400,  
            
            # Saturation
            'score_saturation_threshold': 0.95,
            'saturation_count_trigger': 10, # Scaled down for 20 nodes
            
            # Steiner tree settings
            'use_articulation_points': True,
            'use_shortest_paths': True,
            'shortest_path_node_limit': 50,
            'max_path_length': 3,
        }

    def _reset_query_state(self):
        """Reset per-query state (called at start of each search)."""
        self.subgraphs = []
        self.retrieved_context_subgraphs = []
        self.current_context_key_embedding = None
        self.current_keywords = ""
        self.current_iteration = 0
        self.iteration_metrics = []

        # NEW: Track per-operation success
        self._op_stats = defaultdict(lambda: {'attempted': 0, 'added': 0})

    
    def _get_cache_statistics(self) -> Dict:
        """Generate detailed cache statistics."""
        stats = {
            'kg_queries': {},
            'vector_searches': {},
            'summary': {}
        }
        
        # KG Query Stats
        for op_type, counts in self._cache_stats['kg_queries'].items():
            total = counts['hits'] + counts['misses']
            if total > 0:
                hit_rate = counts['hits'] / total * 100
                stats['kg_queries'][op_type] = {
                    'hits': counts['hits'],
                    'misses': counts['misses'],
                    'total': total,
                    'hit_rate': hit_rate
                }
        
        # Vector Search Stats
        for op_type, counts in self._cache_stats['vector_searches'].items():
            total = counts['hits'] + counts['misses']
            if total > 0:
                hit_rate = counts['hits'] / total * 100
                stats['vector_searches'][op_type] = {
                    'hits': counts['hits'],
                    'misses': counts['misses'],
                    'total': total,
                    'hit_rate': hit_rate
                }
        
        # Overall Summary
        kg_total_hits = sum(c['hits'] for c in self._cache_stats['kg_queries'].values())
        kg_total_misses = sum(c['misses'] for c in self._cache_stats['kg_queries'].values())
        kg_total = kg_total_hits + kg_total_misses
        
        vec_total_hits = sum(c['hits'] for c in self._cache_stats['vector_searches'].values())
        vec_total_misses = sum(c['misses'] for c in self._cache_stats['vector_searches'].values())
        vec_total = vec_total_hits + vec_total_misses
        
        stats['summary'] = {
            'kg_queries': {
                'total_hits': kg_total_hits,
                'total_misses': kg_total_misses,
                'total': kg_total,
                'hit_rate': (kg_total_hits / kg_total * 100) if kg_total > 0 else 0
            },
            'vector_searches': {
                'total_hits': vec_total_hits,
                'total_misses': vec_total_misses,
                'total': vec_total,
                'hit_rate': (vec_total_hits / vec_total * 100) if vec_total > 0 else 0
            }
        }
        
        return stats
    
    def get_iteration_metrics(self) -> List[Dict]:
        """
        Get evaluation metrics for all iterations.
        
        Returns:
            List of iteration evaluation results
        """
        return self.iteration_metrics
    
    def _log_final_best_subgraph_evaluation(self, time_reference: str):
        """
        Evaluate ONLY the final best subgraph for official retrieval accuracy.
        This is called AFTER all iterations complete.
        
        This provides the single "retrieval accuracy" metric that should be used
        for performance evaluation, while iteration_metrics contains all subgraphs
        for debugging purposes.
        
        Args:
            time_reference: Time reference string from QA data
        """
        if not time_reference or time_reference.strip() in ["N/A", "", "None", "None-None"]:
            return
        
        if not self.subgraphs:
            print(f"  📊 Final Evaluation: No subgraphs to evaluate")
            return
        
        # Select the final best subgraph using the same logic as select_best_subgraphs
        best_subgraph = None
        best_score = -1
        
        for sg in self.subgraphs:
            if len(sg.nodes) < 2:  # Skip broken graphs
                continue
            score = self._calculate_answerability_score(sg)
            if score > best_score:
                best_score = score
                best_subgraph = sg
        
        if best_subgraph is None:
            print(f"  📊 Final Evaluation: No valid subgraphs found")
            return
        
        # Evaluate the best subgraph
        best_result = self._evaluate_single_subgraph(best_subgraph, time_reference.strip())
        
        # Add to iteration_metrics as a special final entry
        final_data = {
            'iteration': 'final_best_subgraph',
            'note': 'This is the official retrieval accuracy metric (best subgraph only)',
            'total_subgraphs_in_pool': len(self.subgraphs),
            'best_subgraph': best_result,
            'answerability_score': best_score
        }
        self.iteration_metrics.append(final_data)
        
        # Print summary
        if best_result and best_result['overlap'] is not None:
            overlap = best_result['overlap']
            print(f"\n  🎯 FINAL BEST SUBGRAPH EVALUATION")
            print(f"     Subgraph ID: {best_result['subgraph_id']}")
            print(f"     ⭐ Retrieval Accuracy (Overlap): {overlap:.3f}")
            print(f"     Answerability Score: {best_score:.4f}")
            print(f"     Composition: {best_result['num_events']}E/{best_result['num_objects']}O/{best_result['num_nodes']}N, {best_result['num_edges']}Edges")
            print(f"     (Selected from pool of {len(self.subgraphs)} subgraphs)")
        else:
            print(f"  🎯 FINAL: Best subgraph evaluation complete - no valid overlap")

    
    def _evaluate_single_subgraph(self, subgraph: Subgraph, time_reference: str) -> Dict:
        """
        Evaluate overlap for a SINGLE subgraph.
        
        DEBUG + SELF-HEALING MODE: 
        1. Scans for bad nodes (missing time/empty objects).
        2. BREAKPOINT triggers on detection to allow inspection.
        3. Prunes bad nodes to prevent crash and continues execution.
        
        Args:
            subgraph: The subgraph to evaluate
            time_reference: Time reference string from QA data
        
        Returns:
            Dictionary with evaluation results for this subgraph
        """
        # --- 1. CLEANUP PASS: Identify and remove bad nodes ---
        nodes_to_remove = []
        
        # Iterate safely over copy of items to allow modification later
        for node_id, node in list(subgraph.nodes.items()):
            # A. Validate EVENTS
            if node.type == 'event':
                has_valid_time = False
                
                # Check 'duration' [start, end]
                if 'duration' in node.metadata and isinstance(node.metadata['duration'], (list, tuple)) and len(node.metadata['duration']) >= 2:
                    float(node.metadata['duration'][0])
                    float(node.metadata['duration'][1])
                    has_valid_time = True
                if not has_valid_time and 'start_time' in node.metadata:
                    float(node.metadata['start_time'])
                    has_valid_time = True
                if not has_valid_time and 'timestamps' in node.metadata and isinstance(node.metadata['timestamps'], (list, tuple)) and len(node.metadata['timestamps']) >= 2:
                    float(node.metadata['timestamps'][0])
                    has_valid_time = True
                if not has_valid_time:
                    print("\n" + "!"*60)
                    print(f"🚨 FOUND CORRUPT EVENT NODE: {node_id}")
                    print(f"   Node Type: {node.type}")
                    print(f"   Metadata keys: {list(node.metadata.keys())}")
                    print(f"   Full Metadata: {node.metadata}")
                    print("!"*60)
                    print("🛑 Pausing for inspection. Type 'c' to PRUNE this node and continue.")
                    # breakpoint() # <--- EXECUTION STOPS HERE
                    
                    # Mark for removal
                    nodes_to_remove.append(node_id)

            # B. Validate OBJECTS
            elif node.type == 'object':
                # If object has absolutely no metadata or content
                if not node.content and not node.metadata:
                    print("\n" + "!"*60)
                    print(f"🚨 FOUND EMPTY OBJECT NODE: {node_id}")
                    print("!"*60)
                    print("🛑 Pausing for inspection. Type 'c' to PRUNE this node and continue.")
                    # breakpoint() # <--- EXECUTION STOPS HERE
                    
                    nodes_to_remove.append(node_id)
        
        # Execute removal (Safe Pruning)
        if nodes_to_remove:
            print(f"✂️  Pruning {len(nodes_to_remove)} bad nodes from subgraph {subgraph.id}...")
            for node_id in nodes_to_remove:
                subgraph.remove_node(node_id)

        # --- 2. EVALUATION PASS (Safe on cleaned graph) ---
        time_list = []
        for node in subgraph.nodes.values():
            # We can now safely access metadata because bad nodes are gone
            start_time_seconds = 0.0
            end_time_seconds = 0.0
            found_time = False
            if node.type == 'event':
                meta = node.metadata
                if 'duration' in meta:
                    start_time_seconds = float(meta['duration'][0])
                    end_time_seconds = float(meta['duration'][1])
                    found_time = True
            elif node.type == 'object':
                for duration in node.metadata["durations"]:
                    start_time_seconds_i = float(duration[0])
                    end_time_seconds_i = float(duration[1])
                    if end_time_seconds_i > start_time_seconds_i:
                        time_list.append((start_time_seconds_i, end_time_seconds_i))
            
            if found_time and end_time_seconds > start_time_seconds:
                time_list.append((start_time_seconds, end_time_seconds))
        
        # Evaluate overlap
        overlap = overlap_reference_helper(time_reference, time_list)
        
        # Collect statistics
        objects = subgraph.get_nodes_by_type('object')
        result = {
            'subgraph_id': subgraph.id,
            'overlap': overlap,
            'num_nodes': len(subgraph.nodes),
            'num_events': len(subgraph.get_nodes_by_type('event')),
            'num_objects': len(objects),
            'num_edges': len(subgraph.edges)
        }
        
        return result
    
    def _evaluate_subgraphs(self, time_reference: str) -> List[Dict]:
        """
        Evaluate overlap for each subgraph separately.
        
        This method evaluates ALL subgraphs and is kept for backward compatibility
        or special debugging purposes. For iteration evaluation, use 
        _evaluate_single_subgraph() on the best subgraph instead.
        """
        results = []
        for subgraph in self.subgraphs:
            result = self._evaluate_single_subgraph(subgraph, time_reference)
            results.append(result)
        return results
    
    def _log_iteration_evaluation(self, iteration_label: str, time_reference: str):
        """
        Evaluate ALL subgraphs and store results in iteration_metrics (for debugging).
        
        Args:
            iteration_label: Label for this evaluation (e.g., "iteration_0_seeds", "iteration_1")
            time_reference: Time reference string from QA data
        """
        if not time_reference or time_reference.strip() in ["N/A", "", "None", "None-None"]:
            # No time reference available, skip evaluation
            return
        
        # Evaluate all subgraphs (for debugging)
        subgraph_results = self._evaluate_subgraphs(time_reference.strip())
        
        # Store in iteration_metrics
        iteration_data = {
            'iteration': iteration_label,
            'num_subgraphs': len(self.subgraphs),
            'subgraphs': subgraph_results
        }
        self.iteration_metrics.append(iteration_data)
        
        # Print summary
        if subgraph_results:
            overlaps = [r['overlap'] for r in subgraph_results if r['overlap'] is not None]
            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
                max_overlap = max(overlaps)
                print(f"  📊 {iteration_label}: {len(subgraph_results)} subgraphs evaluated")
                print(f"     Average overlap: {avg_overlap:.3f}, Max overlap: {max_overlap:.3f}")
                for i, result in enumerate(subgraph_results):
                    if result['overlap'] is not None:
                        print(f"     Subgraph {i+1} ({result['subgraph_id']}): overlap={result['overlap']:.3f}, "
                              f"{result['num_events']}E/{result['num_objects']}O/{result['num_nodes']}N")
            else:
                print(f"  📊 {iteration_label}: {len(subgraph_results)} subgraphs (no valid overlaps)")

    def search(self, query: str, query_embedding: np.ndarray, max_iterations: int = 15, time_reference: Optional[str] = None):
        """
        Run graph search with STRICT BREAKPOINTS for empty/dead graphs.
        """
        print(f"--- Starting Search: '{query}' ---")
        self._reset_query_state()

        self.query_embedding = query_embedding
        
        # 1. Initialization
        self._initial_exploration(query)
        
        # Evaluate after seed initialization
        if time_reference:
            print(">> Running Initial Evaluation (Self-Healing)...")
            self._log_iteration_evaluation("iteration_0_seeds", time_reference)
        
        graph_size_history = []
        
        # 2. Iterative Exploration Loop
        for i in range(max_iterations):
            self.current_iteration = i + 1
            print(f"\n--- Iteration {self.current_iteration} ---")
            
            
            # Expansion
            for subgraph in self.subgraphs[:]:
                
                # ====================================================================
                # M1: CONSTRAINED PROPAGATION
                # ====================================================================
                if self.constrained_propagation:
                    # Get all nodes by type
                    all_events = subgraph.get_nodes_by_type('event')
                    all_objects = subgraph.get_nodes_by_type('object')
                    
                    # Sort by score descending (highest scores first)
                    all_events.sort(key=lambda n: n.score, reverse=True)
                    all_objects.sort(key=lambda n: n.score, reverse=True)
                    
                    # Take only top-k
                    active_events = all_events[:self.top_k_events]
                    active_objects = all_objects[:self.top_k_objects]
                    
                    # Debug output
                    print(f"  [M1] Constrained propagation: "
                        f"{len(active_events)}/{len(all_events)} events, "
                        f"{len(active_objects)}/{len(all_objects)} objects")
                else:
                    # Original behavior: expand from ALL nodes
                    active_events = subgraph.get_nodes_by_type('event')
                    active_objects = subgraph.get_nodes_by_type('object')


                # ====================================================================
                # M3: ADAPTIVE THRESHOLD
                # ====================================================================
                score_threshold = self._calculate_adaptive_threshold(subgraph)
                
                if self.adaptive_threshold:
                    print(f"  [M3] Adaptive threshold: {score_threshold:.4f} "
                        f"({self.threshold_percentile}th percentile)")
                
                # Filter nodes by adaptive threshold
                active_events = [e for e in active_events if e.score >= score_threshold]
                active_objects = [o for o in active_objects if o.score >= score_threshold]
                
                if self.adaptive_threshold:
                    print(f"  [M3] After threshold filter: "
                        f"{len(active_events)} events, {len(active_objects)} objects")


                # ====================================================================
                # EXPANSION OPERATIONS (unchanged)
                # ====================================================================
                
                # 1. Expand Events (Structure + Vector)
                for event in active_events:
                    self._op_event_to_object(event, subgraph)
                    self._op_vector_event_to_event(event, subgraph)
                
                # 2. Expand Objects (Structure + Relation + Vector)
                for obj in active_objects:
                    self._op_object_to_event(obj, subgraph)
                    self._op_relation_expansion(obj, subgraph)
                    self._op_vector_object_to_object(obj, subgraph)
                
                subgraph.update_iteration_stats()
            

            # Merging
            self._check_and_merge_subgraphs()
            
            # Pruning
            self._adaptive_pruning()
            
            # Evaluation
            if time_reference:
                self._log_iteration_evaluation(f"iteration_{self.current_iteration}", time_reference)
            
            # Convergence
            current_subgraph_count = len([sg for sg in self.subgraphs if len(sg.nodes) > 0])
            graph_size_history.append(current_subgraph_count)
            if len(graph_size_history) >= 2:
                if all(count == current_subgraph_count for count in graph_size_history[-2:]):
                    print(f"\n🛑 Graph converged at {current_subgraph_count} subgraphs.")
                    break

        # 3. Post-Processing
        print(f"\n--- Post-Processing ---")
        self._finalize_all_subgraphs()

        # 4. Final Best Subgraph Evaluation (Official Retrieval Accuracy)
        if time_reference:
            print(f"\n--- Final Best Subgraph Evaluation ---")
            self._log_final_best_subgraph_evaluation(time_reference)

        print("\n📊 Operation Statistics (Diagnosis):")
        for op, stats in self._op_stats.items():
            print(f"  - {op}: Attempted {stats['attempted']} -> Added {stats['added']}")
        
        valid_subgraphs = [sg for sg in self.subgraphs if len(sg.nodes) > 0]
        if not valid_subgraphs:
            print("❌ FINAL ERROR: No non-empty subgraphs remain for aggregation!")
            return "No information found.", []


        answer, final_subgraphs = self._aggregation(query)
        
        # 4. Save Context (existing logic)
        if self.context_graph is not None and self.current_context_key_embedding is not None and self.subgraphs:
            best_subgraphs = self.select_best_subgraphs(top_k=1, max_total_nodes_budget=999999)
            if best_subgraphs:
                best_subgraph = best_subgraphs[0]
                if len(best_subgraph.nodes) > 0:
                    self.context_graph.add_context(self.current_context_key_embedding, best_subgraph, keywords=self.current_keywords)
        
        return answer, final_subgraphs


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_context_neighbors(self, source_id: str, edge_type: str) -> Set[str]:
        """Finds neighbors in retrieved Context Graphs (History)."""
        neighbors = set()
        for sg in self.retrieved_context_subgraphs:
            if sg.has_node(source_id):
                for edge in sg.edges:
                    if edge.source_id == source_id and edge.type == edge_type:
                        neighbors.add(edge.target_id)
                    elif edge.target_id == source_id and edge.type == edge_type:
                         neighbors.add(edge.source_id)
        return neighbors

    def _calculate_adaptive_threshold(self, subgraph: Subgraph) -> float:
        """
        M3: Calculate adaptive threshold based on score distribution.
        
        Instead of fixed threshold (0.001), use percentile-based threshold
        that adapts to the current score distribution.
        
        Args:
            subgraph: Current subgraph to analyze
            
        Returns:
            float: Threshold value to use for filtering nodes
        """
        if not self.adaptive_threshold:
            return 0.001  # Default fixed threshold if M3 disabled
        
        # Get all node scores
        scores = [node.score for node in subgraph.nodes.values()]
        
        if len(scores) == 0:
            return 0.001  # Fallback if no nodes
        
        # Sort scores to calculate percentile
        scores.sort()
        
        # Calculate percentile index
        # Example: 80th percentile of [0.1, 0.2, 0.3, 0.4, 0.5] is 0.4
        percentile_idx = int(len(scores) * (self.threshold_percentile / 100.0))
        percentile_idx = min(percentile_idx, len(scores) - 1)  # Stay within bounds
        
        threshold = scores[percentile_idx]
        
        # Safety bounds: never go below 0.01 or above 0.5
        # This prevents:
        # - Too aggressive filtering (< 0.01 would keep almost nothing)
        # - Too permissive filtering (> 0.5 would keep everything)
        threshold = max(0.01, min(0.5, threshold))
        
        return threshold

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    # 1. Event -> Object (Name: 'event_object')
    def _op_event_to_object(self, source: Node, subgraph: Subgraph):
        if source.id in self._kg_query_cache['objects_in_event']:
            kg_ids = set(self._kg_query_cache['objects_in_event'][source.id])
            self._cache_stats['kg_queries']['objects_in_event']['hits'] += 1
        else:
            kg_ids = set(self.kg.get_objects_in_event(source.id))
            self._kg_query_cache['objects_in_event'][source.id] = list(kg_ids)
            self._cache_stats['kg_queries']['objects_in_event']['misses'] += 1
        
        hub_size = len(kg_ids)
        if hub_size == 0: return

        # M4: Add embeddings
        energy = self.scorer.calculate_energy_transfer(
            source.score, 'event_object', self.current_iteration, 
            hub_size=hub_size,
            parent_embedding=source.embedding,        # NEW
            query_embedding=self.query_embedding      # NEW
        )
        for oid in kg_ids:
            self._update_or_create_node(oid, 'object', energy, subgraph, source, 'event_object')

    # 2. Object -> Event (Name: 'object_event')
    def _op_object_to_event(self, source: Node, subgraph: Subgraph):
        if source.id in self._kg_query_cache['events_with_object']:
            kg_ids = set(self._kg_query_cache['events_with_object'][source.id])
            self._cache_stats['kg_queries']['events_with_object']['hits'] += 1
        else:
            kg_ids = set(self.kg.get_events_containing_object(source.id))
            self._kg_query_cache['events_with_object'][source.id] = list(kg_ids)
            self._cache_stats['kg_queries']['events_with_object']['misses'] += 1
        
        if source.id in self._kg_query_cache['event_count_per_object']:
            global_count = self._kg_query_cache['event_count_per_object'][source.id]
            self._cache_stats['kg_queries']['event_count_per_object']['hits'] += 1
        else:
            global_count = self.kg.get_global_event_count_for_object(source.id)
            self._kg_query_cache['event_count_per_object'][source.id] = global_count
            self._cache_stats['kg_queries']['event_count_per_object']['misses'] += 1
        
        if len(kg_ids) == 0: return
        
        # M4: Add embeddings
        energy = self.scorer.calculate_energy_transfer(
            source.score, 'object_event', self.current_iteration, 
            global_uniqueness=global_count,
            parent_embedding=source.embedding,        # NEW
            query_embedding=self.query_embedding      # NEW
        )
        for eid in kg_ids:
            self._update_or_create_node(eid, 'event', energy, subgraph, source, 'object_event')

    # 3. Structure Object (Name: 'structure_object')
    def _op_relation_expansion(self, source: Node, subgraph: Subgraph):
        # 1. Fetch Relation Data (used to find targets, not added as nodes)
        if source.id in self._kg_query_cache['relations_for_object']:
            relations = self._kg_query_cache['relations_for_object'][source.id]
            self._cache_stats['kg_queries']['relations_for_object']['hits'] += 1
        else:
            relations = self.kg.get_relations_containing_object(source.id)
            self._kg_query_cache['relations_for_object'][source.id] = relations
            self._cache_stats['kg_queries']['relations_for_object']['misses'] += 1
        
        if not relations:
            return

        # 2. Calculate Energy for the Target Objects
        # We treat this as a 1-hop structural connection
        hub_size = len(relations)
        # M4: Add embeddings
        energy_target = self.scorer.calculate_energy_transfer(
            source.score, 'structure_object', 
            current_iteration=self.current_iteration, 
            hub_size=hub_size,
            parent_embedding=source.embedding,        # NEW
            query_embedding=self.query_embedding      # NEW
        )
        
        for rel_node in relations:
            # Check metadata to find the OTHER object ID
            e1 = rel_node.metadata.get('entity1')
            e2 = rel_node.metadata.get('entity2')
            
            # Identify target
            if e1 == source.id:
                target_id = e2
            elif e2 == source.id:
                target_id = e1
            else:
                continue # Should not happen given get_relations logic
                
            if target_id and target_id != source.id:
                # Add Target Object Node directly
                # Edge Type: 'structure_object'
                self._update_or_create_node(target_id, 'object', energy_target, subgraph, source, 'structure_object')

    # 4. Context Event (Name: 'context_event')
    def _op_context_event_expansion(self, source: Node, subgraph: Subgraph):
        target_ids = self._get_context_neighbors(source.id, 'context_event')
        if not target_ids: return

        # M4: Add embeddings
        energy = self.scorer.calculate_energy_transfer(
            source.score, 'context_event', self.current_iteration,
            parent_embedding=source.embedding,        # NEW
            query_embedding=self.query_embedding      # NEW
        ) 
        for tid in target_ids:
            self._update_or_create_node(tid, 'event', energy, subgraph, source, 'context_event')

    # 5. Vector Object (Name: 'vector_object')
    def _op_vector_object_to_object(self, source: Node, subgraph: Subgraph):
        if not self._is_valid_embedding(source.embedding): return
        
        if source.id in self._vector_search_cache['object_to_object_by_node']:
            results = self._vector_search_cache['object_to_object_by_node'][source.id]
            self._cache_stats['vector_searches']['object_to_object']['hits'] += 1
        else:
            results = self.kg.search_objects_by_embedding(source.embedding, top_k=5)
            self._vector_search_cache['object_to_object_by_node'][source.id] = results
            self._cache_stats['vector_searches']['object_to_object']['misses'] += 1
        
        for res_node in results:
            # M4: Add query_embedding (node_embedding and parent_embedding already passed)
            energy = self.scorer.calculate_energy_transfer(
                source.score, 'vector_object', self.current_iteration,
                node_embedding=res_node.embedding, 
                parent_embedding=source.embedding,
                query_embedding=self.query_embedding      # NEW
            )
            self._update_or_create_node(res_node.id, 'object', energy, subgraph, source, 'vector_object', node_data=res_node)

    # 6. Vector Event (Name: 'vector_event')
    def _op_vector_event_to_event(self, source: Node, subgraph: Subgraph):
        if not self._is_valid_embedding(source.embedding): return
        
        if source.id in self._vector_search_cache['event_to_event_by_node']:
            results = self._vector_search_cache['event_to_event_by_node'][source.id]
            self._cache_stats['vector_searches']['event_to_event']['hits'] += 1
        else:
            results = self.kg.search_events_by_embedding(source.embedding, top_k=5)
            self._vector_search_cache['event_to_event_by_node'][source.id] = results
            self._cache_stats['vector_searches']['event_to_event']['misses'] += 1
        
        for res_node in results:
            # M4: Add query_embedding (node_embedding and parent_embedding already passed)
            energy = self.scorer.calculate_energy_transfer(
                source.score, 'vector_event', self.current_iteration,
                node_embedding=res_node.embedding, 
                parent_embedding=source.embedding,
                query_embedding=self.query_embedding      # NEW
            )
            self._update_or_create_node(res_node.id, 'event', energy, subgraph, source, 'vector_event', node_data=res_node)

    def _op_vector_relation_expansion(self, source: Node, subgraph: Subgraph):
        """
        [PLACEHOLDER] Future Operation:
        Map Object -> Relation (A) -> [Vector Search] -> Relation (B) -> Target Object
        
        This will allow hopping between similar relations (e.g., "holding cup" -> "holding bottle")
        even if they are not structurally connected.
        """
        # TODO: Implement relation-to-relation vector search
        pass

    def _is_valid_embedding(self, embedding) -> bool:
        """Check if embedding is valid (not None and no NaN values)."""
        if embedding is None:
            return False
        if isinstance(embedding, np.ndarray):
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                return False
        return True
    # ------------------------------------------------------------------
    # Core Update Logic
    # ------------------------------------------------------------------
    def _update_or_create_node(self, node_id, node_type, incoming_energy, subgraph, parent_node, op_type, node_data=None):
        if node_id == parent_node.id: return False
        
        self._op_stats[op_type]['attempted'] += 1

        final_energy = incoming_energy

        is_newly_created = False
        target_node = subgraph.get_node(node_id)

        if target_node:
            old_score = target_node.score
            new_score = self.scorer.accumulate_score(old_score, final_energy)
            target_node.score = new_score
            subgraph.add_edge(Edge(parent_node.id, node_id, op_type, final_energy))
            
            if (new_score - old_score) < 0.001: 
                return False
        else:
            if node_data: new_node = node_data
            elif node_type == 'event': new_node = self.kg.get_event_node_by_id(node_id)
            elif node_type == 'object': new_node = self.kg.get_object_node_by_id(node_id)
            else: new_node = None
            
            if new_node is None: new_node = Node(id=node_id, type=node_type, score=final_energy)
            else: new_node.score = final_energy
            
            subgraph.add_node(new_node)
            subgraph.add_edge(Edge(parent_node.id, node_id, op_type, final_energy))
            target_node = new_node
            is_newly_created = True

        self._op_stats[op_type]['added'] += 1

        if is_newly_created:
            self._instant_local_triangulation(target_node, subgraph)
            
        return True

    def _instant_local_triangulation(self, new_node: Node, subgraph: Subgraph):
        """
        Checks KG *AND* Context Graph for immediate connections 
        to nodes ALREADY existing in the subgraph.
        """
        triangulation_tasks = [] 

        if new_node.type == 'event':
            # 1. Check KG Objects (Structure)
            kg_obj_ids = self.kg.get_objects_in_event(new_node.id)
            if kg_obj_ids:
                hub_size = len(kg_obj_ids)
                # M4: Add embeddings
                energy = self.scorer.calculate_energy_transfer(
                    new_node.score, 'event_to_object', 
                    current_iteration=self.current_iteration, 
                    hub_size=hub_size,
                    parent_embedding=new_node.embedding,      # NEW
                    query_embedding=self.query_embedding      # NEW
                )
                for tid in kg_obj_ids:
                    triangulation_tasks.append((tid, energy, 'event_to_object', 'object'))

            # 2. Check Context Events (History)
            ctx_evt_ids = self._get_context_neighbors(new_node.id, 'event_to_event')
            if ctx_evt_ids:
                # M4: Add embeddings
                energy = self.scorer.calculate_energy_transfer(
                    new_node.score, 'context_event_to_event', 
                    current_iteration=self.current_iteration,
                    parent_embedding=new_node.embedding,      # NEW
                    query_embedding=self.query_embedding      # NEW
                )
                for tid in ctx_evt_ids:
                    triangulation_tasks.append((tid, energy, 'context_event_to_event', 'event'))

        elif new_node.type == 'object':
            # 1. Check KG Events (Structure)
            kg_evt_ids = self.kg.get_events_containing_object(new_node.id)
            if kg_evt_ids:
                global_count = self.kg.get_global_event_count_for_object(new_node.id)
                # M4: Add embeddings
                energy = self.scorer.calculate_energy_transfer(
                    new_node.score, 'object_to_event', 
                    current_iteration=self.current_iteration, 
                    global_uniqueness=global_count,
                    parent_embedding=new_node.embedding,      # NEW
                    query_embedding=self.query_embedding      # NEW
                )
                for tid in kg_evt_ids:
                    triangulation_tasks.append((tid, energy, 'object_to_event', 'event'))
            
            # 2. Check Context Relations (Object -> Object)
            ctx_rel_ids = self._get_context_neighbors(new_node.id, 'relation')
            if ctx_rel_ids:
                # M4: Add embeddings
                energy = self.scorer.calculate_energy_transfer(
                    new_node.score, 'context_relation', 
                    current_iteration=self.current_iteration,
                    parent_embedding=new_node.embedding,      # NEW
                    query_embedding=self.query_embedding      # NEW
                )
                for tid in ctx_rel_ids:
                    triangulation_tasks.append((tid, energy, 'context_relation', 'object'))

        for target_id, energy, op_type, target_type in triangulation_tasks:
            if subgraph.has_node(target_id):
                self._update_or_create_node(
                    node_id=target_id, node_type=target_type, 
                    incoming_energy=energy, 
                    subgraph=subgraph, parent_node=new_node, op_type=op_type
                )

    def _initial_exploration(self, query: str):
        """
        Intelligent seed initialization using LLM-generated keywords.
        
        Args:
            query: Natural language query (text)
        
        Steps:
        1. Use LLM to extract keywords/rewrite query (like tri_view_retrieval)
        2. Retrieve seed nodes using keyword-based search
        3. Check connectivity between seeds via KG/Context
        4. Group connected seeds into same subgraph
        5. Create separate subgraphs for disconnected clusters
        """
        # Use LLM prompts for better retrieval (following tri_view_retrieval pattern)
        if self.llm and self.kg.embedding_model:
            print("🔍 Using LLM for keyword extraction...")
            batch_inputs = []
            
            # Extract keywords for event search
            keywords_prompt = PROMPTS["keyword_extraction"].format(input_text=query)
            batch_inputs.append({"text": keywords_prompt})
            
            # Rewrite query for entity/object search
            rewrite_entity_prompt = PROMPTS["query_rewrite_for_entity_retrieval"].format(input_text=query)
            batch_inputs.append({"text": rewrite_entity_prompt})
            
            # Generate keywords
            batch_outputs = self.llm.batch_generate_response(batch_inputs)
            keywords_response = batch_outputs[0]
            rewrite_entity_response = batch_outputs[1]
            
            print(f"  Event keywords: {keywords_response}")
            print(f"  Entity query: {rewrite_entity_response}")
            
            # Format as comma-separated keyword string
            self.current_keywords = f"{keywords_response}, {rewrite_entity_response}"
            
            # Search using keywords (text-based search)
            # UPDATED: Use top_k=5 to match AVA's Tri-View Retrieval precision (was 45)
            # AVA settings: top_k_for_events = 5, top_k_for_entities = 5
            init_events = self.kg.search_events_by_description(keywords_response, top_k=10)
            init_objects = self.kg.search_objects_by_description(rewrite_entity_response, top_k=10)

        # --- DEBUG CHECKPOINT A ---
        print(f"📊 CHECKPOINT A: Retrieval Results")
        print(f"   - Retrieved Events: {len(init_events)}")
        print(f"   - Retrieved Objects: {len(init_objects)}")
        
        if len(init_events) == 0 and len(init_objects) == 0:
            print("❌ FAILURE: Gatekeeper filtered out ALL candidates. No valid nodes found.")
            print("🛑 Pausing to inspect why VDB returned nothing valid.")
            # breakpoint() # <--- CHECK HERE: If you hit this, VDB data is very bad.
            return
        
        if init_events or init_objects:
            # Group seeds by connectivity (also returns connections to avoid re-querying KG)
            seed_groups, seed_connections = self._group_connected_seeds(init_events, init_objects)
            
            for group_id, (events, objects) in enumerate(seed_groups):
                sg = Subgraph(id=f"kg_seed_group_{group_id}")
            
                # Add nodes with proper scoring and mark as seeds
                for node in events + objects:
                    initial_energy = node.score * self.scorer.trust_map['init'] 
                    node.score = self.scorer.accumulate_score(0.0, initial_energy)
                    node.metadata['is_seed'] = True  # Mark as seed for pruning protection
                    sg.add_node(node)
                
                # Add edges between connected seeds (reuse pre-computed connections)
                self._connect_seeds_in_group(sg, events, objects, seed_connections)
                
                self.subgraphs.append(sg)
                print(f"Created subgraph '{sg.id}' with {len(sg.nodes)} seeds ({len(events)} events, {len(objects)} objects)")


    def _group_connected_seeds(self, init_events: List[Node], init_objects: List[Node]) -> Tuple[List[Tuple[List[Node], List[Node]]], Dict[str, Set[str]]]:
        """
        Group seed nodes by connectivity via KG/Context relationships.
        Returns: 
            - List of (events, objects) tuples, each representing a connected group
            - Dict of seed_connections (node_id -> set of connected node_ids) for reuse
        """
        
        # Build connectivity graph between seeds
        seed_connections = defaultdict(set)  # node_id -> set of connected node_ids
        all_seeds = {node.id: node for node in init_events + init_objects}
        print(f"\n[GROUPING] All seeds ({len(all_seeds)}): {list(all_seeds.keys())}")
        
        # Check Event <-> Object connections via KG
        # This processes all events in init_events and finds their objects
        for event in init_events:
            # Use original_id if available for KG lookup
            obj_ids = self.kg.get_objects_in_event(event.id)
            for oid in obj_ids:
                if oid in all_seeds:
                    seed_connections[event.id].add(oid)
                    seed_connections[oid].add(event.id)
        
        # NOTE: Object <-> Event check is REMOVED because it's completely redundant:
        # 1. Event->Object loop (above) already processes ALL events in init_events
        # 2. all_seeds = init_events + init_objects, so any event in all_seeds MUST be in init_events
        # 3. Therefore, all Event-Object connections for retrieved seeds are already found above
        
        # Find connected components using Union-Find
        # WHY Union-Find? Because we need TRANSITIVE connections:
        # If event_1 <-> obj_A <-> event_2, all three should be in same group
        # Union-Find finds these connected components efficiently
        parent = {sid: sid for sid in all_seeds}
        print(f"\n[GROUPING] Initial parent map: {parent}")
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Unite connected seeds
        for seed_id, connected_ids in seed_connections.items():
            for connected_id in connected_ids:
                union(seed_id, connected_id)
        
        
        # Group seeds by root
        groups = defaultdict(lambda: {'events': [], 'objects': []})
        print(f"\n[GROUPING] Finding roots and grouping:")
        for node in init_events:
            root = find(node.id)
            print(f"  Event {node.id} -> root: {root}")
            groups[root]['events'].append(node)
        
        for node in init_objects:
            root = find(node.id)
            print(f"  Object {node.id} -> root: {root}")
            groups[root]['objects'].append(node)
        
        print(f"\n[GROUPING] Groups formed:")
        for root, group_data in groups.items():
            event_ids = [e.id for e in group_data['events']]
            object_ids = [o.id for o in group_data['objects']]
            print(f"  Group (root={root}): events={event_ids}, objects={object_ids}")
        
        # Convert to list of tuples
        result = [(g['events'], g['objects']) for g in groups.values()]
        
        print(f"Grouped {len(init_events)} events and {len(init_objects)} objects into {len(result)} connected groups")
        return result, dict(seed_connections)
    
    def _connect_seeds_in_group(self, subgraph: Subgraph, events: List[Node], objects: List[Node], 
                                seed_connections: Dict[str, Set[str]]):
        """
        Add edges between seeds that are connected via KG.
        
        Uses pre-computed seed_connections (from _group_connected_seeds) to avoid re-querying KG.
        Note: seed_connections contains connections for ALL seeds, but we only add edges
        for nodes that are actually in this subgraph (checked via subgraph.has_node()).
        
        Args:
            subgraph: The subgraph to add edges to
            events: List of event nodes in this group
            objects: List of object nodes in this group
            seed_connections: Pre-computed connections dict (node_id -> set of connected node_ids)
        """
        # Add edges using pre-computed connections
        # Event -> Object edges
        for event in events:
            connected_obj_ids = seed_connections.get(event.id, set())
            for oid in connected_obj_ids:
                if subgraph.has_node(oid):
                    # Still need to query for hub_size calculation
                    all_obj_ids = self.kg.get_objects_in_event(event.id)
                    # M4: Add embeddings
                    energy = self.scorer.calculate_energy_transfer(
                        event.score, 'event_to_object', 
                        current_iteration=self.current_iteration, 
                        hub_size=len(all_obj_ids),
                        parent_embedding=event.embedding,     # NEW
                        query_embedding=self.query_embedding  # NEW
                    )
                    subgraph.add_edge(Edge(event.id, oid, 'seed_connection', energy))
        
        # Object -> Event edges
        for obj in objects:
            connected_evt_ids = seed_connections.get(obj.id, set())
            for eid in connected_evt_ids:
                if subgraph.has_node(eid):
                    global_count = self.kg.get_global_event_count_for_object(obj.id)
                    # M4: Add embeddings
                    energy = self.scorer.calculate_energy_transfer(
                        obj.score, 'object_to_event', 
                        current_iteration=self.current_iteration, 
                        global_uniqueness=global_count,
                        parent_embedding=obj.embedding,       # NEW
                        query_embedding=self.query_embedding  # NEW
                    )
                    subgraph.add_edge(Edge(obj.id, eid, 'seed_connection', energy))
    
    def _check_and_merge_subgraphs(self):
        """
        Check if any subgraphs share nodes and should be merged.
        This is called after each expansion iteration.
        """
        if len(self.subgraphs) <= 1:
            return
        
        # Build node -> subgraph index mapping
        node_to_subgraphs = {}
        for sg in self.subgraphs:
            for node_id in sg.nodes:
                if node_id not in node_to_subgraphs:
                    node_to_subgraphs[node_id] = []
                node_to_subgraphs[node_id].append(sg)
        
        # Find subgraphs that share nodes
        merge_pairs = set()
        for node_id, subgraph_list in node_to_subgraphs.items():
            if len(subgraph_list) > 1:
                # This node exists in multiple subgraphs - they should merge
                for i in range(len(subgraph_list)):
                    for j in range(i+1, len(subgraph_list)):
                        pair = tuple(sorted([subgraph_list[i].id, subgraph_list[j].id]))
                        merge_pairs.add(pair)
        
        if not merge_pairs:
            return
        
        print(f"  Found {len(merge_pairs)} subgraph pairs to merge")
        
        # Group subgraphs that need to be merged together
        from collections import defaultdict
        parent = {sg.id: sg.id for sg in self.subgraphs}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for sg1_id, sg2_id in merge_pairs:
            union(sg1_id, sg2_id)
        
        # Group subgraphs by merge root
        merge_groups = defaultdict(list)
        sg_by_id = {sg.id: sg for sg in self.subgraphs}
        for sg in self.subgraphs:
            root = find(sg.id)
            merge_groups[root].append(sg)
        
        # Perform merges
        new_subgraphs = []
        for root, subgraphs_to_merge in merge_groups.items():
            if len(subgraphs_to_merge) == 1:
                new_subgraphs.append(subgraphs_to_merge[0])
            else:
                merged = self._merge_multiple_subgraphs(subgraphs_to_merge)
                new_subgraphs.append(merged)
                print(f"  Merged {len(subgraphs_to_merge)} subgraphs into '{merged.id}'")
        
        self.subgraphs = new_subgraphs
    
    def _merge_multiple_subgraphs(self, subgraphs: List[Subgraph]) -> Subgraph:
        """Merge multiple subgraphs into one"""
        sorted_ids = sorted([sg.id for sg in subgraphs])
        id_string = "".join(sorted_ids)
        short_hash = hashlib.md5(id_string.encode()).hexdigest()[:8]
        merged_id = f"merged_{len(subgraphs)}_subgraphs_{short_hash}"
        
        merged = Subgraph(id=merged_id)
        
        # Merge nodes (keep highest score for duplicates)
        for sg in subgraphs:
            for node in sg.nodes.values():
                if merged.has_node(node.id):
                    # Update score to max
                    existing = merged.get_node(node.id)
                    existing.score = max(existing.score, node.score)
                else:
                    merged.add_node(node)
        
        # Merge edges (avoid duplicates)
        for sg in subgraphs:
            for edge in sg.edges:
                merged.add_edge(edge)
        
        print(f"    Created merged subgraph with {len(merged.nodes)} nodes, {len(merged.edges)} edges")
        return merged
    
    def _finalize_all_subgraphs(self):
        """
        Cross-subgraph finalization:
        Create mutual_object_link edges between events that share objects,
        even if they're in different subgraphs.
        
        FIX: Build FULL BIDIRECTIONAL adjacency map (not just Events).
        """
        # 1. Build FULL Adjacency List (Node ID -> Set of Neighbors)
        # We don't care about types yet, just connectivity
        adj = {} 
        
        for sg in self.subgraphs:
            for edge in sg.edges:
                # Add Source -> Target
                if edge.source_id not in adj:
                    adj[edge.source_id] = set()
                adj[edge.source_id].add(edge.target_id)
                
                # Add Target -> Source (Undirected traversal)
                if edge.target_id not in adj:
                    adj[edge.target_id] = set()
                adj[edge.target_id].add(edge.source_id)

        # 2. Collect all events to start BFS from
        all_events = []
        for sg in self.subgraphs: 
            all_events.extend(sg.get_nodes_by_type('event'))
            
        existing_edges_global = set()
        # Pre-fill existing to avoid dupes
        for sg in self.subgraphs:
            for e in sg.edges: 
                if e.type == 'mutual_object_link': 
                    existing_edges_global.add(tuple(sorted((e.source_id, e.target_id))))

        mutual_links_created = 0
        MAX_HOPS = 3

        # Get base score from Scorer config
        mutual_link_base_score = self.scorer.trust_map.get('mutual_object_link', 0.8)
        
        # 3. BFS Logic
        for start_node in all_events:
            queue = deque([(start_node.id, 0)]) # (current_id, dist)
            visited = {start_node.id}
            
            while queue:
                curr_id, dist = queue.popleft()
                
                # Constraint: Don't go too deep (e.g., max 3 hops: E->O->O->E)
                if dist > MAX_HOPS:
                    continue

                # If we moved and hit another EVENT, we found a link
                if curr_id != start_node.id:
                    # Check node type (we need to find which subgraph owns this node to check type)
                    curr_node_type = None
                    for sg in self.subgraphs:
                        n = sg.get_node(curr_id)
                        if n: 
                            curr_node_type = n.type
                            break
                    
                    if curr_node_type == 'event':
                        # Valid Mutual Link!
                        pair_key = tuple(sorted((start_node.id, curr_id)))
                        if pair_key not in existing_edges_global:
                            link_score = mutual_link_base_score * (0.9 ** dist)
                            
                            # Add to the subgraph containing start_node
                            for target_sg in self.subgraphs:
                                if target_sg.has_node(start_node.id):
                                    target_sg.add_edge(Edge(start_node.id, curr_id, 'mutual_object_link', link_score))
                                    existing_edges_global.add(pair_key)
                                    mutual_links_created += 1
                                    break
                        continue # Stop this branch (don't go Event->Event->Object)

                # Explore neighbors using the FULL adj list
                if curr_id in adj:
                    for neighbor_id in adj[curr_id]:
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            queue.append((neighbor_id, dist + 1))
                            
        print(f"Created {mutual_links_created} cross-subgraph mutual_object_link edges")
        self._check_and_merge_subgraphs()
    
    # ------------------------------------------------------------------
    # Steiner Tree Pruning
    # ------------------------------------------------------------------
    def _should_prune_subgraph(self, subgraph: Subgraph) -> bool:
        """Adaptive check if pruning is needed"""
        config = self.pruning_config
        total = len(subgraph.nodes)
        
        # Don't prune tiny graphs
        if total < 20: return False
        
        # Trigger based on strict total budget
        if total > config['max_total_nodes']: return True
        
        # Trigger if edges are exploding (prevent loops)
        if len(subgraph.edges) > config['max_edges']: return True
        
        return False
    
    def _identify_terminals(self, subgraph: Subgraph) -> Set[str]:
        """
        Identify high-value nodes using Dynamic Ratio Selection.
        
        - Keeps Seeds.
        - Fills remaining budget with highest-scoring nodes (Event OR Object).
        - Does NOT enforce fixed event/object counts.
        """
        config = self.pruning_config
        max_total = config['max_total_nodes']
        terminals = set()
        
        # ====================================================================
        # 1. Identify seeds with M7 Prize-Based Protection
        # ====================================================================
        all_seeds = [n for n in subgraph.nodes.values() if n.metadata.get('is_seed', False)]
        
        if self.prize_based_seeds and len(all_seeds) > 0:
            # M7: Sort seeds by score descending
            all_seeds.sort(key=lambda n: n.score, reverse=True)
            
            # Protect only top-k seeds
            num_protected = min(self.top_k_protected, len(all_seeds))
            protected_seeds = all_seeds[:num_protected]
            
            # Assign descending prizes to protected seeds
            for i, seed in enumerate(protected_seeds):
                prize = num_protected - i  # k, k-1, k-2, ..., 1
                
                # Boost score with prize (multiplicative)
                boost_factor = 1.0 + (prize / (num_protected * 2))
                seed.score *= boost_factor
                
                # Add to terminals
                terminals.add(seed.id)
            
            # Debug output
            print(f"  [M7] Prize-based seeds: protected {num_protected}/{len(all_seeds)} "
                f"seeds (prizes: {num_protected} to 1)")
            
            # Unprotected seeds are NOT added to terminals (can compete with other nodes)
            if len(all_seeds) > num_protected:
                unprotected_count = len(all_seeds) - num_protected
                print(f"  [M7] {unprotected_count} low-scoring seeds competing with non-seeds")
        else:
            # Original behavior: protect ALL seeds
            for seed in all_seeds:
                terminals.add(seed.id)
            
            if all_seeds and not self.prize_based_seeds:
                print(f"  [M7] Protected all {len(all_seeds)} seeds (M7 disabled)")
        
        # ====================================================================
        # 2. Reserve buffer for Steiner Bridges (20% or min 5)
        # ====================================================================
        bridge_buffer = max(5, int(max_total * 0.2))
        available_slots = max_total - len(terminals) - bridge_buffer
        
        if available_slots <= 0:
            return terminals
        
        # ====================================================================
        # 3. Dynamic Selection: Sort ALL non-terminal nodes by score
        # ====================================================================
        # This includes:
        # - Non-seed nodes (events and objects)
        # - Unprotected seeds (if M7 enabled)
        candidates = []
        for node in subgraph.nodes.values():
            # Skip nodes already in terminals (protected seeds)
            if node.id in terminals:
                continue
            # Include all other nodes (non-seeds + unprotected seeds)
            candidates.append(node)
        
        # Sort by score descending (boosted protected seeds already in terminals)
        candidates.sort(key=lambda n: n.score, reverse=True)
        
        # Take top N candidates to fill available slots
        for node in candidates[:available_slots]:
            terminals.add(node.id)
        
        return terminals
    
    def _find_steiner_nodes(self, subgraph: Subgraph, terminals: Set[str], max_count: int) -> Set[str]:
        """
        Find weak nodes that are critical bridges between terminals.
        
        STRICT BUDGET MODE:
        - Strictly caps the number of Steiner nodes to 'max_count'.
        - If candidates > max_count, returns the highest scoring ones.
        """
        config = self.pruning_config
        steiner_nodes = set()
        
        # Optimization: If no budget for bridges, return immediately
        if max_count <= 0:
            return set()

        node_count = len(subgraph.nodes)
        
        # Build NetworkX graph
        G = nx.Graph()
        for node_id in subgraph.nodes:
            G.add_node(node_id)
        for edge in subgraph.edges:
            G.add_edge(edge.source_id, edge.target_id)
        
        # Method 1: Articulation Points (Fast & Critical)
        if config['use_articulation_points']:
            try:
                articulation_points = set(nx.articulation_points(G))
                for ap in articulation_points:
                    if ap not in terminals:
                        steiner_nodes.add(ap)
            except:
                pass
        
        # Method 2: Shortest Paths (Gap Filling)
        if (config['use_shortest_paths'] and 
            len(terminals) > 1 and 
            node_count <= config.get('shortest_path_node_limit', 300) and
            len(steiner_nodes) < max_count): # Optimization check
            
            terminals_list = list(terminals)
            max_path_length = config['max_path_length']
            
            # Limit pairs
            max_pairs = min(50, len(terminals_list) * (len(terminals_list) - 1) // 2)
            pairs_checked = 0
            
            for i, t1 in enumerate(terminals_list):
                if pairs_checked >= max_pairs: break
                for t2 in terminals_list[i+1:]:
                    if pairs_checked >= max_pairs: break
                    try:
                        path = nx.shortest_path(G, t1, t2)
                        if len(path) <= max_path_length + 2:
                            for node_id in path[1:-1]:
                                if node_id not in terminals:
                                    steiner_nodes.add(node_id)
                        pairs_checked += 1
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pairs_checked += 1
                        continue

        # STRICT BUDGET ENFORCEMENT
        if len(steiner_nodes) > max_count:
            # Sort by score and keep top 'max_count'
            steiner_node_objects = [subgraph.nodes[nid] for nid in steiner_nodes if nid in subgraph.nodes]
            steiner_node_objects.sort(key=lambda n: n.score, reverse=True)
            steiner_nodes = set(n.id for n in steiner_node_objects[:max_count])
        
        return steiner_nodes
    
    def _prune_subgraph_steiner(self, subgraph: Subgraph) -> Dict:
        """
        Prune subgraph using Steiner Tree approach with STRICT TOTAL BUDGET.
        """
        config = self.pruning_config
        max_total_nodes = config['max_total_nodes'] # e.g., 300
        
        original_nodes = len(subgraph.nodes)
        
        # Step 1: Identify Terminals (High Value Nodes)
        terminals = self._identify_terminals(subgraph)
        
        # Step 1.5: Emergency Trim if Terminals > Budget
        # This happens if seeds + max_events + max_objects > max_total_nodes
        if len(terminals) > max_total_nodes:
            # Must trim terminals, but protect seeds
            protected_seeds = {nid for nid in terminals if subgraph.nodes[nid].metadata.get('is_seed', False)}
            expendable_terminals = [nid for nid in terminals if nid not in protected_seeds]
            
            # Sort expendable by score
            expendable_terminals.sort(key=lambda nid: subgraph.nodes[nid].score, reverse=True)
            
            # Calculate how many expendables we can keep
            allowance = max_total_nodes - len(protected_seeds)
            if allowance > 0:
                kept_expendables = set(expendable_terminals[:allowance])
                terminals = protected_seeds.union(kept_expendables)
            else:
                # Edge case: Seeds alone exceed budget (rare, but possible)
                terminals = protected_seeds
        
        # Step 2: Calculate Remaining Budget for Bridges
        remaining_budget = max(0, max_total_nodes - len(terminals))
        
        # Step 3: Find Steiner nodes (capped by remaining budget)
        steiner_nodes = self._find_steiner_nodes(subgraph, terminals, max_count=remaining_budget)
        
        # Step 4: Determine Final Keep List
        keep_nodes = terminals.union(steiner_nodes)
        
        # Step 5: Execute Pruning
        nodes_to_remove = []
        for node_id in list(subgraph.nodes.keys()):
            if node_id not in keep_nodes:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            subgraph.remove_node(node_id)
        
        # Clean edges
        edges_to_remove = [e for e in subgraph.edges if e.source_id not in subgraph.nodes or e.target_id not in subgraph.nodes]
        for edge in edges_to_remove:
            subgraph.edges.remove(edge)
        
        # Stats
        stats = {
            'original_nodes': original_nodes,
            'terminals': len(terminals),
            'steiner_nodes': len(steiner_nodes),
            'removed_nodes': len(nodes_to_remove),
            'final_nodes': len(subgraph.nodes),
            'final_events': len(subgraph.get_nodes_by_type('event')),
            'final_objects': len(subgraph.get_nodes_by_type('object')),
            'final_edges': len(subgraph.edges),
            'original_edges': 0, 'original_events': 0, 'original_objects': 0, 'pruning_ratio': 0 # placeholders
        }
        return stats
    
    def _adaptive_pruning(self) -> int:
        """
        Check all subgraphs and prune those that need it.
        Returns number of subgraphs pruned.
        """
        pruned_count = 0
        config = self.pruning_config
        
        for subgraph in self.subgraphs:
            if self._should_prune_subgraph(subgraph):
                node_count = len(subgraph.nodes)
                edge_count = len(subgraph.edges)
                
                # Log why pruning was triggered
                reasons = []
                if node_count > config['max_total_nodes']:
                    reasons.append(f"nodes>{config['max_total_nodes']}")
                if edge_count > config['max_edges']:
                    reasons.append(f"edges>{config['max_edges']}")
                
                print(f"\n🔪 Pruning '{subgraph.id}' ({', '.join(reasons)})...")
                stats = self._prune_subgraph_steiner(subgraph)
                
                print(f"  Nodes: {stats['original_nodes']} → {stats['final_nodes']} "
                      f"({stats['removed_nodes']} removed, {stats['pruning_ratio']:.1%} pruned)")
                print(f"  Events: {stats['original_events']} → {stats['final_events']}")
                print(f"  Objects: {stats['original_objects']} → {stats['final_objects']}")
                print(f"  Edges: {stats['original_edges']} → {stats['final_edges']}")
                print(f"  Kept: {stats['terminals']} terminals + {stats['steiner_nodes']} Steiner nodes")
                
                pruned_count += 1
        
        return pruned_count

    # ------------------------------------------------------------------
    # Subgraph Selection & Ranking (The Curator)
    # ------------------------------------------------------------------
    def select_best_subgraphs(self, 
                              max_total_nodes_budget: int = 300, 
                              top_k: Optional[int] = None) -> List[Subgraph]:
        """
        Rank subgraphs by 'Total Relevance Energy' and select the best ones.
        Includes a 'Soft Budget' to prevent empty results.
        """
        scored_graphs = []
        
        # 1. Score all subgraphs
        for sg in self.subgraphs:
            score = self._calculate_answerability_score(sg)
            scored_graphs.append((sg, score))
            
            # Debug Print (Simplified)
            n_count = len(sg.nodes)
            print(f"  Subgraph '{sg.id}': Total Energy={score:.4f} (Nodes: {n_count})")
        
        # 2. Sort by Score Descending (Highest Energy First)
        scored_graphs.sort(key=lambda x: x[1], reverse=True)
        
        selected_subgraphs = []
        current_node_count = 0
        
        # 3. Selection Loop
        for i, (sg, score) in enumerate(scored_graphs):
            # Filter out broken graphs
            if len(sg.nodes) < 2: 
                continue
            
            # Check top_k limit
            if top_k is not None and len(selected_subgraphs) >= top_k:
                break
            
            # === CRITICAL: SOFT BUDGET LOGIC ===
            # Always take the FIRST (best) subgraph, even if it exceeds budget.
            # This prevents returning empty results if Pruning was slightly off.
            if len(selected_subgraphs) == 0:
                selected_subgraphs.append(sg)
                current_node_count += len(sg.nodes)
                if current_node_count > max_total_nodes_budget:
                    print(f"    ⚠️  Taking best subgraph despite budget overflow ({len(sg.nodes)} > {max_total_nodes_budget})")
                continue

            # For subsequent graphs, enforce strict budget
            if current_node_count + len(sg.nodes) <= max_total_nodes_budget:
                selected_subgraphs.append(sg)
                current_node_count += len(sg.nodes)
            else:
                # If top_k is explicit, allow overflow to satisfy it
                if top_k is not None and len(selected_subgraphs) < top_k:
                    selected_subgraphs.append(sg)
                    current_node_count += len(sg.nodes)
                    print(f"    ⚠️  Exceeded budget to satisfy top_k={top_k} requirement")
                continue
                
        return selected_subgraphs

    def _calculate_answerability_score(self, subgraph: Subgraph) -> float:
        """
        Simplest Effective Heuristic: Total Relevance Energy.
        
        Score = Sum(Node Scores)
        
        Why?
        1. Pruning has already filtered out the "garbage" (low scoring nodes).
        2. Remaining nodes are the "Top N" best candidates.
        3. A subgraph with MORE high-scoring nodes provides MORE context for RAG.
        """
        nodes = list(subgraph.nodes.values())
        if not nodes: return 0.0
        
        # Penalize tiny fragments that are likely pruning artifacts
        if len(nodes) < 3: 
            return 0.0
            
        # Metric: Total Energy (Sum of scores)
        # This naturally balances quality and quantity.
        total_energy = sum(n.score for n in nodes)
        
        return total_energy

    def _aggregation(self, query, selection_mode: str = 'budget'):
        """
        Aggregate final results by selecting the best subgraphs.
        
        Args:
            selection_mode: 
                - 'best': Select only the single best subgraph (top_k=1)
                - 'top3': Select top 3 subgraphs regardless of budget (top_k=3)
                - 'budget': Select as many as fit in budget (default, max_total_nodes_budget=300)
                - 'all': Return all subgraphs (no filtering)
        """
        print(f"\n--- Aggregation: Selecting Best Subgraphs (mode={selection_mode}) ---")
        
        if selection_mode == 'best':
            # Select only the single best subgraph
            final_selection = self.select_best_subgraphs(top_k=1, max_total_nodes_budget=999999)
            print(f"✓ Selected THE BEST subgraph for Final Prompt.")
        elif selection_mode == 'top3':
            # Select top 3 subgraphs (ignore budget)
            final_selection = self.select_best_subgraphs(top_k=3, max_total_nodes_budget=999999)
            print(f"✓ Selected top {len(final_selection)} subgraphs for Final Prompt.")
        elif selection_mode == 'budget':
            # Select best graphs fitting into ~300 nodes context window
            final_selection = self.select_best_subgraphs(max_total_nodes_budget=500)
            print(f"✓ Selected {len(final_selection)} subgraphs within budget for Final Prompt.")
        else:  # 'all'
            final_selection = self.subgraphs
            print(f"✓ Using all {len(final_selection)} subgraphs (no filtering).")
        
        # In a real app, here you would format 'final_selection' into text for the LLM
        return "Aggregated Answer", final_selection

if __name__ == "__main__":
    print("Graph Engine Main Entry Point")