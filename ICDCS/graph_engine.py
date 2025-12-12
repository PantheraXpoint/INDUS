import numpy as np
import math
import json
from typing import List, Dict, Set, Optional, Tuple, Deque
from collections import deque, defaultdict
import networkx as nx
from graph_interfaces import Node, Edge, Subgraph, KnowledgeGraphInterface, ContextGraphInterface, get_global_system_counts
from graph_scorer import GraphScorer
from .prompt import PROMPTS

class GraphEngine:
    def __init__(self, 
                 knowledge_graph: KnowledgeGraphInterface, 
                 context_graph: Optional[ContextGraphInterface] = None, 
                 scorer: Optional[GraphScorer] = None,
                 llm = None):  # LLM for query rewriting/keyword extraction
        self.kg = knowledge_graph
        self.context_graph = context_graph
        self.scorer = scorer
        self.llm = llm
        self.subgraphs: List[Subgraph] = []
        
        # Session State
        self.current_context_key_embedding: Optional[np.ndarray] = None  # Keyword embedding for context
        self.current_keywords: str = ""  # Comma-separated keywords from LLM
        self.retrieved_context_subgraphs: List[Subgraph] = []

        # Pruning Configuration
        self.pruning_config = {
            # RANK-BASED BUDGETS (Score-Saturation Proof)
            # These define how many TOP nodes to keep (regardless of scores)
            'max_event_nodes': 15,          # Keep TOP 50 events (excl. seeds)
            'max_object_nodes': 35,        # Keep TOP 100 objects (excl. seeds)
            'max_edges': 150,               # Edge count threshold for pruning trigger
            'max_total_nodes': 70,         # Total node threshold for pruning trigger
            
            # Saturation detection (when to trigger pruning)
            'score_saturation_threshold': 0.95,
            'saturation_count_trigger': 10,
            
            # Steiner tree settings (bridge protection)
            'use_articulation_points': True,    # Find critical connection points
            'use_shortest_paths': True,         # Find bridge nodes (disabled on large graphs)
            'shortest_path_node_limit': 100,    # Skip shortest paths if > 300 nodes
            'max_path_length': 3,               # Max path length for bridge detection
        }

    def search(self, query: str, query_embedding: np.ndarray, max_iterations: int = 10):
        """
        Run graph search for the given query.
        
        Args:
            query: Natural language query text
            query_embedding: Query embedding (kept for signature compatibility, but not used internally)
            max_iterations: Maximum exploration iterations
        
        Note: query_embedding parameter is kept for backward compatibility but is not used.
              The search uses keyword embeddings extracted via LLM instead.
        """
        print(f"--- Starting Search: '{query}' ---")
        self.current_iteration = 0  # Track iteration for progressive pruning
        
        # 1. Initialization (Create Anchor Subgraphs with intelligent grouping)
        self._initial_exploration(query)
        print(f"Initialized with {len(self.subgraphs)} subgraphs")
        
        # State tracking for convergence detection (subgraph count stability)
        graph_size_history = []  # Tracks number of subgraphs per iteration
        
        # 2. Iterative Exploration Loop (The Wave)
        for i in range(max_iterations):
            self.current_iteration = i + 1
            print(f"\n--- Iteration {self.current_iteration} ---")
            print(f"Active subgraphs: {len(self.subgraphs)}")
            
            # Early stopping check: If graph is too large, stop
            total_nodes = sum(len(sg.nodes) for sg in self.subgraphs)
            total_edges = sum(len(sg.edges) for sg in self.subgraphs)
            print(f"  Total graph size: {total_nodes} nodes, {total_edges} edges")
            
            if total_nodes > 5000:
                print(f"⚠️  Graph exceeded size limit ({total_nodes} > 10000 nodes). Stopping early.")
                break
            
            # Process each subgraph with Shape-Aware Strategy
            for subgraph in self.subgraphs[:]:  # Copy list since we might modify it
                # Get active nodes
                active_events = subgraph.get_nodes_by_type('event')
                active_objects = subgraph.get_nodes_by_type('object')
                
                # DIAGNOSIS: Determine strategy based on graph shape
                strategy_mode = self._determine_strategy(subgraph)
                mass_ratio = self._calculate_mass_ratio(subgraph)
                density = self._calculate_density(subgraph)
                velocity = subgraph.get_velocity()
                
                # print(f"  Subgraph '{subgraph.id}': {len(active_events)}E, {len(active_objects)}O | "
                #       f"Strategy={strategy_mode} (R={mass_ratio:.2f}, D={density:.2f}, V={velocity:.3f})")
                
                # EXECUTION: Strategy-based operation dispatch
                if strategy_mode == 'GROUNDING':
                    # Too many events, need objects
                    # PRIMARY: Event→Object (structural grounding)
                    # FALLBACK: Vector E→E (find similar events that might have objects)
                    self._process_event_stream(active_events, subgraph, 
                                               strategy_mode='GROUNDING',
                                               allowed_ops=['event_to_object', 'vector_event'])
                
                elif strategy_mode == 'BRIDGING':
                    # Too many objects, need events
                    # PRIMARY: Object→Event (find narratives)
                    # FALLBACK: Vector O→O (find similar objects with more events)
                    self._process_object_stream(active_objects, subgraph,
                                                strategy_mode='BRIDGING',
                                                allowed_ops=['object_to_event', 'vector_object'])
                
                elif strategy_mode == 'LEAPING':
                    # Dense + stagnant, need scene jump - TOP 3 events, Vector/Context E→E
                    self._process_event_stream(active_events, subgraph,
                                               strategy_mode='LEAPING',
                                               allowed_ops=['vector_event', 'context_event'],
                                               limit_top_k=3)
                
                elif strategy_mode == 'TRIANGULATION':
                    # Loose chain, need cross-links - Object→Object relations
                    self._process_object_stream(active_objects, subgraph,
                                                strategy_mode='TRIANGULATION',
                                                allowed_ops=['relation', 'vector_object'])
                
                else:  # BALANCED
                    # Healthy state - run standard mixed operations
                    self._process_event_stream(active_events, subgraph, strategy_mode='BALANCED')
                    self._process_object_stream(active_objects, subgraph, strategy_mode='BALANCED')
                
                # Update iteration stats for velocity calculation next iteration
                subgraph.update_iteration_stats()
            
            # Check for natural merges after expansion
            self._check_and_merge_subgraphs()
            print(f"After merge check: {len(self.subgraphs)} subgraphs remaining")
            
            # Adaptive pruning (check if any subgraph needs pruning)
            self._adaptive_pruning()
            
            # === CONVERGENCE CHECK: Subgraph Stability ===
            # Measure current system state by subgraph count
            current_subgraph_count = len(self.subgraphs)
            
            # Track history
            graph_size_history.append(current_subgraph_count)
            
            # Check if last 2 iterations have the same subgraph count (convergence)
            if len(graph_size_history) >= 2:
                last_2 = graph_size_history[-2:]
                if all(count == current_subgraph_count for count in last_2):
                    print(f"\n🛑 Graph converged (subgraph count stable for 2 iterations):")
                    print(f"   Stable state: {current_subgraph_count} subgraphs")
                    print(f"   Stopping early at iteration {self.current_iteration}/{max_iterations}")
                    break  # Early exit - no new merges or splits happening

        # 3. Post-Processing: Densify Graph (Cross-Subgraph Mutual Links)
        print(f"\n--- Post-Processing: Cross-Subgraph Finalization ---")
        self._finalize_all_subgraphs()
        
        # 4. Final Aggregation
        answer, final_subgraphs = self._aggregation(query)
        
        # 5. Save best subgraph to context for future queries
        if self.current_context_key_embedding is not None and self.subgraphs:
            best_subgraphs = self.select_best_subgraphs(top_k=1, max_total_nodes_budget=999999)
            if best_subgraphs:
                best_subgraph = best_subgraphs[0]
                print(f"\n💾 Saving best subgraph to context graph...")
                self.context_graph.add_context(
                    self.current_context_key_embedding,
                    best_subgraph,
                    keywords=self.current_keywords
                )
        
        return answer, final_subgraphs

    # ------------------------------------------------------------------
    # Dispatchers
    # ------------------------------------------------------------------
    def _process_event_stream(self, events: List[Node], subgraph: Subgraph, 
                             strategy_mode: str = 'BALANCED',
                             allowed_ops: List[str] = None, 
                             limit_top_k: int = None):
        """
        Process events with strategy-aware operation filtering.
        
        Args:
            strategy_mode: Current strategy for dynamic trust adjustment
            allowed_ops: List of allowed operations (filters execution)
            limit_top_k: If set, only process top K highest-scoring events
        """
        # Default to all operations if not specified
        if allowed_ops is None:
            allowed_ops = ['event_to_object', 'context_event', 'vector_event']
        
        # Apply TOP-K filtering (used in LEAPING mode)
        if limit_top_k is not None:
            events = sorted(events, key=lambda e: e.score, reverse=True)[:limit_top_k]
        
        # Process events with filtered operations
        for event in events:
            # if event.score < 0.15: continue
            
            # Execute only allowed operations
            if 'event_to_object' in allowed_ops:
                self._op_event_to_object(event, subgraph, strategy_mode)
            
            if 'context_event' in allowed_ops:
                self._op_context_event_expansion(event, subgraph, strategy_mode)
            
            if 'vector_event' in allowed_ops:
                self._op_vector_event_to_event(event, subgraph, strategy_mode)

    def _process_object_stream(self, objects: List[Node], subgraph: Subgraph,
                               strategy_mode: str = 'BALANCED',
                               allowed_ops: List[str] = None,
                               limit_top_k: int = None):
        """
        Process objects with strategy-aware operation filtering.
        
        Args:
            strategy_mode: Current strategy for dynamic trust adjustment
            allowed_ops: List of allowed operations (filters execution)
            limit_top_k: If set, only process top K highest-scoring objects
        """
        # Default to all operations if not specified
        if allowed_ops is None:
            allowed_ops = ['object_to_event', 'relation', 'vector_object']
        
        # Apply TOP-K filtering if specified
        if limit_top_k is not None:
            objects = sorted(objects, key=lambda o: o.score, reverse=True)[:limit_top_k]
        
        # Process objects with filtered operations
        for obj in objects:
            # if obj.score < 0.15: continue
            
            # Execute only allowed operations
            if 'object_to_event' in allowed_ops:
                self._op_object_to_event(obj, subgraph, strategy_mode)
            
            if 'relation' in allowed_ops:
                self._op_relation_expansion(obj, subgraph, strategy_mode)
            
            if 'vector_object' in allowed_ops:
                self._op_vector_object_to_object(obj, subgraph, strategy_mode)

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

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def _op_event_to_object(self, source: Node, subgraph: Subgraph, strategy_mode: str = 'BALANCED'):
        kg_ids = set(self.kg.get_objects_in_event(source.id))
        ctx_ids = self._get_context_neighbors(source.id, 'event_to_object')
        obj_ids = kg_ids.union(ctx_ids)
        
        hub_size = len(obj_ids)
        if hub_size == 0:
            print(f"  [EVENT_TO_OBJECT] Event {source.id} has no objects")
            return

        raw_energy = self.scorer.calculate_energy_transfer(
            source_score=source.score, op_type='event_to_object', 
            strategy_mode=strategy_mode, current_iteration=self.current_iteration,
            hub_size=hub_size
        )

        added_count = 0
        for oid in obj_ids:
            if self._update_or_create_node(oid, 'object', raw_energy, subgraph, source, 'event_to_object'):
                added_count += 1

        # if added_count > 0:
        #     print(f"  [EVENT_TO_OBJECT] Event {source.id} → Added {added_count}/{len(obj_ids)} objects (strategy={strategy_mode})")

    def _op_object_to_event(self, source: Node, subgraph: Subgraph, strategy_mode: str = 'BALANCED'):
        kg_ids = set(self.kg.get_events_containing_object(source.id))
        global_count = self.kg.get_global_event_count_for_object(source.id)
        
        if len(kg_ids) == 0:
            print(f"  [OBJECT_TO_EVENT] Object {source.id} appears in no events")
            return
        
        raw_energy = self.scorer.calculate_energy_transfer(
            source_score=source.score, op_type='object_to_event', 
            strategy_mode=strategy_mode, current_iteration=self.current_iteration,
            global_uniqueness=global_count
        )

        added_count = 0
        for eid in kg_ids:
            if self._update_or_create_node(eid, 'event', raw_energy, subgraph, source, 'object_to_event'):
                added_count += 1
        
        # if added_count > 0:
        #     print(f"  [OBJECT_TO_EVENT] Object {source.id} → Added {added_count}/{len(kg_ids)} events (strategy={strategy_mode})")

    def _op_relation_expansion(self, source: Node, subgraph: Subgraph, strategy_mode: str = 'BALANCED'):
        """Object → Object via Context Graph only (no KG structure)"""
        target_ids = self._get_context_neighbors(source.id, 'relation')
        if not target_ids: return

        energy = self.scorer.calculate_energy_transfer(
            source.score, 'context_relation', 
            strategy_mode=strategy_mode, current_iteration=self.current_iteration
        )
        
        for tid in target_ids:
            self._update_or_create_node(tid, 'object', energy, subgraph, source, 'context_relation')

    def _op_context_event_expansion(self, source: Node, subgraph: Subgraph, strategy_mode: str = 'BALANCED'):
        """Event → Event via Context Graph only (no KG structure)"""
        target_ids = self._get_context_neighbors(source.id, 'event_to_event')
        if not target_ids: return

        energy = self.scorer.calculate_energy_transfer(
            source.score, 'context_event_to_event', 
            strategy_mode=strategy_mode, current_iteration=self.current_iteration
        ) 

        for tid in target_ids:
            self._update_or_create_node(tid, 'event', energy, subgraph, source, 'context_event_to_event')

    def _op_vector_object_to_object(self, source: Node, subgraph: Subgraph, strategy_mode: str = 'BALANCED'):
        # if source.embedding is None:
        #     print(f"  [VECTOR_OBJECT] BLOCKED: Object {source.id} has no embedding (Lobotomy Bug!)")
        #     return
        
        results = self.kg.search_objects_by_embedding(source.embedding, top_k=5)
        # print(f"  [VECTOR_OBJECT] Object {source.id} → Found {len(results)} similar objects (strategy={strategy_mode})")
        
        added_count = 0
        for res_node in results:
            energy = self.scorer.calculate_energy_transfer(
                source.score, 'vector_object', strategy_mode=strategy_mode,
                current_iteration=self.current_iteration,
                node_embedding=res_node.embedding,
                parent_embedding=source.embedding
                # query_embedding removed - not used for vector similarity scoring
            )
            if self._update_or_create_node(res_node.id, 'object', energy, subgraph, source, 'vector_object', node_data=res_node):
                added_count += 1
        
        # if added_count > 0:
        #     print(f"  [VECTOR_OBJECT] ✓ Added {added_count}/{len(results)} new object links")

    def _op_vector_event_to_event(self, source: Node, subgraph: Subgraph, strategy_mode: str = 'BALANCED'):
        # if source.embedding is None:
        #     print(f"  [VECTOR_EVENT] BLOCKED: Event {source.id} has no embedding (Lobotomy Bug!)")
        #     return
        
        results = self.kg.search_events_by_embedding(source.embedding, top_k=5)
        # print(f"  [VECTOR_EVENT] Event {source.id} → Found {len(results)} similar events (strategy={strategy_mode})")
        
        added_count = 0
        for res_node in results:
            energy = self.scorer.calculate_energy_transfer(
                source.score, 'vector_event', strategy_mode=strategy_mode,
                current_iteration=self.current_iteration,
                node_embedding=res_node.embedding,
                parent_embedding=source.embedding
                # query_embedding removed - not used for vector similarity scoring
            )
            if self._update_or_create_node(res_node.id, 'event', energy, subgraph, source, 'vector_event', node_data=res_node):
                added_count += 1
        
        # if added_count > 0:
        #     print(f"  [VECTOR_EVENT] ✓ Added {added_count}/{len(results)} new event links")

    # ------------------------------------------------------------------
    # Post-Processing Logic (Corrected for Deep Paths)
    # ------------------------------------------------------------------
    def _finalize_subgraph_structure(self, subgraph: Subgraph):
        """
        Creates edges between Events that are connected by a path 
        containing ONLY Object nodes (of any length).
        
        Algorithm: BFS from every Event node, traversing only Object nodes.
        """
        # 1. Build Adjacency List for the current subgraph (Undirected view for connectivity)
        adj: Dict[str, Set[str]] = {node_id: set() for node_id in subgraph.nodes}
        for edge in subgraph.edges:
            if edge.source_id in adj and edge.target_id in adj:
                adj[edge.source_id].add(edge.target_id)
                adj[edge.target_id].add(edge.source_id)

        events = subgraph.get_nodes_by_type('event')
        existing_edges = set((e.source_id, e.target_id) for e in subgraph.edges)

        # 2. BFS from each event to find other reachable events via Objects
        for start_event in events:
            queue = deque([(start_event.id, 0)]) # (current_id, path_length)
            visited = {start_event.id}
            
            while queue:
                curr_id, dist = queue.popleft()
                
                # If we moved more than 0 steps and hit an Event, we found a link
                if curr_id != start_event.id:
                    curr_node = subgraph.get_node(curr_id)
                    if curr_node and curr_node.type == 'event':
                        # Found an event! Create Mutual Link if it doesn't exist
                        if (start_event.id, curr_id) not in existing_edges:
                            # Score logic: Decay based on path length
                            link_score = 0.8 * (0.9 ** dist) 
                            subgraph.add_edge(Edge(start_event.id, curr_id, 'mutual_object_link', link_score))
                            existing_edges.add((start_event.id, curr_id))
                        continue # Stop this branch, don't go through the event to other objects

                # Explore neighbors
                for neighbor_id in adj[curr_id]:
                    if neighbor_id not in visited:
                        neighbor_node = subgraph.get_node(neighbor_id)
                        if not neighbor_node: continue
                        
                        # Only traverse through OBJECTS. 
                        # Or if we hit an EVENT, we process it (above) but don't add to queue (stop branch)
                        if neighbor_node.type == 'object':
                            visited.add(neighbor_id)
                            queue.append((neighbor_id, dist + 1))
                        elif neighbor_node.type == 'event':
                            # We still want to "visit" the event to check the condition above,
                            # but we do NOT add it to the queue to continue traversal.
                            # We just add to queue to be processed by the check at the start of loop.
                            visited.add(neighbor_id)
                            queue.append((neighbor_id, dist + 1))

    # ------------------------------------------------------------------
    # Core Update Logic
    # ------------------------------------------------------------------
    def _update_or_create_node(self, node_id, node_type, incoming_energy, subgraph, parent_node, op_type, node_data=None):
        if node_id == parent_node.id: return False

        # Normalize event IDs (strip sentence index)
        normalized_id = node_id
        if node_type == 'event':
            normalized_id = self.kg.normalize_event_id(node_id)

        global_counts = get_global_system_counts(self.subgraphs)
        balance_mult = self.scorer.get_balance_multiplier(node_type, global_counts)
        final_energy = incoming_energy * balance_mult
        
        # if final_energy < 0.1: return False

        is_newly_created = False
        target_node = subgraph.get_node(normalized_id)

        if target_node:
            old_score = target_node.score
            new_score = self.scorer.accumulate_score(old_score, final_energy)
            
            # Update score even if small change (important for vector edges with damped energy)
            target_node.score = new_score
            subgraph.add_edge(Edge(parent_node.id, normalized_id, op_type, final_energy))
            
            if (new_score - old_score) < 0.001: 
                return False
        else:
            # Fetch full node data with metadata if not provided
            if node_data:
                new_node = node_data
                new_node.score = final_energy
            else:
                # Fetch from database
                if node_type == 'event':
                    new_node = self.kg.get_event_node_by_id(node_id)
                elif node_type == 'object':
                    new_node = self.kg.get_object_node_by_id(node_id)
                else:
                    new_node = Node(id=normalized_id, type=node_type, score=final_energy)
                new_node.score = final_energy
            
            subgraph.add_node(new_node)
            subgraph.add_edge(Edge(parent_node.id, normalized_id, op_type, final_energy))
            target_node = new_node
            is_newly_created = True

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
            # Use original_id if available for KG lookup
            event_id_for_lookup = new_node.metadata.get('original_id', new_node.id)
            kg_obj_ids = self.kg.get_objects_in_event(event_id_for_lookup)
            if kg_obj_ids:
                hub_size = len(kg_obj_ids)
                energy = self.scorer.calculate_energy_transfer(
                    new_node.score, 'event_to_object', 
                    current_iteration=self.current_iteration, hub_size=hub_size
                )
                for tid in kg_obj_ids: triangulation_tasks.append((tid, energy, 'event_to_object', 'object'))

            # 2. Check Context Events (History)
            ctx_evt_ids = self._get_context_neighbors(new_node.id, 'event_to_event')
            if ctx_evt_ids:
                energy = self.scorer.calculate_energy_transfer(
                    new_node.score, 'context_event_to_event', 
                    current_iteration=self.current_iteration
                )
                # Normalize context event IDs
                normalized_ctx_evt_ids = [self.kg.normalize_event_id(eid) for eid in ctx_evt_ids]
                for tid in normalized_ctx_evt_ids: triangulation_tasks.append((tid, energy, 'context_event_to_event', 'event'))

        elif new_node.type == 'object':
            # 1. Check KG Events (Structure)
            kg_evt_ids = self.kg.get_events_containing_object(new_node.id)
            if kg_evt_ids:
                global_count = self.kg.get_global_event_count_for_object(new_node.id)
                energy = self.scorer.calculate_energy_transfer(
                    new_node.score, 'object_to_event', 
                    current_iteration=self.current_iteration, global_uniqueness=global_count
                )
                # Normalize event IDs from KG lookup
                normalized_kg_evt_ids = [self.kg.normalize_event_id(eid) for eid in kg_evt_ids]
                for tid in normalized_kg_evt_ids: triangulation_tasks.append((tid, energy, 'object_to_event', 'event'))
            
            # 2. Check Context Relations (Object -> Object)
            ctx_rel_ids = self._get_context_neighbors(new_node.id, 'relation')
            if ctx_rel_ids:
                energy = self.scorer.calculate_energy_transfer(
                    new_node.score, 'context_relation', 
                    current_iteration=self.current_iteration
                )
                for tid in ctx_rel_ids: triangulation_tasks.append((tid, energy, 'context_relation', 'object'))

        for target_id, energy, op_type, target_type in triangulation_tasks:
            # For events, check both normalized and with possible suffix
            normalized_target_id = target_id
            if target_type == 'event':
                normalized_target_id = self.kg.normalize_event_id(target_id)
            
            if subgraph.has_node(normalized_target_id):
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
            
            # Compute keyword embedding for context storage/retrieval
            # self.current_context_key_embedding = self.kg.embedding_model.get_text_features(
            #     [self.current_keywords]
            # )[0]
            # print(f"  📝 Context keywords: {self.current_keywords}")
            
            # Search using keywords (text-based search)
            init_events = self.kg.search_events_by_description(keywords_response, top_k=5)
            init_objects = self.kg.search_objects_by_description(rewrite_entity_response, top_k=5)
        else:
            # Fallback: Use raw query text directly (no keyword extraction)
            print("⚠️  LLM not available, using direct query text for search")
            
            # Fallback context key
            self.current_keywords = query
            if self.kg.embedding_model:
                self.current_context_key_embedding = self.kg.embedding_model.get_text_features([query])[0]
            
            init_events = self.kg.search_events_by_description(query, top_k=5)
            init_objects = self.kg.search_objects_by_description(query, top_k=5)
        
        # Normalize event IDs in seed nodes
        for event in init_events:
            original_id = event.id
            event.id = self.kg.normalize_event_id(original_id)
            if 'original_id' not in event.metadata:
                event.metadata['original_id'] = original_id
        
        if init_events or init_objects:
            # Group seeds by connectivity (also returns connections to avoid re-querying KG)
            seed_groups, seed_connections = self._group_connected_seeds(init_events, init_objects)
            
            # Create separate subgraph for each connected group
            initial_counts = {'event': len(init_events), 'object': len(init_objects)}
            for group_id, (events, objects) in enumerate(seed_groups):
                sg = Subgraph(id=f"kg_seed_group_{group_id}")
            
                # Add nodes with proper scoring and mark as seeds
                for node in events + objects:
                    balance_mult = self.scorer.get_balance_multiplier(node.type, initial_counts)
                    initial_energy = node.score * self.scorer.trust_map['init'] 
                    node.score = self.scorer.accumulate_score(0.0, initial_energy * balance_mult)
                    node.metadata['is_seed'] = True  # Mark as seed for pruning protection
                    sg.add_node(node)
                
                # Add edges between connected seeds (reuse pre-computed connections)
                self._connect_seeds_in_group(sg, events, objects, seed_connections)
                
                self.subgraphs.append(sg)
                print(f"Created subgraph '{sg.id}' with {len(sg.nodes)} seeds ({len(events)} events, {len(objects)} objects)")

        # Retrieve context subgraphs from previous queries as REFERENCES (not active subgraphs)
        # Use KEYWORD embedding (not raw query embedding)
        if self.current_context_key_embedding is not None:
            self.retrieved_context_subgraphs = self.context_graph.search_context(
                self.current_context_key_embedding, top_k=3
            )
        else:
            self.retrieved_context_subgraphs = []
        
        if self.retrieved_context_subgraphs:
            print(f"📂 Retrieved {len(self.retrieved_context_subgraphs)} context subgraphs from past queries")
            for ctx_sg in self.retrieved_context_subgraphs:
                print(f"  - Context '{ctx_sg.id}': {len(ctx_sg.nodes)} nodes, {len(ctx_sg.edges)} edges")
        else:
            print("No context subgraphs retrieved (first query or no matches)")

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
            event_id_for_lookup = event.metadata.get('original_id', event.id)
            obj_ids = self.kg.get_objects_in_event(event_id_for_lookup)
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
                    event_id_for_lookup = event.metadata.get('original_id', event.id)
                    all_obj_ids = self.kg.get_objects_in_event(event_id_for_lookup)
                    energy = self.scorer.calculate_energy_transfer(
                        event.score, 'event_to_object', 
                        current_iteration=self.current_iteration, hub_size=len(all_obj_ids)
                    )
                    subgraph.add_edge(Edge(event.id, oid, 'seed_connection', energy))
        
        # Object -> Event edges
        for obj in objects:
            connected_evt_ids = seed_connections.get(obj.id, set())
            # Normalize event IDs
            normalized_evt_ids = [self.kg.normalize_event_id(eid) for eid in connected_evt_ids]
            for eid in normalized_evt_ids:
                if subgraph.has_node(eid):
                    global_count = self.kg.get_global_event_count_for_object(obj.id)
                    energy = self.scorer.calculate_energy_transfer(
                        obj.score, 'object_to_event', 
                        current_iteration=self.current_iteration, global_uniqueness=global_count
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
        merged_id = "_".join([sg.id for sg in subgraphs])
        merged = Subgraph(id=f"merged_{merged_id}")
        
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
        
        # 3. BFS Logic
        for start_node in all_events:
            queue = deque([(start_node.id, 0)]) # (current_id, dist)
            visited = {start_node.id}
            
            while queue:
                curr_id, dist = queue.popleft()
                
                # Constraint: Don't go too deep (e.g., max 3 hops: E->O->O->E)
                if dist > 3:
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
                            link_score = 0.8 * (0.9 ** dist)
                            
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
    # Shape-Aware Strategy Selection
    # ------------------------------------------------------------------
    def _calculate_mass_ratio(self, subgraph: Subgraph) -> float:
        """Calculate Events / Objects ratio"""
        events = len(subgraph.get_nodes_by_type('event'))
        objects = len(subgraph.get_nodes_by_type('object'))
        
        if objects == 0:
            return float('inf') if events > 0 else 1.0
        
        return events / objects
    
    def _calculate_density(self, subgraph: Subgraph) -> float:
        """Calculate graph density: 2 * Edges / (Nodes * (Nodes-1))"""
        nodes = len(subgraph.nodes)
        edges = len(subgraph.edges)
        
        if nodes <= 1:
            return 0.0
        
        max_possible_edges = nodes * (nodes - 1) / 2
        return (2 * edges) / (nodes * (nodes - 1)) if max_possible_edges > 0 else 0.0
    
    def _determine_strategy(self, subgraph: Subgraph) -> str:
        """
        Determine the strategy mode based on graph shape metrics.
        
        Returns one of: 'GROUNDING', 'BRIDGING', 'LEAPING', 'TRIANGULATION', 'BALANCED'
        """
        # Calculate shape metrics
        mass_ratio = self._calculate_mass_ratio(subgraph)
        density = self._calculate_density(subgraph)
        velocity = subgraph.get_velocity()
        
        # Priority-based mode selection
        
        # Mode A: GROUNDING (too many events, few objects - "ghostly verbs")
        if mass_ratio > 1.2:
            return 'GROUNDING'
        
        # Mode B: BRIDGING (too many objects, few events - "disconnected nouns")
        if mass_ratio < 0.6:
            return 'BRIDGING'
        
        # Mode C: LEAPING (dense cluster + stagnant - "explored fully, need to jump")
        # High density threshold: 0.3, Low velocity threshold: 0.05
        if density > 0.3 and velocity < 0.05:
            return 'LEAPING'
        
        # Mode D: TRIANGULATION (loose chain - "need cross-validation")
        # Low density threshold: 0.1
        if density < 0.1:
            return 'TRIANGULATION'
        
        # Default: BALANCED (healthy graph state)
        return 'BALANCED'
    
    # ------------------------------------------------------------------
    # Steiner Tree Pruning
    # ------------------------------------------------------------------
    def _should_prune_subgraph(self, subgraph: Subgraph) -> bool:
        """Adaptive check if pruning is needed for this subgraph"""
        config = self.pruning_config
        
        # Count nodes by type
        event_count = len(subgraph.get_nodes_by_type('event'))
        object_count = len(subgraph.get_nodes_by_type('object'))
        edge_count = len(subgraph.edges)
        total_nodes = len(subgraph.nodes)
        
        # Check size thresholds (AGGRESSIVE)
        if total_nodes > config['max_total_nodes']:
            return True
        if event_count > config['max_event_nodes']:
            return True
        if object_count > config['max_object_nodes']:
            return True
        if edge_count > config['max_edges']:
            return True
        
        # Check score saturation
        saturated_count = sum(
            1 for node in subgraph.nodes.values()
            if node.score >= config['score_saturation_threshold']
        )
        if saturated_count > config['saturation_count_trigger']:
            return True
        
        return False
    
    def _identify_terminals(self, subgraph: Subgraph) -> Set[str]:
        """
        Identify high-value nodes that MUST be kept using Rank-Based Selection.
        
        NEW LOGIC (Score-Saturation Proof):
        - Select TOP N nodes by score (budget enforcement)
        - Seeds are always protected as "Honorary Terminals"
        - Guarantees exact budget compliance regardless of score distribution
        """
        config = self.pruning_config
        terminals = set()
        
        # Separate nodes by type
        events = [n for n in subgraph.nodes.values() if n.type == 'event']
        objects = [n for n in subgraph.nodes.values() if n.type == 'object']
        
        # Identify seeds (honorary terminals - always protected)
        seed_events = [n for n in events if n.metadata.get('is_seed', False)]
        seed_objects = [n for n in objects if n.metadata.get('is_seed', False)]
        
        for seed in seed_events + seed_objects:
            terminals.add(seed.id)
        
        # Rank-based selection: TOP N non-seed nodes by score
        non_seed_events = [n for n in events if not n.metadata.get('is_seed', False)]
        non_seed_objects = [n for n in objects if not n.metadata.get('is_seed', False)]
        
        # Sort by score descending
        non_seed_events.sort(key=lambda n: n.score, reverse=True)
        non_seed_objects.sort(key=lambda n: n.score, reverse=True)
        
        # Take TOP N (budget enforcement)
        max_event_budget = config['max_event_nodes']
        max_object_budget = config['max_object_nodes']
        
        # Select top non-seeds (note: seeds don't count against budget)
        for node in non_seed_events[:max_event_budget]:
            terminals.add(node.id)
        
        for node in non_seed_objects[:max_object_budget]:
            terminals.add(node.id)
        
        return terminals
    
    def _find_steiner_nodes(self, subgraph: Subgraph, terminals: Set[str]) -> Set[str]:
        """
        Find weak nodes that are critical bridges between terminals (optimized for speed).
        
        NEW: Caps Steiner nodes at 1.5x terminals to prevent graph explosion.
        """
        config = self.pruning_config
        steiner_nodes = set()
        node_count = len(subgraph.nodes)
        
        # Build NetworkX graph for analysis
        G = nx.Graph()
        for node_id in subgraph.nodes:
            G.add_node(node_id)
        for edge in subgraph.edges:
            G.add_edge(edge.source_id, edge.target_id)
        
        # Method 1: Find articulation points (critical bridges) - ALWAYS DO THIS (fast)
        if config['use_articulation_points']:
            try:
                articulation_points = set(nx.articulation_points(G))
                for ap in articulation_points:
                    if ap not in terminals:
                        steiner_nodes.add(ap)
            except:
                pass  # Graph might be empty or have issues
        
        # Method 2: Find nodes on shortest paths between terminals - SKIP IF GRAPH IS LARGE (slow)
        # Only do this for small graphs to avoid performance issues
        if (config['use_shortest_paths'] and 
            len(terminals) > 1 and 
            node_count <= config.get('shortest_path_node_limit', 300)):
            
            terminals_list = list(terminals)
            max_path_length = config['max_path_length']
            
            # Limit number of terminal pairs to check (avoid O(n^2) explosion)
            max_pairs = min(50, len(terminals_list) * (len(terminals_list) - 1) // 2)
            pairs_checked = 0
            
            for i, t1 in enumerate(terminals_list):
                if pairs_checked >= max_pairs:
                    break
                for t2 in terminals_list[i+1:]:
                    if pairs_checked >= max_pairs:
                        break
                    try:
                        path = nx.shortest_path(G, t1, t2)
                        if len(path) <= max_path_length + 2:  # +2 for endpoints
                            for node_id in path[1:-1]:  # Exclude endpoints
                                if node_id not in terminals:
                                    steiner_nodes.add(node_id)
                        pairs_checked += 1
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pairs_checked += 1
                        continue
        elif node_count > config.get('shortest_path_node_limit', 300):
            # Skip shortest paths on large graphs for performance
            pass
        
        # BUDGET CAP: Steiner nodes at most 1.5x terminals (prevents explosion)
        max_steiner_nodes = int(len(terminals) * 1.5)
        
        if len(steiner_nodes) > max_steiner_nodes:
            # Sort Steiner nodes by score and keep only top N
            steiner_node_objects = [subgraph.nodes[nid] for nid in steiner_nodes if nid in subgraph.nodes]
            steiner_node_objects.sort(key=lambda n: n.score, reverse=True)
            steiner_nodes = set(n.id for n in steiner_node_objects[:max_steiner_nodes])
        
        return steiner_nodes
    
    def _prune_subgraph_steiner(self, subgraph: Subgraph) -> Dict:
        """
        Prune subgraph using Steiner Tree approach.
        Returns statistics about the pruning operation.
        """
        config = self.pruning_config
        
        original_nodes = len(subgraph.nodes)
        original_edges = len(subgraph.edges)
        original_events = len(subgraph.get_nodes_by_type('event'))
        original_objects = len(subgraph.get_nodes_by_type('object'))
        
        # Step 1: Identify terminals (must keep)
        terminals = self._identify_terminals(subgraph)
        
        # Step 2: Find Steiner nodes (weak but critical bridges)
        steiner_nodes = self._find_steiner_nodes(subgraph, terminals)
        
        # Step 3: Determine nodes to keep
        keep_nodes = terminals.union(steiner_nodes)
        
        # Step 4: FORCED BINARY PRUNING (Score-Saturation Proof)
        # If NOT (terminal OR steiner) → DELETE (no score check!)
        nodes_to_remove = []
        for node_id in list(subgraph.nodes.keys()):
            if node_id not in keep_nodes:
                nodes_to_remove.append(node_id)
        
        # Step 5: Remove nodes (edges are automatically removed in Subgraph class)
        for node_id in nodes_to_remove:
            subgraph.remove_node(node_id)
        
        # Step 6: Clean up orphaned edges (if any)
        edges_to_remove = []
        for edge in subgraph.edges:
            if edge.source_id not in subgraph.nodes or edge.target_id not in subgraph.nodes:
                edges_to_remove.append(edge)
        for edge in edges_to_remove:
            subgraph.edges.remove(edge)
        
        # Compile statistics
        stats = {
            'original_nodes': original_nodes,
            'original_edges': original_edges,
            'original_events': original_events,
            'original_objects': original_objects,
            'terminals': len(terminals),
            'steiner_nodes': len(steiner_nodes),
            'removed_nodes': len(nodes_to_remove),
            'final_nodes': len(subgraph.nodes),
            'final_edges': len(subgraph.edges),
            'final_events': len(subgraph.get_nodes_by_type('event')),
            'final_objects': len(subgraph.get_nodes_by_type('object')),
            'pruning_ratio': len(nodes_to_remove) / original_nodes if original_nodes > 0 else 0
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
                event_count = len(subgraph.get_nodes_by_type('event'))
                object_count = len(subgraph.get_nodes_by_type('object'))
                edge_count = len(subgraph.edges)
                
                # Log why pruning was triggered
                reasons = []
                if node_count > config['max_total_nodes']:
                    reasons.append(f"nodes>{config['max_total_nodes']}")
                if event_count > config['max_event_nodes']:
                    reasons.append(f"events>{config['max_event_nodes']}")
                if object_count > config['max_object_nodes']:
                    reasons.append(f"objects>{config['max_object_nodes']}")
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
        Rank subgraphs by 'Answerability Score' and select the best ones.
        
        Args:
            max_total_nodes_budget: Total node budget across all selected subgraphs
            top_k: If specified, select at most this many subgraphs (e.g., top_k=1 for best only)
                   When top_k is set, it takes priority over budget constraints.
        
        Returns:
            List of selected subgraphs, ranked by score (best first)
        """
        scored_graphs = []
        
        for sg in self.subgraphs:
            score = self._calculate_answerability_score(sg)
            scored_graphs.append((sg, score))
            
            # Debug print with component breakdown
            events = len(sg.get_nodes_by_type('event'))
            objects = len(sg.get_nodes_by_type('object'))
            nodes = list(sg.nodes.values())
            
            # Calculate components for debugging
            confidence = sum(n.score for n in nodes) / len(nodes) if nodes else 0
            structure = self._calculate_density(sg)
            n_count = len(nodes)
            e_count = events
            o_count = objects
            
            # Mass calculation (same logic as in _calculate_answerability_score)
            max_budget = self.pruning_config['max_total_nodes']
            if n_count < 10:
                mass = n_count / 10
            else:
                mass = 0.7 + (0.3 * min(n_count, max_budget) / max_budget)
            
            # Balance calculation
            if e_count == 0 or o_count == 0:
                balance = 0.0
            else:
                ratio = e_count / o_count
                target_ratio = self.pruning_config['max_event_nodes'] / self.pruning_config['max_object_nodes']
                deviation = abs(ratio - target_ratio)
                balance = math.exp(-deviation * 2)
            
            print(f"  Subgraph '{sg.id}': Score={score:.4f}")
            print(f"    Nodes: {n_count} (E:{events}, O:{objects}, ratio:{ratio:.2f}) | Conf:{confidence:.3f}, Struct:{structure:.3f}, Mass:{mass:.3f}, Bal:{balance:.3f}")
        
        # Sort by Score Descending
        scored_graphs.sort(key=lambda x: x[1], reverse=True)
        
        selected_subgraphs = []
        current_node_count = 0
        
        for sg, score in scored_graphs:
            # Filter out empty or broken graphs
            if len(sg.nodes) < 2: 
                continue
            
            # Check top_k limit first (if specified)
            if top_k is not None and len(selected_subgraphs) >= top_k:
                break
            
            # Check budget constraint
            if current_node_count + len(sg.nodes) <= max_total_nodes_budget:
                selected_subgraphs.append(sg)
                current_node_count += len(sg.nodes)
            else:
                # If top_k is set and we haven't reached it, take the graph anyway (ignore budget)
                if top_k is not None and len(selected_subgraphs) < top_k:
                    selected_subgraphs.append(sg)
                    current_node_count += len(sg.nodes)
                    print(f"    ⚠️  Exceeded budget to satisfy top_k={top_k} requirement")
                # Otherwise skip this graph
                continue
                
        return selected_subgraphs

    def _calculate_answerability_score(self, subgraph: Subgraph) -> float:
        """
        Calculates a 0.0-1.0 score indicating how useful this subgraph is for RAG.
        Formula: 0.35*Confidence + 0.25*Structure + 0.25*Mass + 0.15*Balance
        
        NOTE: Pruning config already controls size limits (max 70 nodes).
        This function only ranks quality among pruned subgraphs.
        """
        nodes = list(subgraph.nodes.values())
        if not nodes: return 0.0
        
        n_count = len(nodes)
        e_count = len([n for n in nodes if n.type == 'event'])
        o_count = len([n for n in nodes if n.type == 'object'])
        
        # 1. CONFIDENCE (Avg Node Score) [0-1]
        # Measures relevance to query (higher scores = more relevant nodes)
        confidence = sum(n.score for n in nodes) / len(nodes)
        
        # 2. STRUCTURE (Density) [0-1]
        # Measures connectivity (higher = more cross-references, better narrative)
        structure = self._calculate_density(subgraph)
        
        # 3. MASS (Relative Completeness) [0-1]
        # Measures how complete the subgraph is relative to pruning budget
        # Since pruning caps at max_total_nodes=70, we score relative to that
        # This rewards graphs that use more of the available budget
        max_budget = self.pruning_config['max_total_nodes']
        
        # Linear score: more nodes = more comprehensive (up to budget)
        # Minimum threshold: at least 10 nodes for valid context
        if n_count < 10:
            mass = n_count / 10  # Penalty for very small graphs (< 10 nodes)
        else:
            # Score from 0.7 to 1.0 as we approach budget
            # 10 nodes = 0.7, budget nodes = 1.0
            mass = 0.7 + (0.3 * min(n_count, max_budget) / max_budget)
        
        # 4. BALANCE (Event/Object Ratio) [0-1]
        # Target ratio from pruning config: 15 events / 35 objects = 0.43
        # We want ratio close to this target
        if e_count == 0 or o_count == 0:
            balance = 0.0  # Broken graph (all one type)
        else:
            ratio = e_count / o_count
            target_ratio = self.pruning_config['max_event_nodes'] / self.pruning_config['max_object_nodes']
            
            # Gaussian-like penalty for deviation from target
            # Perfect at target_ratio (0.43), penalty as we deviate
            deviation = abs(ratio - target_ratio)
            balance = math.exp(-deviation * 2)  # Smooth decay
        
        # WEIGHTED SUM
        # Confidence: 35% - Quality of retrieved nodes
        # Structure: 25% - How well connected
        # Mass: 25% - How complete relative to budget
        # Balance: 15% - Event/Object ratio health
        final_score = (0.35 * confidence) + (0.25 * structure) + (0.25 * mass) + (0.15 * balance)
        
        return final_score

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
            final_selection = self.select_best_subgraphs(max_total_nodes_budget=300)
            print(f"✓ Selected {len(final_selection)} subgraphs within budget for Final Prompt.")
        else:  # 'all'
            final_selection = self.subgraphs
            print(f"✓ Using all {len(final_selection)} subgraphs (no filtering).")
        
        # In a real app, here you would format 'final_selection' into text for the LLM
        return "Aggregated Answer", final_selection

if __name__ == "__main__":
    print("Graph Engine Main Entry Point")