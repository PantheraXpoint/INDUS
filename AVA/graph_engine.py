import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Deque
from collections import deque
from graph_interfaces import Node, Edge, Subgraph, KnowledgeGraphInterface, ContextGraphInterface, get_global_system_counts
from graph_scorer import GraphScorer

class GraphEngine:
    def __init__(self, 
                 knowledge_graph: KnowledgeGraphInterface, 
                 context_graph: ContextGraphInterface, 
                 scorer: GraphScorer):
        self.kg = knowledge_graph
        self.context_graph = context_graph
        self.scorer = scorer
        self.subgraphs: List[Subgraph] = []
        
        # Session State
        self.current_query_embedding: Optional[np.ndarray] = None
        self.retrieved_context_subgraphs: List[Subgraph] = []

    def search(self, query: str, query_embedding: np.ndarray, max_iterations: int = 2):
        print(f"--- Starting Search: '{query}' ---")
        self.current_query_embedding = query_embedding
        
        # 1. Initialization (Create Anchor Subgraph)
        self._initial_exploration(query_embedding)
        
        # 2. Iterative Exploration Loop (The Wave)
        for i in range(max_iterations):
            print(f"--- Iteration {i+1} ---")
            for subgraph in self.subgraphs:
                # Process Snapshots (Active Lists)
                active_events = subgraph.get_nodes_by_type('event')
                active_objects = subgraph.get_nodes_by_type('object')
                
                # --- Stream A: Event Operations ---
                self._process_event_stream(active_events, subgraph)
                
                # --- Stream B: Object Operations ---
                self._process_object_stream(active_objects, subgraph)

        # 3. Post-Processing: Densify Graph (Deep Object Paths)
        for subgraph in self.subgraphs:
            self._finalize_subgraph_structure(subgraph)
        
        # 4. Final Aggregation
        return self._aggregation(query)

    # ------------------------------------------------------------------
    # Dispatchers
    # ------------------------------------------------------------------
    def _process_event_stream(self, events: List[Node], subgraph: Subgraph):
        for event in events:
            if event.score < 0.15: continue
            
            # Op 2: Event -> Object (Structure)
            self._op_event_to_object(event, subgraph)

            # Op 7: Event -> Event (Context)
            self._op_context_event_expansion(event, subgraph)
            
            # Op 6: Event -> Event (Vector/Bridge)
            if event.score > 0.6: 
                self._op_vector_event_to_event(event, subgraph)

    def _process_object_stream(self, objects: List[Node], subgraph: Subgraph):
        for obj in objects:
            if obj.score < 0.15: continue
            
            # Op 4: Object -> Event (Structure)
            self._op_object_to_event(obj, subgraph)
            
            # Op 5: Relation (Context Only)
            self._op_relation_expansion(obj, subgraph)
            
            # Op 3: Object -> Object (Vector)
            if obj.score > 0.7: 
                self._op_vector_object_to_object(obj, subgraph)

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
    def _op_event_to_object(self, source: Node, subgraph: Subgraph):
        kg_ids = set(self.kg.get_objects_in_event(source.id))
        ctx_ids = self._get_context_neighbors(source.id, 'event_to_object')
        obj_ids = kg_ids.union(ctx_ids)
        
        hub_size = len(obj_ids)
        if hub_size == 0: return

        raw_energy = self.scorer.calculate_energy_transfer(
            source_score=source.score, op_type='event_to_object', hub_size=hub_size
        )

        for oid in obj_ids:
            self._update_or_create_node(oid, 'object', raw_energy, subgraph, source, 'event_to_object')

    def _op_object_to_event(self, source: Node, subgraph: Subgraph):
        kg_ids = set(self.kg.get_events_containing_object(source.id))
        global_count = self.kg.get_global_event_count_for_object(source.id)
        
        raw_energy = self.scorer.calculate_energy_transfer(
            source_score=source.score, op_type='object_to_event', global_uniqueness=global_count
        )

        for eid in kg_ids:
            self._update_or_create_node(eid, 'event', raw_energy, subgraph, source, 'object_to_event')

    def _op_relation_expansion(self, source: Node, subgraph: Subgraph):
        target_ids = self._get_context_neighbors(source.id, 'relation')
        if not target_ids: return

        energy = self.scorer.calculate_energy_transfer(source.score, 'relation')
        
        for tid in target_ids:
            self._update_or_create_node(tid, 'object', energy, subgraph, source, 'relation')

    def _op_context_event_expansion(self, source: Node, subgraph: Subgraph):
        target_ids = self._get_context_neighbors(source.id, 'event_to_event')
        if not target_ids: return

        energy = self.scorer.calculate_energy_transfer(source.score, 'relation') 

        for tid in target_ids:
            self._update_or_create_node(tid, 'event', energy, subgraph, source, 'event_to_event_context')

    def _op_vector_object_to_object(self, source: Node, subgraph: Subgraph):
        if source.embedding is None: return
        results = self.kg.search_objects_by_embedding(source.embedding, top_k=5)
        
        for res_node in results:
            energy = self.scorer.calculate_energy_transfer(
                source.score, 'vector_object', node_embedding=res_node.embedding,
                parent_embedding=source.embedding, query_embedding=self.current_query_embedding
            )
            self._update_or_create_node(res_node.id, 'object', energy, subgraph, source, 'vector_object', node_data=res_node)

    def _op_vector_event_to_event(self, source: Node, subgraph: Subgraph):
        if source.embedding is None: return
        results = self.kg.search_events_by_embedding(source.embedding, top_k=5)
        
        for res_node in results:
            energy = self.scorer.calculate_energy_transfer(
                source.score, 'vector_event', node_embedding=res_node.embedding,
                parent_embedding=source.embedding, query_embedding=self.current_query_embedding
            )
            self._update_or_create_node(res_node.id, 'event', energy, subgraph, source, 'vector_event', node_data=res_node)

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

        global_counts = get_global_system_counts(self.subgraphs)
        balance_mult = self.scorer.get_balance_multiplier(node_type, global_counts)
        final_energy = incoming_energy * balance_mult
        
        if final_energy < 0.1: return False

        is_newly_created = False
        target_node = subgraph.get_node(node_id)

        if target_node:
            old_score = target_node.score
            new_score = self.scorer.accumulate_score(old_score, final_energy)
            if (new_score - old_score) < 0.01:
                target_node.score = new_score
                return False
            target_node.score = new_score
            subgraph.add_edge(Edge(parent_node.id, node_id, op_type, final_energy))
        else:
            if node_data: new_node = node_data; new_node.score = final_energy
            else: new_node = Node(id=node_id, type=node_type, score=final_energy)
            
            subgraph.add_node(new_node)
            subgraph.add_edge(Edge(parent_node.id, node_id, op_type, final_energy))
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
            kg_obj_ids = self.kg.get_objects_in_event(new_node.id)
            if kg_obj_ids:
                hub_size = len(kg_obj_ids)
                energy = self.scorer.calculate_energy_transfer(
                    new_node.score, 'event_to_object', hub_size=hub_size
                )
                for tid in kg_obj_ids: triangulation_tasks.append((tid, energy, 'event_to_object', 'object'))

            # 2. Check Context Events (History)
            ctx_evt_ids = self._get_context_neighbors(new_node.id, 'event_to_event')
            if ctx_evt_ids:
                energy = self.scorer.calculate_energy_transfer(new_node.score, 'relation')
                for tid in ctx_evt_ids: triangulation_tasks.append((tid, energy, 'event_to_event_context', 'event'))

        elif new_node.type == 'object':
            # 1. Check KG Events (Structure)
            kg_evt_ids = self.kg.get_events_containing_object(new_node.id)
            if kg_evt_ids:
                global_count = self.kg.get_global_event_count_for_object(new_node.id)
                energy = self.scorer.calculate_energy_transfer(
                    new_node.score, 'object_to_event', global_uniqueness=global_count
                )
                for tid in kg_evt_ids: triangulation_tasks.append((tid, energy, 'object_to_event', 'event'))
            
            # 2. Check Context Relations (Object -> Object)
            ctx_rel_ids = self._get_context_neighbors(new_node.id, 'relation')
            if ctx_rel_ids:
                energy = self.scorer.calculate_energy_transfer(new_node.score, 'relation')
                for tid in ctx_rel_ids: triangulation_tasks.append((tid, energy, 'relation', 'object'))

        for target_id, energy, op_type, target_type in triangulation_tasks:
            if subgraph.has_node(target_id):
                self._update_or_create_node(
                    node_id=target_id, node_type=target_type, 
                    incoming_energy=energy, 
                    subgraph=subgraph, parent_node=new_node, op_type=op_type
                )

    def _initial_exploration(self, query_emb: np.ndarray):
        init_events = self.kg.search_events_by_embedding(query_emb, top_k=5)
        init_objects = self.kg.search_objects_by_embedding(query_emb, top_k=5)
        
        if init_events or init_objects:
            sg_kg = Subgraph(id="kg_bootstrapped")
            initial_counts = {'event': len(init_events), 'object': len(init_objects)}
            
            for node_list in [init_events, init_objects]:
                for node in node_list:
                    balance_mult = self.scorer.get_balance_multiplier(node.type, initial_counts)
                    initial_energy = node.score * self.scorer.trust_map['init'] 
                    node.score = self.scorer.accumulate_score(0.0, initial_energy * balance_mult)
                    sg_kg.add_node(node)
            self.subgraphs.append(sg_kg)

        self.retrieved_context_subgraphs = self.context_graph.search_context(query_emb, top_k=3)
        for ctx_sg in self.retrieved_context_subgraphs:
            self.subgraphs.append(ctx_sg)

    def _aggregation(self, query):
        return "Aggregated Answer", self.subgraphs

if __name__ == "__main__":
    print("Graph Engine Main Entry Point")