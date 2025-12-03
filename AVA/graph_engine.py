import numpy as np
from typing import List, Dict, Set, Optional, Tuple
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
        
        # 3. Final Aggregation
        return self._aggregation(query)

    # ------------------------------------------------------------------
    # Dispatchers
    # ------------------------------------------------------------------
    def _process_event_stream(self, events: List[Node], subgraph: Subgraph):
        for event in events:
            # Efficiency Heuristic: Don't expand weak events
            if event.score < 0.15: continue
            
            # Op 2: Event -> Object (Structure)
            self._op_event_to_object(event, subgraph)
            
            # Op 6: Event -> Event (Vector/Bridge) - High Cost, High Threshold
            if event.score > 0.6: 
                self._op_vector_event_to_event(event, subgraph)

    def _process_object_stream(self, objects: List[Node], subgraph: Subgraph):
        for obj in objects:
            if obj.score < 0.15: continue
            
            # Op 4: Object -> Event (Structure)
            self._op_object_to_event(obj, subgraph)
            
            # Op 5: Relation
            self._op_relation_expansion(obj, subgraph)
            
            # Op 3: Object -> Object (Vector) - Risky
            if obj.score > 0.7: 
                self._op_vector_object_to_object(obj, subgraph)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_context_neighbors(self, source_id: str, edge_type: str) -> Set[str]:
        """Finds neighbors in retrieved Context Graphs (History)."""
        neighbors = set()
        for sg in self.retrieved_context_subgraphs:
            if sg.contains(source_id):
                for edge in sg.edges:
                    if edge.source_id == source_id and edge.type == edge_type:
                        neighbors.add(edge.target_id)
        return neighbors

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def _op_event_to_object(self, source: Node, subgraph: Subgraph):
        # 1. Retrieval (KG + Context)
        kg_ids = set(self.kg.get_objects_in_event(source.id))
        ctx_ids = self._get_context_neighbors(source.id, 'event_to_object')
        obj_ids = kg_ids.union(ctx_ids)
        
        hub_size = len(obj_ids)
        if hub_size == 0: return

        # 2. Energy
        raw_energy = self.scorer.calculate_energy_transfer(
            source_score=source.score, op_type='event_to_object', hub_size=hub_size
        )

        # 3. Update
        for oid in obj_ids:
            self._update_or_create_node(oid, 'object', raw_energy, subgraph, source, 'event_to_object')

    def _op_object_to_event(self, source: Node, subgraph: Subgraph):
        # 1. Retrieval
        kg_ids = set(self.kg.get_events_containing_object(source.id))
        # Logic to reverse context edges if needed (simplified here)
        evt_ids = kg_ids 
        
        global_count = self.kg.get_global_event_count_for_object(source.id)
        
        # 2. Energy
        raw_energy = self.scorer.calculate_energy_transfer(
            source_score=source.score, op_type='object_to_event', global_uniqueness=global_count
        )

        # 3. Update
        for eid in evt_ids:
            self._update_or_create_node(eid, 'event', raw_energy, subgraph, source, 'object_to_event')

    def _op_relation_expansion(self, source: Node, subgraph: Subgraph):
        rels = self.kg.get_relations_involving_object(source.id)
        energy = self.scorer.calculate_energy_transfer(source.score, 'relation')
        for rel in rels:
            self._update_or_create_node(rel['target_id'], 'object', energy, subgraph, source, 'relation')

    def _op_vector_object_to_object(self, source: Node, subgraph: Subgraph):
        if source.embedding is None: return
        results = self.kg.search_objects_by_embedding(source.embedding, top_k=5)
        
        for res_node in results:
            # Compass Rule / Plot Twist applied here
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
    # Core Update Logic
    # ------------------------------------------------------------------
    def _update_or_create_node(self, node_id, node_type, incoming_energy, subgraph, parent_node, op_type, node_data=None):
        """
        Handles Cycle Breaking (Delta), Accumulation, and ILT Trigger.
        """
        # 1. Backtrack Guard
        if node_id == parent_node.id: return False

        # 2. Thermostat (Using Global Counts)
        global_counts = get_global_system_counts(self.subgraphs)
        balance_mult = self.scorer.get_balance_multiplier(node_type, global_counts)
        final_energy = incoming_energy * balance_mult
        
        # 3. Pruning Threshold
        if final_energy < 0.1: return False

        is_newly_created = False
        target_node = subgraph.get_node(node_id)

        if target_node:
            # UPDATE EXISTING (Accumulation)
            old_score = target_node.score
            new_score = self.scorer.accumulate_score(old_score, final_energy)
            
            # Delta Guard: Stop propagation if update is tiny
            if (new_score - old_score) < 0.01:
                target_node.score = new_score
                return False
            
            target_node.score = new_score
            subgraph.add_edge(Edge(parent_node.id, node_id, op_type, final_energy))
        else:
            # CREATE NEW
            if node_data: new_node = node_data; new_node.score = final_energy
            else: new_node = Node(id=node_id, type=node_type, score=final_energy)
            
            subgraph.add_node(new_node)
            subgraph.add_edge(Edge(parent_node.id, node_id, op_type, final_energy))
            target_node = new_node
            is_newly_created = True

        # 4. Instant Local Triangulation (ILT)
        if is_newly_created:
            self._instant_local_triangulation(target_node, subgraph)
            
        return True

    def _instant_local_triangulation(self, new_node: Node, subgraph: Subgraph):
        """
        Connects new_node to existing neighbors immediately.
        Uses Flow-Based Edge Scoring.
        """
        if new_node.type == 'event':
            candidate_ids = self.kg.get_objects_in_event(new_node.id)
            target_type = 'object'; edge_type = 'event_to_object'
            
            # CRITICAL: Calculate Hub Penalty for energy conservation
            hub_size = len(candidate_ids) 
            if hub_size == 0: return
            transfer_energy = self.scorer.calculate_energy_transfer(
                source_score=new_node.score, op_type='event_to_object', hub_size=hub_size
            )
        else:
            candidate_ids = self.kg.get_events_containing_object(new_node.id)
            target_type = 'event'; edge_type = 'object_to_event'
            
            global_count = self.kg.get_global_event_count_for_object(new_node.id)
            transfer_energy = self.scorer.calculate_energy_transfer(
                source_score=new_node.score, op_type='object_to_event', global_uniqueness=global_count
            )

        # Iterate only through candidates that ALREADY EXIST in the subgraph
        for target_id in candidate_ids:
            if subgraph.has_node(target_id):
                
                # 1. Update Node First (Accumulate Score)
                self._update_or_create_node(
                    node_id=target_id, node_type=target_type, 
                    incoming_energy=transfer_energy, 
                    subgraph=subgraph, parent_node=new_node, op_type=edge_type
                )
                
                # 2. Update Edge Second (Record Flow Energy)
                subgraph.add_edge(Edge(
                    source_id=new_node.id, target_id=target_id, 
                    type=edge_type, score=transfer_energy
                ))

    def _initial_exploration(self, query_emb: np.ndarray):
        # A. KG Retrieval
        init_events = self.kg.search_events_by_embedding(query_emb, top_k=5)
        init_objects = self.kg.search_objects_by_embedding(query_emb, top_k=5)
        
        if init_events or init_objects:
            sg_kg = Subgraph(id="kg_bootstrapped")
            initial_counts = {'event': len(init_events), 'object': len(init_objects)}
            
            for node_list in [init_events, init_objects]:
                for node in node_list:
                    # Apply Thermostat logic to initial scores
                    balance_mult = self.scorer.get_balance_multiplier(node.type, initial_counts)
                    initial_energy = node.score * self.scorer.trust_map['init'] 
                    node.score = self.scorer.accumulate_score(0.0, initial_energy * balance_mult)
                    sg_kg.add_node(node)
            self.subgraphs.append(sg_kg)

        # B. Context Retrieval
        self.retrieved_context_subgraphs = self.context_graph.search_context(query_emb, top_k=3)
        for ctx_sg in self.retrieved_context_subgraphs:
            self.subgraphs.append(ctx_sg)

    def _aggregation(self, query):
        return "Aggregated Answer", self.subgraphs

# ==========================================
# 4. EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("Graph Engine Main Entry Point")
    # You can instantiate mocks here to test logic
    # kg = KnowledgeGraphInterface()
    # scorer = GraphScorer()
    # engine = GraphEngine(kg, None, scorer)