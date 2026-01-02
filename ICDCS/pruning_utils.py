#!/usr/bin/env python3
"""
Standalone pruning utilities for subgraphs.

This module contains pruning algorithms that can be applied to subgraphs
independently of the graph engine, allowing post-processing and experimentation
with different pruning strategies.
"""

import networkx as nx
from typing import Set, Dict, List
from graph_interfaces import Subgraph, Node


def prune_subgraph_steiner(subgraph: Subgraph, 
                           max_nodes: int = 20, 
                           pruning_config: Dict = None) -> Dict:
    """
    Prune subgraph using Steiner Tree approach with connectivity preservation.
    
    Strategy:
    1. Identify high-value terminal nodes (seeds + top-scoring nodes)
    2. Find bridge nodes (articulation points, shortest paths) that connect terminals
    3. Keep terminals + bridges, remove the rest
    
    Args:
        subgraph: Subgraph to prune (modified in-place)
        max_nodes: Maximum number of nodes to keep
        pruning_config: Optional configuration dict with keys:
            - use_articulation_points (bool): Use articulation point detection
            - use_shortest_paths (bool): Use shortest path bridge finding
            - shortest_path_node_limit (int): Max nodes for shortest path calculation
            - max_path_length (int): Maximum path length to consider
            - prize_based_seeds (bool): Use prize-based seed protection (M7)
            - top_k_protected (int): Number of top seeds to protect (M7)
    
    Returns:
        Dictionary with pruning statistics
    """
    if pruning_config is None:
        pruning_config = {
            'use_articulation_points': True,
            'use_shortest_paths': True,
            'shortest_path_node_limit': 50,
            'max_path_length': 3,
            'prize_based_seeds': False,
            'top_k_protected': 5
        }
    
    original_nodes = len(subgraph.nodes)
    original_edges = len(subgraph.edges)
    original_events = len(subgraph.get_nodes_by_type('event'))
    original_objects = len(subgraph.get_nodes_by_type('object'))
    
    # If already within budget, no pruning needed
    if original_nodes <= max_nodes:
        return {
            'original_nodes': original_nodes,
            'final_nodes': original_nodes,
            'removed_nodes': 0,
            'final_events': original_events,
            'final_objects': original_objects,
            'final_edges': original_edges,
            'original_edges': original_edges,
            'original_events': original_events,
            'original_objects': original_objects,
            'terminals': original_nodes,
            'steiner_nodes': 0,
            'pruning_ratio': 0.0
        }
    
    # Step 1: Identify Terminals
    terminals = _identify_terminals(subgraph, max_nodes, pruning_config)
    
    # Step 2: Calculate Remaining Budget for Bridges
    remaining_budget = max(0, max_nodes - len(terminals))
    
    # Step 3: Find Steiner nodes (capped by remaining budget)
    steiner_nodes = _find_steiner_nodes(subgraph, terminals, remaining_budget, pruning_config)
    
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
    edges_to_remove = [e for e in subgraph.edges 
                       if e.source_id not in subgraph.nodes or e.target_id not in subgraph.nodes]
    for edge in edges_to_remove:
        subgraph.edges.remove(edge)
    
    # Calculate statistics
    final_nodes = len(subgraph.nodes)
    final_edges = len(subgraph.edges)
    final_events = len(subgraph.get_nodes_by_type('event'))
    final_objects = len(subgraph.get_nodes_by_type('object'))
    
    stats = {
        'original_nodes': original_nodes,
        'original_edges': original_edges,
        'original_events': original_events,
        'original_objects': original_objects,
        'terminals': len(terminals),
        'steiner_nodes': len(steiner_nodes),
        'removed_nodes': len(nodes_to_remove),
        'final_nodes': final_nodes,
        'final_events': final_events,
        'final_objects': final_objects,
        'final_edges': final_edges,
        'pruning_ratio': len(nodes_to_remove) / original_nodes if original_nodes > 0 else 0
    }
    
    return stats


def prune_subgraph_topk(subgraph: Subgraph, max_nodes: int = 20) -> Dict:
    """
    Simple top-k pruning: Keep only the highest-scoring N nodes.
    
    This is a naive baseline that ignores connectivity. It simply takes
    the top-k nodes by score and discards edges to removed nodes.
    
    Args:
        subgraph: Subgraph to prune (modified in-place)
        max_nodes: Maximum number of nodes to keep
        
    Returns:
        Dictionary with pruning statistics
    """
    original_nodes = len(subgraph.nodes)
    original_edges = len(subgraph.edges)
    original_events = len(subgraph.get_nodes_by_type('event'))
    original_objects = len(subgraph.get_nodes_by_type('object'))
    
    # If already within budget, no pruning needed
    if original_nodes <= max_nodes:
        return {
            'original_nodes': original_nodes,
            'final_nodes': original_nodes,
            'removed_nodes': 0,
            'final_events': original_events,
            'final_objects': original_objects,
            'final_edges': original_edges,
            'original_edges': original_edges,
            'original_events': original_events,
            'original_objects': original_objects,
            'pruning_ratio': 0.0
        }
    
    # Sort all nodes by score (descending)
    all_nodes = list(subgraph.nodes.values())
    all_nodes.sort(key=lambda n: n.score, reverse=True)
    
    # Keep top-k nodes
    keep_nodes = set(n.id for n in all_nodes[:max_nodes])
    
    # Remove nodes not in keep set
    nodes_to_remove = [nid for nid in list(subgraph.nodes.keys()) if nid not in keep_nodes]
    
    for node_id in nodes_to_remove:
        subgraph.remove_node(node_id)
    
    # Clean edges (remove edges with missing endpoints)
    edges_to_remove = [e for e in subgraph.edges 
                      if e.source_id not in subgraph.nodes or e.target_id not in subgraph.nodes]
    for edge in edges_to_remove:
        subgraph.edges.remove(edge)
    
    # Calculate statistics
    final_nodes = len(subgraph.nodes)
    final_edges = len(subgraph.edges)
    final_events = len(subgraph.get_nodes_by_type('event'))
    final_objects = len(subgraph.get_nodes_by_type('object'))
    
    stats = {
        'original_nodes': original_nodes,
        'original_edges': original_edges,
        'original_events': original_events,
        'original_objects': original_objects,
        'removed_nodes': len(nodes_to_remove),
        'final_nodes': final_nodes,
        'final_events': final_events,
        'final_objects': final_objects,
        'final_edges': final_edges,
        'pruning_ratio': len(nodes_to_remove) / original_nodes if original_nodes > 0 else 0
    }
    
    return stats


def prune_subgraph_event_centric(subgraph: Subgraph, max_nodes: int = 20) -> Dict:
    """
    Event-centric pruning (mimics final_answer.py selection logic).
    
    Strategy:
    1. Select top-k events by score
    2. For each selected event, include associated objects (max 5 per event)
    3. Dynamically adjust to fit within max_nodes budget
    
    This mirrors the logic in final_answer.py:format_nodes_as_segments() where:
    - Events are the primary units
    - Objects are grouped under events with limit of 5
    
    Args:
        subgraph: Subgraph to prune (modified in-place)
        max_nodes: Maximum number of nodes to keep
        
    Returns:
        Dictionary with pruning statistics
    """
    original_nodes = len(subgraph.nodes)
    original_edges = len(subgraph.edges)
    original_events = len(subgraph.get_nodes_by_type('event'))
    original_objects = len(subgraph.get_nodes_by_type('object'))
    
    # If already within budget, no pruning needed
    if original_nodes <= max_nodes:
        return {
            'original_nodes': original_nodes,
            'final_nodes': original_nodes,
            'removed_nodes': 0,
            'final_events': original_events,
            'final_objects': original_objects,
            'final_edges': original_edges,
            'original_edges': original_edges,
            'original_events': original_events,
            'original_objects': original_objects,
            'pruning_ratio': 0.0
        }
    
    # Strategy: Event-centric selection (like final_answer.py)
    # 1. Sort events by score (descending)
    all_events = subgraph.get_nodes_by_type('event')
    all_events.sort(key=lambda n: n.score, reverse=True)
    
    # 2. Build event-to-objects mapping from edges
    event_to_objects = {}
    for edge in subgraph.edges:
        source_node = subgraph.get_node(edge.source_id)
        target_node = subgraph.get_node(edge.target_id)
        
        if source_node and target_node:
            # Event -> Object edges
            if source_node.type == 'event' and target_node.type == 'event':
                if edge.source_id not in event_to_objects:
                    event_to_objects[edge.source_id] = []
                event_to_objects[edge.source_id].append(target_node)
    
    # 3. Sort objects within each event by score (to pick best ones)
    for event_id, objects in event_to_objects.items():
        objects.sort(key=lambda n: n.score, reverse=True)
    
    # 4. Select events and their objects
    # Like final_answer.py: limit to 5 objects per event
    keep_nodes = set()
    max_objects_per_event = 20
    
    for event in all_events:
        # Check if we have budget for this event
        if len(keep_nodes) >= max_nodes:
            break
        
        # Add event
        keep_nodes.add(event.id)
        
        # Add associated objects (limit to 5, like final_answer.py)
        associated_objects = event_to_objects.get(event.id, [])
        objects_to_add = associated_objects[:max_objects_per_event]
        
        for obj in objects_to_add:
            if len(keep_nodes) >= max_nodes:
                break
            if obj.id not in keep_nodes:
                keep_nodes.add(obj.id)
    
    # 5. Remove nodes not in keep set
    nodes_to_remove = [nid for nid in list(subgraph.nodes.keys()) if nid not in keep_nodes]
    
    for node_id in nodes_to_remove:
        subgraph.remove_node(node_id)
    
    # 6. Clean edges (remove edges with missing endpoints)
    edges_to_remove = [e for e in subgraph.edges 
                      if e.source_id not in subgraph.nodes or e.target_id not in subgraph.nodes]
    for edge in edges_to_remove:
        subgraph.edges.remove(edge)
    
    # Calculate statistics
    final_nodes = len(subgraph.nodes)
    final_edges = len(subgraph.edges)
    final_events = len(subgraph.get_nodes_by_type('event'))
    final_objects = len(subgraph.get_nodes_by_type('object'))
    
    stats = {
        'original_nodes': original_nodes,
        'original_edges': original_edges,
        'original_events': original_events,
        'original_objects': original_objects,
        'removed_nodes': len(nodes_to_remove),
        'final_nodes': final_nodes,
        'final_events': final_events,
        'final_objects': final_objects,
        'final_edges': final_edges,
        'pruning_ratio': len(nodes_to_remove) / original_nodes if original_nodes > 0 else 0
    }
    
    return stats


# ============================================================================
# Helper Functions (Internal)
# ============================================================================

def _identify_terminals(subgraph: Subgraph, max_total: int, config: Dict) -> Set[str]:
    """
    Identify high-value nodes using Dynamic Ratio Selection.
    
    - Keeps Seeds (optionally with M7 prize-based protection).
    - Fills remaining budget with highest-scoring nodes (Event OR Object).
    - Does NOT enforce fixed event/object counts.
    """
    terminals = set()
    
    # ====================================================================
    # 1. Identify seeds with M7 Prize-Based Protection
    # ====================================================================
    all_seeds = [n for n in subgraph.nodes.values() if n.metadata.get('is_seed', False)]
    
    if config.get('prize_based_seeds', False) and len(all_seeds) > 0:
        # M7: Sort seeds by score descending
        all_seeds.sort(key=lambda n: n.score, reverse=True)
        
        # Protect only top-k seeds
        top_k_protected = config.get('top_k_protected', 5)
        num_protected = min(top_k_protected, len(all_seeds))
        protected_seeds = all_seeds[:num_protected]
        
        # Assign descending prizes to protected seeds
        for i, seed in enumerate(protected_seeds):
            prize = num_protected - i  # k, k-1, k-2, ..., 1
            
            # Boost score with prize (multiplicative)
            boost_factor = 1.0 + (prize / (num_protected * 2))
            seed.score *= boost_factor
            
            # Add to terminals
            terminals.add(seed.id)
        
        # Unprotected seeds are NOT added to terminals (can compete with other nodes)
    else:
        # Original behavior: protect ALL seeds
        for seed in all_seeds:
            terminals.add(seed.id)
    
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
    candidates = []
    for node in subgraph.nodes.values():
        # Skip nodes already in terminals (protected seeds)
        if node.id in terminals:
            continue
        # Include all other nodes (non-seeds + unprotected seeds)
        candidates.append(node)
    
    # Sort by score descending
    candidates.sort(key=lambda n: n.score, reverse=True)
    
    # Take top N candidates to fill available slots
    for node in candidates[:available_slots]:
        terminals.add(node.id)
    
    return terminals


def _find_steiner_nodes(subgraph: Subgraph, terminals: Set[str], 
                        max_count: int, config: Dict) -> Set[str]:
    """
    Find weak nodes that are critical bridges between terminals.
    
    Uses articulation points and shortest paths to identify connectivity-critical nodes.
    Strictly caps the number of Steiner nodes to 'max_count'.
    """
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
    if config.get('use_articulation_points', True):
        try:
            articulation_points = set(nx.articulation_points(G))
            for ap in articulation_points:
                if ap not in terminals:
                    steiner_nodes.add(ap)
        except:
            pass
    
    # Method 2: Shortest Paths (Gap Filling)
    if (config.get('use_shortest_paths', True) and 
        len(terminals) > 1 and 
        node_count <= config.get('shortest_path_node_limit', 300) and
        len(steiner_nodes) < max_count):
        
        terminals_list = list(terminals)
        max_path_length = config.get('max_path_length', 3)
        
        # Limit pairs
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
        steiner_node_objects = [subgraph.nodes[nid] for nid in steiner_nodes 
                               if nid in subgraph.nodes]
        steiner_node_objects.sort(key=lambda n: n.score, reverse=True)
        steiner_nodes = set(n.id for n in steiner_node_objects[:max_count])
    
    return steiner_nodes


# ============================================================================
# Standalone Test/Demo
# ============================================================================

if __name__ == "__main__":
    print("Pruning Utilities Module")
    print("="*60)
    print("This module provides standalone pruning functions:")
    print("  - prune_subgraph_steiner(): Connectivity-aware pruning")
    print("  - prune_subgraph_topk(): Simple top-k by score")
    print("  - prune_subgraph_event_centric(): Event-focused pruning")
    print("")
    print("Import and use:")
    print("  from ICDCS.pruning_utils import prune_subgraph_steiner")
    print("  stats = prune_subgraph_steiner(subgraph, max_nodes=20)")

