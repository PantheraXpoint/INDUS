#!/usr/bin/env python3
"""
Export subgraphs to JSON with detailed metadata for inspection and debugging
"""

import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ICDCS.graph_interfaces import Subgraph, Node, Edge
from typing import List, Dict


def export_subgraph_to_json(subgraph: Subgraph, output_path: str = None, pruning_config: Dict = None) -> Dict:
    """
    Export a single subgraph to a detailed JSON structure.
    
    Returns a dictionary that can be saved to JSON with all details:
    - Nodes with full metadata
    - Edges with types and scores
    - Statistics and connectivity info
    - Pruning configuration (for post-processing)
    
    Args:
        subgraph: The subgraph to export
        output_path: Optional path to save JSON file
        pruning_config: Optional pruning configuration dict to save for later use
    """
    data = {
        'subgraph_id': subgraph.id,
        'timestamp': datetime.now().isoformat(),
        'statistics': {
            'total_nodes': len(subgraph.nodes),
            'total_edges': len(subgraph.edges),
            'event_count': len(subgraph.get_nodes_by_type('event')),
            'object_count': len(subgraph.get_nodes_by_type('object')),
        },
        'nodes': [],
        'edges': [],
        'edge_type_summary': {},
        'pruning_config': pruning_config if pruning_config else {}
    }
    
    # Export nodes with full details (SORTED BY SCORE DESCENDING)
    # This ensures consistency with top-k selection in final_answer.py
    sorted_nodes = sorted(subgraph.nodes.values(), key=lambda n: n.score, reverse=True)
    
    for node in sorted_nodes:
        node_data = {
            'id': node.id,
            'type': node.type,
            'score': float(node.score),
            'content': node.content,
            'metadata': {}
        }
        
        # Add metadata (convert non-serializable types)
        if node.metadata:
            for key, value in node.metadata.items():
                try:
                    # Try to serialize, skip if it fails
                    json.dumps(value)
                    node_data['metadata'][key] = value
                except (TypeError, ValueError):
                    node_data['metadata'][key] = str(value)
        
        data['nodes'].append(node_data)
    
    # Export edges with full details
    edge_types = {}
    for edge in subgraph.edges:
        edge_data = {
            'source_id': edge.source_id,
            'target_id': edge.target_id,
            'type': edge.type,
            'score': float(edge.score),
            'metadata': {}
        }
        
        # Get source and target types
        source_node = subgraph.get_node(edge.source_id)
        target_node = subgraph.get_node(edge.target_id)
        if source_node and target_node:
            edge_data['source_type'] = source_node.type
            edge_data['target_type'] = target_node.type
            edge_data['connection'] = f"{source_node.type}→{target_node.type}"
        
        # Add metadata
        if edge.metadata:
            for key, value in edge.metadata.items():
                try:
                    json.dumps(value)
                    edge_data['metadata'][key] = value
                except (TypeError, ValueError):
                    edge_data['metadata'][key] = str(value)
        
        data['edges'].append(edge_data)
        
        # Count edge types
        edge_types[edge.type] = edge_types.get(edge.type, 0) + 1
    
    # Add edge type summary
    data['edge_type_summary'] = edge_types
    
    # Nodes are already sorted by score (descending) from the loop above
    # This order matches the top-k selection logic used in:
    # 1. _prune_subgraph_topk() in graph_engine.py
    # 2. final_answer.py when it slices nodes[:N]
    # DO NOT re-sort here to maintain score-based order
    
    # Sort edges by type for readability
    data['edges'].sort(key=lambda x: x['type'])
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        display_id = (subgraph.id[:50] + '...') if len(subgraph.id) > 50 else subgraph.id
        print(f"📄 Exported subgraph '{display_id}' to {output_path}")
    
    return data


def export_all_subgraphs(subgraphs: List[Subgraph], output_dir: str, query: str = None, qa_data: dict = None):
    """
    Export all subgraphs to JSON files in a directory.
    Creates one file per subgraph plus a summary file.
    
    Args:
        subgraphs: List of subgraphs to export
        output_dir: Directory to save files
        query: Query text
        qa_data: Optional dictionary containing 'options', 'answer', 'time_reference', etc.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export individual subgraphs
    exported_files = []
    all_subgraphs_data = []
    
    for i, subgraph in enumerate(subgraphs):
        # Create simple filename (avoid long subgraph IDs from merge history)
        filename = f"subgraph_{i}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Export
        data = export_subgraph_to_json(subgraph, filepath)
        exported_files.append(filename)
        all_subgraphs_data.append(data)
    
    # Create summary file
    summary = {
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'num_subgraphs': len(subgraphs),
        'total_statistics': {
            'total_nodes': sum(len(sg.nodes) for sg in subgraphs),
            'total_edges': sum(len(sg.edges) for sg in subgraphs),
            'total_events': sum(len(sg.get_nodes_by_type('event')) for sg in subgraphs),
            'total_objects': sum(len(sg.get_nodes_by_type('object')) for sg in subgraphs),
        },
        'subgraph_files': exported_files,
        'subgraphs_summary': []
    }
    
    # Add QA data if provided
    if qa_data:
        summary['options'] = qa_data.get('options', [])
        summary['ground_truth_answer'] = qa_data.get('answer', 'N/A')
        summary['time_reference'] = qa_data.get('time_reference', 'N/A')
    
    # Add per-subgraph summary
    for i, sg in enumerate(subgraphs):
        summary['subgraphs_summary'].append({
            'index': i,
            'id': sg.id,
            'file': exported_files[i],
            'nodes': len(sg.nodes),
            'edges': len(sg.edges),
            'events': len(sg.get_nodes_by_type('event')),
            'objects': len(sg.get_nodes_by_type('object'))
        })
    
    # Save summary
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Export Summary:")
    print(f"   Total Subgraphs: {len(subgraphs)}")
    print(f"   Output Directory: {output_dir}")
    print(f"   Summary File: {summary_path}")
    print(f"   Individual Files: {len(exported_files)}")
    
    return summary_path


def export_comparison_report(subgraphs: List[Subgraph], baseline_results: List[Dict], 
                             output_path: str, query: str = None):
    """
    Export a comparison report between graph engine results and baseline (tri_view_retrieval).
    """
    # Graph engine stats
    ge_events = set()
    ge_objects = set()
    for sg in subgraphs:
        ge_events.update([n.id for n in sg.get_nodes_by_type('event')])
        ge_objects.update([n.id for n in sg.get_nodes_by_type('object')])
    
    # Baseline stats
    baseline_events = set([str(r['event_id'][0]) for r in baseline_results])
    baseline_objects = set()
    for result in baseline_results:
        baseline_objects.update([str(e['id']) for e in result['entities']])
    
    # Create comparison report
    report = {
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'graph_engine': {
            'num_subgraphs': len(subgraphs),
            'total_nodes': sum(len(sg.nodes) for sg in subgraphs),
            'total_edges': sum(len(sg.edges) for sg in subgraphs),
            'unique_events': len(ge_events),
            'unique_objects': len(ge_objects),
            'event_ids': sorted(list(ge_events)),
            'object_ids': sorted(list(ge_objects))
        },
        'baseline_tri_view': {
            'num_results': len(baseline_results),
            'unique_events': len(baseline_events),
            'unique_objects': len(baseline_objects),
            'event_ids': sorted(list(baseline_events)),
            'object_ids': sorted(list(baseline_objects))
        },
        'comparison': {
            'events': {
                'graph_engine_only': sorted(list(ge_events - baseline_events)),
                'baseline_only': sorted(list(baseline_events - ge_events)),
                'common': sorted(list(ge_events & baseline_events)),
                'graph_engine_expansion': len(ge_events) - len(baseline_events)
            },
            'objects': {
                'graph_engine_only': sorted(list(ge_objects - baseline_objects)),
                'baseline_only': sorted(list(baseline_objects - ge_objects)),
                'common': sorted(list(ge_objects & baseline_objects)),
                'graph_engine_expansion': len(ge_objects) - len(baseline_objects)
            }
        }
    }
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📈 Comparison Report saved to: {output_path}")
    print(f"   Events: {len(ge_events)} (GE) vs {len(baseline_events)} (baseline)")
    print(f"   Objects: {len(ge_objects)} (GE) vs {len(baseline_objects)} (baseline)")
    
    return report


def create_visualization_data(subgraphs: List[Subgraph], output_path: str):
    """
    Create a simplified JSON format optimized for visualization tools (e.g., D3.js, Cytoscape).
    """
    viz_data = {
        'nodes': [],
        'edges': [],
        'metadata': {
            'num_subgraphs': len(subgraphs),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    node_ids = set()
    
    for sg_idx, sg in enumerate(subgraphs):
        # Add nodes
        for node in sg.nodes.values():
            if node.id not in node_ids:
                viz_data['nodes'].append({
                    'id': node.id,
                    'label': node.content[:30] if node.content else node.id,
                    'type': node.type,
                    'score': float(node.score),
                    'subgraph': sg.id,
                    'subgraph_index': sg_idx
                })
                node_ids.add(node.id)
        
        # Add edges
        for edge in sg.edges:
            viz_data['edges'].append({
                'source': edge.source_id,
                'target': edge.target_id,
                'type': edge.type,
                'score': float(edge.score),
                'subgraph': sg.id
            })
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, indent=2, ensure_ascii=False)
    
    print(f"🎨 Visualization data saved to: {output_path}")
    return viz_data


if __name__ == "__main__":
    print("Subgraph Export Module")
    print("Import and use:")
    print("  from AVA.export_subgraph import export_all_subgraphs")
    print("  export_all_subgraphs(subgraphs, 'output_dir', query='your query')")

