#!/usr/bin/env python3
"""
Test script for Graph Engine with real databases
Leverages tri_view_retrieval for initial exploration
"""

import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.JinaCLIP import JinaCLIP
from embeddings.object_search import SearchSystem
from llms.init_model import init_model
from AVA.utils import tri_view_retrieval
from AVA.graph_interfaces import KnowledgeGraphInterface, ContextGraphInterface, Node
from AVA.graph_scorer import GraphScorer
from AVA.graph_engine import GraphEngine
from AVA.analyze_subgraph import print_analysis_report
from AVA.export_subgraph import export_all_subgraphs, export_comparison_report, create_visualization_data
import numpy as np

def test_graph_engine_full_pipeline(query: str, db_paths: dict):
    """
    Full test pipeline:
    1. tri_view_retrieval (existing)
    2. Graph engine search (new)
    3. Comparison
    """
    print("=" * 80)
    print("GRAPH ENGINE FULL PIPELINE TEST")
    print("=" * 80)
    print(f"Query: '{query}'")
    
    # Initialize
    print("\n📦 Initializing components...")
    embedding_model = JinaCLIP("jinaai/jina-clip-v1")
    llm = init_model('qwenvl', 1)
    
    # # Search systems for tri_view_retrieval
    # object_search_system = SearchSystem(
    #     db_paths['object_db'],
    #     db_paths['sqlite_db'],
    #     embedding_model
    # )
    # event_search_system = SearchSystem(
    #     db_paths['event_db'],
    #     None,
    #     embedding_model
    # )
    
    # Graph components
    kg = KnowledgeGraphInterface(
        object_faiss_db_path=db_paths['object_db'],
        event_faiss_db_path=db_paths['event_db'],
        object_sqlite_db_path=db_paths['sqlite_db'],
        embedding_model=embedding_model,  # Pass embedding model for text search
        embedding_dim=768
    )
    # Context graph: stores past query subgraphs for retrieval
    ctx = ContextGraphInterface(
        db_path="database/test_context.db",
        embedding_dim=768
    )
    scorer = GraphScorer()
    engine = GraphEngine(kg, ctx, scorer, llm=llm)  # Pass LLM for keyword extraction
    
    # # Step 1: Run tri_view_retrieval (baseline)
    # print("\n" + "=" * 80)
    # print("STEP 1: Running tri_view_retrieval (Baseline)")
    # print("=" * 80)
    
    # search_results = tri_view_retrieval(
    #     query,
    #     event_search_system,
    #     object_search_system,
    #     llm,
    #     "both"
    # )
    
    # print(f"\n✅ tri_view_retrieval found {len(search_results)} events")
    # for i, result in enumerate(search_results[:3], 1):
    #     print(f"  {i}. Event {result['event_id']}: score={result['score']:.3f}, entities={len(result['entities'])}")
    
    # Step 2: Run Graph Engine
    print("\n" + "=" * 80)
    print("STEP 2: Running Graph Engine Search")
    print("=" * 80)
    
    query_embedding = embedding_model.get_text_features([query])[0]
    answer, subgraphs = engine.search(query, query_embedding, max_iterations=20)
    
    print(f"\n✅ Graph Engine generated {len(subgraphs)} subgraphs")
    
    # Select the best subgraph for final output
    print("\n🏆 Selecting Best Subgraph...")
    best_subgraphs = engine.select_best_subgraphs(top_k=1, max_total_nodes_budget=999999)
    best_subgraph = best_subgraphs[0] if best_subgraphs else None
    
    if best_subgraph:
        print(f"   Selected: '{best_subgraph.id}' with {len(best_subgraph.nodes)} nodes, {len(best_subgraph.edges)} edges")
    
    # # Step 3: Analyze subgraphs with comprehensive tools
    # print("\n" + "=" * 80)
    # print("STEP 3: Deep Analysis of Subgraphs")
    # print("=" * 80)
    
    # for i, sg in enumerate(subgraphs, 1):
    #     print(f"\n{'▼' * 80}")
    #     print(f"Subgraph {i} of {len(subgraphs)}")
    #     print(f"{'▼' * 80}")
    #     print_analysis_report(sg, detailed=True)
    
    # # Step 4: Comparison
    # print("\n" + "=" * 80)
    # print("STEP 4: Comparison")
    # print("=" * 80)
    
    # # Count unique nodes in graph engine
    # all_graph_events = set()
    # all_graph_objects = set()
    # for sg in subgraphs:
    #     all_graph_events.update([n.id for n in sg.get_nodes_by_type('event')])
    #     all_graph_objects.update([n.id for n in sg.get_nodes_by_type('object')])
    
    # # Count in tri_view
    # triview_events = set([str(r['event_id'][0]) for r in search_results])
    # triview_objects = set()
    # for result in search_results:
    #     triview_objects.update([str(e['id']) for e in result['entities']])
    
    # print(f"\n📈 Coverage Comparison:")
    # print(f"   tri_view_retrieval:")
    # print(f"     Events: {len(triview_events)}")
    # print(f"     Objects: {len(triview_objects)}")
    # print(f"\n   Graph Engine:")
    # print(f"     Events: {len(all_graph_events)} ({len(all_graph_events) - len(triview_events):+d} vs baseline)")
    # print(f"     Objects: {len(all_graph_objects)} ({len(all_graph_objects) - len(triview_objects):+d} vs baseline)")
    
    # # Step 5: Export to JSON for inspection
    # print("\n" + "=" * 80)
    # print("STEP 5: Exporting Results to JSON")
    # print("=" * 80)
    
    # # Create output directory
    # output_dir = "graph_engine_output"
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Export the BEST subgraph first (for downstream LLM consumption)
    # if best_subgraph:
    #     from AVA.export_subgraph import export_subgraph_to_json
    #     best_path = os.path.join(output_dir, "best_subgraph.json")
    #     export_subgraph_to_json(best_subgraph, best_path)
    #     print(f"✅ Best subgraph exported to: {best_path}")
    
    # # Export all subgraphs (for analysis/debugging)
    # summary_path = export_all_subgraphs(subgraphs, output_dir, query=query)
    
    # # Export comparison report
    # comparison_path = os.path.join(output_dir, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    # export_comparison_report(subgraphs, search_results, comparison_path, query=query)
    
    # # Export visualization data
    # viz_path = os.path.join(output_dir, f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    # create_visualization_data(subgraphs, viz_path)
    
    # print(f"\n✅ All results exported to: {output_dir}/")
    
    # print("\n" + "=" * 80)
    # print("✅ TEST COMPLETED")
    # print("=" * 80)
    
    return search_results, subgraphs

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python test_graph_engine.py <query> <object_db> <event_db> <sqlite_db>")
        print("\nExample:")
        print("  python AVA/test_graph_engine.py \\")
        print("    'person walking on the street' \\")
        print("    database/citytour1/object_embeddings.db \\")
        print("    database/citytour1/event_embeddings.db \\")
        print("    database/citytour1/tracked_objects.db")
        sys.exit(1)
    
    query = sys.argv[1]
    db_paths = {
        'object_db': sys.argv[2],
        'event_db': sys.argv[3],
        'sqlite_db': sys.argv[4]
    }
    
    test_graph_engine_full_pipeline(query, db_paths)