#!/usr/bin/env python3
"""
Automated benchmark script for AVA100 dataset.
Runs graph engine tests for all queries in citytour, ego, traffic, and wildlife datasets.
"""

import sys
import os
import json
import gc
from datetime import datetime
from pathlib import Path
import psutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.JinaCLIP import JinaCLIP
from llms.init_model import init_model
from ICDCS.graph_interfaces import KnowledgeGraphInterface, ContextGraphInterface
from ICDCS.graph_scorer import GraphScorer
from ICDCS.graph_engine import GraphEngine
from ICDCS.export_subgraph import export_subgraph_to_json, export_all_subgraphs
from ICDCS.final_answer import generate_question_list


class AVA100Benchmark:
    def __init__(self, base_data_dir="datas/AVA100", base_db_dir="AVA_cache/AVA100", output_dir="ava100_results", 
                 use_cache=True, memory_threshold_percent=80.0, max_total_nodes=50, port=8000, config=''):
        # Get project root directory (parent of ICDCS directory)
        project_root = Path(__file__).parent.parent
        self.base_data_dir = project_root / base_data_dir
        self.base_db_dir = project_root / base_db_dir
        self.output_dir = project_root / output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        self.memory_threshold = memory_threshold_percent
        self.max_total_nodes = max_total_nodes
        self.config = config
        # Dataset configurations
        # Order: ego first (video indices 1-2), then citytour (3-4), traffic (5-6), wildlife (7-8)
        self.datasets = ["ego", "citytour", "wildlife", "traffic"]
        
        # Reusable graph components (kept in memory until threshold exceeded)
        self.current_graph_engine = None
        self.current_video_key = None
        self.current_kg = None
        self.current_ctx = None
        self.current_scorer = None
        self.active_connection_aliases = []
        
        # Build mapping from video_index (1-8) to video_key by reading config.json files
        self.video_index_to_key = self._build_video_index_mapping()
        
        # Initialize models once (reuse across queries)
        print("=" * 80)
        print("INITIALIZING MODELS")
        print("=" * 80)
        self.embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        # self.llm = init_model('qwenvl', 1)
        print("✅ Models initialized\n")
        print(f"💾 Memory threshold: {self.memory_threshold}%")
        self.llm = init_model("qwenvl_vllm", num_gpus=1, model_type="Qwen/Qwen2.5-14B-Instruct-AWQ", port=port)
    
    def _build_video_index_mapping(self):
        """
        Build mapping from video_index (1-8) to video_key by reading config.json files.
        
        Returns:
            Dict mapping video_index (int) -> video_key (str)
            Example: {1: "ego1", 2: "ego2", 3: "citytour1", ...}
        """
        mapping = {}
        
        # Check video indices 1-8
        for video_index in range(1, 9):
            config_path = self.base_db_dir / str(video_index) / "config.json"
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        source_path = config.get('source_path', '')
                        
                        # Extract video_key from source_path
                        # Example: "datas/AVA100/videos/ego2.mp4" -> "ego2"
                        if source_path:
                            # Get filename without extension
                            filename = Path(source_path).stem
                            mapping[video_index] = filename
                            print(f"📋 Mapped video_index {video_index} -> {filename}")
                except Exception as e:
                    print(f"⚠️  Failed to read config.json for video_index {video_index}: {e}")
            else:
                print(f"⚠️  config.json not found for video_index {video_index} at {config_path}")
        
        print(f"✅ Built mapping for {len(mapping)} videos\n")
        return mapping
    
    def get_video_index_from_key(self, video_key):
        """
        Get video_index (1-8) for a given video_key by looking up the mapping.
        
        Args:
            video_key: e.g., "ego1", "citytour2", etc.
        
        Returns:
            video_index (int) or None if not found
        """
        for video_index, mapped_key in self.video_index_to_key.items():
            if mapped_key == video_key:
                return video_index
        return None
    
    def load_dataset_json(self, dataset_name):
        """Load JSON file for a specific dataset."""
        json_path = self.base_data_dir / f"{dataset_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_path}. Please check the path.")
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def get_kg_dir(self, video_index):
        """Get the Knowledge Graph directory for a specific video.
        
        Args:
            video_index: Video index (1-8) corresponding to folder in AVA_cache/AVA100/
        
        Returns:
            Path to the kg directory: AVA_cache/AVA100/{video_index}/kg
        """
        # AVA creates a 'kg' folder inside the video's work directory
        # Path structure: AVA_cache/AVA100/{video_index}/kg
        return str(self.base_db_dir / str(video_index) / "kg")
    
    def is_video_empty(self, kg_dir):
        """
        Check if a video's kg directory is empty (has no events or entities).
        
        Args:
            kg_dir: Path to the kg directory
        
        Returns:
            True if video is empty (no events or entities), False otherwise
        """
        vdb_events_path = Path(kg_dir) / "vdb_events.json"
        vdb_entities_path = Path(kg_dir) / "vdb_entities.json"
        
        # Check if files exist
        if not vdb_events_path.exists() or not vdb_entities_path.exists():
            return True
        
        try:
            # Check events
            with open(vdb_events_path, 'r') as f:
                events_data = json.load(f)
                events_list = events_data.get('data', [])
                if not events_list or len(events_list) == 0:
                    return True
            
            # Check entities
            with open(vdb_entities_path, 'r') as f:
                entities_data = json.load(f)
                entities_list = entities_data.get('data', [])
                if not entities_list or len(entities_list) == 0:
                    return True
            
            return False
        except (json.JSONDecodeError, KeyError, Exception) as e:
            # If we can't read the files, consider it empty
            return True
    
    def is_query_cached(self, video_key, question_id):
        """
        Check if a query has already been processed.
        
        Args:
            video_key: e.g., "citytour1"
            question_id: Question ID
        
        Returns:
            Tuple (is_cached: bool, cached_result: dict or None)
        """
        if not self.use_cache:
            return False, None
        
        # Check if best_subgraph.json exists
        best_path = self.output_dir / video_key / f"q{question_id}" / "best_subgraph.json"
        
        if best_path.exists():
            try:
                with open(best_path, 'r') as f:
                    cached_data = json.load(f)
                
                # Extract query metadata to reconstruct result
                query_meta = cached_data.get('query_metadata', {})
                
                cached_result = {
                    'video_key': video_key,
                    'question_id': question_id,
                    'query': query_meta.get('query', 'N/A'),
                    'options': query_meta.get('options', []),
                    'ground_truth_answer': query_meta.get('ground_truth_answer', 'N/A'),
                    'time_reference': query_meta.get('time_reference', 'N/A'),
                    'success': True,
                    'num_subgraphs': query_meta.get('num_total_subgraphs', 0),
                    'best_subgraph_nodes': cached_data.get('total_nodes', 0),
                    'best_subgraph_edges': cached_data.get('total_edges', 0),
                    'processing_time_seconds': query_meta.get('processing_time_seconds', 0),
                    'output_path': str(best_path),
                    'all_subgraphs_dir': str(best_path.parent),
                    'cached': True  # Flag to indicate this was loaded from cache
                }
                
                return True, cached_result
            except Exception as e:
                print(f"  ⚠️  Failed to load cache for {video_key} Q{question_id}: {e}")
                return False, None
        
        return False, None
    
    def get_memory_usage_percent(self):
        """Get current RAM usage as percentage."""
        return psutil.virtual_memory().percent
    
    def check_and_cleanup_if_needed(self):
        """Check memory usage and cleanup if over threshold."""
        mem_usage = self.get_memory_usage_percent()
        
        if mem_usage > self.memory_threshold:
            print(f"\n⚠️  Memory usage {mem_usage:.1f}% exceeds threshold {self.memory_threshold}%")
            print("🧹 Cleaning up graph engine instances...")
            self.cleanup_graph_components()
            
            # Force garbage collection
            gc.collect()
            
            new_mem_usage = self.get_memory_usage_percent()
            print(f"✅ Memory after cleanup: {new_mem_usage:.1f}%\n")
            return True
        return False
    
    def cleanup_graph_components(self):
        """Cleanup current graph engine and connections."""
        
        # Clear instances
        self.current_graph_engine = None
        self.current_kg = None
        self.current_ctx = None
        self.current_scorer = None
        self.current_video_key = None
    
    def get_or_create_graph_engine(self, video_key, kg_dir):
        """
        Get existing graph engine or create new one.
        Reuses instance if same video and memory is OK.
        
        Args:
            video_key: Video identifier (e.g., "citytour1")
            kg_dir: Knowledge Graph directory
        
        Returns:
            GraphEngine instance (reused or newly created)
        """
        # Check memory first
        mem_usage = self.get_memory_usage_percent()
        if mem_usage > self.memory_threshold and self.current_graph_engine:
            print(f"⚠️  Memory {mem_usage:.1f}% > {self.memory_threshold}% - forcing cleanup")
            self.cleanup_graph_components()
        
        # If different video, cleanup old components
        if self.current_video_key != video_key:
            if self.current_graph_engine:
                print(f"📦 Switching video: {self.current_video_key} → {video_key}")
                self.cleanup_graph_components()
        
        # Create new instance if needed
        if self.current_graph_engine is None:
            print(f"🔧 Initializing graph engine for {video_key}... (mem: {mem_usage:.1f}%)")
            
            
            # Initialize graph components (KnowledgeGraphInterface creates its own connections)
            self.current_kg = KnowledgeGraphInterface(
                working_dir=kg_dir,
                embedding_model=self.embedding_model,
                embedding_dim=768
            )
            
            # Context graph: video-specific to accumulate query history
            # self.current_ctx = ContextGraphInterface(
            #     db_path=f"database/{video_key}_context.db",
            #     embedding_dim=768
            # )
            # self.current_ctx = None
            # self.active_connection_aliases.append(f'milvus_{video_key}_context')
            if self.config == 'm4':
                self.current_scorer = GraphScorer(base_decay=0.8, alpha=0.7, beta=0.3) # M4
            else:
                self.current_scorer = GraphScorer(base_decay=0.8)
            self.current_graph_engine = GraphEngine(
                self.current_kg, 
                self.current_ctx, 
                self.current_scorer, 
                llm=self.llm,
                constrained_propagation=False, # M1
                top_k_events=15,
                top_k_objects=15,
                adaptive_threshold=False, # M3
                threshold_percentile=80,
                prize_based_seeds=False, # M7
                top_k_protected=5,
                max_total_nodes=self.max_total_nodes
            )
            self.current_video_key = video_key
            
            print(f"✅ Graph engine ready for {video_key}")
        else:
            print(f"♻️  Reusing graph engine for {video_key} (mem: {mem_usage:.1f}%)")
        
        return self.current_graph_engine
    
    def run_single_query(self, video_key, query, question_id, qa_data, kg_dir, max_iterations=10, postfix: str = ''):
        """
        Run graph engine for a single query using reusable engine instance.
        
        Args:
            video_key: e.g., "citytour1"
            query: The question text
            question_id: Question ID from dataset
            qa_data: Full question data including options, answer, time_reference
            kg_dir: Knowledge Graph directory
            max_iterations: Max graph exploration iterations
        
        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 80)
        print(f"PROCESSING: {video_key} - Question {question_id}")
        print("=" * 80)
        print(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
        
        try:
            # Get or create graph engine (memory-aware reuse)
            engine = self.get_or_create_graph_engine(video_key, kg_dir)
            
            # Run graph engine
            # Note: query_embedding parameter is kept for compatibility but not used internally
            query_embedding = self.embedding_model.get_text_features([query])[0]
            time_reference = qa_data.get('time_reference', 'N/A')
            start_time = datetime.now()
            answer, subgraphs = engine.search(query, query_embedding, max_iterations=max_iterations, time_reference=time_reference)
            end_time = datetime.now()
            
            # Create output directory
            output_subdir = self.output_dir / video_key / f"q{question_id}"
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Select best subgraph and get all ranked by score
            all_ranked = engine.select_best_subgraphs(top_k=len(subgraphs), max_total_nodes_budget=999999)
            best_subgraph = all_ranked[0] if all_ranked else None
            
            # Export all subgraphs in ranked order (best to worst)
            if all_ranked:
                print(f"\n📊 Exporting all {len(all_ranked)} subgraphs (ranked by Answerability Score)...")
                
                # Show ranking
                for i, sg in enumerate(all_ranked[:5]):  # Show top 5
                    score = engine._calculate_answerability_score(sg)
                    print(f"   #{i+1}: subgraph_{i} - Score: {score:.4f} ({len(sg.nodes)}N, {len(sg.edges)}E)")
                if len(all_ranked) > 5:
                    print(f"   ... and {len(all_ranked) - 5} more")
                
                summary_path = export_all_subgraphs(all_ranked, str(output_subdir), query=query, qa_data=qa_data)
                print(f"   All subgraphs exported to: {output_subdir}/")
            
            # Export STAGE 1 ONLY
            # Stage 1: Full exploration (best subgraph after iterative expansion + pruning)
            # Stages 2-4 will be generated on-demand by calculate_accuracy.py
            stage1_sg = getattr(engine, 'best_subgraph_stage1', None)
            
            # Save Stage 1 as "best_subgraph.json"
            if best_subgraph:
                best_path = output_subdir / f"best_subgraph{postfix}.json"
                
                # Export base subgraph data WITH pruning config
                best_data = export_subgraph_to_json(
                    best_subgraph, 
                    str(best_path),
                    pruning_config=engine.pruning_config  # Include pruning config for calculate_accuracy.py
                )
                
                # Add query metadata to the JSON
                with open(best_path, 'r') as f:
                    best_data = json.load(f)
                
                # Enrich with query information
                best_data['query_metadata'] = {
                    'video_key': video_key,
                    'question_id': question_id,
                    'query': query,
                    'options': qa_data.get('options', []),
                    'ground_truth_answer': qa_data.get('answer', 'N/A'),
                    'time_reference': qa_data.get('time_reference', 'N/A'),
                    'processing_time_seconds': (end_time - start_time).total_seconds(),
                    'max_iterations': max_iterations,
                    'num_total_subgraphs': len(all_ranked),
                    'note': 'Stage 1: Best subgraph after full exploration. Stages 2-4 generated on-demand by calculate_accuracy.py'
                }
                
                # Save enriched data
                with open(best_path, 'w') as f:
                    json.dump(best_data, f, indent=2)
                
                print(f"\n✅ Stage 1 subgraph saved to: {best_path}")
                print(f"   Nodes: {len(best_subgraph.nodes)}, Edges: {len(best_subgraph.edges)}")
                print(f"   Note: Stages 2-4 will be generated on-demand by calculate_accuracy.py")
            
            # Save detailed cache statistics to file
            cache_stats = engine._get_cache_statistics()
            cache_stats_path = output_subdir / f"cache_statistics{postfix}.json"
            with open(cache_stats_path, 'w') as f:
                json.dump(cache_stats, f, indent=2)
            print(f"📊 Cache statistics saved to: {cache_stats_path}")
            
            # Save iteration-by-iteration evaluation metrics
            iteration_metrics = engine.get_iteration_metrics()
            iteration_log_path = output_subdir / f"iteration_log{postfix}.json"
            log_data = {
                'video_key': video_key,
                'question_id': question_id,
                'query': query,
                'time_reference': time_reference,
                'iterations': iteration_metrics
            }
            
            # Always save the log, even if empty (to indicate evaluation was skipped)
            with open(iteration_log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            if iteration_metrics:
                print(f"📈 Iteration evaluation log saved to: {iteration_log_path} ({len(iteration_metrics)} iterations)")
            else:
                # Log was saved but empty - likely because time_reference was invalid
                if time_reference and time_reference.strip() not in ["N/A", "", "None", "None-None"]:
                    print(f"⚠️  Iteration log saved but empty (no iterations evaluated) to: {iteration_log_path}")
                else:
                    print(f"⚠️  Iteration log saved but empty (time_reference='{time_reference}' is invalid) to: {iteration_log_path}")
            
            # Compile results
            result = {
                'video_key': video_key,
                'question_id': question_id,
                'query': query,
                'options': qa_data.get('options', []),
                'ground_truth_answer': qa_data.get('answer', 'N/A'),
                'time_reference': qa_data.get('time_reference', 'N/A'),
                'success': True,
                'num_subgraphs': len(all_ranked),
                'best_subgraph_nodes': len(best_subgraph.nodes) if best_subgraph else 0,
                'best_subgraph_edges': len(best_subgraph.edges) if best_subgraph else 0,
                'processing_time_seconds': (end_time - start_time).total_seconds(),
                'output_path': str(output_subdir / f"best_subgraph{postfix}.json") if best_subgraph else None,
                'all_subgraphs_dir': str(output_subdir),
                'subgraphs_ranked': True,  # Indicate exported subgraphs are ranked
                'stage1_nodes': len(stage1_sg.nodes) if stage1_sg else 0,
                'note': 'Stage 1 only. Stages 2-4 generated on-demand by calculate_accuracy.py'
            }
            
            print(f"⏱️  Processing time: {result['processing_time_seconds']:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"\n❌ ERROR processing {video_key} Q{question_id}: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'video_key': video_key,
                'question_id': question_id,
                'query': query,
                'options': qa_data.get('options', []),
                'ground_truth_answer': qa_data.get('answer', 'N/A'),
                'time_reference': qa_data.get('time_reference', 'N/A'),
                'success': False,
                'error': str(e),
                'processing_time_seconds': 0
            }
        
        # Note: No finally block - connections are kept alive for reuse
        # Cleanup is handled by memory monitoring in get_or_create_graph_engine()
    
    def run_dataset(self, dataset_name, max_iterations=10, limit_per_video=None):
        """
        Run all queries for a specific dataset.
        
        Args:
            dataset_name: e.g., "citytour", "ego", "traffic", "wildlife"
            max_iterations: Max graph exploration iterations
            limit_per_video: Limit number of questions per video (for testing)
        
        Returns:
            List of results
        """
        print("\n" + "🎯" * 40)
        print(f"RUNNING DATASET: {dataset_name.upper()}")
        print("🎯" * 40 + "\n")
        
        # Load dataset JSON
        dataset_json = self.load_dataset_json(dataset_name)
        
        all_results = []
        
        # Process each video in the dataset
        for video_data in dataset_json:
            video_key = video_data['video_key']
            questions = video_data['qa']
            
            # Map video_key to video_index (1-8) using config.json mapping
            video_index = self.get_video_index_from_key(video_key)
            if video_index is None:
                print(f"⚠️  Skipping {video_key}: No video_index found in AVA_cache/AVA100/ (video not preprocessed)")
                continue
            
            # Check if KG directory exists
            kg_dir = self.get_kg_dir(video_index)
            if not os.path.exists(kg_dir):
                print(f"⚠️  Skipping {video_key} (video_index={video_index}): KG directory not found at {kg_dir}")
                continue
            
            # Check if video is empty (no events or entities)
            if self.is_video_empty(kg_dir):
                print(f"⚠️  Skipping {video_key} (video_index={video_index}): Video has no events or entities (empty data)")
                continue
            
            print(f"\n📹 Processing video: {video_key} (video_index={video_index}, {len(questions)} questions)")
            
            # Limit questions if specified (for testing)
            if limit_per_video:
                questions = questions[:limit_per_video]
                print(f"   (Limited to {limit_per_video} questions for testing)")
            
            # Process each question
            for qa in questions:
                question = qa["query"]
                options = qa["options"]
                concat_question = f"{question}\n{options[0]}\n{options[1]}\n{options[2]}\n{options[3]}"
                query = concat_question
                question_list = generate_question_list(query, "Reasoning", self.llm)
                question_list[0] = query
                question_id = qa['question_id']
                for idx, question in enumerate(question_list):
                    # Check cache first
                    # is_cached, cached_result = self.is_query_cached(video_key, question_id)
                    
                    # if is_cached:
                    #     print(f"\n✅ CACHED: {video_key} - Question {question_id}")
                    #     print(f"   Query: {query[:100]}..." if len(query) > 100 else f"   Query: {query}")
                    #     print(f"   Loaded from: {cached_result['output_path']}")
                    #     print(f"   Nodes: {cached_result['best_subgraph_nodes']}, Edges: {cached_result['best_subgraph_edges']}")
                    #     result = cached_result
                    # else:
                    # Process the query
                    result = self.run_single_query(
                        video_key=video_key,
                        query=question,
                        question_id=question_id,
                        qa_data=qa,
                        kg_dir=kg_dir, 
                        max_iterations=max_iterations,
                        postfix=f"_sup{idx}" if idx > 0 else ""
                    )
                    
                    # Note: options, ground_truth_answer, and time_reference are now included
                    # directly in run_single_query() and is_query_cached()
                    
                    all_results.append(result)
            
            # After processing all questions for this video, check memory
            # (don't force cleanup, just check - cleanup will happen naturally if needed)
            mem_usage = self.get_memory_usage_percent()
            print(f"\n📊 Completed {video_key}: {len(questions)} queries processed (mem: {mem_usage:.1f}%)")
        
        # After processing all videos in dataset, cleanup
        if self.current_graph_engine:
            print(f"\n🧹 Dataset complete - cleaning up graph components...")
            self.cleanup_graph_components()
        
        return all_results
    
    def run_all_datasets(self, max_iterations=20, limit_per_video=None):
        """
        Run all datasets in AVA100.
        
        Args:
            max_iterations: Max graph exploration iterations
            limit_per_video: Limit number of questions per video (for testing)
        """
        start_time = datetime.now()
        all_results = []
        
        for dataset_name in self.datasets:
            results = self.run_dataset(
                dataset_name=dataset_name,
                max_iterations=max_iterations,
                limit_per_video=limit_per_video
            )
            all_results.extend(results)
        
        end_time = datetime.now()
        
        # Final cleanup of any remaining instances
        if self.current_graph_engine:
            print(f"\n🧹 All datasets complete - final cleanup...")
            self.cleanup_graph_components()
        
        # Calculate cache statistics
        cached_count = sum(1 for r in all_results if r.get('cached', False))
        processed_count = len(all_results) - cached_count
        
        # Save summary
        summary = {
            'total_queries': len(all_results),
            'successful_queries': sum(1 for r in all_results if r['success']),
            'failed_queries': sum(1 for r in all_results if not r['success']),
            'cached_queries': cached_count,
            'processed_queries': processed_count,
            'total_time_seconds': (end_time - start_time).total_seconds(),
            'results': all_results
        }
        
        summary_path = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 80)
        print("🎉 BENCHMARK COMPLETE")
        print("=" * 80)
        print(f"Total queries: {summary['total_queries']}")
        print(f"Successful: {summary['successful_queries']}")
        print(f"Failed: {summary['failed_queries']}")
        print(f"Cached (reused): {summary['cached_queries']}")
        print(f"Processed (new): {summary['processed_queries']}")
        print(f"Total time: {summary['total_time_seconds'] / 60:.2f} minutes")
        print(f"Summary saved to: {summary_path}")
        print("=" * 80)
        
        return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AVA100 benchmark for graph engine")
    parser.add_argument('--dataset', type=str, choices=['citytour', 'ego', 'traffic', 'wildlife', 'all'],
                        default='all', help='Which dataset to run (default: all)')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Max graph exploration iterations (default: 10)')
    parser.add_argument('--limit-per-video', type=int, default=None,
                        help='Limit number of questions per video for testing (default: None)')
    parser.add_argument('--output-dir', type=str, default='ava100_results',
                        help='Output directory for results (default: ava100_results)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable cache and reprocess all queries (default: use cache)')
    parser.add_argument('--force-rerun', action='store_true',
                        help='Force rerun all queries (same as --no-cache)')
    parser.add_argument('--memory-threshold', type=float, default=80.0,
                        help='Memory usage threshold percent for cleanup (default: 80.0)')
    parser.add_argument('--max-total-nodes', type=int, default=50,
                        help='Maximum total nodes for graph engine (default: 50)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port for LLM (default: 8000)')
    parser.add_argument('--config', type=str, default='',
                        help='Config for LLM (default: "")')
    
    args = parser.parse_args()
    
    # Determine cache usage
    use_cache = not (args.no_cache or args.force_rerun)
    
    # Initialize benchmark
    benchmark = AVA100Benchmark(
        output_dir=args.output_dir, 
        use_cache=use_cache,
        memory_threshold_percent=args.memory_threshold,
        max_total_nodes=args.max_total_nodes,
        port=args.port,
        config=args.config
    )
    
    if not use_cache:
        print("⚠️  Cache disabled - all queries will be reprocessed\n")
    
    # Run benchmark
    if args.dataset == 'all':
        benchmark.run_all_datasets(
            max_iterations=args.max_iterations,
            limit_per_video=args.limit_per_video
        )
    else:
        results = benchmark.run_dataset(
            dataset_name=args.dataset,
            max_iterations=args.max_iterations,
            limit_per_video=args.limit_per_video
        )
        
        # Save results for single dataset
        summary_path = benchmark.output_dir / f"{args.dataset}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {summary_path}")


if __name__ == "__main__":
    main()

