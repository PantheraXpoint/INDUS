#!/usr/bin/env python3
"""
Calculate overall retrieval accuracy from iteration logs.

This script scans all iteration_log.json files in the results directory
and extracts the final best subgraph overlap (retrieval accuracy).

NEW: Supports on-demand 4-stage pruning evaluation. If iteration_log.json
only contains Stage 1 data, this script will load the Stage 1 subgraph,
apply Stages 2-4 pruning, and calculate accuracies for all stages.

NEW: Supports baseline evaluation from sorted_SA_score_result.json files.
Each path in the baseline is evaluated separately (not combined).
Use --evaluate-baseline to compare your method against the baseline.
Use --baseline-only to evaluate baseline results independently (no benchmark results needed).

Usage:
    # Evaluate your method results
    python calculate_accuracy.py --results-dir ava100_results
    python calculate_accuracy.py --results-dir ava100_results --output accuracy_report.json
    python calculate_accuracy.py --results-dir ava100_results --llm-budget 20
    
    # Force recalculate overlaps (fixes old reports with accuracy > 1.0)
    python calculate_accuracy.py --results-dir ava100_results --force-recalculate --output fixed_report.json
    
    # Compare your method vs baseline
    python calculate_accuracy.py --results-dir ava100_results --evaluate-baseline --cache-dir AVA_cache
    
    # Evaluate baseline only
    python calculate_accuracy.py --baseline-only --cache-dir AVA_cache --output baseline_report.json
"""

import json
import argparse
import sys
import os
import copy
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ICDCS.graph_interfaces import Subgraph
from ICDCS.pruning_utils import prune_subgraph_steiner, prune_subgraph_topk, prune_subgraph_event_centric
from ICDCS.export_subgraph import export_subgraph_to_json

# Import evaluation functions from time_ref.py
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
        
        # Merge overlapping intervals within the reference range to avoid counting overlaps multiple times
        relevant_intervals = []
        for start, end in time_list:
            if end <= start:
                continue
            # Clip to reference range
            s = max(start, ref_start)
            e = min(end, ref_end)
            if e > s:
                relevant_intervals.append((s, e))
        
        if not relevant_intervals:
            return 0.0
        
        # Sort intervals and merge overlapping ones
        relevant_intervals.sort()
        merged = [relevant_intervals[0]]
        for current_start, current_end in relevant_intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                # Overlapping or adjacent, merge
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # Non-overlapping, add as new interval
                merged.append((current_start, current_end))
        
        # Calculate total covered length
        total_covered = sum(end - start for start, end in merged)
        
        # Cap at 1.0 (100% coverage maximum)
        return min(1.0, total_covered / ref_length) if ref_length > 0 else 0.0
    
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


def evaluate_baseline_overlap(sorted_sa_path: Path, time_reference: str) -> Dict:
    """
    Evaluate overlap for baseline method using sorted_SA_score_result.json.
    
    Evaluates EACH path separately and returns per-path accuracies.
    
    Args:
        sorted_sa_path: Path to sorted_SA_score_result.json
        time_reference: Time reference string from QA data
    
    Returns:
        Dictionary with per-path evaluation results
    """
    try:
        with open(sorted_sa_path, 'r') as f:
            sa_results = json.load(f)
    except Exception as e:
        return {
            'per_path_overlaps': [],
            'num_paths': 0,
            'error': str(e)
        }
    
    if not sa_results or len(sa_results) == 0:
        return {
            'per_path_overlaps': [],
            'num_paths': 0,
            'error': 'No SA results found'
        }
    
    # Evaluate each path separately
    per_path_overlaps = []
    
    for path_idx, result in enumerate(sa_results):
        frame_durations = result.get('frame_durations', [])
        
        if not frame_durations:
            per_path_overlaps.append({
                'path_idx': path_idx,
                'overlap': None,
                'num_frames': 0,
                'depth': result.get('depth'),
                'action': result.get('action'),
                'error': 'No frame durations'
            })
            continue
        
        # Convert frame durations to time intervals for this path
        time_list = []
        for duration in frame_durations:
            if len(duration) == 2:
                start, end = duration
                if end > start:
                    time_list.append((int(start), int(end)))
        
        if not time_list:
            per_path_overlaps.append({
                'path_idx': path_idx,
                'overlap': None,
                'num_frames': len(frame_durations),
                'depth': result.get('depth'),
                'action': result.get('action'),
                'error': 'No valid frame durations'
            })
            continue
        
        # Evaluate overlap for this path
        overlap = overlap_reference_helper(time_reference, time_list)
        
        per_path_overlaps.append({
            'path_idx': path_idx,
            'overlap': overlap,
            'num_frames': len(frame_durations),
            'depth': result.get('depth'),
            'action': result.get('action'),
            'final_score': result.get('final_score', {}),
            'error': None
        })
    
    return {
        'per_path_overlaps': per_path_overlaps,
        'num_paths': len(sa_results),
        'error': None
    }


def evaluate_subgraph_overlap(subgraph: Subgraph, time_reference: str) -> Dict:
    """
    Evaluate overlap for a single subgraph.
    
    Args:
        subgraph: The subgraph to evaluate
        time_reference: Time reference string from QA data
    
    Returns:
        Dictionary with evaluation results (overlap, node counts, etc.)
    """
    time_list = []
    
    for node in subgraph.nodes.values():
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
            if 'durations' in node.metadata:
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
    result = {
        'subgraph_id': subgraph.id,
        'overlap': overlap,
        'num_nodes': len(subgraph.nodes),
        'num_events': len(subgraph.get_nodes_by_type('event')),
        'num_objects': len(subgraph.get_nodes_by_type('object')),
        'num_edges': len(subgraph.edges)
    }
    
    return result


def generate_stages_from_subgraph(subgraph_path: Path, 
                                   time_reference: str, 
                                   llm_budget: int = 20,
                                   pruning_config: Dict = None,
                                   save_to_disk: bool = False) -> Dict:
    """
    Generate 4-stage evaluation results from a Stage 1 subgraph JSON.
    
    This is the core function that enables on-demand pruning evaluation.
    It loads a Stage 1 subgraph, applies different pruning strategies,
    and returns accuracy metrics for all 4 stages.
    
    Args:
        subgraph_path: Path to best_subgraph.json (Stage 1)
        time_reference: Time reference from QA data
        llm_budget: Target node count for LLM stages (default: 20)
        pruning_config: Optional pruning configuration (uses defaults if None)
        save_to_disk: If True, save Stage 2/3/4 subgraphs to disk (default: False)
    
    Returns:
        Dict with stage1/2/3/4 accuracy results and metadata
    """
    # Load Stage 1 subgraph from JSON
    try:
        with open(subgraph_path, 'r') as f:
            stage1_data = json.load(f)
    except Exception as e:
        print(f"  ⚠️  Failed to load {subgraph_path}: {e}")
        return None
    
    # Reconstruct subgraph object
    try:
        stage1_subgraph = Subgraph.from_json(stage1_data)
    except Exception as e:
        print(f"  ⚠️  Failed to reconstruct subgraph from {subgraph_path}: {e}")
        return None
    
    # Use saved pruning_config if available, otherwise use defaults
    if pruning_config is None:
        pruning_config = stage1_data.get('pruning_config', {
            'use_articulation_points': True,
            'use_shortest_paths': True,
            'shortest_path_node_limit': 50,
            'max_path_length': 3,
            'prize_based_seeds': False,
            'top_k_protected': 5
        })
    
    # Stage 1: Evaluate original (full exploration)
    stage1_result = evaluate_subgraph_overlap(stage1_subgraph, time_reference)
    
    # If graph is already within budget, all stages are identical
    if len(stage1_subgraph.nodes) <= llm_budget:
        return {
            'stage1': stage1_result,
            'stage2': stage1_result,
            'stage3': stage1_result,
            'stage4': stage1_result,
            'selection_loss': 0.0,
            'note': f'Graph already within budget ({len(stage1_subgraph.nodes)} ≤ {llm_budget})'
        }
    
    # Stage 2: Steiner pruning for LLM
    stage2_subgraph = copy.deepcopy(stage1_subgraph)
    stage2_stats = prune_subgraph_steiner(stage2_subgraph, max_nodes=llm_budget, pruning_config=pruning_config)
    stage2_result = evaluate_subgraph_overlap(stage2_subgraph, time_reference)
    
    # Stage 3: Top-k pruning (naive baseline)
    stage3_subgraph = copy.deepcopy(stage1_subgraph)
    stage3_stats = prune_subgraph_topk(stage3_subgraph, max_nodes=llm_budget)
    stage3_result = evaluate_subgraph_overlap(stage3_subgraph, time_reference)
    
    # Stage 4: Event-centric pruning (final_answer.py logic)
    stage4_subgraph = copy.deepcopy(stage1_subgraph)
    stage4_stats = prune_subgraph_event_centric(stage4_subgraph, max_nodes=llm_budget)
    stage4_result = evaluate_subgraph_overlap(stage4_subgraph, time_reference)
    
    # Save pruned subgraphs to disk if requested
    if save_to_disk:
        output_dir = subgraph_path.parent
        
        # Save Stage 2 (Steiner)
        stage2_path = output_dir / "best_subgraph_stage2.json"
        stage2_data = export_subgraph_to_json(stage2_subgraph, str(stage2_path), pruning_config)
        # Add metadata
        with open(stage2_path, 'r') as f:
            stage2_data = json.load(f)
        stage2_data['pruning_metadata'] = {
            'method': 'steiner',
            'llm_budget': llm_budget,
            'pruning_stats': stage2_stats,
            'note': 'Stage 2: LLM-pruned with Steiner tree connectivity preservation'
        }
        with open(stage2_path, 'w') as f:
            json.dump(stage2_data, f, indent=2)
        
        # Save Stage 3 (Top-k)
        stage3_path = output_dir / "best_subgraph_stage3.json"
        stage3_data = export_subgraph_to_json(stage3_subgraph, str(stage3_path), pruning_config)
        # Add metadata
        with open(stage3_path, 'r') as f:
            stage3_data = json.load(f)
        stage3_data['pruning_metadata'] = {
            'method': 'topk',
            'llm_budget': llm_budget,
            'pruning_stats': stage3_stats,
            'note': 'Stage 3: Simple top-k pruning (naive baseline)'
        }
        with open(stage3_path, 'w') as f:
            json.dump(stage3_data, f, indent=2)
        
        # Save Stage 4 (Event-centric)
        stage4_path = output_dir / "best_subgraph_stage4.json"
        stage4_data = export_subgraph_to_json(stage4_subgraph, str(stage4_path), pruning_config)
        # Add metadata
        with open(stage4_path, 'r') as f:
            stage4_data = json.load(f)
        stage4_data['pruning_metadata'] = {
            'method': 'event_centric',
            'llm_budget': llm_budget,
            'pruning_stats': stage4_stats,
            'note': 'Stage 4: Event-centric pruning (final_answer.py logic)'
        }
        with open(stage4_path, 'w') as f:
            json.dump(stage4_data, f, indent=2)
        
        print(f"     💾 Saved pruned subgraphs to {output_dir}/")
        print(f"        - best_subgraph_stage2.json ({stage2_result['num_nodes']} nodes)")
        print(f"        - best_subgraph_stage3.json ({stage3_result['num_nodes']} nodes)")
        print(f"        - best_subgraph_stage4.json ({stage4_result['num_nodes']} nodes)")
    
    # Calculate selection loss and comparisons
    selection_loss = None
    if stage1_result['overlap'] is not None and stage2_result['overlap'] is not None:
        selection_loss = stage1_result['overlap'] - stage2_result['overlap']
    
    # Determine best method
    best_overlap = max(
        stage2_result['overlap'] or 0,
        stage3_result['overlap'] or 0,
        stage4_result['overlap'] or 0
    )
    best_methods = []
    if stage2_result['overlap'] == best_overlap:
        best_methods.append("Steiner")
    if stage3_result['overlap'] == best_overlap:
        best_methods.append("Top-k")
    if stage4_result['overlap'] == best_overlap:
        best_methods.append("Event-Centric")
    
    return {
        'stage1': stage1_result,
        'stage2': stage2_result,
        'stage3': stage3_result,
        'stage4': stage4_result,
        'selection_loss': selection_loss,
        'pruning_stats': {
            'stage2': stage2_stats,
            'stage3': stage3_stats,
            'stage4': stage4_stats
        },
        'best_methods': best_methods,
        'best_overlap': best_overlap,
        'note': 'Generated on-demand from Stage 1 subgraph'
    }


def extract_accuracy_from_log(log_path: Path, llm_budget: int = 20, enable_on_demand: bool = True, save_pruned_subgraphs: bool = False, evaluate_baseline: bool = False, cache_dir: Path = None, force_recalculate: bool = False) -> Dict:
    """
    Extract accuracy metrics from a single iteration_log.json file.
    
    Supports multiple formats:
    - NEW FORMAT: 'stage_1', 'stage_2', 'stage_3', and 'stage_4' iteration entries (four-stage evaluation)
      * Stage 1: Full exploration with Steiner pruning
      * Stage 2: LLM budget with Steiner pruning
      * Stage 3: LLM budget with simple top-k pruning (naive baseline)
      * Stage 4: LLM budget with event-centric pruning (final_answer.py logic)
    - OLD FORMAT: 'final_best_subgraph' iteration entry (single-stage)
    - VERY OLD FORMAT: Reconstructs from last iteration's subgraphs
    - ON-DEMAND MODE: If only Stage 1 exists, loads best_subgraph.json and generates stages 2-4 dynamically
    
    Args:
        log_path: Path to iteration_log.json
        llm_budget: Target node count for LLM stages (default: 20)
        enable_on_demand: If True, generate missing stages from best_subgraph.json (default: True)
        save_pruned_subgraphs: If True, save Stage 2/3/4 subgraphs to disk (default: False)
        evaluate_baseline: If True, also evaluate baseline method from sorted_SA_score_result.json (default: False)
        cache_dir: Directory containing AVA_cache (for baseline evaluation)
        force_recalculate: If True, recalculate overlaps from subgraphs instead of using pre-computed values (default: False)
    
    Returns:
        Dict with accuracy info, or None if no valid data found
    """
    try:
        with open(log_path, 'r') as f:
            data = json.load(f)
        
        # Extract metadata
        video_key = data.get('video_key', 'unknown')
        question_id = data.get('question_id', -1)
        query = data.get('query', '')
        time_reference = data.get('time_reference', 'N/A')
        
        iterations = data.get('iterations', [])
        
        if not iterations:
            return {
                'video_key': video_key,
                'question_id': question_id,
                'query': query,
                'time_reference': time_reference,
                'status': 'no_iterations',
                'stage1_accuracy': None,
                'stage2_accuracy': None,
                'stage3_accuracy': None,
                'stage4_accuracy': None,
                'selection_loss': None
            }
        
        # Find final evaluations (there may be 1, 2, 3, or 4 depending on evaluation mode)
        # NEW FORMAT: Look for 'stage_1', 'stage_2', 'stage_3', and 'stage_4'
        stage1_entries = [x for x in iterations if x.get('iteration') == 'stage_1']
        stage2_entries = [x for x in iterations if x.get('iteration') == 'stage_2']
        stage3_entries = [x for x in iterations if x.get('iteration') == 'stage_3']
        stage4_entries = [x for x in iterations if x.get('iteration') == 'stage_4']
        
        # BACKWARD COMPATIBILITY: Try old 'final_best_subgraph' format
        if not stage1_entries and not stage2_entries and not stage3_entries and not stage4_entries:
            final_entries = [x for x in iterations if x.get('iteration') == 'final_best_subgraph']
        else:
            # Use new format: stage1 + stage2 + stage3 + stage4
            final_entries = stage1_entries + stage2_entries + stage3_entries + stage4_entries
        
        # BACKWARD COMPATIBILITY: If no final entries at all, try VERY OLD format
        # VERY OLD format: Find the last iteration and extract the best subgraph from all subgraphs
        if not final_entries:
            # Find last non-seed iteration
            regular_iterations = [x for x in iterations if x.get('iteration', '').startswith('iteration_') 
                                  and 'seed' not in x.get('iteration', '')]
            
            if not regular_iterations:
                # Try seed iteration as fallback
                regular_iterations = [x for x in iterations if 'seed' in x.get('iteration', '')]
            
            if regular_iterations:
                last_iter = regular_iterations[-1]
                subgraphs = last_iter.get('subgraphs', [])
                
                if subgraphs:
                    # Find best subgraph by max overlap (old format doesn't have answerability score)
                    best_subgraph = max(subgraphs, key=lambda sg: sg.get('overlap', 0) if sg.get('overlap') is not None else -1)
                    
                    # Create a synthetic final entry in new format
                    final_entries = [{
                        'iteration': 'final_best_subgraph',
                        'note': 'Reconstructed from old format (last iteration, best overlap)',
                        'total_subgraphs_in_pool': len(subgraphs),
                        'best_subgraph': best_subgraph,
                        'answerability_score': None
                    }]
        
        if not final_entries:
            return {
                'video_key': video_key,
                'question_id': question_id,
                'query': query,
                'time_reference': time_reference,
                'status': 'no_final_evaluation',
                'stage1_accuracy': None,
                'stage2_accuracy': None,
                'stage3_accuracy': None,
                'selection_loss': None
            }
        
        # Extract accuracies from explicit stage entries (more robust)
        stage1_accuracy = None
        stage2_accuracy = None
        stage3_accuracy = None
        stage4_accuracy = None
        selection_loss = None
        pruning_method_comparison = None
        all_methods_comparison = None
        stage1_nodes = stage1_events = stage1_objects = 0
        stage2_nodes = stage2_events = stage2_objects = 0
        stage3_nodes = stage3_events = stage3_objects = 0
        stage4_nodes = stage4_events = stage4_objects = 0
        
        # Stage 1: After full exploration
        # Try explicit stage_1 entries first (new format)
        if stage1_entries:
            best_sg = stage1_entries[0].get('best_subgraph', {})
            stage1_accuracy = best_sg.get('overlap')
            stage1_nodes = best_sg.get('num_nodes', 0)
            stage1_events = best_sg.get('num_events', 0)
            stage1_objects = best_sg.get('num_objects', 0)
        # Fallback: first entry in final_entries (backward compatibility)
        elif len(final_entries) >= 1:
            best_sg = final_entries[0].get('best_subgraph', {})
            stage1_accuracy = best_sg.get('overlap')
            stage1_nodes = best_sg.get('num_nodes', 0)
            stage1_events = best_sg.get('num_events', 0)
            stage1_objects = best_sg.get('num_objects', 0)
        
        # Stage 2: After LLM pruning (optional)
        # Try explicit stage_2 entries first (new format)
        if stage2_entries:
            best_sg = stage2_entries[0].get('best_subgraph', {})
            stage2_accuracy = best_sg.get('overlap')
            stage2_nodes = best_sg.get('num_nodes', 0)
            stage2_events = best_sg.get('num_events', 0)
            stage2_objects = best_sg.get('num_objects', 0)
            selection_loss = stage2_entries[0].get('selection_loss')
        # Fallback: second entry in final_entries (backward compatibility)
        elif len(final_entries) >= 2:
            best_sg = final_entries[1].get('best_subgraph', {})
            stage2_accuracy = best_sg.get('overlap')
            stage2_nodes = best_sg.get('num_nodes', 0)
            stage2_events = best_sg.get('num_events', 0)
            stage2_objects = best_sg.get('num_objects', 0)
            selection_loss = final_entries[1].get('selection_loss')
        else:
            # No stage 2 means graph was already within budget
            stage2_accuracy = stage1_accuracy
            stage2_nodes = stage1_nodes
            stage2_events = stage1_events
            stage2_objects = stage1_objects
            selection_loss = 0.0
        
        # Stage 3: After top-k pruning (optional - naive baseline)
        # Try explicit stage_3 entries first (new format)
        if stage3_entries:
            best_sg = stage3_entries[0].get('best_subgraph', {})
            stage3_accuracy = best_sg.get('overlap')
            stage3_nodes = best_sg.get('num_nodes', 0)
            stage3_events = best_sg.get('num_events', 0)
            stage3_objects = best_sg.get('num_objects', 0)
            pruning_method_comparison = stage3_entries[0].get('pruning_method_comparison')
        # Fallback: third entry in final_entries (backward compatibility)
        elif len(final_entries) >= 3:
            best_sg = final_entries[2].get('best_subgraph', {})
            stage3_accuracy = best_sg.get('overlap')
            stage3_nodes = best_sg.get('num_nodes', 0)
            stage3_events = best_sg.get('num_events', 0)
            stage3_objects = best_sg.get('num_objects', 0)
            pruning_method_comparison = final_entries[2].get('pruning_method_comparison')
        else:
            # No stage 3
            stage3_accuracy = None
            stage3_nodes = 0
            stage3_events = 0
            stage3_objects = 0
            pruning_method_comparison = None
        
        # Stage 4: After event-centric pruning (optional - final_answer.py logic)
        # Try explicit stage_4 entries first (new format)
        if stage4_entries:
            best_sg = stage4_entries[0].get('best_subgraph', {})
            stage4_accuracy = best_sg.get('overlap')
            stage4_nodes = best_sg.get('num_nodes', 0)
            stage4_events = best_sg.get('num_events', 0)
            stage4_objects = best_sg.get('num_objects', 0)
            all_methods_comparison = stage4_entries[0].get('all_methods_comparison')
        # Fallback: fourth entry in final_entries (backward compatibility)
        elif len(final_entries) >= 4:
            best_sg = final_entries[3].get('best_subgraph', {})
            stage4_accuracy = best_sg.get('overlap')
            stage4_nodes = best_sg.get('num_nodes', 0)
            stage4_events = best_sg.get('num_events', 0)
            stage4_objects = best_sg.get('num_objects', 0)
            all_methods_comparison = final_entries[3].get('all_methods_comparison')
        else:
            # No stage 4
            stage4_accuracy = None
            stage4_nodes = 0
            stage4_events = 0
            stage4_objects = 0
            all_methods_comparison = None
        
        # Determine which format was used (for debugging)
        if stage1_entries or stage2_entries or stage3_entries or stage4_entries:
            format_used = 'new_four_stage'
        elif len([x for x in iterations if x.get('iteration') == 'final_best_subgraph']) > 0:
            format_used = 'old_single_stage'
        else:
            format_used = 'reconstructed'
        
        # ========================================================================
        # FORCE RECALCULATION OF OVERLAPS FROM SUBGRAPHS
        # ========================================================================
        # If force_recalculate is True, reload subgraphs and recalculate overlaps
        # This is useful when the iteration_log.json contains overlaps calculated
        # with an old/buggy version of percentage_overlap function
        if force_recalculate and stage1_accuracy is not None and time_reference != 'N/A':
            subgraph_dir = log_path.parent
            best_subgraph_path = subgraph_dir / "best_subgraph.json"
            
            if best_subgraph_path.exists():
                try:
                    with open(best_subgraph_path, 'r') as f:
                        stage1_data = json.load(f)
                    stage1_subgraph = Subgraph.from_json(stage1_data)
                    
                    # Recalculate Stage 1 overlap
                    stage1_result = evaluate_subgraph_overlap(stage1_subgraph, time_reference)
                    stage1_accuracy = stage1_result['overlap']
                    print(f"  🔄 Recalculated {video_key} Q{question_id} Stage 1: {stage1_accuracy:.3f}")
                    
                    # If other stage subgraphs exist, recalculate them too
                    for stage_num, stage_var in [(2, 'stage2'), (3, 'stage3'), (4, 'stage4')]:
                        stage_path = subgraph_dir / f"best_subgraph_{stage_var}.json"
                        if stage_path.exists():
                            try:
                                with open(stage_path, 'r') as f:
                                    stage_data = json.load(f)
                                stage_subgraph = Subgraph.from_json(stage_data)
                                stage_result = evaluate_subgraph_overlap(stage_subgraph, time_reference)
                                
                                if stage_num == 2:
                                    stage2_accuracy = stage_result['overlap']
                                elif stage_num == 3:
                                    stage3_accuracy = stage_result['overlap']
                                elif stage_num == 4:
                                    stage4_accuracy = stage_result['overlap']
                                    
                                print(f"  🔄 Recalculated {video_key} Q{question_id} Stage {stage_num}: {stage_result['overlap']:.3f}")
                            except Exception as e:
                                print(f"     ⚠️  Failed to recalculate Stage {stage_num}: {e}")
                
                except Exception as e:
                    print(f"     ⚠️  Failed to recalculate overlaps for {video_key} Q{question_id}: {e}")
        
        # ========================================================================
        # ON-DEMAND STAGE GENERATION
        # ========================================================================
        # If we have Stage 1 but missing Stages 2/3/4, generate them dynamically
        # from the best_subgraph.json file
        if enable_on_demand and stage1_accuracy is not None:
            missing_stages = (stage2_accuracy is None or 
                            stage3_accuracy is None or 
                            stage4_accuracy is None)
            
            if missing_stages:
                # Try to load best_subgraph.json from same directory
                subgraph_dir = log_path.parent
                best_subgraph_path = subgraph_dir / "best_subgraph.json"
                
                if best_subgraph_path.exists():
                    print(f"  🔄 Generating missing stages on-demand for {video_key} Q{question_id}...")
                    
                    try:
                        # Generate all stages from Stage 1 subgraph
                        stage_results = generate_stages_from_subgraph(
                            best_subgraph_path, 
                            time_reference,
                            llm_budget=llm_budget,
                            save_to_disk=save_pruned_subgraphs
                        )
                        
                        if stage_results:
                            # Update missing accuracies
                            # Note: We always update from on-demand results since they are freshly generated
                            stage2_accuracy = stage_results['stage2']['overlap']
                            stage2_nodes = stage_results['stage2']['num_nodes']
                            stage2_events = stage_results['stage2']['num_events']
                            stage2_objects = stage_results['stage2']['num_objects']
                            
                            stage3_accuracy = stage_results['stage3']['overlap']
                            stage3_nodes = stage_results['stage3']['num_nodes']
                            stage3_events = stage_results['stage3']['num_events']
                            stage3_objects = stage_results['stage3']['num_objects']
                            
                            stage4_accuracy = stage_results['stage4']['overlap']
                            stage4_nodes = stage_results['stage4']['num_nodes']
                            stage4_events = stage_results['stage4']['num_events']
                            stage4_objects = stage_results['stage4']['num_objects']
                            
                            # Update selection loss if not present
                            if selection_loss is None:
                                selection_loss = stage_results.get('selection_loss')
                            
                            # Update comparison data
                            if all_methods_comparison is None:
                                all_methods_comparison = {
                                    'steiner_overlap': stage2_accuracy,
                                    'topk_overlap': stage3_accuracy,
                                    'eventcentric_overlap': stage4_accuracy,
                                    'best_method': stage_results.get('best_methods', []),
                                    'best_overlap': stage_results.get('best_overlap', 0)
                                }
                            
                            format_used = 'on_demand_generated'
                            print(f"     ✅ Generated stages 2-4 (S2: {stage2_accuracy:.3f}, S3: {stage3_accuracy:.3f}, S4: {stage4_accuracy:.3f})")
                    
                    except Exception as e:
                        print(f"     ⚠️  Failed to generate stages: {e}")
        
        # ========================================================================
        # BASELINE EVALUATION (sorted_SA_score_result.json)
        # ========================================================================
        baseline_per_path_overlaps = []
        baseline_num_paths = 0
        baseline_error = None
        
        if evaluate_baseline and time_reference != 'N/A':
            # Look for sorted_SA_score_result.json in multiple locations:
            # 1. Same directory as iteration_log.json (if they're co-located)
            sorted_sa_path = log_path.parent / "sorted_SA_score_result.json"
            
            # 2. In AVA_cache directory structure (AVA_cache/AVA100/video_id/questions/question_id/)
            if not sorted_sa_path.exists() and cache_dir:
                cache_path = cache_dir / "AVA100" / str(video_key) / "questions" / str(question_id) / "sorted_SA_score_result.json"
                if cache_path.exists():
                    sorted_sa_path = cache_path
            
            # 3. Try to auto-detect cache directory relative to results
            if not sorted_sa_path.exists() and not cache_dir:
                # Assume cache is at same level as results directory
                cache_base = log_path.parents[2] / ".." / "AVA_cache" / "AVA100"
                cache_path = cache_base / str(video_key) / "questions" / str(question_id) / "sorted_SA_score_result.json"
                cache_path = cache_path.resolve()
                if cache_path.exists():
                    sorted_sa_path = cache_path
            
            if sorted_sa_path.exists():
                baseline_result = evaluate_baseline_overlap(sorted_sa_path, time_reference)
                baseline_per_path_overlaps = baseline_result.get('per_path_overlaps', [])
                baseline_num_paths = baseline_result.get('num_paths', 0)
                baseline_error = baseline_result.get('error')
            else:
                baseline_error = f"sorted_SA_score_result.json not found"
                baseline_num_paths = 0
        
        return {
            'video_key': video_key,
            'question_id': question_id,
            'query': query,
            'time_reference': time_reference,
            'status': 'success',
            'format': format_used,
            'stage1_accuracy': stage1_accuracy,
            'stage2_accuracy': stage2_accuracy,
            'stage3_accuracy': stage3_accuracy,
            'stage4_accuracy': stage4_accuracy,
            'selection_loss': selection_loss,
            'pruning_method_comparison': pruning_method_comparison,
            'all_methods_comparison': all_methods_comparison,
            'stage1_nodes': stage1_nodes,
            'stage1_events': stage1_events,
            'stage1_objects': stage1_objects,
            'stage2_nodes': stage2_nodes,
            'stage2_events': stage2_events,
            'stage2_objects': stage2_objects,
            'stage3_nodes': stage3_nodes,
            'stage3_events': stage3_events,
            'stage3_objects': stage3_objects,
            'stage4_nodes': stage4_nodes,
            'stage4_events': stage4_events,
            'stage4_objects': stage4_objects,
            'baseline_per_path_overlaps': baseline_per_path_overlaps,
            'baseline_num_paths': baseline_num_paths,
            'baseline_error': baseline_error,
        }
        
    except Exception as e:
        print(f"⚠️  Error reading {log_path}: {e}")
        return None


def scan_baseline_only(cache_dir: Path) -> List[Dict]:
    """
    Scan AVA_cache directory for baseline evaluation only (no benchmark results needed).
    
    This reads sorted_SA_score_result.json files directly from the cache and evaluates them
    against time references from the AVA100 dataset JSON files.
    
    Args:
        cache_dir: Directory containing AVA_cache
    
    Returns:
        List of accuracy data dictionaries
    """
    all_results = []
    
    # Load AVA100 dataset to get time references
    dataset_files = {
        'ego': 'datas/AVA100/ego.json',
        'citytour': 'datas/AVA100/citytour.json',
        'wildlife': 'datas/AVA100/wildlife.json',
        'traffic': 'datas/AVA100/traffic.json'
    }
    
    dataset_map = {}  # video_key -> qa_list
    for dataset_type, json_path in dataset_files.items():
        if Path(json_path).exists():
            with open(json_path, 'r') as f:
                videos = json.load(f)
                for video in videos:
                    video_key = video.get('video_key')
                    dataset_map[video_key] = video.get('qa', [])
        else:
            print(f"⚠️  Warning: Dataset file not found: {json_path}")
    
    # Find all sorted_SA_score_result.json files in cache
    ava100_cache = cache_dir / "AVA100"
    if not ava100_cache.exists():
        print(f"❌ Error: AVA100 cache directory not found: {ava100_cache}")
        return []
    
    baseline_files = list(ava100_cache.rglob("sorted_SA_score_result.json"))
    print(f"Found {len(baseline_files)} baseline result files")
    
    for baseline_path in sorted(baseline_files):
        # Extract video_id and question_id from path
        # Path structure: AVA_cache/AVA100/{video_id}/questions/{question_id}/sorted_SA_score_result.json
        parts = baseline_path.parts
        try:
            ava100_idx = parts.index("AVA100")
            video_id = parts[ava100_idx + 1]
            question_id = int(parts[ava100_idx + 3])
        except (ValueError, IndexError):
            print(f"⚠️  Could not parse path: {baseline_path}")
            continue
        
        # Read config to get video_key
        config_path = baseline_path.parent.parent.parent / "config.json"
        if not config_path.exists():
            print(f"⚠️  Config not found for {baseline_path}")
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            video_key = config.get('source_path', '').split('/')[-1].split('.')[0]
        
        if video_key not in dataset_map:
            print(f"⚠️  Video key '{video_key}' not found in dataset")
            continue
        
        qa_list = dataset_map[video_key]
        if question_id >= len(qa_list):
            print(f"⚠️  Question {question_id} not found for video {video_key}")
            continue
        
        qa_item = qa_list[question_id]
        query = qa_item.get('question', qa_item.get('query', ''))
        time_reference = qa_item.get('time_reference', 'N/A')
        
        # Skip questions without time reference
        if time_reference == 'N/A' or not time_reference:
            continue
        
        # Evaluate baseline overlap (per-path)
        baseline_result = evaluate_baseline_overlap(baseline_path, time_reference)
        baseline_per_path_overlaps = baseline_result.get('per_path_overlaps', [])
        baseline_num_paths = baseline_result.get('num_paths', 0)
        baseline_error = baseline_result.get('error')
        
        if baseline_error:
            print(f"⚠️  Error evaluating {video_key} Q{question_id}: {baseline_error}")
            continue
        
        result = {
            'video_key': video_key,
            'question_id': question_id,
            'query': query,
            'time_reference': time_reference,
            'status': 'success',
            'format': 'baseline_only',
            'baseline_per_path_overlaps': baseline_per_path_overlaps,
            'baseline_num_paths': baseline_num_paths,
            'baseline_error': baseline_error,
            # Set stage accuracies to None (not applicable for baseline-only mode)
            'stage1_accuracy': None,
            'stage2_accuracy': None,
            'stage3_accuracy': None,
            'stage4_accuracy': None,
            'selection_loss': None,
            'stage1_nodes': 0,
            'stage1_events': 0,
            'stage1_objects': 0,
            'stage2_nodes': 0,
            'stage2_events': 0,
            'stage2_objects': 0,
            'stage3_nodes': 0,
            'stage3_events': 0,
            'stage3_objects': 0,
            'stage4_nodes': 0,
            'stage4_events': 0,
            'stage4_objects': 0,
        }
        
        all_results.append(result)
        # Show summary of per-path results
        valid_overlaps = [p['overlap'] for p in baseline_per_path_overlaps if p['overlap'] is not None]
        if valid_overlaps:
            avg_overlap = sum(valid_overlaps) / len(valid_overlaps)
            max_overlap = max(valid_overlaps)
            print(f"  ✓ {video_key} Q{question_id}: {len(valid_overlaps)} paths, avg={avg_overlap:.3f}, max={max_overlap:.3f}")
        else:
            print(f"  ⚠️  {video_key} Q{question_id}: No valid paths")
    
    return all_results


def scan_results_directory(results_dir: Path, llm_budget: int = 20, enable_on_demand: bool = True, save_pruned_subgraphs: bool = False, evaluate_baseline: bool = False, cache_dir: Path = None, force_recalculate: bool = False) -> List[Dict]:
    """
    Scan all iteration_log.json files in the results directory.
    
    Args:
        results_dir: Directory containing results
        llm_budget: Target node count for LLM stages (default: 20)
        enable_on_demand: Enable on-demand stage generation (default: True)
        save_pruned_subgraphs: Save Stage 2/3/4 subgraphs to disk (default: False)
        evaluate_baseline: Evaluate baseline method from sorted_SA_score_result.json (default: False)
        cache_dir: Directory containing AVA_cache (for baseline evaluation)
        force_recalculate: Recalculate overlaps from subgraphs instead of using pre-computed values (default: False)
    
    Returns:
        List of accuracy data dictionaries
    """
    all_results = []
    
    # Find all iteration_log.json files
    log_files = list(results_dir.rglob("iteration_log.json"))
    
    print(f"Found {len(log_files)} iteration log files")
    if enable_on_demand:
        print(f"📊 On-demand stage generation: ENABLED (LLM budget: {llm_budget} nodes)")
        if save_pruned_subgraphs:
            print(f"💾 Saving pruned subgraphs: ENABLED")
    else:
        print(f"📊 On-demand stage generation: DISABLED")
    
    if force_recalculate:
        print(f"🔄 Force recalculate overlaps: ENABLED (will reload subgraphs)")
    
    if evaluate_baseline:
        print(f"📊 Baseline evaluation: ENABLED")
        if cache_dir:
            print(f"📂 Cache directory: {cache_dir}")
    
    for log_path in sorted(log_files):
        accuracy_data = extract_accuracy_from_log(log_path, llm_budget=llm_budget, enable_on_demand=enable_on_demand, save_pruned_subgraphs=save_pruned_subgraphs, evaluate_baseline=evaluate_baseline, cache_dir=cache_dir, force_recalculate=force_recalculate)
        if accuracy_data:
            all_results.append(accuracy_data)
    
    return all_results


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate overall statistics from all results.
    
    Returns:
        Dictionary with aggregated statistics
    """
    # Filter valid results (have stage1_accuracy OR baseline_per_path_overlaps)
    valid_results = [r for r in results if r.get('stage1_accuracy') is not None or r.get('baseline_per_path_overlaps')]
    
    # Group by video
    by_video = defaultdict(list)
    for r in valid_results:
        by_video[r['video_key']].append(r)
    
    # Calculate overall stats
    stage1_accuracies = [r['stage1_accuracy'] for r in valid_results if r.get('stage1_accuracy') is not None]
    stage2_accuracies = [r['stage2_accuracy'] for r in valid_results if r.get('stage2_accuracy') is not None]
    stage3_accuracies = [r['stage3_accuracy'] for r in valid_results if r.get('stage3_accuracy') is not None]
    stage4_accuracies = [r['stage4_accuracy'] for r in valid_results if r.get('stage4_accuracy') is not None]
    selection_losses = [r['selection_loss'] for r in valid_results if r.get('selection_loss') is not None]
    
    # Collect baseline per-question metrics (two approaches)
    baseline_per_question_avg = []  # Average of all 13 paths (expected performance)
    baseline_per_question_first = []  # First path only (baseline's own best score selection)
    baseline_per_question_stats = []
    for r in valid_results:
        per_path = r.get('baseline_per_path_overlaps', [])
        if per_path:
            valid_path_overlaps = [p['overlap'] for p in per_path if p.get('overlap') is not None]
            if valid_path_overlaps:
                avg_overlap = sum(valid_path_overlaps) / len(valid_path_overlaps)
                first_overlap = valid_path_overlaps[0]  # File is sorted by final_score descending
                baseline_per_question_avg.append(avg_overlap)
                baseline_per_question_first.append(first_overlap)
                baseline_per_question_stats.append({
                    'video_key': r['video_key'],
                    'question_id': r['question_id'],
                    'num_paths': len(valid_path_overlaps),
                    'avg_overlap': avg_overlap,
                    'first_path_overlap': first_overlap,
                    'min_overlap': min(valid_path_overlaps),
                    'max_overlap': max(valid_path_overlaps)
                })
    
    # Per-video stats
    video_stats = {}
    for video_key, video_results in by_video.items():
        video_stage1 = [r['stage1_accuracy'] for r in video_results if r.get('stage1_accuracy') is not None]
        video_stage2 = [r['stage2_accuracy'] for r in video_results if r.get('stage2_accuracy') is not None]
        video_stage3 = [r['stage3_accuracy'] for r in video_results if r.get('stage3_accuracy') is not None]
        video_stage4 = [r['stage4_accuracy'] for r in video_results if r.get('stage4_accuracy') is not None]
        
        # Collect baseline per-question metrics for this video
        video_baseline_avg = []
        video_baseline_first = []
        for r in video_results:
            per_path = r.get('baseline_per_path_overlaps', [])
            if per_path:
                valid_overlaps = [p['overlap'] for p in per_path if p.get('overlap') is not None]
                if valid_overlaps:
                    video_baseline_avg.append(sum(valid_overlaps) / len(valid_overlaps))
                    video_baseline_first.append(valid_overlaps[0])
        
        video_stats[video_key] = {
            'num_questions': len(video_results),
            'stage1_avg_accuracy': sum(video_stage1) / len(video_stage1) if video_stage1 else 0,
            'stage2_avg_accuracy': sum(video_stage2) / len(video_stage2) if video_stage2 else 0,
            'stage3_avg_accuracy': sum(video_stage3) / len(video_stage3) if video_stage3 else 0,
            'stage4_avg_accuracy': sum(video_stage4) / len(video_stage4) if video_stage4 else 0,
            'baseline_avg_of_avg': sum(video_baseline_avg) / len(video_baseline_avg) if video_baseline_avg else 0,
            'baseline_avg_of_first': sum(video_baseline_first) / len(video_baseline_first) if video_baseline_first else 0,
            'baseline_num_questions': len(video_baseline_avg),
            'stage1_max': max(video_stage1) if video_stage1 else 0,
            'stage1_min': min(video_stage1) if video_stage1 else 0,
        }
    
    # Status breakdown
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r['status']] += 1
    
    # Format breakdown (for debugging)
    format_counts = defaultdict(int)
    for r in valid_results:
        format_counts[r.get('format', 'unknown')] += 1
    
    return {
        'total_questions': len(results),
        'valid_questions': len(valid_results),
        'invalid_questions': len(results) - len(valid_results),
        'status_breakdown': dict(status_counts),
        'format_breakdown': dict(format_counts),
        'stage1_overall': {
            'avg_accuracy': sum(stage1_accuracies) / len(stage1_accuracies) if stage1_accuracies else 0,
            'max_accuracy': max(stage1_accuracies) if stage1_accuracies else 0,
            'min_accuracy': min(stage1_accuracies) if stage1_accuracies else 0,
            'num_perfect': sum(1 for a in stage1_accuracies if a >= 0.99),
            'num_good': sum(1 for a in stage1_accuracies if 0.7 <= a < 0.99),
            'num_fair': sum(1 for a in stage1_accuracies if 0.3 <= a < 0.7),
            'num_poor': sum(1 for a in stage1_accuracies if a < 0.3),
        } if stage1_accuracies else None,
        'stage2_overall': {
            'avg_accuracy': sum(stage2_accuracies) / len(stage2_accuracies) if stage2_accuracies else 0,
            'max_accuracy': max(stage2_accuracies) if stage2_accuracies else 0,
            'min_accuracy': min(stage2_accuracies) if stage2_accuracies else 0,
            'num_perfect': sum(1 for a in stage2_accuracies if a >= 0.99),
            'num_good': sum(1 for a in stage2_accuracies if 0.7 <= a < 0.99),
            'num_fair': sum(1 for a in stage2_accuracies if 0.3 <= a < 0.7),
            'num_poor': sum(1 for a in stage2_accuracies if a < 0.3),
        } if stage2_accuracies else None,
        'stage3_overall': {
            'avg_accuracy': sum(stage3_accuracies) / len(stage3_accuracies) if stage3_accuracies else 0,
            'max_accuracy': max(stage3_accuracies) if stage3_accuracies else 0,
            'min_accuracy': min(stage3_accuracies) if stage3_accuracies else 0,
            'num_perfect': sum(1 for a in stage3_accuracies if a >= 0.99),
            'num_good': sum(1 for a in stage3_accuracies if 0.7 <= a < 0.99),
            'num_fair': sum(1 for a in stage3_accuracies if 0.3 <= a < 0.7),
            'num_poor': sum(1 for a in stage3_accuracies if a < 0.3),
        } if stage3_accuracies else None,
        'stage4_overall': {
            'avg_accuracy': sum(stage4_accuracies) / len(stage4_accuracies) if stage4_accuracies else 0,
            'max_accuracy': max(stage4_accuracies) if stage4_accuracies else 0,
            'min_accuracy': min(stage4_accuracies) if stage4_accuracies else 0,
            'num_perfect': sum(1 for a in stage4_accuracies if a >= 0.99),
            'num_good': sum(1 for a in stage4_accuracies if 0.7 <= a < 0.99),
            'num_fair': sum(1 for a in stage4_accuracies if 0.3 <= a < 0.7),
            'num_poor': sum(1 for a in stage4_accuracies if a < 0.3),
        } if stage4_accuracies else None,
        'selection_loss': {
            'avg_loss': sum(selection_losses) / len(selection_losses) if selection_losses else 0,
            'max_loss': max(selection_losses) if selection_losses else 0,
            'num_with_loss': sum(1 for sl in selection_losses if sl > 0.01),
        } if selection_losses else None,
        'baseline_overall': {
            'avg_of_avg_paths': sum(baseline_per_question_avg) / len(baseline_per_question_avg) if baseline_per_question_avg else 0,
            'avg_of_first_path': sum(baseline_per_question_first) / len(baseline_per_question_first) if baseline_per_question_first else 0,
            'max_of_first_path': max(baseline_per_question_first) if baseline_per_question_first else 0,
            'min_of_first_path': min(baseline_per_question_first) if baseline_per_question_first else 0,
            'num_questions': len(baseline_per_question_avg),
            'num_perfect_first': sum(1 for a in baseline_per_question_first if a >= 0.99),
            'num_good_first': sum(1 for a in baseline_per_question_first if 0.7 <= a < 0.99),
            'num_fair_first': sum(1 for a in baseline_per_question_first if 0.3 <= a < 0.7),
            'num_poor_first': sum(1 for a in baseline_per_question_first if a < 0.3),
            'per_question_stats': baseline_per_question_stats
        } if baseline_per_question_avg else None,
        'per_video': video_stats
    }


def print_report(stats: Dict, detailed_results: List[Dict] = None):
    """Print a human-readable report."""
    print("\n" + "="*80)
    print("RETRIEVAL ACCURACY REPORT")
    print("="*80)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total questions: {stats['total_questions']}")
    print(f"  Valid evaluations: {stats['valid_questions']}")
    print(f"  Invalid/skipped: {stats['invalid_questions']}")
    
    if stats['invalid_questions'] > 0:
        print(f"\n  Status breakdown:")
        for status, count in stats['status_breakdown'].items():
            print(f"    {status}: {count}")
    
    # Format breakdown (shows which evaluation format was used)
    if stats.get('format_breakdown'):
        print(f"\n  Format breakdown:")
        format_labels = {
            'new_four_stage': 'New (four-stage evaluation)',
            'on_demand_generated': 'On-demand (generated from Stage 1)',
            'new_two_stage': 'New (two-stage evaluation - legacy)',
            'old_single_stage': 'Old (single-stage)',
            'reconstructed': 'Reconstructed (very old)'
        }
        for format_key, count in stats['format_breakdown'].items():
            label = format_labels.get(format_key, format_key)
            print(f"    {label}: {count}")
    
    # Stage 1 results
    if stats.get('stage1_overall') and stats['valid_questions'] > 0:
        stage1 = stats['stage1_overall']
        print(f"\n🎯 STAGE 1: After Full Exploration")
        print(f"  Average Accuracy: {stage1['avg_accuracy']:.3f}")
        print(f"  Max Accuracy: {stage1['max_accuracy']:.3f}")
        print(f"  Min Accuracy: {stage1['min_accuracy']:.3f}")
        print(f"\n  Performance Distribution:")
        print(f"    Perfect (≥0.99): {stage1['num_perfect']} ({stage1['num_perfect']/stats['valid_questions']*100:.1f}%)")
        print(f"    Good (0.7-0.99): {stage1['num_good']} ({stage1['num_good']/stats['valid_questions']*100:.1f}%)")
        print(f"    Fair (0.3-0.7):  {stage1['num_fair']} ({stage1['num_fair']/stats['valid_questions']*100:.1f}%)")
        print(f"    Poor (<0.3):     {stage1['num_poor']} ({stage1['num_poor']/stats['valid_questions']*100:.1f}%)")
    elif not stats.get('baseline_overall'):
        print(f"\n⚠️  No valid evaluations found. All questions are missing iteration data.")
    
    # Stage 2 results (if available)
    if stats['stage2_overall'] and stats['valid_questions'] > 0:
        stage2 = stats['stage2_overall']
        print(f"\n🎯 STAGE 2: After LLM Pruning (Steiner)")
        print(f"  Average Accuracy: {stage2['avg_accuracy']:.3f}")
        print(f"  Max Accuracy: {stage2['max_accuracy']:.3f}")
        print(f"  Min Accuracy: {stage2['min_accuracy']:.3f}")
        print(f"\n  Performance Distribution:")
        print(f"    Perfect (≥0.99): {stage2['num_perfect']} ({stage2['num_perfect']/stats['valid_questions']*100:.1f}%)")
        print(f"    Good (0.7-0.99): {stage2['num_good']} ({stage2['num_good']/stats['valid_questions']*100:.1f}%)")
        print(f"    Fair (0.3-0.7):  {stage2['num_fair']} ({stage2['num_fair']/stats['valid_questions']*100:.1f}%)")
        print(f"    Poor (<0.3):     {stage2['num_poor']} ({stage2['num_poor']/stats['valid_questions']*100:.1f}%)")
    
    # Stage 3 results (if available)
    if stats['stage3_overall'] and stats['valid_questions'] > 0:
        stage3 = stats['stage3_overall']
        print(f"\n🎯 STAGE 3: After Top-k Pruning (Naive Baseline)")
        print(f"  Average Accuracy: {stage3['avg_accuracy']:.3f}")
        print(f"  Max Accuracy: {stage3['max_accuracy']:.3f}")
        print(f"  Min Accuracy: {stage3['min_accuracy']:.3f}")
        print(f"\n  Performance Distribution:")
        print(f"    Perfect (≥0.99): {stage3['num_perfect']} ({stage3['num_perfect']/stats['valid_questions']*100:.1f}%)")
        print(f"    Good (0.7-0.99): {stage3['num_good']} ({stage3['num_good']/stats['valid_questions']*100:.1f}%)")
        print(f"    Fair (0.3-0.7):  {stage3['num_fair']} ({stage3['num_fair']/stats['valid_questions']*100:.1f}%)")
        print(f"    Poor (<0.3):     {stage3['num_poor']} ({stage3['num_poor']/stats['valid_questions']*100:.1f}%)")
    
    # Stage 4 results (if available)
    if stats['stage4_overall'] and stats['valid_questions'] > 0:
        stage4 = stats['stage4_overall']
        print(f"\n🎯 STAGE 4: After Event-Centric Pruning (final_answer.py logic)")
        print(f"  Average Accuracy: {stage4['avg_accuracy']:.3f}")
        print(f"  Max Accuracy: {stage4['max_accuracy']:.3f}")
        print(f"  Min Accuracy: {stage4['min_accuracy']:.3f}")
        print(f"\n  Performance Distribution:")
        print(f"    Perfect (≥0.99): {stage4['num_perfect']} ({stage4['num_perfect']/stats['valid_questions']*100:.1f}%)")
        print(f"    Good (0.7-0.99): {stage4['num_good']} ({stage4['num_good']/stats['valid_questions']*100:.1f}%)")
        print(f"    Fair (0.3-0.7):  {stage4['num_fair']} ({stage4['num_fair']/stats['valid_questions']*100:.1f}%)")
        print(f"    Poor (<0.3):     {stage4['num_poor']} ({stage4['num_poor']/stats['valid_questions']*100:.1f}%)")
        
        # Show comparison if all stages exist
        if stats['stage2_overall'] and stats['stage3_overall']:
            stage2_avg = stats['stage2_overall']['avg_accuracy']
            stage3_avg = stats['stage3_overall']['avg_accuracy']
            stage4_avg = stage4['avg_accuracy']
            
            print(f"\n  📊 All Pruning Methods Comparison:")
            print(f"     Steiner (Stage 2):       {stage2_avg:.3f}")
            print(f"     Top-k (Stage 3):         {stage3_avg:.3f}")
            print(f"     Event-Centric (Stage 4): {stage4_avg:.3f}")
            
            # Find best
            best_avg = max(stage2_avg, stage3_avg, stage4_avg)
            best_methods = []
            if stage2_avg == best_avg:
                best_methods.append("Steiner")
            if stage3_avg == best_avg:
                best_methods.append("Top-k")
            if stage4_avg == best_avg:
                best_methods.append("Event-Centric")
            
            print(f"     Best Method(s):          {', '.join(best_methods)} ({best_avg:.3f})")
        elif stats['stage2_overall'] and stats['stage3_overall']:
            # Show original 2-way comparison if Stage 4 doesn't exist
            stage2_avg = stats['stage2_overall']['avg_accuracy']
            stage3_avg = stats['stage3_overall']['avg_accuracy']
            diff = stage2_avg - stage3_avg
            print(f"\n  📊 Pruning Method Comparison:")
            print(f"     Steiner avg: {stage2_avg:.3f}")
            print(f"     Top-k avg:   {stage3_avg:.3f}")
            print(f"     Difference:  {diff:+.3f} ({'Steiner better' if diff > 0 else 'Top-k better' if diff < 0 else 'Tied'})")
    
    # Baseline results (if available)
    if stats.get('baseline_overall'):
        baseline = stats['baseline_overall']
        print(f"\n🎯 BASELINE: sorted_SA_score_result.json")
        print(f"  Questions Evaluated: {baseline['num_questions']}")
        print(f"\n  📊 Approach 1: Average of all 13 paths (expected performance)")
        print(f"     Avg Accuracy: {baseline['avg_of_avg_paths']:.3f}")
        print(f"\n  📊 Approach 2: First path only (baseline's best score selection)")
        print(f"     Avg Accuracy: {baseline['avg_of_first_path']:.3f}")
        print(f"     Max Accuracy: {baseline['max_of_first_path']:.3f}")
        print(f"     Min Accuracy: {baseline['min_of_first_path']:.3f}")
        print(f"     Note: File is sorted by final_score (descending), first path = baseline's best")
        print(f"\n  Performance Distribution (First Path):")
        total = baseline['num_questions']
        print(f"    Perfect (≥0.99): {baseline['num_perfect_first']} ({baseline['num_perfect_first']/total*100:.1f}%)")
        print(f"    Good (0.7-0.99): {baseline['num_good_first']} ({baseline['num_good_first']/total*100:.1f}%)")
        print(f"    Fair (0.3-0.7):  {baseline['num_fair_first']} ({baseline['num_fair_first']/total*100:.1f}%)")
        print(f"    Poor (<0.3):     {baseline['num_poor_first']} ({baseline['num_poor_first']/total*100:.1f}%)")
    
    # Selection loss (if available)
    if stats['selection_loss']:
        sl = stats['selection_loss']
        print(f"\n📉 Selection Loss (Stage 1 → Stage 2):")
        print(f"  Average Loss: {sl['avg_loss']:.3f}")
        print(f"  Max Loss: {sl['max_loss']:.3f}")
        print(f"  Questions with loss > 0.01: {sl['num_with_loss']}")
    
    # Per-video breakdown
    if stats['per_video']:
        print(f"\n📹 Per-Video Breakdown:")
        for video_key, vstats in sorted(stats['per_video'].items()):
            print(f"\n  {video_key}:")
            print(f"    Questions: {vstats['num_questions']}")
            if vstats.get('baseline_avg_of_avg', 0) > 0:
                print(f"    Baseline Avg (all paths):    {vstats['baseline_avg_of_avg']:.3f}")
                print(f"    Baseline Avg (first path):   {vstats['baseline_avg_of_first']:.3f} ({vstats.get('baseline_num_questions', 0)} questions)")
            if vstats.get('stage1_avg_accuracy', 0) > 0:
                print(f"    Stage 1 Avg: {vstats['stage1_avg_accuracy']:.3f} (range: {vstats['stage1_min']:.3f} - {vstats['stage1_max']:.3f})")
            if vstats.get('stage2_avg_accuracy', 0) > 0:
                print(f"    Stage 2 Avg (Steiner):       {vstats['stage2_avg_accuracy']:.3f}")
            if vstats.get('stage3_avg_accuracy', 0) > 0:
                print(f"    Stage 3 Avg (Top-k):         {vstats['stage3_avg_accuracy']:.3f}")
            if vstats.get('stage4_avg_accuracy', 0) > 0:
                print(f"    Stage 4 Avg (Event-Centric): {vstats['stage4_avg_accuracy']:.3f}")
    else:
        print(f"\n📹 No valid per-video statistics available")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Calculate retrieval accuracy from iteration logs or baseline results")
    parser.add_argument('--results-dir', type=str, default='ava100_results',
                        help='Directory containing results (default: ava100_results, not needed for --baseline-only)')
    parser.add_argument('--output', type=str, default='accuracy_report.json',
                        help='Output file for detailed report (default: accuracy_report.json)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Optional: Export detailed results to CSV')
    parser.add_argument('--llm-budget', type=int, default=20,
                        help='Target node count for LLM pruning stages (default: 20)')
    parser.add_argument('--no-on-demand', action='store_true',
                        help='Disable on-demand stage generation (only use pre-computed stages)')
    parser.add_argument('--save-pruned-subgraphs', action='store_true',
                        help='Save Stage 2/3/4 pruned subgraphs to disk for inspection')
    parser.add_argument('--evaluate-baseline', action='store_true',
                        help='Evaluate baseline method from sorted_SA_score_result.json')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory containing AVA_cache (for baseline evaluation, default: auto-detect)')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Evaluate baseline only (no benchmark results needed, reads from AVA_cache)')
    parser.add_argument('--force-recalculate', action='store_true',
                        help='Recalculate overlaps from subgraphs instead of using pre-computed values from logs')
    
    args = parser.parse_args()
    
    # Handle baseline-only mode
    if args.baseline_only:
        cache_dir = Path(args.cache_dir) if args.cache_dir else Path('AVA_cache')
        
        if not cache_dir.exists():
            print(f"❌ Error: Cache directory not found: {cache_dir}")
            print(f"   Please specify correct path with --cache-dir")
            return
        
        print(f"📂 Baseline-Only Mode: Scanning cache directory: {cache_dir}")
        all_results = scan_baseline_only(cache_dir)
    else:
        # Normal mode: scan benchmark results
        results_dir = Path(args.results_dir)
        
        if not results_dir.exists():
            print(f"❌ Error: Results directory not found: {results_dir}")
            return
        
        print(f"📂 Scanning results directory: {results_dir}")
        
        # Scan all logs
        enable_on_demand = not args.no_on_demand
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        all_results = scan_results_directory(
            results_dir, 
            llm_budget=args.llm_budget, 
            enable_on_demand=enable_on_demand,
            save_pruned_subgraphs=args.save_pruned_subgraphs,
            evaluate_baseline=args.evaluate_baseline,
            cache_dir=cache_dir,
            force_recalculate=args.force_recalculate
        )
    
    # Calculate statistics
    stats = calculate_statistics(all_results)
    
    # Print report
    print_report(stats, all_results)
    
    # Save detailed report
    output_path = Path(args.output)
    report_data = {
        'statistics': stats,
        'detailed_results': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n✅ Detailed report saved to: {output_path}")
    
    # Optional CSV export
    if args.csv:
        import csv
        csv_path = Path(args.csv)
        
        with open(csv_path, 'w', newline='') as f:
            if all_results:
                fieldnames = all_results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
        
        print(f"✅ CSV export saved to: {csv_path}")


if __name__ == "__main__":
    main()

