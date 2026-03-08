#!/usr/bin/env python3
"""
Analyze the relationship between time retrieval accuracy and answer correctness.

This script:
1. Loads query results from query_CA_ava100_qwenvl.json
2. For each question, retrieves baseline time retrieval overlap from sorted_SA_score_result.json
3. Compares time retrieval accuracy with answer correctness
4. Generates visualizations and statistics showing the correlation

Usage:
    python insight_2.py --query-results outputs/query_CA_ava100_qwenvl.json --cache-dir AVA_cache
    python insight_2.py --query-results outputs/query_CA_ava100_qwenvl.json --cache-dir AVA_cache --output insights_report.json
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation functions from calculate_accuracy.py
try:
    from ICDCS.calculate_accuracy import (
        evaluate_baseline_overlap,
        overlap_reference_helper,
        time_to_seconds,
        percentage_overlap
    )
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
        
        # Merge overlapping intervals within the reference range
        relevant_intervals = []
        for start, end in time_list:
            if end <= start:
                continue
            s = max(start, ref_start)
            e = min(end, ref_end)
            if e > s:
                relevant_intervals.append((s, e))
        
        if not relevant_intervals:
            return 0.0
        
        relevant_intervals.sort()
        merged = [relevant_intervals[0]]
        for current_start, current_end in relevant_intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        total_covered = sum(end - start for start, end in merged)
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
        """Evaluate overlap for baseline method using sorted_SA_score_result.json."""
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
        
        per_path_overlaps = []
        # breakpoint()
        for path_idx, result in enumerate(sa_results):
            frame_durations = result.get('frame_durations', [])
            
            if not frame_durations:
                per_path_overlaps.append({
                    'path_idx': path_idx,
                    'overlap': None,
                    'num_frames': 0,
                    'error': 'No frame durations'
                })
                continue
            
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
                    'error': 'No valid frame durations'
                })
                continue
            
            overlap = overlap_reference_helper(time_reference, time_list)
            
            per_path_overlaps.append({
                'path_idx': path_idx,
                'overlap': overlap,
                'num_frames': len(frame_durations),
                'error': None
            })
        
        return {
            'per_path_overlaps': per_path_overlaps,
            'num_paths': len(sa_results),
            'error': None
        }


def load_ava100_dataset() -> Tuple[Dict, Dict]:
    """Load AVA100 dataset to get time references."""
    dataset_files = {
        'ego': 'datas/AVA100/ego.json',
        'citytour': 'datas/AVA100/citytour.json',
        'wildlife': 'datas/AVA100/wildlife.json',
        'traffic': 'datas/AVA100/traffic.json'
    }
    
    dataset_map = {}  # (video_id, question_id) -> qa_item
    video_id_to_key = {}  # video_id -> video_key
    
    for idx, (dataset_type, json_path) in enumerate(dataset_files.items()):
        json_path_obj = Path(json_path)
        if json_path_obj.exists():
            with open(json_path_obj, 'r') as f:
                videos = json.load(f)
                for video in videos:
                    video_key = video.get('video_key')
                    video_id = int(video.get('video_id')) + idx*2
                    
                    # Build video_id to video_key mapping
                    if video_id is not None and video_key:
                        video_id_to_key[video_id] = video_key
                    
                    qa_list = video.get('qa', [])
                    for q_idx, qa_item in enumerate(qa_list):
                        # Use video_id as primary key (it's numeric and consistent)
                        if video_id is not None:
                            dataset_map[(video_id, q_idx)] = qa_item
                        # Also map by video_key as fallback
                        if video_key:
                            dataset_map[(video_key, q_idx)] = qa_item
        else:
            print(f"⚠️  Warning: Dataset file not found: {json_path}")
    
    return dataset_map, video_id_to_key


def find_baseline_file(cache_dir: Path, video_id: int, question_id: int, video_id_to_key: Dict) -> Optional[Path]:
    """Find sorted_SA_score_result.json file for given video_id and question_id."""
    # Try multiple paths
    possible_paths = []
    
    # Path 1: Using video_id directly
    possible_paths.append(cache_dir / "AVA100" / str(video_id) / "questions" / str(question_id) / "sorted_SA_score_result.json")
    
    # Path 2: Using video_key if available
    if video_id in video_id_to_key:
        video_key = video_id_to_key[video_id]
        possible_paths.append(cache_dir / "AVA100" / video_key / "questions" / str(question_id) / "sorted_SA_score_result.json")
    
    # Path 3: Try to find by scanning (if video_id doesn't match directory name)
    breakpoint()
    for path in possible_paths:
        if path.exists():
            return path
    
    # Path 4: Try to find config.json to get video_key
    for parent_dir in [cache_dir / "AVA100" / str(video_id), cache_dir / "AVA100"]:
        if parent_dir.exists():
            for video_folder in parent_dir.iterdir():
                if video_folder.is_dir():
                    config_path = video_folder / "config.json"
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                source_path = config.get('source_path', '')
                                video_key = source_path.split('/')[-1].split('.')[0]
                                
                                # Check if this matches our video_id somehow
                                # (This is a fallback, may not always work)
                                sa_path = video_folder / "questions" / str(question_id) / "sorted_SA_score_result.json"
                                if sa_path.exists():
                                    return sa_path
                        except:
                            continue
    
    return None


def analyze_query_results(query_results_path: Path, cache_dir: Path) -> List[Dict]:
    """
    Analyze query results and correlate with baseline time retrieval accuracy.
    
    Args:
        query_results_path: Path to query_CA_ava100_qwenvl.json
        cache_dir: Directory containing AVA_cache
    
    Returns:
        List of analysis results with overlap and correctness info
    """
    # Load query results
    with open(query_results_path, 'r') as f:
        query_results = json.load(f)
    
    # Load AVA100 dataset for time references
    dataset_map, video_id_to_key = load_ava100_dataset()

    # Find baseline files
    ava100_cache = cache_dir / "AVA100"
    if not ava100_cache.exists():
        print(f"❌ Error: AVA100 cache directory not found: {ava100_cache}")
        return []
    
    analysis_results = []
    
    print(f"📊 Analyzing {len(query_results)} query results...")
    
    # Track statistics during processing
    stats_tracker = {
        'processed': 0,
        'no_time_ref': 0,
        'baseline_not_found': 0,
        'baseline_error': 0,
        'success': 0
    }
    
    for query_item in query_results:
        stats_tracker['processed'] += 1
        video_id = query_item.get('video_id')
        question_id = query_item.get('question_id')
        answer_gt = query_item.get('answer', '').strip()
        response = query_item.get('response', '').strip()
        time_processed = query_item.get('time_processed', 0)
        
        # Check if answer is correct
        is_correct = (answer_gt == response)
        
        # Get time reference from dataset
        time_reference = None
        if (video_id, question_id) in dataset_map:
            qa_item = dataset_map[(video_id, question_id)]
            time_reference = qa_item.get('time_reference', 'N/A')
        else:
            # Try with video_key if video_id_to_key mapping exists
            if video_id in video_id_to_key:
                video_key = video_id_to_key[video_id]
                if (video_key, question_id) in dataset_map:
                    qa_item = dataset_map[(video_key, question_id)]
                    time_reference = qa_item.get('time_reference', 'N/A')
        
        if time_reference is None or time_reference == 'N/A':
            # Skip if no time reference
            stats_tracker['no_time_ref'] += 1
            analysis_results.append({
                'video_id': video_id,
                'question_id': question_id,
                'answer_gt': answer_gt,
                'response': response,
                'is_correct': is_correct,
                'time_processed': time_processed,
                'time_reference': None,
                'overlap_first_path': None,
                'overlap_avg_all_paths': None,
                'overlap_max_all_paths': None,
                'num_paths': 0,
                'error': 'No time reference'
            })
            continue
        
        # Find baseline file
        baseline_path = find_baseline_file(cache_dir, video_id, question_id, video_id_to_key)

        if baseline_path is None or not baseline_path.exists():
            stats_tracker['baseline_not_found'] += 1
            analysis_results.append({
                'video_id': video_id,
                'question_id': question_id,
                'answer_gt': answer_gt,
                'response': response,
                'is_correct': is_correct,
                'time_processed': time_processed,
                'time_reference': time_reference,
                'overlap_first_path': None,
                'overlap_avg_all_paths': None,
                'overlap_max_all_paths': None,
                'num_paths': 0,
                'error': 'Baseline file not found'
            })
            continue
        
        # Evaluate baseline overlap
        baseline_result = evaluate_baseline_overlap(baseline_path, time_reference)
        
        if baseline_result.get('error'):
            stats_tracker['baseline_error'] += 1
            analysis_results.append({
                'video_id': video_id,
                'question_id': question_id,
                'answer_gt': answer_gt,
                'response': response,
                'is_correct': is_correct,
                'time_processed': time_processed,
                'time_reference': time_reference,
                'overlap_first_path': None,
                'overlap_avg_all_paths': None,
                'overlap_max_all_paths': None,
                'num_paths': baseline_result.get('num_paths', 0),
                'error': baseline_result.get('error')
            })
            continue
        
        # Extract overlap metrics
        per_path_overlaps = baseline_result.get('per_path_overlaps', [])
        valid_overlaps = [p['overlap'] for p in per_path_overlaps if p.get('overlap') is not None]
        
        overlap_first_path = valid_overlaps[0] if len(valid_overlaps) > 0 else None
        overlap_avg_all_paths = np.mean(valid_overlaps) if len(valid_overlaps) > 0 else None
        overlap_max_all_paths = max(valid_overlaps) if len(valid_overlaps) > 0 else None
        
        analysis_results.append({
            'video_id': video_id,
            'question_id': question_id,
            'answer_gt': answer_gt,
            'response': response,
            'is_correct': is_correct,
            'time_processed': time_processed,
            'time_reference': time_reference,
            'overlap_first_path': overlap_first_path,
            'overlap_avg_all_paths': overlap_avg_all_paths,
            'overlap_max_all_paths': overlap_max_all_paths,
            'num_paths': baseline_result.get('num_paths', 0),
            'error': None
        })
        stats_tracker['success'] += 1
    
    # Print processing summary
    print(f"\n📊 Processing Summary:")
    print(f"  Total processed: {stats_tracker['processed']}")
    print(f"  Successful (with overlap): {stats_tracker['success']}")
    print(f"  No time reference: {stats_tracker['no_time_ref']}")
    print(f"  Baseline file not found: {stats_tracker['baseline_not_found']}")
    print(f"  Baseline evaluation error: {stats_tracker['baseline_error']}")
    
    return analysis_results


def calculate_statistics(analysis_results: List[Dict]) -> Dict:
    """Calculate statistics about the relationship between overlap and correctness."""
    # Filter valid results (have overlap data)
    valid_results = [r for r in analysis_results if r.get('overlap_first_path') is not None]
    
    # Count errors by type
    error_counts = defaultdict(int)
    for r in analysis_results:
        if r.get('error'):
            error_counts[r['error']] += 1
    
    if len(valid_results) == 0:
        return {
            'error': 'No valid results with overlap data',
            'total_questions': len(analysis_results),
            'valid_questions': 0,
            'error_breakdown': dict(error_counts)
        }
    
    # Separate by correctness
    correct_results = [r for r in valid_results if r['is_correct']]
    incorrect_results = [r for r in valid_results if not r['is_correct']]
    
    # Extract overlap values
    correct_overlaps_first = [r['overlap_first_path'] for r in correct_results]
    incorrect_overlaps_first = [r['overlap_first_path'] for r in incorrect_results]
    
    correct_overlaps_avg = [r['overlap_avg_all_paths'] for r in correct_results if r.get('overlap_avg_all_paths') is not None]
    incorrect_overlaps_avg = [r['overlap_avg_all_paths'] for r in incorrect_results if r.get('overlap_avg_all_paths') is not None]
    
    correct_overlaps_max = [r['overlap_max_all_paths'] for r in correct_results if r.get('overlap_max_all_paths') is not None]
    incorrect_overlaps_max = [r['overlap_max_all_paths'] for r in incorrect_results if r.get('overlap_max_all_paths') is not None]
    
    # Calculate statistics
    stats = {
        'total_questions': len(analysis_results),
        'valid_questions': len(valid_results),
        'correct_answers': len(correct_results),
        'incorrect_answers': len(incorrect_results),
        'overall_accuracy': len(correct_results) / len(valid_results) if len(valid_results) > 0 else 0,
        
        'first_path_overlap': {
            'correct': {
                'mean': np.mean(correct_overlaps_first) if correct_overlaps_first else None,
                'median': np.median(correct_overlaps_first) if correct_overlaps_first else None,
                'std': np.std(correct_overlaps_first) if correct_overlaps_first else None,
                'min': min(correct_overlaps_first) if correct_overlaps_first else None,
                'max': max(correct_overlaps_first) if correct_overlaps_first else None,
            },
            'incorrect': {
                'mean': np.mean(incorrect_overlaps_first) if incorrect_overlaps_first else None,
                'median': np.median(incorrect_overlaps_first) if incorrect_overlaps_first else None,
                'std': np.std(incorrect_overlaps_first) if incorrect_overlaps_first else None,
                'min': min(incorrect_overlaps_first) if incorrect_overlaps_first else None,
                'max': max(incorrect_overlaps_first) if incorrect_overlaps_first else None,
            },
            'difference': {
                'mean_diff': (np.mean(correct_overlaps_first) - np.mean(incorrect_overlaps_first)) if (correct_overlaps_first and incorrect_overlaps_first) else None,
            }
        },
        
        'avg_all_paths_overlap': {
            'correct': {
                'mean': np.mean(correct_overlaps_avg) if correct_overlaps_avg else None,
                'median': np.median(correct_overlaps_avg) if correct_overlaps_avg else None,
                'std': np.std(correct_overlaps_avg) if correct_overlaps_avg else None,
            },
            'incorrect': {
                'mean': np.mean(incorrect_overlaps_avg) if incorrect_overlaps_avg else None,
                'median': np.median(incorrect_overlaps_avg) if incorrect_overlaps_avg else None,
                'std': np.std(incorrect_overlaps_avg) if incorrect_overlaps_avg else None,
            },
        },
        
        'max_all_paths_overlap': {
            'correct': {
                'mean': np.mean(correct_overlaps_max) if correct_overlaps_max else None,
                'median': np.median(correct_overlaps_max) if correct_overlaps_max else None,
                'std': np.std(correct_overlaps_max) if correct_overlaps_max else None,
            },
            'incorrect': {
                'mean': np.mean(incorrect_overlaps_max) if incorrect_overlaps_max else None,
                'median': np.median(incorrect_overlaps_max) if incorrect_overlaps_max else None,
                'std': np.std(incorrect_overlaps_max) if incorrect_overlaps_max else None,
            },
        },
    }
    
    # Calculate correlation
    if correct_overlaps_first and incorrect_overlaps_first:
        try:
            from scipy.stats import mannwhitneyu, pearsonr
            
            # Binary correctness (1 for correct, 0 for incorrect) vs overlap
            correctness_binary = [1] * len(correct_overlaps_first) + [0] * len(incorrect_overlaps_first)
            all_overlaps_first = correct_overlaps_first + incorrect_overlaps_first
            
            try:
                correlation, p_value = pearsonr(all_overlaps_first, correctness_binary)
                stats['correlation'] = {
                    'pearson_r': correlation,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except:
                stats['correlation'] = None
            
            # Mann-Whitney U test (non-parametric test for difference in distributions)
            try:
                u_statistic, u_p_value = mannwhitneyu(correct_overlaps_first, incorrect_overlaps_first, alternative='two-sided')
                stats['mann_whitney_u'] = {
                    'u_statistic': u_statistic,
                    'p_value': u_p_value,
                    'significant': u_p_value < 0.05
                }
            except:
                stats['mann_whitney_u'] = None
        except ImportError:
            stats['correlation'] = None
            stats['mann_whitney_u'] = None
    
    return stats


def create_visualizations(analysis_results: List[Dict], output_dir: Path):
    """Create visualization plots."""
    # Filter valid results
    valid_results = [r for r in analysis_results if r.get('overlap_first_path') is not None]
    
    if len(valid_results) == 0:
        print("⚠️  No valid results to plot (no overlap data available)")
        print("   Skipping visualization generation.")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Extract data
    correct_overlaps = [r['overlap_first_path'] for r in valid_results if r['is_correct']]
    incorrect_overlaps = [r['overlap_first_path'] for r in valid_results if not r['is_correct']]
    
    # 1. Box plot: Overlap distribution by correctness
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Box plot
    ax1 = axes[0, 0]
    data_to_plot = [correct_overlaps, incorrect_overlaps]
    bp = ax1.boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax1.set_ylabel('Time Retrieval Overlap (First Path)')
    ax1.set_title('Distribution of Time Retrieval Overlap by Answer Correctness')
    ax1.grid(True, alpha=0.3)
    
    # Violin plot
    ax2 = axes[0, 1]
    data_dict = {
        'Correct': correct_overlaps,
        'Incorrect': incorrect_overlaps
    }
    parts = ax2.violinplot([correct_overlaps, incorrect_overlaps], positions=[0, 1], 
                          showmeans=True, showmedians=True)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Correct', 'Incorrect'])
    ax2.set_ylabel('Time Retrieval Overlap (First Path)')
    ax2.set_title('Violin Plot: Overlap Distribution by Correctness')
    ax2.grid(True, alpha=0.3)
    
    # Histogram overlay
    ax3 = axes[1, 0]
    ax3.hist(correct_overlaps, bins=20, alpha=0.6, label='Correct', color='green', density=True)
    ax3.hist(incorrect_overlaps, bins=20, alpha=0.6, label='Incorrect', color='red', density=True)
    ax3.set_xlabel('Time Retrieval Overlap (First Path)')
    ax3.set_ylabel('Density')
    ax3.set_title('Histogram: Overlap Distribution by Correctness')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Scatter plot with jitter
    ax4 = axes[1, 1]
    correct_y = [1] * len(correct_overlaps)
    incorrect_y = [0] * len(incorrect_overlaps)
    
    # Add jitter
    np.random.seed(42)
    correct_y_jittered = [y + np.random.normal(0, 0.05) for y in correct_y]
    incorrect_y_jittered = [y + np.random.normal(0, 0.05) for y in incorrect_y]
    
    ax4.scatter(correct_overlaps, correct_y_jittered, alpha=0.5, label='Correct', color='green', s=30)
    ax4.scatter(incorrect_overlaps, incorrect_y_jittered, alpha=0.5, label='Incorrect', color='red', s=30)
    ax4.set_xlabel('Time Retrieval Overlap (First Path)')
    ax4.set_ylabel('Answer Correctness')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Incorrect', 'Correct'])
    ax4.set_title('Scatter Plot: Overlap vs Correctness')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overlap_vs_correctness.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_dir / 'overlap_vs_correctness.png'}")
    plt.close()
    
    # 2. Threshold analysis: Accuracy at different overlap thresholds
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thresholds = np.arange(0, 1.01, 0.05)
    accuracies = []
    counts = []
    
    for threshold in thresholds:
        above_threshold = [r for r in valid_results if r['overlap_first_path'] >= threshold]
        if len(above_threshold) > 0:
            correct_count = sum(1 for r in above_threshold if r['is_correct'])
            accuracy = correct_count / len(above_threshold)
            accuracies.append(accuracy)
            counts.append(len(above_threshold))
        else:
            accuracies.append(0)
            counts.append(0)
    
    ax.plot(thresholds, accuracies, marker='o', linewidth=2, markersize=6, label='Accuracy')
    ax.set_xlabel('Time Retrieval Overlap Threshold')
    ax.set_ylabel('Answer Accuracy')
    ax.set_title('Answer Accuracy at Different Time Retrieval Overlap Thresholds')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add count annotations
    ax2 = ax.twinx()
    ax2.bar(thresholds, counts, alpha=0.3, color='gray', label='Question Count')
    ax2.set_ylabel('Number of Questions', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_threshold.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_dir / 'accuracy_by_threshold.png'}")
    plt.close()
    
    # 3. Correlation heatmap (if we have multiple overlap metrics)
    try:
        import pandas as pd
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Prepare data for correlation
        overlap_data = {
            'First Path': [r['overlap_first_path'] for r in valid_results],
            'Avg All Paths': [r.get('overlap_avg_all_paths', 0) for r in valid_results],
            'Max All Paths': [r.get('overlap_max_all_paths', 0) for r in valid_results],
            'Correctness': [1 if r['is_correct'] else 0 for r in valid_results]
        }
        
        df = pd.DataFrame(overlap_data)
        correlation_matrix = df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix: Overlap Metrics vs Correctness')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization: {output_dir / 'correlation_heatmap.png'}")
        plt.close()
    except ImportError:
        print("⚠️  pandas not available, skipping correlation heatmap")


def print_report(stats: Dict):
    """Print a human-readable report."""
    print("\n" + "="*80)
    print("TIME RETRIEVAL vs ANSWER CORRECTNESS ANALYSIS")
    print("="*80)
    
    # Check for error case
    if 'error' in stats:
        print(f"\n❌ Error: {stats['error']}")
        print(f"\n📊 Summary:")
        print(f"  Total questions processed: {stats.get('total_questions', 0)}")
        print(f"  Valid questions (with overlap data): {stats.get('valid_questions', 0)}")
        
        if 'error_breakdown' in stats and stats['error_breakdown']:
            print(f"\n  Error breakdown:")
            for error_type, count in stats['error_breakdown'].items():
                print(f"    {error_type}: {count}")
        
        print("\n⚠️  No overlap data available for analysis.")
        print("   Possible reasons:")
        print("   - Baseline files (sorted_SA_score_result.json) not found")
        print("   - Time references missing from dataset")
        print("   - Cache directory path incorrect")
        print("\n" + "="*80)
        return
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total questions: {stats['total_questions']}")
    print(f"  Valid questions (with overlap data): {stats['valid_questions']}")
    
    if stats['valid_questions'] > 0:
        print(f"  Correct answers: {stats['correct_answers']} ({stats['correct_answers']/stats['valid_questions']*100:.1f}%)")
        print(f"  Incorrect answers: {stats['incorrect_answers']} ({stats['incorrect_answers']/stats['valid_questions']*100:.1f}%)")
        print(f"  Overall accuracy: {stats['overall_accuracy']:.3f}")
    else:
        print(f"  ⚠️  No valid questions with overlap data")
    
    if 'first_path_overlap' in stats:
        fp = stats['first_path_overlap']
        print(f"\n🎯 First Path Overlap Statistics:")
        
        if fp['correct']['mean'] is not None:
            print(f"\n  Correct Answers:")
            print(f"    Mean overlap:   {fp['correct']['mean']:.3f}")
            print(f"    Median overlap: {fp['correct']['median']:.3f}")
            print(f"    Std deviation:  {fp['correct']['std']:.3f}")
            print(f"    Range:          [{fp['correct']['min']:.3f}, {fp['correct']['max']:.3f}]")
        
        if fp['incorrect']['mean'] is not None:
            print(f"\n  Incorrect Answers:")
            print(f"    Mean overlap:   {fp['incorrect']['mean']:.3f}")
            print(f"    Median overlap: {fp['incorrect']['median']:.3f}")
            print(f"    Std deviation:  {fp['incorrect']['std']:.3f}")
            print(f"    Range:          [{fp['incorrect']['min']:.3f}, {fp['incorrect']['max']:.3f}]")
        
        if fp['difference']['mean_diff'] is not None:
            print(f"\n  Difference (Correct - Incorrect):")
            print(f"    Mean difference: {fp['difference']['mean_diff']:.3f}")
    
    if 'correlation' in stats and stats['correlation']:
        corr = stats['correlation']
        print(f"\n📈 Correlation Analysis:")
        print(f"  Pearson correlation: {corr['pearson_r']:.3f}")
        print(f"  P-value: {corr['p_value']:.4f}")
        print(f"  Significant: {'Yes' if corr['significant'] else 'No'} (p < 0.05)")
    
    if 'mann_whitney_u' in stats and stats['mann_whitney_u']:
        mw = stats['mann_whitney_u']
        print(f"\n📊 Statistical Test (Mann-Whitney U):")
        print(f"  U-statistic: {mw['u_statistic']:.1f}")
        print(f"  P-value: {mw['p_value']:.4f}")
        print(f"  Significant difference: {'Yes' if mw['significant'] else 'No'} (p < 0.05)")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze relationship between time retrieval and answer correctness")
    parser.add_argument('--query-results', type=str, required=True,
                        help='Path to query_CA_ava100_qwenvl.json')
    parser.add_argument('--cache-dir', type=str, default='AVA_cache',
                        help='Directory containing AVA_cache (default: AVA_cache)')
    parser.add_argument('--output', type=str, default='insight_2_report.json',
                        help='Output file for detailed report (default: insight_2_report.json)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save visualizations (default: current directory)')
    
    args = parser.parse_args()
    
    query_results_path = Path(args.query_results)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    
    if not query_results_path.exists():
        print(f"❌ Error: Query results file not found: {query_results_path}")
        return
    
    if not cache_dir.exists():
        print(f"❌ Error: Cache directory not found: {cache_dir}")
        return
    
    # Analyze results
    print(f"📂 Loading query results from: {query_results_path}")
    print(f"📂 Using cache directory: {cache_dir}")
    
    analysis_results = analyze_query_results(query_results_path, cache_dir)
    
    # Calculate statistics
    stats = calculate_statistics(analysis_results)

    # Print report
    print_report(stats)
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    create_visualizations(analysis_results, output_dir)
    
    # Save detailed report
    output_path = Path(args.output)
    report_data = {
        'statistics': stats,
        'detailed_results': analysis_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n✅ Detailed report saved to: {output_path}")


if __name__ == "__main__":
    main()

