#!/usr/bin/env python3
"""
Calculate overall retrieval accuracy from iteration logs.

This script scans all iteration_log.json files in the results directory
and extracts the final best subgraph overlap (retrieval accuracy).

Usage:
    python calculate_accuracy.py --results-dir ava100_results
    python calculate_accuracy.py --results-dir ava100_results --output accuracy_report.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def extract_accuracy_from_log(log_path: Path) -> Dict:
    """
    Extract accuracy metrics from a single iteration_log.json file.
    
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
                'selection_loss': None
            }
        
        # Find final evaluations (there may be 1 or 2 depending on two-stage evaluation)
        final_entries = [x for x in iterations if x.get('iteration') == 'final_best_subgraph']
        
        # BACKWARD COMPATIBILITY: If no final_best_subgraph entry, try OLD format
        # OLD format: Find the last iteration and extract the best subgraph from all subgraphs
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
                'selection_loss': None
            }
        
        # Extract accuracies
        stage1_accuracy = None
        stage2_accuracy = None
        selection_loss = None
        
        # Stage 1: After full exploration (always present)
        if len(final_entries) >= 1:
            best_sg = final_entries[0].get('best_subgraph', {})
            stage1_accuracy = best_sg.get('overlap')
            stage1_nodes = best_sg.get('num_nodes', 0)
            stage1_events = best_sg.get('num_events', 0)
            stage1_objects = best_sg.get('num_objects', 0)
        
        # Stage 2: After LLM pruning (optional - only if two-stage evaluation is enabled)
        if len(final_entries) >= 2:
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
        
        return {
            'video_key': video_key,
            'question_id': question_id,
            'query': query,
            'time_reference': time_reference,
            'status': 'success',
            'stage1_accuracy': stage1_accuracy,
            'stage2_accuracy': stage2_accuracy,
            'selection_loss': selection_loss,
            'stage1_nodes': stage1_nodes,
            'stage1_events': stage1_events,
            'stage1_objects': stage1_objects,
            'stage2_nodes': stage2_nodes,
            'stage2_events': stage2_events,
            'stage2_objects': stage2_objects,
        }
        
    except Exception as e:
        print(f"⚠️  Error reading {log_path}: {e}")
        return None


def scan_results_directory(results_dir: Path) -> List[Dict]:
    """
    Scan all iteration_log.json files in the results directory.
    
    Returns:
        List of accuracy data dictionaries
    """
    all_results = []
    
    # Find all iteration_log.json files
    log_files = list(results_dir.glob("*/q*/iteration_log.json"))
    
    print(f"Found {len(log_files)} iteration log files")
    
    for log_path in sorted(log_files):
        accuracy_data = extract_accuracy_from_log(log_path)
        if accuracy_data:
            all_results.append(accuracy_data)
    
    return all_results


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate overall statistics from all results.
    
    Returns:
        Dictionary with aggregated statistics
    """
    # Filter valid results (have stage1_accuracy)
    valid_results = [r for r in results if r['stage1_accuracy'] is not None]
    
    # Group by video
    by_video = defaultdict(list)
    for r in valid_results:
        by_video[r['video_key']].append(r)
    
    # Calculate overall stats
    stage1_accuracies = [r['stage1_accuracy'] for r in valid_results]
    stage2_accuracies = [r['stage2_accuracy'] for r in valid_results if r['stage2_accuracy'] is not None]
    selection_losses = [r['selection_loss'] for r in valid_results if r['selection_loss'] is not None]
    
    # Per-video stats
    video_stats = {}
    for video_key, video_results in by_video.items():
        video_stage1 = [r['stage1_accuracy'] for r in video_results]
        video_stage2 = [r['stage2_accuracy'] for r in video_results if r['stage2_accuracy'] is not None]
        
        video_stats[video_key] = {
            'num_questions': len(video_results),
            'stage1_avg_accuracy': sum(video_stage1) / len(video_stage1) if video_stage1 else 0,
            'stage2_avg_accuracy': sum(video_stage2) / len(video_stage2) if video_stage2 else 0,
            'stage1_max': max(video_stage1) if video_stage1 else 0,
            'stage1_min': min(video_stage1) if video_stage1 else 0,
        }
    
    # Status breakdown
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r['status']] += 1
    
    return {
        'total_questions': len(results),
        'valid_questions': len(valid_results),
        'invalid_questions': len(results) - len(valid_results),
        'status_breakdown': dict(status_counts),
        'stage1_overall': {
            'avg_accuracy': sum(stage1_accuracies) / len(stage1_accuracies) if stage1_accuracies else 0,
            'max_accuracy': max(stage1_accuracies) if stage1_accuracies else 0,
            'min_accuracy': min(stage1_accuracies) if stage1_accuracies else 0,
            'num_perfect': sum(1 for a in stage1_accuracies if a >= 0.99),
            'num_good': sum(1 for a in stage1_accuracies if 0.7 <= a < 0.99),
            'num_fair': sum(1 for a in stage1_accuracies if 0.3 <= a < 0.7),
            'num_poor': sum(1 for a in stage1_accuracies if a < 0.3),
        },
        'stage2_overall': {
            'avg_accuracy': sum(stage2_accuracies) / len(stage2_accuracies) if stage2_accuracies else 0,
            'max_accuracy': max(stage2_accuracies) if stage2_accuracies else 0,
            'min_accuracy': min(stage2_accuracies) if stage2_accuracies else 0,
            'num_perfect': sum(1 for a in stage2_accuracies if a >= 0.99),
            'num_good': sum(1 for a in stage2_accuracies if 0.7 <= a < 0.99),
            'num_fair': sum(1 for a in stage2_accuracies if 0.3 <= a < 0.7),
            'num_poor': sum(1 for a in stage2_accuracies if a < 0.3),
        } if stage2_accuracies else None,
        'selection_loss': {
            'avg_loss': sum(selection_losses) / len(selection_losses) if selection_losses else 0,
            'max_loss': max(selection_losses) if selection_losses else 0,
            'num_with_loss': sum(1 for sl in selection_losses if sl > 0.01),
        } if selection_losses else None,
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
    
    # Stage 1 results
    if stats['valid_questions'] > 0:
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
    else:
        print(f"\n⚠️  No valid evaluations found. All questions are missing iteration data.")
    
    # Stage 2 results (if available)
    if stats['stage2_overall'] and stats['valid_questions'] > 0:
        stage2 = stats['stage2_overall']
        print(f"\n🎯 STAGE 2: After LLM Pruning")
        print(f"  Average Accuracy: {stage2['avg_accuracy']:.3f}")
        print(f"  Max Accuracy: {stage2['max_accuracy']:.3f}")
        print(f"  Min Accuracy: {stage2['min_accuracy']:.3f}")
        print(f"\n  Performance Distribution:")
        print(f"    Perfect (≥0.99): {stage2['num_perfect']} ({stage2['num_perfect']/stats['valid_questions']*100:.1f}%)")
        print(f"    Good (0.7-0.99): {stage2['num_good']} ({stage2['num_good']/stats['valid_questions']*100:.1f}%)")
        print(f"    Fair (0.3-0.7):  {stage2['num_fair']} ({stage2['num_fair']/stats['valid_questions']*100:.1f}%)")
        print(f"    Poor (<0.3):     {stage2['num_poor']} ({stage2['num_poor']/stats['valid_questions']*100:.1f}%)")
    
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
            print(f"    Stage 1 Avg: {vstats['stage1_avg_accuracy']:.3f} (range: {vstats['stage1_min']:.3f} - {vstats['stage1_max']:.3f})")
            if vstats['stage2_avg_accuracy'] > 0:
                print(f"    Stage 2 Avg: {vstats['stage2_avg_accuracy']:.3f}")
    else:
        print(f"\n📹 No valid per-video statistics available")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Calculate retrieval accuracy from iteration logs")
    parser.add_argument('--results-dir', type=str, default='ava100_results',
                        help='Directory containing results (default: ava100_results)')
    parser.add_argument('--output', type=str, default='accuracy_report.json',
                        help='Output file for detailed report (default: accuracy_report.json)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Optional: Export detailed results to CSV')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        return
    
    print(f"📂 Scanning results directory: {results_dir}")
    
    # Scan all logs
    all_results = scan_results_directory(results_dir)
    
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

