#!/usr/bin/env python3
"""
Knowledge Graph Statistics Analysis
Analyzes the distribution of objects per event and events per object
across all videos in AVA_cache/AVA100/
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Try to import seaborn for better styling (optional)
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("Note: seaborn not available, using matplotlib default style")
    plt.style.use('default')

# Set style for better-looking plots
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def load_json_file(file_path: str) -> dict:
    """Load a JSON file, handling large files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_video(video_id: int, base_path: str = "AVA_cache/AVA100", video_name: str = None) -> Dict:
    """
    Analyze a single video's knowledge graph statistics.
    
    Returns:
        Dictionary with statistics for objects per event and events per object
    """
    kg_path = Path(base_path) / str(video_id) / "kg"
    
    events_file = kg_path / "vdb_events.json"
    entities_file = kg_path / "vdb_entities.json"
    
    if not events_file.exists() or not entities_file.exists():
        print(f"Warning: Missing files for video {video_id}")
        return None
    
    # Load data
    print(f"Loading data for video {video_id}...")
    events_data = load_json_file(str(events_file))
    entities_data = load_json_file(str(entities_file))
    
    events = events_data.get("data", [])
    entities = entities_data.get("data", [])
    
    print(f"  - Found {len(events)} events and {len(entities)} entities")
    
    # Build event -> objects mapping
    event_to_objects = defaultdict(list)
    object_to_events_count = []
    
    for entity in entities:
        entity_events = entity.get("events", [])
        num_events = len(entity_events)
        object_to_events_count.append(num_events)
        
        # Add this entity to all its events
        for event_id in entity_events:
            event_to_objects[event_id].append(entity["id"])
    
    # Calculate objects per event
    objects_per_event = [len(event_to_objects[event["id"]]) for event in events]

    # Sanity check: total entity-event links must match from both sides
    # (L = sum(events_per_object) = sum(objects_per_event)); can differ if entity["events"] references missing event IDs
    L_from_entities = sum(object_to_events_count)
    L_from_events = sum(objects_per_event)
    if L_from_entities != L_from_events:
        n_orphan = L_from_entities - L_from_events
        print(f"  - Note: {n_orphan} entity->event ref(s) point to event IDs not in events list (L_ent={L_from_entities}, L_ev={L_from_events})")

    # Calculate statistics
    def calc_stats(values: List[float]) -> Dict:
        if not values:
            return {
                "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0,
                "q25": 0, "q75": 0, "total": 0
            }
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "total": len(values)
        }
    
    stats = {
        "video_id": video_id,
        "video_name": video_name or str(video_id),
        "num_events": len(events),
        "num_entities": len(entities),
        "objects_per_event": calc_stats(objects_per_event),
        "events_per_object": calc_stats(object_to_events_count),
        "raw_objects_per_event": objects_per_event,
        "raw_events_per_object": object_to_events_count
    }
    
    return stats


def analyze_all_videos(base_path: str = "AVA_cache/AVA100", video_ids: List[int] = None, dataset_name: str = None) -> Dict:
    """Analyze all videos and return aggregated statistics."""
    if video_ids is None:
        # Auto-detect video folders
        base = Path(base_path)
        video_ids = sorted([int(d.name) for d in base.iterdir() 
                           if d.is_dir() and d.name.isdigit() and (base / d.name / "kg").exists()])
    
    all_stats = []
    for video_id in video_ids:
        video_name = f"{dataset_name or 'Video'}-{video_id}"
        stats = analyze_video(video_id, base_path, video_name)
        if stats:
            all_stats.append(stats)
    
    return {
        "per_video": all_stats,
        "video_ids": video_ids,
        "dataset_name": dataset_name or base_path
    }


def print_statistics(all_stats: Dict):
    """Print formatted statistics to console."""
    print("\n" + "="*80)
    print(f"KNOWLEDGE GRAPH STATISTICS ANALYSIS - {all_stats.get('dataset_name', 'Unknown')}")
    print("="*80)
    
    for stats in all_stats["per_video"]:
        video_id = stats["video_id"]
        print(f"\n{'='*80}")
        print(f"VIDEO {video_id}")
        print(f"{'='*80}")
        print(f"Total Events: {stats['num_events']}")
        print(f"Total Entities (Objects): {stats['num_entities']}")
        
        print(f"\n--- Objects per Event ---")
        ope = stats["objects_per_event"]
        print(f"  Mean:   {ope['mean']:.2f}")
        print(f"  Median: {ope['median']:.2f}")
        print(f"  Std:    {ope['std']:.2f}")
        print(f"  Min:    {ope['min']:.0f}")
        print(f"  Max:    {ope['max']:.0f}")
        print(f"  Q25:    {ope['q25']:.2f}")
        print(f"  Q75:    {ope['q75']:.2f}")
        
        print(f"\n--- Events per Object ---")
        epo = stats["events_per_object"]
        print(f"  Mean:   {epo['mean']:.2f}")
        print(f"  Median: {epo['median']:.2f}")
        print(f"  Std:    {epo['std']:.2f}")
        print(f"  Min:    {epo['min']:.0f}")
        print(f"  Max:    {epo['max']:.0f}")
        print(f"  Q25:    {epo['q25']:.2f}")
        print(f"  Q75:    {epo['q75']:.2f}")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS (Across All Videos)")
    print(f"{'='*80}")
    
    all_objects_per_event = []
    all_events_per_object = []
    
    for stats in all_stats["per_video"]:
        all_objects_per_event.extend(stats["raw_objects_per_event"])
        all_events_per_object.extend(stats["raw_events_per_object"])
    
    def print_overall_stats(name: str, values: List[float]):
        arr = np.array(values)
        print(f"\n--- {name} ---")
        print(f"  Mean:   {np.mean(arr):.2f}")
        print(f"  Median: {np.median(arr):.2f}")
        print(f"  Std:    {np.std(arr):.2f}")
        print(f"  Min:    {np.min(arr):.0f}")
        print(f"  Max:    {np.max(arr):.0f}")
        print(f"  Q25:    {np.percentile(arr, 25):.2f}")
        print(f"  Q75:    {np.percentile(arr, 75):.2f}")
        print(f"  Total samples: {len(values)}")
    
    print_overall_stats("Objects per Event (Overall)", all_objects_per_event)
    print_overall_stats("Events per Object (Overall)", all_events_per_object)

    # Explain why mean(events_per_object) can look similar across videos
    print("\n--- Why mean(events_per_object) is similar across videos ---")
    print("  mean(events_per_object) = total_entity_event_links / num_entities.")
    print("  So if every video has ~2.5 links per entity on average (pipeline/data property),")
    print("  that mean will be ~2.5 for all. The computation is correct; use median/std for comparison.")


def create_visualizations(all_stats: Dict, output_dir: str = "ECML-PKDD/kg_analysis_output"):
    """Create visualization plots for the statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    per_video_stats = all_stats["per_video"]
    num_videos = len(per_video_stats)
    
    # Prepare data for plotting
    video_ids = [s["video_id"] for s in per_video_stats]
    
    # 1. Objects per Event - Box plot across videos
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Box plot: Objects per Event by video
    ax1 = axes[0, 0]
    data_ope = [s["raw_objects_per_event"] for s in per_video_stats]
    bp1 = ax1.boxplot(data_ope, labels=[f"Video {vid}" for vid in video_ids], 
                      patch_artist=True, showmeans=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_title('Distribution of Objects per Event (by Video)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Video ID', fontsize=12)
    ax1.set_ylabel('Number of Objects', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Histogram: Objects per Event (overall)
    ax2 = axes[0, 1]
    all_ope = []
    for s in per_video_stats:
        all_ope.extend(s["raw_objects_per_event"])
    ax2.hist(all_ope, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(np.mean(all_ope), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(all_ope):.2f}')
    ax2.axvline(np.median(all_ope), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(all_ope):.2f}')
    ax2.set_title('Overall Distribution: Objects per Event', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Objects per Event', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Box plot: Events per Object by video
    ax3 = axes[1, 0]
    data_epo = [s["raw_events_per_object"] for s in per_video_stats]
    bp2 = ax3.boxplot(data_epo, labels=[f"Video {vid}" for vid in video_ids], 
                      patch_artist=True, showmeans=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    ax3.set_title('Distribution of Events per Object (by Video)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Video ID', fontsize=12)
    ax3.set_ylabel('Number of Events', fontsize=12)
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Histogram: Events per Object (overall)
    ax4 = axes[1, 1]
    all_epo = []
    for s in per_video_stats:
        all_epo.extend(s["raw_events_per_object"])
    ax4.hist(all_epo, bins=50, edgecolor='black', alpha=0.7, color='salmon')
    ax4.axvline(np.mean(all_epo), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(all_epo):.2f}')
    ax4.axvline(np.median(all_epo), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(all_epo):.2f}')
    ax4.set_title('Overall Distribution: Events per Object', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Number of Events per Object', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kg_statistics_overview.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/kg_statistics_overview.png")
    plt.close()
    
    # 2. Summary statistics comparison across videos
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Mean comparison
    ax1 = axes[0, 0]
    means_ope = [s["objects_per_event"]["mean"] for s in per_video_stats]
    means_epo = [s["events_per_object"]["mean"] for s in per_video_stats]
    x = np.arange(len(video_ids))
    width = 0.35
    ax1.bar(x - width/2, means_ope, width, label='Objects per Event', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, means_epo, width, label='Events per Object', color='coral', alpha=0.8)
    ax1.set_xlabel('Video ID', fontsize=12)
    ax1.set_ylabel('Mean Value', fontsize=12)
    ax1.set_title('Mean Comparison Across Videos', fontsize=14, fontweight='bold')
    ax1.text(0.02, 0.98, 'mean(events/object)=L/num_entities (often ~2.5)', transform=ax1.transAxes, fontsize=9, va='top')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Video {vid}" for vid in video_ids])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Median comparison
    ax2 = axes[0, 1]
    medians_ope = [s["objects_per_event"]["median"] for s in per_video_stats]
    medians_epo = [s["events_per_object"]["median"] for s in per_video_stats]
    ax2.bar(x - width/2, medians_ope, width, label='Objects per Event', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, medians_epo, width, label='Events per Object', color='coral', alpha=0.8)
    ax2.set_xlabel('Video ID', fontsize=12)
    ax2.set_ylabel('Median Value', fontsize=12)
    ax2.set_title('Median Comparison Across Videos', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Video {vid}" for vid in video_ids])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Max comparison
    ax3 = axes[1, 0]
    maxs_ope = [s["objects_per_event"]["max"] for s in per_video_stats]
    maxs_epo = [s["events_per_object"]["max"] for s in per_video_stats]
    ax3.bar(x - width/2, maxs_ope, width, label='Objects per Event', color='steelblue', alpha=0.8)
    ax3.bar(x + width/2, maxs_epo, width, label='Events per Object', color='coral', alpha=0.8)
    ax3.set_xlabel('Video ID', fontsize=12)
    ax3.set_ylabel('Max Value', fontsize=12)
    ax3.set_title('Max Comparison Across Videos', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Video {vid}" for vid in video_ids])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Standard deviation comparison
    ax4 = axes[1, 1]
    stds_ope = [s["objects_per_event"]["std"] for s in per_video_stats]
    stds_epo = [s["events_per_object"]["std"] for s in per_video_stats]
    ax4.bar(x - width/2, stds_ope, width, label='Objects per Event', color='steelblue', alpha=0.8)
    ax4.bar(x + width/2, stds_epo, width, label='Events per Object', color='coral', alpha=0.8)
    ax4.set_xlabel('Video ID', fontsize=12)
    ax4.set_ylabel('Standard Deviation', fontsize=12)
    ax4.set_title('Standard Deviation Comparison Across Videos', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"Video {vid}" for vid in video_ids])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kg_statistics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/kg_statistics_comparison.png")
    plt.close()
    
    # 3. Violin plots for better distribution visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data for violin plots
    video_labels = [f"Video {vid}" for vid in video_ids]
    
    ax1 = axes[0]
    parts = ax1.violinplot(data_ope, positions=range(len(video_ids)), 
                           showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    ax1.set_xticks(range(len(video_ids)))
    ax1.set_xticklabels(video_labels, rotation=45, ha='right')
    ax1.set_ylabel('Number of Objects', fontsize=12)
    ax1.set_title('Objects per Event Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[1]
    parts = ax2.violinplot(data_epo, positions=range(len(video_ids)), 
                           showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    ax2.set_xticks(range(len(video_ids)))
    ax2.set_xticklabels(video_labels, rotation=45, ha='right')
    ax2.set_ylabel('Number of Events', fontsize=12)
    ax2.set_title('Events per Object Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kg_statistics_violin.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/kg_statistics_violin.png")
    plt.close()
    
    print(f"\nAll visualizations saved to: {output_dir}/")


def save_json_report(all_stats: Dict, output_file: str = "ECML-PKDD/kg_analysis_output/statistics_report.json"):
    """Save statistics to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Remove raw data for JSON (too large)
    export_stats = []
    for stats in all_stats["per_video"]:
        export_stat = {k: v for k, v in stats.items() 
                      if k not in ["raw_objects_per_event", "raw_events_per_object"]}
        export_stats.append(export_stat)
    
    report = {
        "per_video": export_stats,
        "video_ids": all_stats["video_ids"],
        "dataset_name": all_stats.get("dataset_name", "Unknown")
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Saved JSON report: {output_file}")


def analyze_dataset(dataset_path: str, dataset_name: str, output_subdir: str = None):
    """Analyze a single dataset."""
    print("\n" + "="*80)
    print(f"Analyzing dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print("="*80)
    
    # Analyze all videos in this dataset
    all_stats = analyze_all_videos(dataset_path, dataset_name=dataset_name)
    
    if not all_stats["per_video"]:
        print(f"No video data found in {dataset_path}!")
        return None
    
    print(f"\nFound {len(all_stats['per_video'])} videos with knowledge graph data.")
    
    # Print statistics
    print_statistics(all_stats)
    
    # Create output directory for this dataset
    output_dir = f"ECML-PKDD/kg_analysis_output/{output_subdir or dataset_name}"
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(all_stats, output_dir=output_dir)
    
    # Save JSON report
    output_file = f"{output_dir}/statistics_report.json"
    save_json_report(all_stats, output_file=output_file)
    
    return all_stats


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Knowledge Graph Statistics')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'AVA100', 'LVBench', 'VideoMME'],
                       help='Which dataset to analyze (default: all)')
    parser.add_argument('--base-dir', type=str, default='AVA_cache',
                       help='Base directory containing datasets (default: AVA_cache)')
    
    args = parser.parse_args()
    
    print("Starting Knowledge Graph Statistics Analysis...")
    
    datasets_to_analyze = []
    
    if args.dataset == 'all':
        # Analyze all three datasets
        datasets_to_analyze = [
            ('AVA100', 'AVA100'),
            ('LVBench', 'LVBench'),
            ('VideoMME', 'VideoMME')
        ]
    else:
        datasets_to_analyze = [(args.dataset, args.dataset)]
    
    results = {}
    for dataset_name, dataset_folder in datasets_to_analyze:
        dataset_path = f"{args.base_dir}/{dataset_folder}"
        if not Path(dataset_path).exists():
            print(f"\nWarning: Dataset path {dataset_path} does not exist. Skipping...")
            continue
        
        result = analyze_dataset(dataset_path, dataset_name, output_subdir=dataset_name)
        if result:
            results[dataset_name] = result
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*80)
    for dataset_name, result in results.items():
        print(f"\n{dataset_name}:")
        print(f"  - Videos analyzed: {len(result['per_video'])}")
        print(f"  - Total events: {sum(s['num_events'] for s in result['per_video'])}")
        print(f"  - Total entities: {sum(s['num_entities'] for s in result['per_video'])}")
        print(f"  - Output: ECML-PKDD/kg_analysis_output/{dataset_name}/")
    
    print("\n" + "="*80)
    print("All analyses complete!")
    print("="*80)


if __name__ == "__main__":
    main()

