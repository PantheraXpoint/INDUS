#!/usr/bin/env python3
"""
VDB Relations Statistics Analysis
Analyzes the distribution of relations per object (and unique neighbors per object)
across all videos, using vdb_entities.json (object headcount) and vdb_relations.json.
Stats are computed over all objects: entities with no relations count as 0.
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    plt.style.use('default')

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def load_json_file(file_path: str) -> dict:
    """Load a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calc_stats(values: List[float]) -> Dict:
    """Compute mean, median, std, min, max, q25, q75 for a list of values."""
    if not values:
        return {
            "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0,
            "q25": 0, "q75": 0, "total": 0
        }
    arr = np.array(values, dtype=float)
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


def analyze_video(video_id, base_path: str = "AVA_cache/AVA100", video_name: str = None) -> Dict:
    """
    Analyze a single video's vdb_relations statistics.
    Uses vdb_entities.json for object headcount; entities not in any relation get 0.
    Returns dict with relations_per_object and unique_neighbors_per_object stats.
    """
    kg_path = Path(base_path) / str(video_id) / "kg"
    entities_file = kg_path / "vdb_entities.json"
    relations_file = kg_path / "vdb_relations.json"

    if not entities_file.exists() or not relations_file.exists():
        print(f"Warning: Missing entities or relations for video {video_id}")
        return None

    print(f"Loading data for video {video_id}...")
    entities_data = load_json_file(str(entities_file))
    relations_data = load_json_file(str(relations_file))

    entities = entities_data.get("data", [])
    relations = relations_data.get("data", [])

    # Object headcount from entities
    entity_ids = [e["id"] for e in entities]
    num_entities = len(entity_ids)
    entity_index = {eid: i for i, eid in enumerate(entity_ids)}

    # Degree: number of relations each entity participates in (entity1 or entity2)
    degree = [0] * num_entities
    # Neighbors: set of distinct other-entity indices per entity
    neighbors = defaultdict(set)

    for rel in relations:
        e1, e2 = rel.get("entity1"), rel.get("entity2")
        if e1 is None or e2 is None:
            continue
        i1 = entity_index.get(e1)
        i2 = entity_index.get(e2)
        if i1 is not None:
            degree[i1] += 1
            if i2 is not None and i2 != i1:
                neighbors[i1].add(i2)
        if i2 is not None:
            degree[i2] += 1
            if i1 is not None and i1 != i2:
                neighbors[i2].add(i1)

    # Unique neighbors count per object (by object headcount; 0 if no relations)
    unique_neighbors = [len(neighbors[i]) for i in range(num_entities)]

    num_relations = len(relations)
    num_with_zero_relations = sum(1 for d in degree if d == 0)

    stats = {
        "video_id": video_id,
        "video_name": video_name or str(video_id),
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_entities_with_zero_relations": num_with_zero_relations,
        "relations_per_object": calc_stats(degree),
        "unique_neighbors_per_object": calc_stats(unique_neighbors),
        "raw_relations_per_object": degree,
        "raw_unique_neighbors_per_object": unique_neighbors,
    }
    print(f"  - Entities: {num_entities}, Relations: {num_relations}, Objects with 0 relations: {num_with_zero_relations}")
    return stats


def analyze_all_videos(base_path: str = "AVA_cache/AVA100", video_ids: List = None, dataset_name: str = None) -> Dict:
    """Analyze all videos that have both vdb_entities.json and vdb_relations.json."""
    if video_ids is None:
        base = Path(base_path)
        video_ids = []
        for d in base.iterdir():
            if not d.is_dir():
                continue
            kg = base / d.name / "kg"
            if not kg.exists():
                continue
            if not (kg / "vdb_entities.json").exists() or not (kg / "vdb_relations.json").exists():
                continue
            try:
                video_ids.append(int(d.name))
            except ValueError:
                if d.name.lstrip("-").isdigit():
                    video_ids.append(int(d.name))
                else:
                    video_ids.append(d.name)
        video_ids = sorted(video_ids, key=lambda x: (x if isinstance(x, int) else 0))

    all_stats = []
    for video_id in video_ids:
        video_name = f"{dataset_name or 'Video'}-{video_id}"
        s = analyze_video(video_id, base_path, video_name)
        if s:
            all_stats.append(s)

    return {
        "per_video": all_stats,
        "video_ids": video_ids,
        "dataset_name": dataset_name or base_path,
    }


def print_statistics(all_stats: Dict):
    """Print formatted statistics to console."""
    print("\n" + "=" * 80)
    print(f"VDB RELATIONS STATISTICS - {all_stats.get('dataset_name', 'Unknown')}")
    print("=" * 80)

    for stats in all_stats["per_video"]:
        video_id = stats["video_id"]
        print(f"\n{'='*80}")
        print(f"VIDEO {video_id}")
        print(f"{'='*80}")
        print(f"Total Entities (Objects): {stats['num_entities']}")
        print(f"Total Relations: {stats['num_relations']}")
        print(f"Entities with 0 relations: {stats['num_entities_with_zero_relations']}")

        print(f"\n--- Relations per Object ---")
        rpo = stats["relations_per_object"]
        print(f"  Mean:   {rpo['mean']:.2f}")
        print(f"  Median: {rpo['median']:.2f}")
        print(f"  Std:    {rpo['std']:.2f}")
        print(f"  Min:    {rpo['min']:.0f}")
        print(f"  Max:    {rpo['max']:.0f}")
        print(f"  Q25:    {rpo['q25']:.2f}")
        print(f"  Q75:    {rpo['q75']:.2f}")

        print(f"\n--- Unique Neighbors per Object ---")
        uno = stats["unique_neighbors_per_object"]
        print(f"  Mean:   {uno['mean']:.2f}")
        print(f"  Median: {uno['median']:.2f}")
        print(f"  Std:    {uno['std']:.2f}")
        print(f"  Min:    {uno['min']:.0f}")
        print(f"  Max:    {uno['max']:.0f}")
        print(f"  Q25:    {uno['q25']:.2f}")
        print(f"  Q75:    {uno['q75']:.2f}")

    print(f"\n{'='*80}")
    print("OVERALL STATISTICS (Across All Videos)")
    print(f"{'='*80}")

    all_rpo = []
    all_uno = []
    for stats in all_stats["per_video"]:
        all_rpo.extend(stats["raw_relations_per_object"])
        all_uno.extend(stats["raw_unique_neighbors_per_object"])

    def print_overall(name: str, values: List[float]):
        if not values:
            print(f"\n--- {name} --- (no data)")
            return
        arr = np.array(values)
        print(f"\n--- {name} ---")
        print(f"  Mean:   {np.mean(arr):.2f}")
        print(f"  Median: {np.median(arr):.2f}")
        print(f"  Std:    {np.std(arr):.2f}")
        print(f"  Min:    {np.min(arr):.0f}")
        print(f"  Max:    {np.max(arr):.0f}")
        print(f"  Q25:    {np.percentile(arr, 25):.2f}")
        print(f"  Q75:    {np.percentile(arr, 75):.2f}")
        print(f"  Total samples (objects): {len(values)}")

    print_overall("Relations per Object (Overall)", all_rpo)
    print_overall("Unique Neighbors per Object (Overall)", all_uno)


def create_visualizations(all_stats: Dict, output_dir: str = "ECML-PKDD/kg_analysis_output"):
    """Create plots matching analyze_kg_statistics style (relations per object, unique neighbors)."""
    os.makedirs(output_dir, exist_ok=True)
    per_video_stats = all_stats["per_video"]
    video_ids = [s["video_id"] for s in per_video_stats]

    # 1. Overview: box + hist for relations per object, box + hist for unique neighbors
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Box: Relations per Object by video
    ax1 = axes[0, 0]
    data_rpo = [s["raw_relations_per_object"] for s in per_video_stats]
    bp1 = ax1.boxplot(data_rpo, labels=[f"Video {vid}" for vid in video_ids],
                      patch_artist=True, showmeans=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_title('Distribution of Relations per Object (by Video)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Video ID', fontsize=12)
    ax1.set_ylabel('Number of Relations', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Histogram: Relations per Object (overall)
    ax2 = axes[0, 1]
    all_rpo = []
    for s in per_video_stats:
        all_rpo.extend(s["raw_relations_per_object"])
    if all_rpo:
        ax2.hist(all_rpo, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.axvline(np.mean(all_rpo), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_rpo):.2f}')
        ax2.axvline(np.median(all_rpo), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_rpo):.2f}')
    ax2.set_title('Overall Distribution: Relations per Object', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Relations per Object', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Box: Unique Neighbors per Object by video
    ax3 = axes[1, 0]
    data_uno = [s["raw_unique_neighbors_per_object"] for s in per_video_stats]
    bp2 = ax3.boxplot(data_uno, labels=[f"Video {vid}" for vid in video_ids],
                      patch_artist=True, showmeans=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    ax3.set_title('Distribution of Unique Neighbors per Object (by Video)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Video ID', fontsize=12)
    ax3.set_ylabel('Number of Unique Neighbors', fontsize=12)
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Histogram: Unique Neighbors per Object (overall)
    ax4 = axes[1, 1]
    all_uno = []
    for s in per_video_stats:
        all_uno.extend(s["raw_unique_neighbors_per_object"])
    if all_uno:
        ax4.hist(all_uno, bins=50, edgecolor='black', alpha=0.7, color='salmon')
        ax4.axvline(np.mean(all_uno), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_uno):.2f}')
        ax4.axvline(np.median(all_uno), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_uno):.2f}')
    ax4.set_title('Overall Distribution: Unique Neighbors per Object', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Number of Unique Neighbors per Object', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relations_statistics_overview.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/relations_statistics_overview.png")
    plt.close()

    # 2. Comparison across videos: mean, median, max, std
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x = np.arange(len(video_ids))
    width = 0.35

    # Mean
    ax1 = axes[0, 0]
    means_rpo = [s["relations_per_object"]["mean"] for s in per_video_stats]
    means_uno = [s["unique_neighbors_per_object"]["mean"] for s in per_video_stats]
    ax1.bar(x - width/2, means_rpo, width, label='Relations per Object', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, means_uno, width, label='Unique Neighbors per Object', color='coral', alpha=0.8)
    ax1.set_xlabel('Video ID', fontsize=12)
    ax1.set_ylabel('Mean', fontsize=12)
    ax1.set_title('Mean Comparison Across Videos', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Video {vid}" for vid in video_ids], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Median
    ax2 = axes[0, 1]
    medians_rpo = [s["relations_per_object"]["median"] for s in per_video_stats]
    medians_uno = [s["unique_neighbors_per_object"]["median"] for s in per_video_stats]
    ax2.bar(x - width/2, medians_rpo, width, label='Relations per Object', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, medians_uno, width, label='Unique Neighbors per Object', color='coral', alpha=0.8)
    ax2.set_xlabel('Video ID', fontsize=12)
    ax2.set_ylabel('Median', fontsize=12)
    ax2.set_title('Median Comparison Across Videos', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Video {vid}" for vid in video_ids], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Max
    ax3 = axes[1, 0]
    maxs_rpo = [s["relations_per_object"]["max"] for s in per_video_stats]
    maxs_uno = [s["unique_neighbors_per_object"]["max"] for s in per_video_stats]
    ax3.bar(x - width/2, maxs_rpo, width, label='Relations per Object', color='steelblue', alpha=0.8)
    ax3.bar(x + width/2, maxs_uno, width, label='Unique Neighbors per Object', color='coral', alpha=0.8)
    ax3.set_xlabel('Video ID', fontsize=12)
    ax3.set_ylabel('Max', fontsize=12)
    ax3.set_title('Max Comparison Across Videos', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Video {vid}" for vid in video_ids], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Std
    ax4 = axes[1, 1]
    stds_rpo = [s["relations_per_object"]["std"] for s in per_video_stats]
    stds_uno = [s["unique_neighbors_per_object"]["std"] for s in per_video_stats]
    ax4.bar(x - width/2, stds_rpo, width, label='Relations per Object', color='steelblue', alpha=0.8)
    ax4.bar(x + width/2, stds_uno, width, label='Unique Neighbors per Object', color='coral', alpha=0.8)
    ax4.set_xlabel('Video ID', fontsize=12)
    ax4.set_ylabel('Standard Deviation', fontsize=12)
    ax4.set_title('Std Comparison Across Videos', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"Video {vid}" for vid in video_ids], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relations_statistics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/relations_statistics_comparison.png")
    plt.close()

    # 3. Violin plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    video_labels = [f"Video {vid}" for vid in video_ids]

    ax1 = axes[0]
    parts1 = ax1.violinplot(data_rpo, positions=range(len(video_ids)), showmeans=True, showmedians=True)
    for pc in parts1['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    ax1.set_xticks(range(len(video_ids)))
    ax1.set_xticklabels(video_labels, rotation=45, ha='right')
    ax1.set_ylabel('Number of Relations', fontsize=12)
    ax1.set_title('Relations per Object (Violin)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    parts2 = ax2.violinplot(data_uno, positions=range(len(video_ids)), showmeans=True, showmedians=True)
    for pc in parts2['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    ax2.set_xticks(range(len(video_ids)))
    ax2.set_xticklabels(video_labels, rotation=45, ha='right')
    ax2.set_ylabel('Number of Unique Neighbors', fontsize=12)
    ax2.set_title('Unique Neighbors per Object (Violin)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relations_statistics_violin.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/relations_statistics_violin.png")
    plt.close()

    print(f"\nAll visualizations saved to: {output_dir}/")


def save_json_report(all_stats: Dict, output_file: str = "ECML-PKDD/kg_analysis_output/relations_statistics_report.json"):
    """Save statistics to JSON (excluding raw lists)."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    export_stats = []
    for stats in all_stats["per_video"]:
        export_stat = {k: v for k, v in stats.items()
                      if k not in ["raw_relations_per_object", "raw_unique_neighbors_per_object"]}
        export_stats.append(export_stat)
    report = {
        "per_video": export_stats,
        "video_ids": all_stats["video_ids"],
        "dataset_name": all_stats.get("dataset_name", "Unknown"),
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON report: {output_file}")


def analyze_dataset(dataset_path: str, dataset_name: str, output_subdir: str = None):
    """Run full analysis for one dataset."""
    print("\n" + "=" * 80)
    print(f"Analyzing relations for dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print("=" * 80)

    all_stats = analyze_all_videos(dataset_path, dataset_name=dataset_name)
    if not all_stats["per_video"]:
        print(f"No videos with both vdb_entities and vdb_relations in {dataset_path}!")
        return None

    print(f"\nFound {len(all_stats['per_video'])} videos with relation data.")
    print_statistics(all_stats)

    output_dir = f"ECML-PKDD/kg_analysis_output/{output_subdir or dataset_name}"
    print("\nGenerating visualizations...")
    create_visualizations(all_stats, output_dir=output_dir)
    save_json_report(all_stats, output_file=f"{output_dir}/relations_statistics_report.json")
    return all_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze VDB Relations Statistics (relations per object, etc.)')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'AVA100', 'LVBench', 'VideoMME'],
                        help='Which dataset to analyze (default: all)')
    parser.add_argument('--base-dir', type=str, default='AVA_cache',
                        help='Base directory containing datasets (default: AVA_cache)')
    args = parser.parse_args()

    print("Starting VDB Relations Statistics Analysis...")
    if args.dataset == 'all':
        datasets_to_analyze = [('AVA100', 'AVA100'), ('LVBench', 'LVBench'), ('VideoMME', 'VideoMME')]
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

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)
    for dataset_name, result in results.items():
        print(f"\n{dataset_name}:")
        print(f"  - Videos analyzed: {len(result['per_video'])}")
        print(f"  - Total entities: {sum(s['num_entities'] for s in result['per_video'])}")
        print(f"  - Total relations: {sum(s['num_relations'] for s in result['per_video'])}")
        print(f"  - Output: ECML-PKDD/kg_analysis_output/{dataset_name}/")
    print("\n" + "=" * 80)
    print("All analyses complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
