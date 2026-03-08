#!/usr/bin/env python3
"""
Object Overlap Between Consecutive Events (AVA100).

Measures how many objects are shared between temporally adjacent events, and compares
to a non-adjacent control group (random pairs with |i-j| >= 5) to assess discriminative power.

Uses vdb_events.json and vdb_entities.json (event->objects from entity['events']).
Events are ordered by duration start (duration[0]).

Metrics per pair (A = event_i, B = event_j):
  - shared_count, continuity_forward, continuity_backward, jaccard, share_any
Non-adjacent: same metrics for sampled pairs with abs(i-j) >= 5 (~500 per video).
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

NON_ADJACENT_SAMPLE_SIZE = 500
MIN_GAP_NON_ADJACENT = 5
RANDOM_SEED = 42

# Optional: reuse expand_indus_seed_events path resolution if running from project root
_project_root = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_event_to_objects(entities: List[dict]) -> Dict[str, List[str]]:
    """Build event_id -> [entity_ids] by inverting entity['events'] (same logic as expand_indus_seed_events._build_event_to_entities)."""
    event_to_objects = defaultdict(list)
    for entity in entities:
        eid = entity.get("id") or entity.get("__id__")
        if not eid:
            continue
        for ev_id in entity.get("events") or []:
            if ev_id:
                event_to_objects[ev_id].append(eid)
    return dict(event_to_objects)


def event_start_time(ev: dict) -> float:
    """Start time for ordering. Uses duration[0] if present, else 0."""
    d = ev.get("duration")
    if d is not None and isinstance(d, (list, tuple)) and len(d) >= 1:
        try:
            return float(d[0])
        except (TypeError, ValueError):
            pass
    return 0.0


def analyze_video(video_id: int, base_path: Path) -> Optional[Dict[str, Any]]:
    kg_path = base_path / str(video_id) / "kg"
    events_file = kg_path / "vdb_events.json"
    entities_file = kg_path / "vdb_entities.json"
    if not events_file.exists() or not entities_file.exists():
        return None

    events_data = load_json(events_file)
    entities_data = load_json(entities_file)
    events = events_data.get("data", [])
    entities = entities_data.get("data", [])

    event_to_objects = build_event_to_objects(entities)

    # Sort events by start time (duration[0]) to get timeline order
    def event_id(ev):
        return ev.get("id") or ev.get("__id__")

    events_sorted = sorted(events, key=event_start_time)
    event_ids_ordered = [event_id(ev) for ev in events_sorted if event_id(ev)]
    n_events = len(event_ids_ordered)

    # Event object-count buckets (how many events have 0, 1, 2, ... objects)
    object_counts = [len(event_to_objects.get(eid, [])) for eid in event_ids_ordered]
    n_events_zero_objects = sum(1 for c in object_counts if c == 0)
    n_events_one_object = sum(1 for c in object_counts if c == 1)
    n_events_two_objects = sum(1 for c in object_counts if c == 2)
    n_events_3_to_5 = sum(1 for c in object_counts if 3 <= c <= 5)
    n_events_6_plus = sum(1 for c in object_counts if c >= 6)

    def compute_pair_metrics(set_a: set, set_b: set) -> Tuple[int, float, float, float, int]:
        shared = set_a & set_b
        n_shared = len(shared)
        n_a, n_b = len(set_a), len(set_b)
        union = len(set_a | set_b)
        fwd = n_shared / n_a if n_a else 0.0
        bwd = n_shared / n_b if n_b else 0.0
        jacc = n_shared / union if union else 0.0
        any_ = 1 if n_shared > 0 else 0
        return n_shared, fwd, bwd, jacc, any_

    # Consecutive pairs
    shared_counts = []
    continuity_forward = []
    continuity_backward = []
    jaccards = []
    share_any = []

    # Optional: bucket continuity_forward by size of A (1 obj, 2-5, 6+)
    fwd_by_size_a: Dict[str, List[float]] = {"0": [], "1": [], "2-5": [], "6+": []}

    for i in range(n_events - 1):
        id_a, id_b = event_ids_ordered[i], event_ids_ordered[i + 1]
        set_a = set(event_to_objects.get(id_a, []))
        set_b = set(event_to_objects.get(id_b, []))
        n_shared, fwd, bwd, jacc, any_ = compute_pair_metrics(set_a, set_b)
        shared_counts.append(n_shared)
        continuity_forward.append(fwd)
        continuity_backward.append(bwd)
        jaccards.append(jacc)
        share_any.append(any_)
        n_a = len(set_a)
        if n_a == 0:
            bucket = "0"
        elif n_a == 1:
            bucket = "1"
        elif n_a <= 5:
            bucket = "2-5"
        else:
            bucket = "6+"
        fwd_by_size_a[bucket].append(fwd)

    n_pairs = len(shared_counts)
    continuity_forward_by_size_a = {
        k: float(np.mean(v)) if v else 0.0 for k, v in fwd_by_size_a.items()
    }

    # Non-adjacent control group: sample pairs with abs(i - j) >= MIN_GAP_NON_ADJACENT
    rng = random.Random(RANDOM_SEED)
    valid_non_adjacent = [(i, j) for i in range(n_events) for j in range(i + MIN_GAP_NON_ADJACENT, n_events)]
    if len(valid_non_adjacent) > NON_ADJACENT_SAMPLE_SIZE:
        sampled_pairs = rng.sample(valid_non_adjacent, NON_ADJACENT_SAMPLE_SIZE)
    else:
        sampled_pairs = valid_non_adjacent

    na_shared = []
    na_fwd = []
    na_bwd = []
    na_jacc = []
    na_any = []
    for i, j in sampled_pairs:
        set_a = set(event_to_objects.get(event_ids_ordered[i], []))
        set_b = set(event_to_objects.get(event_ids_ordered[j], []))
        n_shared, fwd, bwd, jacc, any_ = compute_pair_metrics(set_a, set_b)
        na_shared.append(n_shared)
        na_fwd.append(fwd)
        na_bwd.append(bwd)
        na_jacc.append(jacc)
        na_any.append(any_)

    n_na = len(na_shared)

    def stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        a = np.array(arr, dtype=float)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    base_out = {
        "video_id": video_id,
        "num_events": len(events),
        "num_entities": len(entities),
        "n_events_zero_objects": n_events_zero_objects,
        "n_events_one_object": n_events_one_object,
        "n_events_two_objects": n_events_two_objects,
        "n_events_3_to_5": n_events_3_to_5,
        "n_events_6_plus": n_events_6_plus,
        "num_consecutive_pairs": n_pairs,
        "num_non_adjacent_pairs": n_na,
        "continuity_forward_by_size_a": continuity_forward_by_size_a,
    }

    if n_pairs == 0:
        return {
            **base_out,
            "shared_count": stats([]),
            "continuity_forward": stats([]),
            "continuity_backward": stats([]),
            "jaccard": stats([]),
            "share_any_ratio": 0.0,
            "non_adjacent_shared_count": stats(na_shared),
            "non_adjacent_continuity_forward": stats(na_fwd),
            "non_adjacent_continuity_backward": stats(na_bwd),
            "non_adjacent_jaccard": stats(na_jacc),
            "non_adjacent_share_any_ratio": sum(na_any) / n_na if n_na else 0.0,
            "raw_shared_count": [],
            "raw_continuity_forward": [],
            "raw_continuity_backward": [],
            "raw_jaccard": [],
            "raw_share_any": [],
            "non_adjacent_raw_shared_count": na_shared,
            "non_adjacent_raw_continuity_forward": na_fwd,
            "non_adjacent_raw_continuity_backward": na_bwd,
            "non_adjacent_raw_jaccard": na_jacc,
            "non_adjacent_raw_share_any": na_any,
        }

    return {
        **base_out,
        "shared_count": stats(shared_counts),
        "continuity_forward": stats(continuity_forward),
        "continuity_backward": stats(continuity_backward),
        "jaccard": stats(jaccards),
        "share_any_ratio": sum(share_any) / n_pairs,
        "non_adjacent_shared_count": stats(na_shared),
        "non_adjacent_continuity_forward": stats(na_fwd),
        "non_adjacent_continuity_backward": stats(na_bwd),
        "non_adjacent_jaccard": stats(na_jacc),
        "non_adjacent_share_any_ratio": sum(na_any) / n_na if n_na else 0.0,
        "raw_shared_count": shared_counts,
        "raw_continuity_forward": continuity_forward,
        "raw_continuity_backward": continuity_backward,
        "raw_jaccard": jaccards,
        "raw_share_any": share_any,
        "non_adjacent_raw_shared_count": na_shared,
        "non_adjacent_raw_continuity_forward": na_fwd,
        "non_adjacent_raw_continuity_backward": na_bwd,
        "non_adjacent_raw_jaccard": na_jacc,
        "non_adjacent_raw_share_any": na_any,
    }


def analyze_all_videos(base_path: Path, dataset_name: str = "AVA100") -> Dict[str, Any]:
    base = Path(base_path)
    video_ids = sorted([
        int(d.name) for d in base.iterdir()
        if d.is_dir() and d.name.isdigit() and (base / d.name / "kg").exists()
        and (base / d.name / "kg" / "vdb_events.json").exists()
        and (base / d.name / "kg" / "vdb_entities.json").exists()
    ])
    per_video = []
    for vid in video_ids:
        s = analyze_video(vid, base)
        if s is not None:
            per_video.append(s)
    return {"dataset_name": dataset_name, "video_ids": video_ids, "per_video": per_video}


def print_statistics(all_stats: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"OBJECT OVERLAP BETWEEN CONSECUTIVE EVENTS - {all_stats.get('dataset_name', 'Unknown')}")
    print("=" * 80)

    for s in all_stats["per_video"]:
        vid = s["video_id"]
        n_ev = s["num_events"]
        n_pairs = s["num_consecutive_pairs"]
        n_na = s.get("num_non_adjacent_pairs", 0)
        print(f"\n--- Video {vid} --- (events: {n_ev}, entities: {s['num_entities']}, consecutive pairs: {n_pairs}, non-adjacent sample: {n_na})")
        # Event object-count buckets (coverage: what fraction of events can the signal work for?)
        n0 = s.get("n_events_zero_objects", 0)
        n1 = s.get("n_events_one_object", 0)
        n2 = s.get("n_events_two_objects", 0)
        n35 = s.get("n_events_3_to_5", 0)
        n6p = s.get("n_events_6_plus", 0)
        pct0 = 100 * n0 / n_ev if n_ev else 0
        print(f"  Events by object count:  0 obj={n0} ({pct0:.1f}%)  1={n1}  2={n2}  3-5={n35}  6+={n6p}")
        by_size = s.get("continuity_forward_by_size_a", {})
        if by_size:
            print(f"  Continuity forward by |A|:  0→{by_size.get('0',0):.2f}  1→{by_size.get('1',0):.2f}  2-5→{by_size.get('2-5',0):.2f}  6+→{by_size.get('6+',0):.2f}")
        if n_pairs == 0:
            continue
        # Adjacent vs non-adjacent side-by-side
        print(f"  ADJACENT (consecutive)   vs   NON-ADJACENT (|i-j|>={MIN_GAP_NON_ADJACENT})")
        print(f"  Shared count:        mean {s['shared_count']['mean']:.2f} / {s['non_adjacent_shared_count']['mean']:.2f}   (gap: {s['shared_count']['mean'] - s['non_adjacent_shared_count']['mean']:+.2f})")
        print(f"  Continuity forward:  mean {s['continuity_forward']['mean']:.2f} / {s['non_adjacent_continuity_forward']['mean']:.2f}   (gap: {s['continuity_forward']['mean'] - s['non_adjacent_continuity_forward']['mean']:+.2f})")
        print(f"  Continuity backward: mean {s['continuity_backward']['mean']:.2f} / {s['non_adjacent_continuity_backward']['mean']:.2f}   (gap: {s['continuity_backward']['mean'] - s['non_adjacent_continuity_backward']['mean']:+.2f})")
        print(f"  Jaccard:             mean {s['jaccard']['mean']:.2f} / {s['non_adjacent_jaccard']['mean']:.2f}   (gap: {s['jaccard']['mean'] - s['non_adjacent_jaccard']['mean']:+.2f})")
        print(f"  Share any (ratio):   {s['share_any_ratio']:.2f} / {s['non_adjacent_share_any_ratio']:.2f}   (gap: {s['share_any_ratio'] - s['non_adjacent_share_any_ratio']:+.2f})")

    # Overall: event object-count summary
    total_ev = sum(s["num_events"] for s in all_stats["per_video"])
    total_n0 = sum(s.get("n_events_zero_objects", 0) for s in all_stats["per_video"])
    total_n1 = sum(s.get("n_events_one_object", 0) for s in all_stats["per_video"])
    print(f"\n{'='*80}\nOVERALL EVENT COVERAGE (object-count buckets)")
    print("=" * 80)
    pct0 = 100 * total_n0 / total_ev if total_ev else 0
    print(f"  Total events: {total_ev}   |   0 objects: {total_n0} ({pct0:.1f}%)   1 object: {total_n1}")

    # Overall: adjacent vs non-adjacent
    all_shared = [x for s in all_stats["per_video"] for x in s.get("raw_shared_count", [])]
    all_fwd = [x for s in all_stats["per_video"] for x in s.get("raw_continuity_forward", [])]
    all_bwd = [x for s in all_stats["per_video"] for x in s.get("raw_continuity_backward", [])]
    all_jacc = [x for s in all_stats["per_video"] for x in s.get("raw_jaccard", [])]
    all_any = [x for s in all_stats["per_video"] for x in s.get("raw_share_any", [])]
    all_na_shared = [x for s in all_stats["per_video"] for x in s.get("non_adjacent_raw_shared_count", [])]
    all_na_fwd = [x for s in all_stats["per_video"] for x in s.get("non_adjacent_raw_continuity_forward", [])]
    all_na_jacc = [x for s in all_stats["per_video"] for x in s.get("non_adjacent_raw_jaccard", [])]
    all_na_any = [x for s in all_stats["per_video"] for x in s.get("non_adjacent_raw_share_any", [])]

    print(f"\n{'='*80}\nOVERALL  ADJACENT vs NON-ADJACENT  (side-by-side)")
    print("=" * 80)
    if all_shared:
        print(f"  Metric              ADJACENT (n={len(all_shared)})    NON-ADJACENT (n={len(all_na_shared)})    GAP")
        print(f"  Shared count mean    {np.mean(all_shared):.2f}                 {np.mean(all_na_shared):.2f}                  {np.mean(all_shared) - np.mean(all_na_shared):+.2f}")
        print(f"  Continuity fwd mean  {np.mean(all_fwd):.2f}                 {np.mean(all_na_fwd):.2f}                  {np.mean(all_fwd) - np.mean(all_na_fwd):+.2f}")
        print(f"  Jaccard mean         {np.mean(all_jacc):.2f}                 {np.mean(all_na_jacc):.2f}                  {np.mean(all_jacc) - np.mean(all_na_jacc):+.2f}")
        print(f"  Share any ratio      {np.mean(all_any):.2f}                 {np.mean(all_na_any):.2f}                  {np.mean(all_any) - np.mean(all_na_any):+.2f}")


def create_visualizations(all_stats: Dict[str, Any], output_dir: Path) -> None:
    """Create overview plots (optional, requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    os.makedirs(output_dir, exist_ok=True)
    per = all_stats["per_video"]
    vids = [s["video_id"] for s in per]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Shared count by video (box)
    axes[0, 0].boxplot(
        [s.get("raw_shared_count", []) for s in per],
        labels=[str(v) for v in vids],
        patch_artist=True,
    )
    axes[0, 0].set_title("Shared object count (consecutive pairs)")
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha="right")

    # Continuity forward by video (box)
    axes[0, 1].boxplot(
        [s.get("raw_continuity_forward", []) for s in per],
        labels=[str(v) for v in vids],
        patch_artist=True,
    )
    axes[0, 1].set_title("Continuity forward (shared/|A|)")
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha="right")

    # Jaccard by video (box)
    axes[1, 0].boxplot(
        [s.get("raw_jaccard", []) for s in per],
        labels=[str(v) for v in vids],
        patch_artist=True,
    )
    axes[1, 0].set_title("Jaccard (consecutive pairs)")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha="right")

    # Share-any ratio per video (bar)
    axes[1, 1].bar(
        range(len(vids)),
        [s.get("share_any_ratio", 0) for s in per],
        color="steelblue",
        alpha=0.8,
    )
    axes[1, 1].set_xticks(range(len(vids)))
    axes[1, 1].set_xticklabels([str(v) for v in vids], rotation=45, ha="right")
    axes[1, 1].set_ylabel("Ratio of pairs with ≥1 shared object")
    axes[1, 1].set_title("Share any object (per video)")

    plt.tight_layout()
    png = output_dir / "object_overlap_consecutive_events_overview.png"
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png}")

    # Adjacent vs non-adjacent comparison (gap = discriminative power)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(vids))
    w = 0.35
    axes2[0].bar(x - w/2, [s.get("jaccard", {}).get("mean", 0) for s in per], w, label="Adjacent", color="steelblue", alpha=0.8)
    axes2[0].bar(x + w/2, [s.get("non_adjacent_jaccard", {}).get("mean", 0) for s in per], w, label="Non-adjacent", color="coral", alpha=0.8)
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels([str(v) for v in vids], rotation=45, ha="right")
    axes2[0].set_ylabel("Mean Jaccard")
    axes2[0].set_title("Adjacent vs non-adjacent (Jaccard)")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3, axis="y")

    axes2[1].bar(x - w/2, [s.get("share_any_ratio", 0) for s in per], w, label="Adjacent", color="steelblue", alpha=0.8)
    axes2[1].bar(x + w/2, [s.get("non_adjacent_share_any_ratio", 0) for s in per], w, label="Non-adjacent", color="coral", alpha=0.8)
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels([str(v) for v in vids], rotation=45, ha="right")
    axes2[1].set_ylabel("Share-any ratio")
    axes2[1].set_title("Adjacent vs non-adjacent (share any object)")
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    png2 = output_dir / "object_overlap_adjacent_vs_non_adjacent.png"
    plt.savefig(png2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png2}")


def save_json_report(all_stats: Dict[str, Any], output_path: Path) -> None:
    export = []
    for s in all_stats["per_video"]:
        e = {k: v for k, v in s.items() if not k.startswith("raw_") and not k.startswith("non_adjacent_raw_")}
        export.append(e)
    report = {
        "dataset_name": all_stats["dataset_name"],
        "video_ids": all_stats["video_ids"],
        "per_video": export,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Object overlap between consecutive events (AVA100)")
    p.add_argument("--dataset", default="AVA100", help="Dataset folder name under AVA_cache")
    p.add_argument("--base-dir", default=None, help="Base dir (default: project_root/AVA_cache)")
    p.add_argument("--output", default=None, help="Output JSON path (default: ECML-PKDD/.../object_overlap_consecutive_events_report.json)")
    p.add_argument("--no-plot", action="store_true", help="Skip generating overview plot")
    args = p.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else _project_root / "AVA_cache"
    base_path = base_dir / args.dataset
    if not base_path.exists():
        print(f"Path not found: {base_path}")
        return

    print(f"Analyzing: {base_path}")
    all_stats = analyze_all_videos(base_path, dataset_name=args.dataset)
    if not all_stats["per_video"]:
        print("No videos with vdb_events + vdb_entities found.")
        return

    print_statistics(all_stats)

    out_dir = _project_root / "ECML-PKDD" / "kg_analysis_output" / args.dataset
    out_path = Path(args.output) if args.output else out_dir / "object_overlap_consecutive_events_report.json"
    out_dir = out_path.parent
    save_json_report(all_stats, out_path)
    if not args.no_plot:
        create_visualizations(all_stats, out_dir)


if __name__ == "__main__":
    main()
