#!/usr/bin/env python3
"""
Visualize and Evaluate VLM Retrieval Results (standalone script).

Reads a results JSON produced by any of:
  - query_vectorized_retrieval.py (vectorized_retrieval)
  - query_uniform_sampling.py (uniform_sampling)
  - query_vlm_multimode.py (vlm_multimode, config_mode 1-7)

and (1) visualizes frame/event distribution over the video timeline, and
(2) computes ground-truth overlap / accuracy metrics by mode.

Usage:
  python visualize_and_evaluate.py --results outputs/query_*.json [--cache-root AVA_cache] [--dataset lvbench] [--out-dir viz_eval_out] [--limit N]

--------------------------------------------------------------------------------
REQUIRED JSON FIELDS (per result entry) FOR FULL VISUALIZATION + EVALUATION
--------------------------------------------------------------------------------

Common (all pipelines):
  - video_id, question_id, question, answer
  - time_reference (or inside ground_truth_overlap / metadata)
  - video_duration, fps (or in retrieval_info / sampling_info / metadata)

Frames (when mode uses frames):
  - frame_indices: full list of frame indices sent to VLM (not just a sample)
  - similarity_scores (optional): one per frame, for top-k modes (y-axis in plot)

Events (when mode uses events):
  - event_segments: list of [start_sec, end_sec] for each event sent to VLM

Mode / method:
  - config_mode (1-7) for multimode, or method + retrieval_info / sampling_info
    so we can infer: frames-only (1,2), events-only (3,4,5), frames+events (6,7)

--------------------------------------------------------------------------------
WHAT EACH PIPELINE CURRENTLY LOGS / WHAT'S MISSING
--------------------------------------------------------------------------------

1) AVA/vectorized_retrieval.py (query_vectorized_retrieval.py)
    - Logs: retrieval_info with frame_indices_sample (first 100 only),
      similarity_scores_sample (first 100 only), video_duration, fps;
      ground_truth_overlap with time_reference.
    - MISSING: Full frame_indices and full similarity_scores (when > 100).
              Without these, visualization shows only first 100 frames and
              evaluation can still run if we use the sample for overlap, but
              results are incomplete. Add full frame_indices and
              similarity_scores to retrieval_info.

2) AVA/uniform_sampling.py (query_uniform_sampling.py)
    - Logs: sampling_info with frame_indices_sample (first 100 only),
      video_duration, fps; ground_truth_overlap.
    - MISSING: Full frame_indices. time_reference may be only in
              ground_truth_overlap. Add full frame_indices to sampling_info
              (e.g. "frame_indices") for visualization and full evaluation.

3) AVA/ava_multimode.py (query_vlm_multimode.py)
    - Logs: config_mode, metadata with only num_frames, num_events, num_segments.
    - MISSING: frame_indices, similarity_scores (for modes 2,7), event_segments
              (for modes 3,4,5,6,7), video_duration, fps, time_reference in
              the result entry. Currently impossible to visualize or run
              event/frame overlap evaluation from JSON alone. Add to metadata
              (or top-level): frame_indices, similarity_scores (if any),
              event_segments (list of [start_sec, end_sec]), video_duration, fps,
              time_reference.

--------------------------------------------------------------------------------
EVALUATION VARIANTS BY MODE
--------------------------------------------------------------------------------

Time reference → GT events (for overlap checks):
  - Single time point: map to the one event that contains that point.
  - Multiple time points: map each point to its containing event (can be
    multiple events or one if points fall in same event).
  - Time range [start,end]: map to all events that overlap (hit) this range
    (e.g. [20,25] → events with segments [19,21], [21,24], [24,27]).

Frames-only (modes 1, 2; also vectorized_retrieval, uniform_sampling):
  - Variant A: Time reference → GT events. Check if any selected frame falls
    inside those GT events. (Binary + optional percentage.)
  - Variant B: Selected frames → associated events. Check if those events
    overlap the time reference directly (no conversion of time reference to events).
  - Variant C: Frames → associated events; time reference → GT events; check
    overlap between these two sets of event time segments.

Events-only (modes 3, 4, 5):
  - Variant A: Our event segments vs time reference (overlap / contain / intersect).
  - Variant B: Our event segments vs GT events (from time reference).

Frames + events (modes 6, 7): 5-way evaluation
  - (1) Frames vs GT events (frames-only Variant A).
  - (2) Frames→events vs time reference (frames-only Variant B).
  - (3) Frames→events vs GT events (frames-only Variant C).
  - (4) Event segments vs time reference (events-only Variant A).
  - (5) Event segments vs GT events (events-only Variant B).

--------------------------------------------------------------------------------
VISUALIZATION
--------------------------------------------------------------------------------

- Horizontal axis: time (0 to video_duration).
- Vertical axis: y = 1 for frames (green dots). y = 0 for event descriptions
  (purple segments or dots).
- Frames: if similarity_scores exist, plot (time, score) at y=1 (green). If no
  scores (uniform sampling), plot green dots at (time, 1).
- Events: show distribution at y = 0 (purple); no similarity score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional matplotlib for visualization
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# -----------------------------------------------------------------------------
# Time reference parsing and GT events
# -----------------------------------------------------------------------------

def _time_str_to_seconds(time_str: str) -> float:
    """Convert 'MM:SS' or 'HH:MM:SS' to seconds. Handles optional fractional seconds."""
    s = time_str.strip()
    parts = s.split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid time string: {time_str}")


def parse_time_reference_to_interval(
    time_reference: Optional[str],
) -> Optional[Tuple[float, float]]:
    """
    Parse time_reference to a single interval (start_sec, end_sec).
    - "N/A", "", "None" -> None.
    - "MM:SS" or "HH:MM:SS" -> (sec, sec).
    - "start-end" -> (start_sec, end_sec).
    - "t1,t2,..." -> (min_sec, max_sec) for the range covering all points.
    """
    if not time_reference or time_reference.strip() in ("N/A", "", "None", "None-None"):
        return None
    tr = time_reference.strip()
    if "-" in tr:
        parts = tr.split("-", 1)
        start_s, end_s = parts[0].strip(), parts[1].strip()
        if not start_s:
            start_s = end_s
        if not end_s:
            end_s = start_s
        try:
            return (_time_str_to_seconds(start_s), _time_str_to_seconds(end_s))
        except ValueError:
            return None
    if "," in tr:
        points = [_time_str_to_seconds(p.strip()) for p in tr.split(",") if p.strip()]
        if not points:
            return None
        return (min(points), max(points))
    try:
        t = _time_str_to_seconds(tr)
        return (t, t)
    except ValueError:
        return None


def time_reference_to_gt_events(
    time_reference: Optional[str],
    all_events: List[Dict[str, Any]],
) -> List[Tuple[float, float]]:
    """
    Map time_reference to ground-truth events: the set of events that overlap
    or contain the reference.
    - all_events: list of dicts with 'duration' = [start_sec, end_sec].
    - Single point: events that contain that point.
    - Time range: events that overlap (hit) the range.
    - Multiple points: events that contain any of the points (then merge
      overlapping segments into a unique set of intervals).
    Returns list of (start_sec, end_sec) segments (merged, non-overlapping).
    """
    interval = parse_time_reference_to_interval(time_reference)
    if interval is None:
        return []
    ref_start, ref_end = interval

    segments: List[Tuple[float, float]] = []
    for ev in all_events:
        dur = ev.get("duration") or ev.get("metadata", {}).get("duration")
        if not dur or len(dur) < 2:
            continue
        start_sec = float(dur[0])
        end_sec = float(dur[1])
        # Overlap: not (event ends before ref OR event starts after ref)
        if not (end_sec < ref_start or start_sec > ref_end):
            segments.append((start_sec, end_sec))

    if not segments:
        return []
    # Merge overlapping
    segments.sort(key=lambda x: x[0])
    merged = [segments[0]]
    for s, e in segments[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def frames_to_associated_events(
    frame_indices: List[int],
    fps: float,
    all_events: List[Dict[str, Any]],
) -> List[Tuple[float, float]]:
    """
    Map each frame index to the event that contains that frame time; return
    unique event segments as (start_sec, end_sec). Used for frames-only
    Variant B: overlap of these segments with time reference.
    """
    segments: List[Tuple[float, float]] = []
    seen: set = set()
    for fi in frame_indices:
        t_sec = fi / fps
        for ev in all_events:
            dur = ev.get("duration") or ev.get("metadata", {}).get("duration")
            if not dur or len(dur) < 2:
                continue
            start_sec, end_sec = float(dur[0]), float(dur[1])
            if start_sec <= t_sec <= end_sec:
                key = (start_sec, end_sec)
                if key not in seen:
                    seen.add(key)
                    segments.append(key)
                break
    segments.sort(key=lambda x: x[0])
    return segments


def intervals_overlap(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> bool:
    return not (a[1] < b[0] or a[0] > b[1])


def percentage_overlap_interval(
    our_segments: List[Tuple[float, float]],
    ref: Tuple[float, float],
) -> float:
    """Fraction of ref interval covered by our_segments (0..1)."""
    ref_start, ref_end = ref
    if ref_end <= ref_start:
        return 0.0
    ref_len = ref_end - ref_start
    # Clip our segments to ref and merge
    clipped = []
    for s, e in our_segments:
        s2 = max(s, ref_start)
        e2 = min(e, ref_end)
        if e2 > s2:
            clipped.append((s2, e2))
    if not clipped:
        return 0.0
    clipped.sort(key=lambda x: x[0])
    merged = [clipped[0]]
    for s, e in clipped[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    covered = sum(e - s for s, e in merged)
    return min(1.0, covered / ref_len)


# -----------------------------------------------------------------------------
# Load events for a video (vdb_events.json)
# -----------------------------------------------------------------------------

def load_video_events(cache_root: str, dataset: str, video_id: int) -> List[Dict[str, Any]]:
    """Load events from {cache_root}/{dataset}/{video_id}/kg/vdb_events.json (data field)."""
    work_dir = Path(cache_root) / dataset / str(video_id)
    # AVA stores vdb_events.json under kg/
    path = work_dir / "kg" / "vdb_events.json"
    if not path.exists():
        path = work_dir / "vdb_events.json"
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("data", [])
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def _overlap_two_segment_lists(
    segs_a: List[Tuple[float, float]],
    segs_b: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Binary overlap (1 if any pair overlaps), and fraction of segs_b covered by segs_a (0..1)."""
    binary = 1.0 if any(
        intervals_overlap((sa, ea), (sb, eb))
        for sa, ea in segs_a for sb, eb in segs_b
    ) else 0.0
    total_b_len = sum(eb - sb for sb, eb in segs_b)
    if total_b_len <= 0:
        return binary, 0.0
    covered = sum(
        percentage_overlap_interval(segs_a, (sb, eb)) * (eb - sb)
        for sb, eb in segs_b
    )
    return binary, min(1.0, covered / total_b_len)


def evaluate_frames_only(
    frame_indices: List[int],
    time_reference: Optional[str],
    fps: float,
    gt_events: List[Tuple[float, float]],
    all_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Frames-only: Variant A (frames vs GT events), Variant B (frames→events vs time_ref),
    Variant C (frames→associated events vs time_ref→GT events, overlap of event segments).
    """
    out: Dict[str, Any] = {
        "variant_a_frames_vs_gt_events": None,
        "variant_b_frames_events_vs_time_ref": None,
        "variant_c_frames_events_vs_gt_events": None,
    }
    if not time_reference or not frame_indices:
        return out

    # Variant A: any frame inside GT events?
    frame_times_sec = [fi / fps for fi in frame_indices]
    hit_gt = False
    for t in frame_times_sec:
        for (s, e) in gt_events:
            if s <= t <= e:
                hit_gt = True
                break
        if hit_gt:
            break
    out["variant_a_frames_vs_gt_events"] = {
        "binary_hit": 1.0 if hit_gt else 0.0,
        "num_frames_in_gt": sum(1 for t in frame_times_sec if any(s <= t <= e for (s, e) in gt_events)),
        "num_frames": len(frame_indices),
    }

    # Variant B: frames → associated events; overlap with time_ref
    ref_interval = parse_time_reference_to_interval(time_reference)
    assoc: List[Tuple[float, float]] = []
    if all_events:
        assoc = frames_to_associated_events(frame_indices, fps, all_events)
    if ref_interval and assoc:
        binary = 1.0 if any(intervals_overlap((s, e), ref_interval) for s, e in assoc) else 0.0
        pct = percentage_overlap_interval(assoc, ref_interval)
        out["variant_b_frames_events_vs_time_ref"] = {
            "binary_overlap": binary,
            "percentage_overlap": pct,
        }

    # Variant C: frames → associated events; time_ref → GT events; overlap between event segments
    if assoc and gt_events:
        binary_c, pct_c = _overlap_two_segment_lists(assoc, gt_events)
        out["variant_c_frames_events_vs_gt_events"] = {
            "binary_overlap": binary_c,
            "percentage_overlap": pct_c,
        }
    return out


def evaluate_events_only(
    event_segments: List[Tuple[float, float]],
    time_reference: Optional[str],
    gt_events: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """
    Events-only: Variant A (our events vs time_ref), Variant B (our events vs GT events).
    """
    out: Dict[str, Any] = {
        "variant_a_events_vs_time_ref": None,
        "variant_b_events_vs_gt_events": None,
    }
    if not event_segments:
        return out

    ref_interval = parse_time_reference_to_interval(time_reference)
    if ref_interval:
        binary_a = 1.0 if any(intervals_overlap((s, e), ref_interval) for s, e in event_segments) else 0.0
        pct_a = percentage_overlap_interval(event_segments, ref_interval)
        out["variant_a_events_vs_time_ref"] = {"binary_overlap": binary_a, "percentage_overlap": pct_a}

    if gt_events:
        binary_b = 1.0 if any(
            intervals_overlap((s, e), (gs, ge)) for s, e in event_segments for gs, ge in gt_events
        ) else 0.0
        # Percentage: how much of GT is covered by our segments
        total_gt_len = sum(ge - gs for gs, ge in gt_events)
        if total_gt_len > 0:
            covered = 0.0
            for gs, ge in gt_events:
                covered += percentage_overlap_interval(event_segments, (gs, ge)) * (ge - gs)
            pct_b = min(1.0, covered / total_gt_len)
        else:
            pct_b = 0.0
        out["variant_b_events_vs_gt_events"] = {"binary_overlap": binary_b, "percentage_overlap": pct_b}

    return out


def run_evaluation_for_entry(
    entry: Dict[str, Any],
    cache_root: str,
    dataset: str,
) -> Dict[str, Any]:
    """
    Determine mode from entry and compute all applicable evaluation metrics.
    Returns a dict of metrics (and optionally attach to entry).
    """
    video_id = entry.get("video_id")
    time_ref = (
        entry.get("time_reference")
        or (entry.get("ground_truth_overlap") or {}).get("time_reference")
    )
    fps = None
    duration = None
    frame_indices: List[int] = []
    similarity_scores: List[float] = []
    event_segments: List[Tuple[float, float]] = []

    # Resolve fps and duration
    ri = entry.get("retrieval_info") or {}
    si = entry.get("sampling_info") or {}
    meta = entry.get("metadata") or {}
    fps = ri.get("original_fps") or si.get("original_fps") or meta.get("fps")
    duration = ri.get("video_duration") or si.get("video_duration") or meta.get("video_duration")
    if fps is None and si.get("original_fps") is not None:
        fps = si["original_fps"]
    if duration is None and si.get("video_duration") is not None:
        duration = si["video_duration"]

    # Resolve frame_indices and similarity_scores
    config_mode = entry.get("config_mode")
    method = entry.get("method", "")
    if config_mode in (1, 2, 6, 7) or method == "vectorized_retrieval":
        frame_indices = (
            ri.get("frame_indices")
            or ri.get("frame_indices_sample")
            or si.get("frame_indices")
            or si.get("frame_indices_sample")
            or meta.get("frame_indices")
            or []
        )
        similarity_scores = (
            ri.get("similarity_scores")
            or ri.get("similarity_scores_sample")
            or meta.get("similarity_scores")
            or []
        )
    if config_mode in (3, 4, 5, 6, 7):
        segs = meta.get("event_segments") or entry.get("event_segments") or []
        for s in segs:
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                event_segments.append((float(s[0]), float(s[1])))

    all_events = load_video_events(cache_root or "AVA_cache", dataset or "LVBench", video_id or 0)
    gt_events = time_reference_to_gt_events(time_ref, all_events) if time_ref else []

    result: Dict[str, Any] = {
        "has_time_reference": bool(parse_time_reference_to_interval(time_ref)),
        "fps": fps,
        "video_duration": duration,
        "num_frames": len(frame_indices),
        "num_event_segments": len(event_segments),
    }

    # Frames-only (3 variants: A, B, C)
    if frame_indices and not event_segments:
        result["frames_only"] = evaluate_frames_only(
            frame_indices, time_ref, fps or 30.0, gt_events, all_events
        )
    # Events-only (2 variants: A, B)
    elif event_segments and not frame_indices:
        result["events_only"] = evaluate_events_only(event_segments, time_ref, gt_events)
    # Frames + events (config 6, 7): 5-way evaluation = 3 from frames-only + 2 from events-only
    elif frame_indices and event_segments:
        result["frames_only"] = evaluate_frames_only(
            frame_indices, time_ref, fps or 30.0, gt_events, all_events
        )
        result["events_only"] = evaluate_events_only(event_segments, time_ref, gt_events)

    return result


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_distribution(
    entry: Dict[str, Any],
    eval_result: Dict[str, Any],
    out_path: str,
) -> None:
    """
    Plot frame (green, y=1) and event (purple, y=0) distribution over time.
    Frames without scores plotted at y=1; with scores use actual score.
    """
    if not HAS_MATPLOTLIB:
        return
    fps = eval_result.get("fps") or 30.0
    duration = eval_result.get("video_duration") or 0.0
    if duration <= 0:
        return
    ri = entry.get("retrieval_info") or {}
    si = entry.get("sampling_info") or {}
    meta = entry.get("metadata") or {}
    frame_indices = (
        ri.get("frame_indices") or ri.get("frame_indices_sample")
        or si.get("frame_indices") or si.get("frame_indices_sample")
        or meta.get("frame_indices") or []
    )
    similarity_scores = (
        ri.get("similarity_scores") or ri.get("similarity_scores_sample")
        or meta.get("similarity_scores") or []
    )
    event_segments = meta.get("event_segments") or entry.get("event_segments") or []

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.2, 1.4)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Similarity / layer")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(1, color="gray", linestyle="--", alpha=0.5)

    # Frames: green at y=1 (or score if available)
    if frame_indices:
        times = [i / fps for i in frame_indices]
        if len(similarity_scores) == len(frame_indices):
            ax.scatter(times, similarity_scores, c="green", s=20, alpha=0.7, label="Frames (similarity)")
        else:
            ax.scatter(times, [1.0] * len(times), c="green", s=20, alpha=0.7, label="Frames")

    # Events: purple at y=0 (horizontal spans)
    for seg in event_segments:
        if isinstance(seg, (list, tuple)) and len(seg) >= 2:
            s, e = float(seg[0]), float(seg[1])
            ax.axvspan(s, e, ymin=0.0, ymax=0.15, color="purple", alpha=0.5)
    if event_segments:
        ax.scatter([], [], c="purple", s=80, alpha=0.6, label="Events (y=0)")

    # GT time reference
    time_ref = (
        entry.get("time_reference")
        or (entry.get("ground_truth_overlap") or {}).get("time_reference")
    )
    ref_interval = parse_time_reference_to_interval(time_ref)
    if ref_interval:
        ax.axvspan(ref_interval[0], ref_interval[1], ymin=0.8, ymax=1.0, color="red", alpha=0.3, label="GT time ref")

    vid = entry.get("video_id", "?")
    qid = entry.get("question_id", "?")
    ax.set_title(f"Video {vid} Q{qid}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize and evaluate VLM retrieval results from JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--cache-root", default="AVA_cache", help="Cache root (e.g. AVA_cache)")
    parser.add_argument("--dataset", default="LVBench", help="Dataset name under cache (e.g. LVBench)")
    parser.add_argument("--out-dir", default="viz_eval_out", help="Output directory for plots and eval JSON")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of entries to process (0 = all)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization, only evaluation")
    args = parser.parse_args()

    with open(args.results, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("results", [])
        if not entries:
            entries = [data]
    else:
        entries = []
    if not entries:
        print("No entries in JSON.")
        return
    if args.limit > 0:
        entries = entries[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    eval_results: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        ev = run_evaluation_for_entry(entry, args.cache_root, args.dataset)
        ev["_index"] = i
        ev["video_id"] = entry.get("video_id")
        ev["question_id"] = entry.get("question_id")
        eval_results.append(ev)
        if not args.no_viz and HAS_MATPLOTLIB:
            plot_path = os.path.join(args.out_dir, f"viz_v{entry.get('video_id', i)}_q{entry.get('question_id', i)}.png")
            plot_distribution(entry, ev, plot_path)
    out_json = os.path.join(args.out_dir, "evaluation_results.json")
    with open(out_json, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation written to {out_json}. Plots in {args.out_dir}.")


if __name__ == "__main__":
    main()
