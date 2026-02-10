#!/usr/bin/env python3
"""
Calculate retrieval accuracy from ECML-PKDD seed_events JSON.

All times are in SECONDS (video timeline):
- Event duration in the database (seed_events) is already in seconds.
- time_reference in datas (AVA100, LVBench) is converted to seconds from
  "HH:MM:SS" or "MM:SS" format.

Time reference formats supported:
- Single point: "00:1:20", "01:16:50", "4:19"
- Time range:   "00:15-00:19", "04:19-08:41" (start-end in same format)
- Multiple points: "00:1:20,00:24:46,00:47:40" (comma-separated; binary = any hit, percentage = average coverage)

Ground truth from:
- AVA100: datas/AVA100/{citytour,ego,traffic,wildlife}.json
- LVBench: datas/LVBench/LVBench.json

Metrics:
- Binary overlap: fraction of queries with valid time GT where at least one
  retrieved event's duration overlaps the GT time point/range.
- Percentage overlap: (sum of per-query overlap percentages) / (number of queries
  with valid time GT).
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# -----------------------------------------------------------------------------
# Time parsing and overlap (aligned with ICDCS/calc_acc_new.py logic)
# -----------------------------------------------------------------------------

def time_to_seconds(time_str: str) -> int:
    """
    Parse a single time string to integer seconds.
    - HH:MM:SS or H:MM:SS or HH:M:SS etc. (2 or 3 colons) -> hours*3600 + minutes*60 + seconds
    - MM:SS or M:SS (one colon) -> minutes*60 + seconds
    Handles optional leading zeros and spaces.
    """
    time_str = (time_str or "").strip()
    if not time_str:
        raise ValueError("Empty time string")
    parts = re.split(r"[:.]", time_str)
    parts = [p.strip() for p in parts if p.strip().replace(" ", "").isdigit()]
    if not parts:
        raise ValueError(f"Invalid time string: {time_str}")
    if len(parts) == 2:
        # MM:SS
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) >= 3:
        # HH:MM:SS (only first three components)
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(parts[0])


def is_valid_time_reference(time_ref: str) -> bool:
    """True if time_ref can be used for accuracy (not N/A, not malformed)."""
    if not time_ref or time_ref.strip() in ["N/A", "", "None", "None-None"]:
        return False
    time_ref = time_ref.strip()
    if "-" in time_ref:
        parts = time_ref.split("-", 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            return False
        if parts[0].strip() in ["None", ""] or parts[1].strip() in ["None", ""]:
            return False
    try:
        if "-" in time_ref:
            start_time, end_time = time_ref.split("-", 1)
            time_to_seconds(start_time.strip())
            time_to_seconds(end_time.strip())
        elif "," in time_ref:
            points = [p.strip() for p in time_ref.split(",") if p.strip()]
            if not points:
                return False
            for p in points:
                time_to_seconds(p)
        else:
            time_to_seconds(time_ref)
        return True
    except (ValueError, IndexError):
        return False


def _percentage_overlap_single(
    time_list: List[Tuple[int, int]], ref_start: int, ref_end: int
) -> float:
    """Fraction of [ref_start, ref_end] covered by time_list. 0 if ref_end < ref_start."""
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
    relevant = []
    for start, end in time_list:
        if end <= start:
            continue
        s = max(start, ref_start)
        e = min(end, ref_end)
        if e > s:
            relevant.append((s, e))
    if not relevant:
        return 0.0
    relevant.sort()
    merged = [relevant[0]]
    for s, e in relevant[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    total_covered = sum(e - s for s, e in merged)
    return min(1.0, total_covered / ref_length) if ref_length > 0 else 0.0


def has_any_overlap(
    time_list: List[Tuple[int, int]], time_ref: Tuple[int, int]
) -> bool:
    """True if any interval in time_list overlaps [ref_start, ref_end]."""
    ref_start, ref_end = time_ref
    if ref_end < ref_start:
        return False
    if ref_start == ref_end:
        point = ref_start
        for start, end in time_list:
            if end <= start:
                continue
            if start <= point < end:
                return True
        return False
    for start, end in time_list:
        if end <= start:
            continue
        if max(start, ref_start) < min(end, ref_end):
            return True
    return False


def overlap_reference_helper(
    time_ref: str, time_list: List[Tuple[int, int]]
) -> Optional[float]:
    """
    Return fraction of time_ref covered by time_list (0.0..1.0), in seconds.
    Supports: single point (HH:MM:SS), time range (start-end), multiple points (t1,t2,...).
    For multiple points, returns the average coverage of each point.
    Returns None if time_ref is invalid.
    """
    if not time_ref or time_ref.strip() in ["N/A", "", "None", "None-None"]:
        return None
    time_ref = time_ref.strip()
    try:
        # Time range: "start-end" (e.g. "00:15-00:19", "04:19-08:41")
        if "-" in time_ref:
            start_s, end_s = time_ref.split("-", 1)
            start_s = start_s.strip() or end_s.strip()
            end_s = end_s.strip() or start_s
            if not start_s or not end_s:
                return None
            s_sec = time_to_seconds(start_s)
            e_sec = time_to_seconds(end_s)
            return _percentage_overlap_single(time_list, s_sec, e_sec)
        # Multiple time points: "t1,t2,t3" (e.g. "00:1:20,01:16:50")
        if "," in time_ref:
            points = [p.strip() for p in time_ref.split(",") if p.strip()]
            if not points:
                return None
            pcts = []
            for p in points:
                t = time_to_seconds(p)
                pcts.append(_percentage_overlap_single(time_list, t, t))
            return sum(pcts) / len(pcts)
        # Single time point: "00:1:20" or "01:16:50"
        t = time_to_seconds(time_ref)
        return _percentage_overlap_single(time_list, t, t)
    except (ValueError, IndexError):
        return None


def binary_overlap_helper(
    time_ref: str, time_list: List[Tuple[int, int]]
) -> Optional[float]:
    """
    1.0 if any overlap, 0.0 if no overlap, None if invalid time_ref.
    Supports: single point, time range (start-end), multiple points (any point hitting = 1.0).
    All times converted to seconds (HH:MM:SS / MM:SS).
    """
    if not time_ref or time_ref.strip() in ["N/A", "", "None", "None-None"]:
        return None
    time_ref = time_ref.strip()
    try:
        # Time range
        if "-" in time_ref:
            start_s, end_s = time_ref.split("-", 1)
            start_s = start_s.strip() or end_s.strip()
            end_s = end_s.strip() or start_s
            if not start_s or not end_s:
                return None
            s_sec = time_to_seconds(start_s)
            e_sec = time_to_seconds(end_s)
            return 1.0 if has_any_overlap(time_list, (s_sec, e_sec)) else 0.0
        # Multiple points: hit if any point overlaps
        if "," in time_ref:
            for p in time_ref.split(","):
                p = p.strip()
                if not p:
                    continue
                t = time_to_seconds(p)
                if has_any_overlap(time_list, (t, t)):
                    return 1.0
            return 0.0
        # Single point
        t = time_to_seconds(time_ref)
        return 1.0 if has_any_overlap(time_list, (t, t)) else 0.0
    except (ValueError, IndexError):
        return None


# -----------------------------------------------------------------------------
# Ground truth loading
# -----------------------------------------------------------------------------

def load_ground_truth_cache(dataset: str, project_root: Path) -> Dict[Tuple[str, str], str]:
    """
    (video_key, question_id) -> time_reference.
    question_id is string for consistent lookup.
    """
    cache: Dict[Tuple[str, str], str] = {}
    if dataset == "AVA100":
        base = project_root / "datas" / "AVA100"
        for name in ["citytour.json", "ego.json", "traffic.json", "wildlife.json"]:
            path = base / name
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
                if isinstance(data, list):
                    for video in data:
                        video_key = video.get("video_key")
                        if not video_key:
                            continue
                        for qa in video.get("qa", []):
                            qid = qa.get("question_id")
                            if qid is not None:
                                cache[(video_key, str(qid))] = qa.get(
                                    "time_reference", "N/A"
                                )
            except Exception as e:
                print(f"  Warning: failed to load {path}: {e}")
    elif dataset == "LVBench":
        path = project_root / "datas" / "LVBench" / "LVBench.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, list):
                    for video in data:
                        video_key = video.get("key") or video.get("video_key")
                        if not video_key:
                            continue
                        for qa in video.get("qa", []):
                            uid = qa.get("uid") or qa.get("question_id")
                            if uid is not None:
                                cache[(video_key, str(uid))] = qa.get(
                                    "time_reference", "N/A"
                                )
            except Exception as e:
                print(f"  Warning: failed to load {path}: {e}")
    return cache


# -----------------------------------------------------------------------------
# Seed events loading (durations already in seconds)
# -----------------------------------------------------------------------------

def event_durations_to_seconds(
    seed_events: List[Dict],
    dataset: str,
    video_key: str,
) -> List[Dict[str, Any]]:
    """
    Build per-event intervals (in seconds) from seed_events.

    The database stores event duration in SECONDS (video timeline). We use values
    as-is so that overlap is computed in the same unit as time_reference (which
    is converted from HH:MM:SS / MM:SS to seconds).

    Returns:
        List of {"id": <event_id or None>, "interval_sec": [start_sec, end_sec]}
    """
    events_sec: List[Dict[str, Any]] = []
    for ev in seed_events:
        dur = ev.get("duration")
        if not isinstance(dur, (list, tuple)) or len(dur) < 2:
            continue
        a, b = float(dur[0]), float(dur[1])
        if b <= a:
            continue
        a_sec = int(a)
        b_sec = int(b)
        if b_sec <= a_sec:
            b_sec = a_sec + 1
        ev_id = ev.get("id") or ev.get("__id__")
        events_sec.append({"id": ev_id, "interval_sec": [a_sec, b_sec]})
    return events_sec


def time_reference_to_segments_sec(time_ref: str) -> Optional[List[Tuple[int, int]]]:
    """
    Convert time_reference string to list of GT segments in seconds.
    - "start-end" -> [(start_sec, end_sec)]
    - "t1,t2,..." -> [(t1,t1), (t2,t2), ...]
    - "t"         -> [(t,t)]
    Returns None if invalid / N/A.
    """
    if not time_ref or time_ref.strip() in ["N/A", "", "None", "None-None"]:
        return None
    time_ref = time_ref.strip()
    try:
        if "-" in time_ref:
            start_s, end_s = time_ref.split("-", 1)
            start_s = start_s.strip() or end_s.strip()
            end_s = end_s.strip() or start_s
            if not start_s or not end_s:
                return None
            s_sec = time_to_seconds(start_s)
            e_sec = time_to_seconds(end_s)
            return [(s_sec, e_sec)]
        if "," in time_ref:
            points = [p.strip() for p in time_ref.split(",") if p.strip()]
            if not points:
                return None
            return [(time_to_seconds(p), time_to_seconds(p)) for p in points]
        t = time_to_seconds(time_ref)
        return [(t, t)]
    except (ValueError, IndexError):
        return None


def event_hits_gt(interval: Tuple[int, int], gt_segments: List[Tuple[int, int]]) -> bool:
    """
    Check if an event interval (start_sec, end_sec) hits any GT segment.
    - For point GT segments (s==e): hit if start <= s < end.
    - For range GT segments: hit if intervals overlap.
    """
    s, e = interval
    if e <= s:
        return False
    for gs, ge in gt_segments:
        if ge < gs:
            continue
        if gs == ge:
            # point segment
            if s <= gs < e:
                return True
        else:
            if max(s, gs) < min(e, ge):
                return True
    return False


# -----------------------------------------------------------------------------
# Main accuracy computation
# -----------------------------------------------------------------------------

def run_accuracy(
    seed_events_path: Path,
    dataset: str,
    project_root: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compute binary and percentage overlap accuracy.
    All times in seconds: event duration from DB, time_reference converted from HH:MM:SS/MM:SS.

    - Binary: (number of queries with valid time GT that have at least one hit) / (number of queries with valid time GT).
      For multiple time points in one query: hitting ANY one point counts as binary hit.
    - Percentage: For a query with multiple time points, first average the coverage over each point; then
      (sum of these per-query percentages) / (number of queries with valid time GT).
    - Per-query total_event_duration_sec = sum of (end - start) over all event intervals.
    - avg_total_event_duration_sec and avg_num_events are over ALL queries (all entries with video_key/question_id), not just valid time GT.
    """
    if not seed_events_path.exists():
        raise FileNotFoundError(f"Seed events file not found: {seed_events_path}")
    data = json.loads(seed_events_path.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected seed_events JSON to be a list of entries")

    gt_cache = load_ground_truth_cache(dataset, project_root)

    # All-queries stats (for avg total event duration and avg num_events over all queries)
    n_total = 0
    total_duration_sum_all = 0.0
    num_events_sum_all = 0

    valid_entries = []
    per_query = []

    for entry in data:
        video_key = entry.get("video_key")
        question_id = entry.get("question_id")
        if video_key is None or question_id is None:
            continue

        seed_events = entry.get("seed_events") or []
        events_sec = event_durations_to_seconds(seed_events, dataset, video_key)
        intervals = [tuple(ev["interval_sec"]) for ev in events_sec]
        total_duration_sec = sum(e - s for s, e in intervals)
        num_events = len(intervals)
        n_total += 1
        total_duration_sum_all += total_duration_sec
        num_events_sum_all += num_events

        key = (video_key, str(question_id))
        time_ref = gt_cache.get(key)
        if time_ref is None:
            continue
        if not is_valid_time_reference(time_ref):
            continue

        binary = binary_overlap_helper(time_ref, intervals)
        percentage = overlap_reference_helper(time_ref, intervals)

        if binary is None:
            continue
        valid_entries.append(entry)
        gt_segments = time_reference_to_segments_sec(time_ref)
        events_hit = []
        if gt_segments:
            events_hit = [
                ev for ev in events_sec if event_hits_gt(tuple(ev["interval_sec"]), gt_segments)
            ]
        # For percentage, use 0.0 when no overlap (so denominator is all valid)
        pct_val = percentage if percentage is not None else 0.0
        first_interval = intervals[0] if intervals else None
        per_query.append({
            "video_key": video_key,
            "question_id": question_id,
            "time_reference": time_ref,
            "gt_segments_sec": gt_segments,
            "binary_overlap": binary,
            "percentage_overlap": pct_val,
            "num_events": len(intervals),
            "total_event_duration_sec": total_duration_sec,
            "first_event_interval_sec": first_interval,
            "events_hit": events_hit,
        })

    n_valid = len(valid_entries)
    if n_valid > 0 and per_query:
        p0 = per_query[0]
        try:
            tr = p0["time_reference"].strip()
            gt_sec = time_to_seconds(tr.split("-")[0].strip() if "-" in tr else tr)
        except Exception:
            gt_sec = None
        print(f"  [Unit check] First query: time_reference='{p0['time_reference']}' -> {gt_sec} sec; first event interval (sec)={p0.get('first_event_interval_sec')}")
    if n_valid == 0:
        return {
            "dataset": dataset,
            "seed_events_file": str(seed_events_path),
            "num_queries_total": n_total,
            "num_queries_with_valid_time_gt": 0,
            "binary_overlap_accuracy": None,
            "percentage_overlap_accuracy": None,
            "avg_total_event_duration_sec": total_duration_sum_all / n_total if n_total else None,
            "avg_num_events": num_events_sum_all / n_total if n_total else None,
            "per_query": [],
        }

    binary_sum = sum(p["binary_overlap"] for p in per_query)
    percentage_sum = sum(p["percentage_overlap"] for p in per_query)

    result = {
        "dataset": dataset,
        "seed_events_file": str(seed_events_path),
        "num_queries_total": n_total,
        "num_queries_with_valid_time_gt": n_valid,
        "binary_overlap_accuracy": binary_sum / n_valid,
        "percentage_overlap_accuracy": percentage_sum / n_valid,
        "binary_hits": int(binary_sum),
        "percentage_sum": percentage_sum,
        "avg_total_event_duration_sec": total_duration_sum_all / n_total if n_total else None,
        "avg_num_events": num_events_sum_all / n_total if n_total else None,
        "per_query": per_query,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calculate retrieval accuracy from ECML-PKDD seed_events JSON."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["AVA100", "LVBench"],
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--seed-events",
        type=Path,
        default=None,
        help="Path to seed_events_{dataset}.json (default: ECML-PKDD/indus_outputs/seed_events_{dataset}.json)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root (default: parent of ECML-PKDD)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write full result JSON here",
    )
    args = parser.parse_args()

    project_root = args.project_root or (Path(__file__).resolve().parent.parent)
    script_dir = Path(__file__).resolve().parent
    seed_events_path = args.seed_events or (
        script_dir / "indus_outputs" / f"seed_events_{args.dataset}.json"
    )

    print(f"Dataset: {args.dataset}")
    print(f"Seed events: {seed_events_path}")
    print(f"Project root: {project_root}")

    result = run_accuracy(
        seed_events_path=seed_events_path,
        dataset=args.dataset,
        project_root=project_root,
        output_path=args.output,
    )

    n_valid = result["num_queries_with_valid_time_gt"]
    n_total = result.get("num_queries_total", n_valid)
    print(f"\nQueries (total): {n_total}; with valid time GT: {n_valid}")
    if n_valid > 0:
        print(f"Binary overlap accuracy:    {result['binary_overlap_accuracy']:.4f}  ({result['binary_hits']}/{n_valid})")
        print(f"Percentage overlap (avg):  {result['percentage_overlap_accuracy']:.4f}  (sum={result['percentage_sum']:.4f} / {n_valid})")
    if n_total > 0 and result.get("avg_total_event_duration_sec") is not None:
        print(f"Avg total event duration:   {result['avg_total_event_duration_sec']:.2f} sec per query  (over all {n_total} queries)")
        print(f"Avg num events per query:   {result['avg_num_events']:.2f}  (over all {n_total} queries)")
    if n_valid == 0:
        print("No queries with valid time reference; no overlap metrics.")

    if args.output:
        print(f"\nWrote full result to {args.output}")


if __name__ == "__main__":
    main()
