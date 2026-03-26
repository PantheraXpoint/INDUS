#!/usr/bin/env python3
"""
Calculate retrieval accuracy from ECML-PKDD seed_events JSON.

Adds new metrics:
- Precision (evt), IoU (evt), SNR (evt)
- Precision (sec), IoU (sec), SNR (sec)

Definitions (your requested choices):
- For a POINT GT time t: if t is inside ANY retrieved event interval => it is a FULL HIT (coverage=1.0).
  For sec-metrics, we treat point GT as a 1-second window [t, t+1) to define lengths.
- FN for event-IoU uses SEGMENT-level GT:
    * time range => 1 segment
    * multiple points => each point is a segment
    * single point => 1 segment
- IoU(evt) = TP_seg / (TP_seg + FN_seg + FP_evt)
- Precision(evt) = TP_evt / (TP_evt + FP_evt)
- SNR(evt) = TP_evt / FP_evt (inf if FP_evt=0 and TP_evt>0; 0 if both 0)

- Sec-level metrics:
    R = union length of retrieved intervals (seconds)
    G = union length of GT intervals (seconds; for points uses 1-sec windows)
    I = intersection length between retrieved union and GT union
    Precision(sec) = I / R
    IoU(sec)       = I / (R + G - I)
    SNR(sec)       = I / (R - I)   (inf if R==I and I>0; 0 if I==0)
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# -----------------------------------------------------------------------------
# Time parsing and overlap (aligned with your existing logic)
# -----------------------------------------------------------------------------

def f1_score(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

def time_to_seconds(time_str: str) -> int:
    time_str = (time_str or "").strip()
    if not time_str:
        raise ValueError("Empty time string")
    parts = re.split(r"[:.]", time_str)
    parts = [p.strip() for p in parts if p.strip().replace(" ", "").isdigit()]
    if not parts:
        raise ValueError(f"Invalid time string: {time_str}")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) >= 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(parts[0])


def is_valid_time_reference(time_ref: str) -> bool:
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


def has_any_overlap(time_list: List[Tuple[int, int]], time_ref: Tuple[int, int]) -> bool:
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


def overlap_reference_helper(time_ref: str, time_list: List[Tuple[int, int]]) -> Optional[float]:
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
            return _percentage_overlap_single(time_list, s_sec, e_sec)
        if "," in time_ref:
            points = [p.strip() for p in time_ref.split(",") if p.strip()]
            if not points:
                return None
            pcts = []
            for p in points:
                t = time_to_seconds(p)
                pcts.append(_percentage_overlap_single(time_list, t, t))
            return sum(pcts) / len(pcts)
        t = time_to_seconds(time_ref)
        return _percentage_overlap_single(time_list, t, t)
    except (ValueError, IndexError):
        return None


def binary_overlap_helper(time_ref: str, time_list: List[Tuple[int, int]]) -> Optional[float]:
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
            return 1.0 if has_any_overlap(time_list, (s_sec, e_sec)) else 0.0
        if "," in time_ref:
            for p in time_ref.split(","):
                p = p.strip()
                if not p:
                    continue
                t = time_to_seconds(p)
                if has_any_overlap(time_list, (t, t)):
                    return 1.0
            return 0.0
        t = time_to_seconds(time_ref)
        return 1.0 if has_any_overlap(time_list, (t, t)) else 0.0
    except (ValueError, IndexError):
        return None


# -----------------------------------------------------------------------------
# NEW: interval union + intersection (seconds)
# -----------------------------------------------------------------------------

def _normalize_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out = []
    for s, e in intervals:
        if e <= s:
            continue
        out.append((int(s), int(e)))
    out.sort()
    return out


def union_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ivs = _normalize_intervals(intervals)
    if not ivs:
        return []
    merged = [ivs[0]]
    for s, e in ivs[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def union_length(intervals: List[Tuple[int, int]]) -> float:
    u = union_intervals(intervals)
    return float(sum(e - s for s, e in u))


def intersection_length(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> float:
    A = union_intervals(a)
    B = union_intervals(b)
    i = j = 0
    total = 0.0
    while i < len(A) and j < len(B):
        a0, a1 = A[i]
        b0, b1 = B[j]
        s = max(a0, b0)
        e = min(a1, b1)
        if e > s:
            total += (e - s)
        if a1 <= b1:
            i += 1
        else:
            j += 1
    return float(total)


# -----------------------------------------------------------------------------
# Ground truth loading
# -----------------------------------------------------------------------------

def load_ground_truth_cache(dataset: str, project_root: Path) -> Dict[Tuple[str, str], str]:
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
                                cache[(video_key, str(qid))] = qa.get("time_reference", "N/A")
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
                        for idx, qa in enumerate(video.get("qa", [])):
                            uid = qa.get("uid") or qa.get("question_id")
                            uid = str(idx)
                            if uid is not None:
                                cache[(video_key, str(uid))] = qa.get("time_reference", "N/A")
            except Exception as e:
                print(f"  Warning: failed to load {path}: {e}")
    return cache


# -----------------------------------------------------------------------------
# Seed events loading
# -----------------------------------------------------------------------------

def event_durations_to_seconds(seed_events: List[Dict], dataset: str, video_key: str) -> List[Dict[str, Any]]:
    events_ids: set[str] = set()
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
        if ev_id in events_ids:
            continue
        events_ids.add(ev_id)
        events_sec.append({"id": ev_id, "interval_sec": [a_sec, b_sec]})
    return events_sec


def gt_to_segments_sec(time_ref: str) -> Optional[List[Tuple[int, int]]]:
    """
    Convert time_reference to GT segments in seconds.
    For points, return a 1-second window [t, t+1) for sec-level metrics + segment hit counting.
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
            if e_sec < s_sec:
                return None
            return [(s_sec, e_sec)]
        if "," in time_ref:
            points = [p.strip() for p in time_ref.split(",") if p.strip()]
            if not points:
                return None
            out = []
            for p in points:
                t = time_to_seconds(p)
                out.append((t, t + 1))
            return out
        t = time_to_seconds(time_ref)
        return [(t, t + 1)]
    except (ValueError, IndexError):
        return None


def event_hits_any_gt_segment(interval: Tuple[int, int], gt_segments: List[Tuple[int, int]]) -> bool:
    """
    interval hits any GT segment (segments are ranges; for point GT they are 1-sec windows).
    """
    s, e = interval
    if e <= s:
        return False
    for gs, ge in gt_segments:
        if ge <= gs:
            continue
        if max(s, gs) < min(e, ge):
            return True
    return False


def gt_segment_hit_by_retrieval(gt_seg: Tuple[int, int], retrieved_intervals: List[Tuple[int, int]]) -> bool:
    """
    Segment-level hit: whether any retrieved interval overlaps GT segment.
    """
    gs, ge = gt_seg
    for s, e in retrieved_intervals:
        if e <= s:
            continue
        if max(s, gs) < min(e, ge):
            return True
    return False


# -----------------------------------------------------------------------------
# Main accuracy computation
# -----------------------------------------------------------------------------

def run_accuracy(seed_events_path, dataset: str, project_root: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
    data = []
    if isinstance(seed_events_path, Path):
        if not seed_events_path.exists():
            raise FileNotFoundError(f"Seed events file not found: {seed_events_path}")
        data = json.loads(seed_events_path.read_text())
    elif isinstance(seed_events_path, list):
        for entry in seed_events_path:
            data_entries = json.loads(entry.read_text())
            if isinstance(data_entries, dict):
                data_entries = data_entries.get("results", [])
            for idx, data_entry in enumerate(data_entries):
                if (
                    len(data) > idx
                    and data[idx].get("video_key") == data_entry.get("video_key")
                    and data[idx].get("question_id") == data_entry.get("question_id")
                ):
                    data[idx]["seed_events"].extend(data_entry.get("seed_events", []))
                else:
                    data.append(data_entry)
    else:
        raise ValueError(f"Expected seed_events path to be a Path or list: {seed_events_path}")

    gt_cache = load_ground_truth_cache(dataset, project_root)

    # all queries stats (avg duration, avg #events)
    n_total = 0
    total_duration_sum_all = 0.0
    num_events_sum_all = 0

    per_query = []
    n_valid = 0

    # aggregate sums over VALID GT queries
    sum_precision_evt = 0.0
    sum_iou_evt = 0.0
    sum_snr_evt = 0.0

    sum_precision_sec = 0.0
    sum_iou_sec = 0.0
    sum_snr_sec = 0.0
    num_reuse_events_sum_all = 0

    sum_recall_evt = 0.0
    sum_f1_evt = 0.0

    sum_recall_sec = 0.0
    sum_f1_sec = 0.0

    for entry in data:
        video_key = entry.get("video_key")
        question_id = entry.get("question_id")
        if video_key is None or question_id is None:
            continue

        seed_events = entry.get("seed_events") or []
        events_sec = event_durations_to_seconds(seed_events, dataset, video_key)
        intervals = [tuple(ev["interval_sec"]) for ev in events_sec]
        num_reuse_events_sum_all += (entry.get("cache_meta", {}).get("A", {}).get("reused_events", 0)) + (entry.get("cache_meta", {}).get("B", {}).get("reused_events", 0))

        total_duration_sec = sum(e - s for s, e in intervals)
        num_events = len(intervals)
        n_total += 1
        total_duration_sum_all += total_duration_sec
        num_events_sum_all += num_events

        key = (video_key, str(question_id))
        time_ref = gt_cache.get(key)
        if time_ref is None or not is_valid_time_reference(time_ref):
            continue

        binary = binary_overlap_helper(time_ref, intervals)
        percentage = overlap_reference_helper(time_ref, intervals)
        if binary is None:
            continue

        gt_segments = gt_to_segments_sec(time_ref)
        if not gt_segments:
            continue

        n_valid += 1

        # ----------------------------
        # Event-level metrics
        # ----------------------------
        tp_evt = 0
        fp_evt = 0
        for ev in events_sec:
            s, e = ev["interval_sec"]
            if event_hits_any_gt_segment((s, e), gt_segments):
                tp_evt += 1
            else:
                fp_evt += 1

        precision_evt = (tp_evt / (tp_evt + fp_evt)) if (tp_evt + fp_evt) > 0 else 0.0
        snr_evt = float("inf") if (fp_evt == 0 and tp_evt > 0) else (tp_evt / fp_evt if fp_evt > 0 else 0.0)

        # segment-level TP/FN for IoU(evt)
        tp_seg = 0
        fn_seg = 0
        for seg in gt_segments:
            if gt_segment_hit_by_retrieval(seg, intervals):
                tp_seg += 1
            else:
                fn_seg += 1
        denom_evt_iou = tp_seg + fn_seg + fp_evt
        iou_evt = (tp_seg / denom_evt_iou) if denom_evt_iou > 0 else 0.0
        
        recall_evt = (tp_seg / (tp_seg + fn_seg)) if (tp_seg + fn_seg) > 0 else 0.0
        f1_evt = f1_score(precision_evt, recall_evt)

        # ----------------------------
        # Sec-level metrics
        # ----------------------------
        R = union_length(intervals)
        G = union_length(gt_segments)
        I = intersection_length(intervals, gt_segments)
        U = (R + G - I)

        precision_sec = (I / R) if R > 0 else 0.0
        iou_sec = (I / U) if U > 0 else 0.0
        noise = (R - I)
        snr_sec = float("inf") if (noise == 0 and I > 0) else (I / noise if noise > 0 else 0.0)

        recall_sec = (I / G) if G > 0 else 0.0
        f1_sec = f1_score(precision_sec, recall_sec)

        # store per-query hit list (events that hit any GT segment)
        events_hit = [ev for ev in events_sec if event_hits_any_gt_segment(tuple(ev["interval_sec"]), gt_segments)]

        pct_val = percentage if percentage is not None else 0.0

        per_query.append({
            "video_key": video_key,
            "question_id": question_id,
            "time_reference": time_ref,
            "gt_segments_sec": gt_segments,
            "binary_overlap": binary,
            "percentage_overlap": pct_val,
            "num_events": len(intervals),
            "total_event_duration_sec": total_duration_sec,
            "events_hit": events_hit,

            # event-level
            "tp_evt": tp_evt,
            "fp_evt": fp_evt,
            "tp_seg": tp_seg,
            "fn_seg": fn_seg,
            "precision_evt": precision_evt,
            "iou_evt": iou_evt,
            "snr_evt": snr_evt,
            "recall_evt": recall_evt,
            "f1_evt": f1_evt,


            # sec-level
            "retrieved_sec_union": R,
            "gt_sec_union": G,
            "intersection_sec": I,
            "precision_sec": precision_sec,
            "iou_sec": iou_sec,
            "snr_sec": snr_sec,
            "recall_sec": recall_sec,
            "f1_sec": f1_sec,
        })

        sum_precision_evt += precision_evt
        sum_iou_evt += iou_evt
        sum_recall_evt += recall_evt
        sum_f1_evt += f1_evt
        # average SNR: treat inf as a large cap? We'll keep numeric sum ignoring inf, and separately count inf.
        # For simplicity: if inf, don't add to sum and count separately.
        if snr_evt != float("inf"):
            sum_snr_evt += snr_evt

        sum_precision_sec += precision_sec
        sum_iou_sec += iou_sec
        sum_recall_sec += recall_sec
        sum_f1_sec += f1_sec
        if snr_sec != float("inf"):
            sum_snr_sec += snr_sec

    if n_valid == 0:
        result = {
            "dataset": dataset,
            "seed_events_file": str(seed_events_path),
            "num_queries_total": n_total,
            "num_queries_with_valid_time_gt": 0,
            "hit_rate": None,
            "coverage_rate": None,
            "avg_total_event_duration_sec": total_duration_sum_all / n_total if n_total else None,
            "avg_num_events": num_events_sum_all / n_total if n_total else None,
            "cache_hit_rate": num_reuse_events_sum_all/ num_events_sum_all,
            "per_query": [],
        }
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2))
        return result

    # Existing (hit/coverage)
    binary_sum = sum(p["binary_overlap"] for p in per_query)
    percentage_sum = sum(p["percentage_overlap"] for p in per_query)

    # Average SNR: also report how many were inf (optional)
    inf_snr_evt = sum(1 for p in per_query if p["snr_evt"] == float("inf"))
    inf_snr_sec = sum(1 for p in per_query if p["snr_sec"] == float("inf"))
    finite_n_evt = n_valid - inf_snr_evt
    finite_n_sec = n_valid - inf_snr_sec

    result = {
        "dataset": dataset,
        "seed_events_file": str(seed_events_path),
        "num_queries_total": n_total,
        "num_queries_with_valid_time_gt": n_valid,

        # requested headline metrics
        "hit_rate": binary_sum / n_valid,
        "coverage_rate": percentage_sum / n_valid,

        "precision_evt": sum_precision_evt / n_valid,
        "iou_evt": sum_iou_evt / n_valid,
        "snr_evt_avg_finite": (sum_snr_evt / finite_n_evt) if finite_n_evt > 0 else None,
        "snr_evt_inf_count": inf_snr_evt,
        "recall_evt": sum_recall_evt / n_valid,
        "f1_evt": sum_f1_evt / n_valid,

        "precision_sec": sum_precision_sec / n_valid,
        "iou_sec": sum_iou_sec / n_valid,
        "snr_sec_avg_finite": (sum_snr_sec / finite_n_sec) if finite_n_sec > 0 else None,
        "snr_sec_inf_count": inf_snr_sec,

        # existing totals
        "binary_hits": int(binary_sum),
        "percentage_sum": percentage_sum,
        "avg_total_event_duration_sec": total_duration_sum_all / n_total if n_total else None,
        "avg_num_events": num_events_sum_all / n_total if n_total else None,
        "cache_hit_rate": num_reuse_events_sum_all/ num_events_sum_all,
        "recall_sec": sum_recall_sec / n_valid,
        "f1_sec": sum_f1_sec / n_valid,

        "per_query": per_query,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))

    return result


def main():
    parser = argparse.ArgumentParser(description="Calculate retrieval accuracy from ECML-PKDD seed_events JSON.")
    parser.add_argument("--dataset", type=str, choices=["AVA100", "LVBench"], required=True)
    parser.add_argument("--seed-events", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root or (Path(__file__).resolve().parent.parent)
    script_dir = Path(__file__).resolve().parent
    seed_events_path = args.seed_events or (script_dir / "indus_outputs" / f"seed_events_{args.dataset}.json")

    print(f"Dataset: {args.dataset}")
    print(f"Seed events: {seed_events_path}")
    print(f"Project root: {project_root}")

    result = run_accuracy(
        seed_events_path=[seed_events_path],
        dataset=args.dataset,
        project_root=project_root,
        output_path=args.output,
    )

    n_valid = result["num_queries_with_valid_time_gt"]
    n_total = result.get("num_queries_total", n_valid)
    print(f"\nQueries (total): {n_total}; with valid time GT: {n_valid}")

    if n_valid > 0:
        print(f"Hit rate:        {result['hit_rate']:.4f}  ({result['binary_hits']}/{n_valid})")
        print(f"Coverage rate:   {result['coverage_rate']:.4f}")

        print("\nEvent-level:")
        print(f"  Precision(evt): {result['precision_evt']:.4f}")
        print(f"  Recall(evt):    {result['recall_evt']:.4f}")
        print(f"  F1(evt):        {result['f1_evt']:.4f}")
        print(f"  IoU(evt):       {result['iou_evt']:.4f}")
        print(f"  SNR(evt):       {result['snr_evt_avg_finite']:.4f} (finite avg), inf_count={result['snr_evt_inf_count']}")

        print("\nSecond-level:")
        print(f"  Precision(sec): {result['precision_sec']:.4f}")
        print(f"  Recall(sec):    {result['recall_sec']:.4f}")
        print(f"  F1(sec):        {result['f1_sec']:.4f}")
        print(f"  IoU(sec):       {result['iou_sec']:.4f}")
        print(f"  SNR(sec):       {result['snr_sec_avg_finite']:.4f} (finite avg), inf_count={result['snr_sec_inf_count']}")

    if n_total > 0 and result.get("avg_total_event_duration_sec") is not None:
        print(f"\nAvg total event duration: {result['avg_total_event_duration_sec']:.2f} sec per query (over all {n_total} queries)")
        print(f"Avg num events per query: {result['avg_num_events']:.2f} (over all {n_total} queries)")

    print(f"Cache hit rate: {result['cache_hit_rate']:.4f}")
    if args.output:
        print(f"\nWrote full result to {args.output}")


if __name__ == "__main__":
    main()