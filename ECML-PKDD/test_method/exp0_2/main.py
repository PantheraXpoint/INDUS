#!/usr/bin/env python3
"""
Temporal clustering: top-k seeds, merge within ±T sec, measure hit rate and total time.
After clustering, expand cluster boundary events (start/end per cluster) via forward_backward
and evt_obj_evt; report hit and time per question for each method.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
from .expansion import initialize_vdbs
from llms.QwenLM import QwenLM
from expand_indus_seed_events import fetch_event_data

from .config import SEED_PATHS, PROJECT_ROOT, TOP_K, THRESHOLD_T, EXPAND_K, EXPAND_T, DATASET, SCRIPT_DIR

def _jsonify(x):
    """Recursively convert non-JSON types (set, tuple, Path, numpy types) into JSON-safe ones."""
    from pathlib import Path

    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list,)):
        return [_jsonify(v) for v in x]
    if isinstance(x, set):
        return sorted(_jsonify(v) for v in x)  # deterministic
    if isinstance(x, tuple):
        return [_jsonify(v) for v in x]
    if isinstance(x, Path):
        return str(x)

    # Optional: handle numpy / torch scalars if you have them
    try:
        import numpy as np
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass

    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
    except Exception:
        pass

    # Fallback: stringify unknown objects
    return str(x)


def _write_expansion_grid_checkpoint(
    results_fb: Dict[Tuple[int, int], Dict[str, Any]],
    results_evo: Dict[Tuple[int, int], Dict[str, Any]],
    path: Path,
    queries_done: int,
    total_queries: int,
    partial: bool,
    run_fb: bool,
    run_evo: bool,
) -> None:
    """Write current grid results to JSON (for checkpoint or final)."""
    path.write_text(json.dumps({
        "partial": partial,
        "queries_done": queries_done,
        "total_queries": total_queries,
        "run_fb": run_fb,
        "run_evo": run_evo,
        "top_k": TOP_K,
        "threshold_T": THRESHOLD_T,
        "results_fb": {f"{k}_{T}": results_fb.get((k, T), {}) for k in TOP_K for T in THRESHOLD_T},
        "results_evo": {f"{k}_{T}": results_evo.get((k, T), {}) for k in TOP_K for T in THRESHOLD_T},
    }, indent=2))


def run_expansion_grid(
    data: List[Dict],
    gt: Dict,
    embedding_model,
    run_fb: bool = True,
    run_evo: bool = True,
    checkpoint_path: Optional[Path] = None,
    fb_mode: str = "full",
    questions: Optional[Dict] = None,
    grounding_detector: Any = None,
    grounding_threshold: float = 0.1,
    llm: QwenLM = None,
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]]]:
    """For each (k, T), run clustering → boundary expansion (fb/evo) → aggregate hit rate and time. Write checkpoint periodically if checkpoint_path set."""
    from .expansion import resolve_kg_dir, get_video_path, expand_and_metrics
    from .grounding import extract_query_objects

    results_fb: Dict[Tuple[int, int], Dict[str, Any]] = {}
    results_evo: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for k in TOP_K:
        for T in THRESHOLD_T:
            kt = (k, T)
            results_fb[kt] = {"hit_count": 0, "total_queries_with_gt": 0, "total_time_sec": 0.0, "n_queries_processed": 0}
            results_evo[kt] = {"hit_count": 0, "total_queries_with_gt": 0, "total_time_sec": 0.0, "n_queries_processed": 0}

    n_entries = len([e for e in data if e.get("video_key") is not None and e.get("question_id") is not None])
    prev_kg_dir = None
    for idx, entry in enumerate(data):
        vk, qid = entry.get("video_key"), entry.get("question_id")
        if vk is None or qid is None:
            continue
        events = entry.get("seed_events") or []
        id_intervals = events_to_id_intervals(events)
        kg_dir = resolve_kg_dir(vk, DATASET) if id_intervals else None
        if not prev_kg_dir or kg_dir != prev_kg_dir:
            prev_kg_dir = kg_dir
            events_vdb, entities_vdb, _ = initialize_vdbs(kg_dir, embedding_model)
        gt_seg = gt.get((vk, str(qid))) if gt else None

        fb_selective = run_fb and fb_mode == "selective"
        qinfo = (questions or {}).get((vk, str(qid))) or {}
        query_objects = extract_query_objects(
            qinfo.get("question", ""), qinfo.get("options", []), llm
        ) if fb_selective else []
        for k in TOP_K:
            top = top_k_id_intervals(id_intervals, k, events)
            for T in THRESHOLD_T:
                kt = (k, T)
                print(f"Processing {vk}, {qid}, {k}, {T}")
                if not id_intervals:
                    results_fb[kt]["n_queries_processed"] += 1
                    results_evo[kt]["n_queries_processed"] += 1
                    if gt_seg is not None:
                        results_fb[kt]["total_queries_with_gt"] += 1
                        results_evo[kt]["total_queries_with_gt"] += 1
                    continue
                merged = merge_gap([x[1] for x in top], T)
                boundary_ids = cluster_boundary_event_ids(top, merged)
                if not boundary_ids or not kg_dir:
                    print(f"[Warning] No boundary ids or kg dir for {vk}, {qid}, {k}, {T}")
                    results_fb[kt]["n_queries_processed"] += 1
                    results_evo[kt]["n_queries_processed"] += 1
                    if gt_seg is not None:
                        results_fb[kt]["total_queries_with_gt"] += 1
                        results_evo[kt]["total_queries_with_gt"] += 1
                    continue
                video_path = get_video_path(vk, DATASET) if fb_selective else None
                try:
                    hit_fb, time_fb, hit_evo, time_evo, fb_events_kept = expand_and_metrics(
                        boundary_ids, gt_seg, events_vdb, entities_vdb, 
                        run_fb=run_fb, run_evo=run_evo,
                        fb_selective=fb_selective, video_path=video_path,
                        query_objects=query_objects or None, grounding_detector=grounding_detector,
                        grounding_threshold=grounding_threshold,
                        original_merged=merged,
                    )
                    fb_events_kept |= set([x[0] for x in top])
                except Exception:
                    hit_fb, time_fb, hit_evo, time_evo, fb_events_kept = False, 0.0, False, 0.0, set()
                results_fb[kt]["n_queries_processed"] += 1
                results_evo[kt]["n_queries_processed"] += 1
                results_fb[kt]["total_time_sec"] += time_fb
                results_evo[kt]["total_time_sec"] += time_evo
                if gt_seg is not None:
                    results_fb[kt]["total_queries_with_gt"] += 1
                    results_evo[kt]["total_queries_with_gt"] += 1
                    if run_fb and hit_fb:
                        results_fb[kt]["hit_count"] += 1
                    if run_evo and hit_evo:
                        results_evo[kt]["hit_count"] += 1
        if (idx + 1) % 1 == 0:
            print(f"  Expansion grid progress: {idx + 1}/{len(data)} queries")
            if checkpoint_path is not None:
                _write_expansion_grid_checkpoint(
                    results_fb, results_evo, checkpoint_path, idx + 1, n_entries, True, run_fb, run_evo
                )
    if checkpoint_path is not None:
        _write_expansion_grid_checkpoint(
            results_fb, results_evo, checkpoint_path, n_entries, n_entries, False, run_fb, run_evo
        )
    return results_fb, results_evo
from .gt import load_gt, load_questions
from .clustering import (
    intervals_from_events,
    top_k_intervals,
    merge_gap,
    hits_gt,
    total_sec,
    events_to_id_intervals,
    top_k_id_intervals,
    cluster_boundary_event_ids,
)
from .report import print_and_save, print_expansion_grid_report

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_seed_only_tables(data: List[Dict], gt: Dict) -> List[Dict[str, Any]]:
    """For (EXPAND_K, EXPAND_T), compute hit and total time from clustered seeds; write CSVs and txt gradually."""
    hit_csv_path = SCRIPT_DIR / "seed_hit_table.csv"
    time_csv_path = SCRIPT_DIR / "seed_time_table.csv"
    tables_path = SCRIPT_DIR / "seed_tables.txt"
    json_path = SCRIPT_DIR / "seed_per_question.json"
    rows = []
    time_lines: List[str] = []
    hit_count = 0
    total_time = 0.0
    n_valid = 0

    with open(hit_csv_path, "w") as fh, open(time_csv_path, "w") as ft, open(tables_path, "w") as ftxt:
        fh.write("video_key,question_id,hit_seed\n")
        ft.write("video_key,question_id,time_seed_sec\n")
        ftxt.write("\n" + "=" * 80 + "\n")
        ftxt.write("Seed-only results (clustered seeds, no expansion)\n")
        ftxt.write(f"Config: k={EXPAND_K}, T={EXPAND_T}\n")
        ftxt.write("=" * 80 + "\n\n--- Hit per question ---\n")
        ftxt.write(f"{'video_key':<14} {'qid':<5} hit_seed\n")

        for entry in data:
            vk, qid = entry.get("video_key"), entry.get("question_id")
            if vk is None or qid is None:
                continue
            events = entry.get("seed_events") or []
            id_intervals = events_to_id_intervals(events)
            if not id_intervals:
                row = {"video_key": vk, "question_id": qid, "hit_seed": False, "time_seed_sec": 0.0, "skip": "no_events"}
                rows.append(row)
                fh.write(f"{vk},{qid},0\n")
                ft.write(f"{vk},{qid},0.00\n")
                ftxt.write(f"{vk:<14} {qid:<5} 0\n")
                time_lines.append(f"{vk:<14} {qid:<5} 0.0\n")
                continue
            top = top_k_id_intervals(id_intervals, EXPAND_K, events)
            merged = merge_gap([x[1] for x in top], EXPAND_T)
            gt_seg = gt.get((vk, str(qid)))
            hit_seed = hits_gt(merged, gt_seg) if gt_seg else False
            time_seed_sec = total_sec(merged)
            row = {"video_key": vk, "question_id": qid, "hit_seed": hit_seed, "time_seed_sec": time_seed_sec, "skip": None}
            rows.append(row)
            fh.write(f"{vk},{qid},{1 if hit_seed else 0}\n")
            ft.write(f"{vk},{qid},{time_seed_sec:.2f}\n")
            ftxt.write(f"{vk:<14} {qid:<5} {1 if hit_seed else 0}\n")
            time_lines.append(f"{vk:<14} {qid:<5} {time_seed_sec:.1f}\n")
            n_valid += 1
            if hit_seed:
                hit_count += 1
            total_time += time_seed_sec
            fh.flush()
            ft.flush()
            ftxt.flush()

        ftxt.write(f"\nSummary: {hit_count}/{n_valid} hit\n")
        ftxt.write("\n--- Time per question (seconds) ---\n")
        ftxt.write(f"{'video_key':<14} {'qid':<5} time_seed_sec\n")
        ftxt.writelines(time_lines)
        ftxt.write(f"\nSummary total time: {total_time:.1f} sec\n")
        ftxt.write(f"\nSeed results: {json_path}\n")
        ftxt.write(f"Seed hit (CSV): {hit_csv_path}\n")
        ftxt.write(f"Seed time (CSV): {time_csv_path}\n")

    json_path.write_text(json.dumps({"expand_k": EXPAND_K, "expand_T": EXPAND_T, "per_question": rows}, indent=2))
    print(f"Seed (before expansion) tables written gradually to: {tables_path}, {hit_csv_path}, {time_csv_path}, {json_path}")
    return rows


def print_seed_tables(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Print seed-only summary to console (files already written gradually by run_seed_only_tables)."""
    n = len([x for x in rows if x.get("skip") is None])
    hit_count = sum(1 for x in rows if x.get("hit_seed"))
    total_time = sum(x["time_seed_sec"] for x in rows)
    print("\n" + "=" * 80)
    print("Seed-only results (clustered seeds, no expansion)")
    print(f"Config: k={EXPAND_K}, T={EXPAND_T}")
    print("=" * 80)
    print(f"\nSummary: {hit_count}/{n} hit")
    print(f"Summary total time: {total_time:.1f} sec")
    print(f"Results saved to: {out_path}")


def run_expansion_tables(
    data: List[Dict],
    gt: Dict,
    embedding_model,
    run_fb: bool = True,
    run_evo: bool = True,
    fb_mode: str = "full",
    questions: Optional[Dict] = None,
    grounding_detector: Any = None,
    grounding_threshold: float = 0.1,
    llm: QwenLM = None,
) -> List[Dict[str, Any]]:
    """For (EXPAND_K, EXPAND_T), get cluster boundary event ids per query, expand with fb and/or evo; write CSVs and txt gradually.
    fb_mode: 'full' | 'selective'. When 'selective', only keep FB-expanded events that see query object (grounding)."""
    from .expansion import resolve_kg_dir, get_video_path, expand_and_metrics
    from .grounding import extract_query_objects

    hit_csv_path = SCRIPT_DIR / "expansion_hit_table.csv"
    time_csv_path = SCRIPT_DIR / "expansion_time_table.csv"
    tables_path = SCRIPT_DIR / "expansion_tables.txt"
    json_path = SCRIPT_DIR / "expansion_per_question.json"
    hit_header = "video_key,question_id" + (",hit_fb" if run_fb else "") + (",hit_evo" if run_evo else "") + "\n"
    time_header = "video_key,question_id" + (",time_fb_sec" if run_fb else "") + (",time_evo_sec" if run_evo else "") + "\n"
    hit_cols = ([], [])
    if run_fb:
        hit_cols[0].append("hit_fb")
    if run_evo:
        hit_cols[0].append("hit_evo")
    time_cols = ([], [])
    if run_fb:
        time_cols[0].append("time_fb_sec")
    if run_evo:
        time_cols[0].append("time_evo_sec")
    rows = []
    time_lines: List[str] = []
    n_valid = 0
    hit_fb_count = 0
    hit_evo_count = 0
    total_fb = 0.0
    total_evo = 0.0

    with open(hit_csv_path, "w") as fh, open(time_csv_path, "w") as ft, open(tables_path, "w") as ftxt:
        fh.write(hit_header)
        ft.write(time_header)
        ftxt.write("\n" + "=" * 80 + "\n")
        ftxt.write("After clustering: expansion from cluster boundary events (forward_backward / evt_obj_evt)\n")
        ftxt.write(f"Cluster config: k={EXPAND_K}, T={EXPAND_T}  |  expansion: fb={run_fb}, evo={run_evo}\n")
        ftxt.write("=" * 80 + "\n\n--- Table 1: Hit per question (1=hit GT, 0=miss) ---\n")
        ftxt.write(f"{'video_key':<14} {'qid':<5} " + " ".join(f"{c:<8}" for c in hit_cols[0]) + "\n")
        prev_kg_dir = None
        final_items: List[Dict[str, Any]] = []
        for idx, entry in enumerate(data):
            vk, qid = entry.get("video_key"), entry.get("question_id")
            if vk is None or qid is None:
                continue
            events = entry.get("seed_events") or []
            id_intervals = events_to_id_intervals(events)
            kg_dir = resolve_kg_dir(vk, DATASET) if id_intervals else None
            if not prev_kg_dir or kg_dir != prev_kg_dir:
                prev_kg_dir = kg_dir
                events_vdb, entities_vdb, _ = initialize_vdbs(kg_dir, embedding_model)
            gt_seg = gt.get((vk, str(qid))) if gt else None
            if not id_intervals:
                row = {"video_key": vk, "question_id": qid, "hit_fb": False, "time_fb_sec": 0.0, "hit_evo": False, "time_evo_sec": 0.0, "skip": "no_events"}
                rows.append(row)
            else:
                top = top_k_id_intervals(id_intervals, EXPAND_K, events)
                merged = merge_gap([x[1] for x in top], EXPAND_T)
                boundary_ids = cluster_boundary_event_ids(top, merged)
                if not boundary_ids:
                    row = {"video_key": vk, "question_id": qid, "hit_fb": False, "time_fb_sec": 0.0, "hit_evo": False, "time_evo_sec": 0.0, "skip": "no_boundary"}
                    rows.append(row)
                else:
                    kg_dir = resolve_kg_dir(vk, DATASET)
                    if not kg_dir:
                        row = {"video_key": vk, "question_id": qid, "hit_fb": False, "time_fb_sec": 0.0, "hit_evo": False, "time_evo_sec": 0.0, "skip": "no_kg"}
                        rows.append(row)
                    else:
                        gt_seg = gt.get((vk, str(qid)))
                        fb_selective = (run_fb and fb_mode == "selective")
                        video_path = get_video_path(vk, DATASET) if fb_selective else None
                        qinfo = (questions or {}).get((vk, str(qid))) or {}
                        query_objects = extract_query_objects(
                            qinfo.get("question", ""), qinfo.get("answer_statements", []), 
                            llm,
                        ) if fb_selective else []
                        try:
                            hit_fb, time_fb, hit_evo, time_evo, fb_events_kept = expand_and_metrics(
                                boundary_ids, gt_seg, events_vdb, entities_vdb,
                                run_fb=run_fb, run_evo=run_evo,
                                fb_selective=fb_selective, video_path=video_path,
                                query_objects=query_objects or None, grounding_detector=grounding_detector,
                                grounding_threshold=grounding_threshold,
                            )
                            fb_events_kept |= set([x[0] for x in top])
                            row = {"video_key": vk, "question_id": qid, "hit_fb": hit_fb, "time_fb_sec": time_fb, "hit_evo": hit_evo, "time_evo_sec": time_evo, "skip": None}
                            rows.append(row)
                            if row.get("skip") is None:
                                n_valid += 1
                                if hit_fb:
                                    hit_fb_count += 1
                                if hit_evo:
                                    hit_evo_count += 1
                                total_fb += time_fb
                                total_evo += time_evo
                        except Exception as e:
                            fb_events_kept = set([x[0] for x in top])
                            row = {"video_key": vk, "question_id": qid, "hit_fb": False, "time_fb_sec": 0.0, "hit_evo": False, "time_evo_sec": 0.0, "skip": str(e)}
                            rows.append(row)
            final_items.append({
                "dataset": DATASET,          # e.g. "AVA100"
                "video_key": vk,
                "question_id": qid,
                "seed_events": fetch_event_data(fb_events_kept, events_vdb),  # <- final kept FB selective expanded events
            })
            r = rows[-1]
            fh.write(f"{r['video_key']},{r['question_id']}" + (f",{1 if r['hit_fb'] else 0}" if run_fb else "") + (f",{1 if r['hit_evo'] else 0}" if run_evo else "") + "\n")
            ft.write(f"{r['video_key']},{r['question_id']}" + (f",{r['time_fb_sec']:.2f}" if run_fb else "") + (f",{r['time_evo_sec']:.2f}" if run_evo else "") + "\n")
            hit_parts = [f"{r['video_key']:<14}", f"{r['question_id']:<5}"]
            if run_fb:
                hit_parts.append(f"{1 if r['hit_fb'] else 0:<8}")
            if run_evo:
                hit_parts.append(f"{1 if r['hit_evo'] else 0:<8}")
            ftxt.write(" ".join(hit_parts) + "\n")
            time_parts = [f"{r['video_key']:<14}", f"{r['question_id']:<5}"]
            if run_fb:
                time_parts.append(f"{r['time_fb_sec']:<14.1f}")
            if run_evo:
                time_parts.append(f"{r['time_evo_sec']:<14.1f}")
            time_lines.append(" ".join(time_parts) + "\n")
            fh.flush()
            ft.flush()
            ftxt.flush()

            if (idx + 1) % 10 == 0:
                print(f"  Expansion progress: {idx + 1}/{len(data)}")

        ftxt.write(f"\nSummary forward_backward: {hit_fb_count}/{n_valid} hit\n" if run_fb else "")
        ftxt.write(f"Summary evt_obj_evt: {hit_evo_count}/{n_valid} hit\n" if run_evo else "")
        ftxt.write("\n--- Table 2: Time per question (seconds) ---\n")
        ftxt.write(f"{'video_key':<14} {'qid':<5} " + " ".join(f"{c:<14}" for c in time_cols[0]) + "\n")
        ftxt.writelines(time_lines)
        if run_fb:
            ftxt.write(f"\nSummary total time forward_backward: {total_fb:.1f} sec\n")
        if run_evo:
            ftxt.write(f"Summary total time evt_obj_evt: {total_evo:.1f} sec\n")
        ftxt.write(f"\nResults saved to: {json_path}\n")
        ftxt.write(f"Tables (text): {tables_path}\n")
        ftxt.write(f"Hit table (CSV): {hit_csv_path}\n")
        ftxt.write(f"Time table (CSV): {time_csv_path}\n")

    json_path.write_text(json.dumps({"expand_k": EXPAND_K, "expand_T": EXPAND_T, "run_fb": run_fb, "run_evo": run_evo, "per_question": rows}, indent=2))
    print(f"Expansion tables written gradually to: {tables_path}, {hit_csv_path}, {time_csv_path}, {json_path}")
    return rows, final_items


def print_expansion_tables(
    rows: List[Dict[str, Any]],
    out_path: Path,
    run_fb: bool = True,
    run_evo: bool = True,
) -> None:
    """Print expansion summary to console (files already written gradually by run_expansion_tables)."""
    n = len([x for x in rows if x.get("skip") is None])
    print("\n" + "=" * 80)
    print("After clustering: expansion from cluster boundary events (forward_backward / evt_obj_evt)")
    print(f"Cluster config: k={EXPAND_K}, T={EXPAND_T}  |  expansion: fb={run_fb}, evo={run_evo}")
    print("=" * 80)
    if run_fb:
        print(f"\nSummary forward_backward: {sum(1 for x in rows if x.get('hit_fb'))}/{n} hit")
        print(f"Summary total time forward_backward: {sum(x['time_fb_sec'] for x in rows):.1f} sec")
    if run_evo:
        print(f"Summary evt_obj_evt: {sum(1 for x in rows if x.get('hit_evo'))}/{n} hit")
        print(f"Summary total time evt_obj_evt: {sum(x['time_evo_sec'] for x in rows):.1f} sec")
    print(f"\nResults saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Temporal clustering + optional expansion (fb / evt_obj_evt)")
    parser.add_argument(
        "--expansion",
        choices=["both", "fb", "evo"],
        default="both",
        help="Expansion method: both (default), fb (forward_backward only), evo (evt_obj_evt only)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run expansion for every (k, T) and print hit rate / total time / avg time tables (like clustering).",
    )
    parser.add_argument(
        "--fb-mode",
        choices=["full", "selective"],
        default="full",
        help="FB expansion: full (all prev/next events) or selective (keep only events that see query object via grounding).",
    )
    args = parser.parse_args()
    run_fb = args.expansion in ("both", "fb")
    run_evo = args.expansion in ("both", "evo")
    fb_mode = args.fb_mode

    path = next((p for p in SEED_PATHS if p.exists()), None)
    if not path:
        raise FileNotFoundError(f"No seed file in {SEED_PATHS}")
    print(f"Loading: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected list")

    gt = load_gt()
    print(f"GT: {len(gt)} queries ({PROJECT_ROOT})")

    results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    n_queries = 0

    for entry in data:
        vk, qid = entry.get("video_key"), entry.get("question_id")
        if vk is None or qid is None:
            continue
        n_queries += 1
        key_gt = (vk, str(qid))
        gt_seg = gt.get(key_gt)
        events = entry.get("seed_events") or []
        intervals = intervals_from_events(events)
        if not intervals:
            continue

        for k in TOP_K:
            top = top_k_intervals(intervals, k, events)
            for T in THRESHOLD_T:
                merged = merge_gap(top, T)
                kt = (k, T)
                if kt not in results:
                    results[kt] = {"hit_count": 0, "total_queries_with_gt": 0, "total_time_sec": 0.0, "n_queries_processed": 0}
                results[kt]["n_queries_processed"] += 1
                results[kt]["total_time_sec"] += total_sec(merged)
                if gt_seg is not None:
                    results[kt]["total_queries_with_gt"] += 1
                    if hits_gt(merged, gt_seg):
                        results[kt]["hit_count"] += 1

    n_with_gt = sum(1 for e in data if (e.get("video_key"), str(e.get("question_id"))) in gt) if gt else 0
    print_and_save(results, n_queries, n_with_gt, bool(gt))

    # Seed-only result table (before expansion) for EXPAND_K, EXPAND_T
    print("\nComputing seed-only result table (clustered seeds, no expansion)...")
    seed_rows = run_seed_only_tables(data, gt)
    seed_out = SCRIPT_DIR / "seed_per_question.json"
    print_seed_tables(seed_rows, seed_out)

    # Expansion from cluster boundaries (forward_backward and/or evt_obj_evt)
    print("\nLoading embedding model for expansion...")
    from embeddings.JinaCLIP import JinaCLIP
    embedding_model = JinaCLIP("jinaai/jina-clip-v1")
    questions = None
    grounding_detector = None
    llm = None
    if run_fb and fb_mode == "selective":
        print("FB mode: selective (grounding filter). Loading questions and grounding detector...")
        questions = load_questions()
        from .grounding import GroundingDetector
        grounding_detector = GroundingDetector()
        llm = QwenLM()
    if args.grid:
        print("Running expansion grid for every (k, T) — this may take a long time...")
        results_fb, results_evo = run_expansion_grid(
            data, gt, embedding_model, run_fb=run_fb, run_evo=run_evo,
            checkpoint_path=SCRIPT_DIR / "expansion_grid_results.json",
            fb_mode=fb_mode, questions=questions, grounding_detector=grounding_detector,
            llm=llm,
        )
        print_expansion_grid_report(
            results_fb, results_evo, n_queries, n_with_gt, run_fb, run_evo, bool(gt)
        )
    else:
        print(f"Running expansion from cluster boundary events (fb={run_fb}, evo={run_evo}, fb_mode={fb_mode})...")
        expand_rows, final_items = run_expansion_tables(
            data, gt, embedding_model, run_fb=run_fb, run_evo=run_evo,
            fb_mode=fb_mode, questions=questions, grounding_detector=grounding_detector,
            llm=llm,
        )
        out_path = SCRIPT_DIR / "expansion_per_question.json"
        final_items_path = SCRIPT_DIR / f"expansion_final_items_{DATASET}_{args.fb_mode}.json"
        final_items_path.write_text(json.dumps(_jsonify(final_items), indent=2, ensure_ascii=False))
        print_expansion_tables(expand_rows, out_path, run_fb=run_fb, run_evo=run_evo)


if __name__ == "__main__":
    main()
