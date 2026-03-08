#!/usr/bin/env python3
"""
Analyze wrong questions per mode, categorized by retrieval vs reasoning failure.

For each mode, identifies wrong questions and splits them into:
  - wrong_due_to_retrieval: No overlap with ground truth (retrieval issue)
  - wrong_due_to_content: Has overlap with ground truth (VLM reasoning issue)

Inputs:
  - JSON files from query_vlm_multimode.py: outputs/query_VLM_multimode_*.json
  - Evaluation results: viz_eval_{dataset}_mode*/evaluation_results.json

Outputs:
  - One JSON file per mode: outputs/wrong_questions_mode{1-7}_{dataset}_{model}.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


QuestionKey = Tuple[int, int]  # (video_id, question_id)


ERROR_PATTERNS = [
    r"internal error",
    r"error happened",
    r"status code",
    r"responsetype\.",
    r"input_length_error",
    r"token_limit",
    r"timeout",
    r"traceback",
    r"exception",
]


def _looks_like_error_response(response: Any) -> bool:
    if not response or not isinstance(response, str):
        return True
    low = response.lower()
    return any(re.search(p, low, re.IGNORECASE) for p in ERROR_PATTERNS)


def extract_predicted_answer(response: Any) -> Optional[str]:
    """Extract predicted letter (A/B/C/D) from a response string. Returns None if not found/invalid/error."""
    if _looks_like_error_response(response):
        return None
    text = str(response).strip()

    # Remove markdown code block if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try JSON first
    try:
        data = json.loads(text)
        ans = data.get("Answer") or data.get("answer")
        if ans and str(ans).strip().upper() in ("A", "B", "C", "D"):
            return str(ans).strip().upper()
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: look for '"Answer": "X"'
    m = re.search(r'"Answer"\s*:\s*"([ABCD])"', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Another fallback: look for a trailing option letter
    m = re.search(r"\b([ABCD])\s*[\.\)]\s*$", text)
    if m:
        return m.group(1).upper()

    return None


def normalize_gt_answer(gt: Any) -> Optional[str]:
    if not gt:
        return None
    s = str(gt).strip().upper()
    return s if s in ("A", "B", "C", "D") else None


def _mode_suffix(retrieval_mode: str) -> str:
    return f"_{retrieval_mode}" if retrieval_mode and retrieval_mode != "tri_view" else ""


def find_mode_files(
    outputs_dir: str,
    dataset: str,
    model: str,
    retrieval_mode: str,
    mode: int,
    processes: Sequence[int],
) -> List[str]:
    """Find all matching files for a given mode and list of processes."""
    suf = _mode_suffix(retrieval_mode)
    paths: List[str] = []
    for p in processes:
        name = f"query_VLM_multimode_{dataset}_{model}_mode{mode}{suf}_process{p}.json"
        path = os.path.join(outputs_dir, name)
        if os.path.exists(path):
            paths.append(path)
    return paths


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_mode_entries(entries: List[Dict[str, Any]]) -> Dict[QuestionKey, Dict[str, Any]]:
    """
    Index entries by (video_id, question_id).
    If duplicates exist, keep the one with a valid parsed predicted answer; otherwise keep the later entry.
    """
    by_key: Dict[QuestionKey, Dict[str, Any]] = {}
    for e in entries:
        try:
            vid = int(e.get("video_id"))
            qid = int(e.get("question_id"))
        except (TypeError, ValueError):
            continue
        key = (vid, qid)
        if key not in by_key:
            by_key[key] = e
            continue

        prev = by_key[key]
        prev_pred = extract_predicted_answer(prev.get("response"))
        new_pred = extract_predicted_answer(e.get("response"))
        if new_pred is not None and prev_pred is None:
            by_key[key] = e
        elif (new_pred is None) == (prev_pred is None):
            # Both parse or both don't parse -> keep newer
            by_key[key] = e
        # else: keep prev
    return by_key


def load_evaluation_results(eval_dir: str) -> Dict[QuestionKey, Dict[str, Any]]:
    """
    Load evaluation_results.json and index by (video_id, question_id).
    Returns dict mapping QuestionKey to evaluation result entry.
    """
    eval_path = os.path.join(eval_dir, "evaluation_results.json")
    if not os.path.exists(eval_path):
        return {}
    
    entries = load_json(eval_path)
    by_key: Dict[QuestionKey, Dict[str, Any]] = {}
    for e in entries:
        try:
            vid = int(e.get("video_id"))
            qid = int(e.get("question_id"))
            by_key[(vid, qid)] = e
        except (TypeError, ValueError):
            continue
    return by_key


def extract_overlap_variants(eval_entry: Dict[str, Any], mode: int) -> Dict[str, Any]:
    """
    Extract all overlap variants for a given mode from evaluation entry.
    Returns dict with prefixed variant names and their values.
    
    Frames-only (modes 1, 2):
      - frames_variant_a_frames_vs_gt_events: {binary_hit, num_frames_in_gt, num_frames}
      - frames_variant_b_frames_events_vs_time_ref: {binary_overlap, percentage_overlap}
      - frames_variant_c_frames_events_vs_gt_events: {binary_overlap, percentage_overlap}
    
    Events-only (modes 3, 4, 5):
      - events_variant_a_events_vs_time_ref: {binary_overlap, percentage_overlap}
      - events_variant_b_events_vs_gt_events: {binary_overlap, percentage_overlap}
    
    Combined (modes 6, 7):
      - All frames variants + all events variants
    """
    variants: Dict[str, Any] = {}
    
    if mode in (1, 2):
        # Frames-only
        frames_only = eval_entry.get("frames_only", {})
        if "variant_a_frames_vs_gt_events" in frames_only:
            variants["frames_variant_a_frames_vs_gt_events"] = frames_only["variant_a_frames_vs_gt_events"]
        if "variant_b_frames_events_vs_time_ref" in frames_only:
            variants["frames_variant_b_frames_events_vs_time_ref"] = frames_only["variant_b_frames_events_vs_time_ref"]
        if "variant_c_frames_events_vs_gt_events" in frames_only:
            variants["frames_variant_c_frames_events_vs_gt_events"] = frames_only["variant_c_frames_events_vs_gt_events"]
    
    elif mode in (3, 4, 5):
        # Events-only
        events_only = eval_entry.get("events_only", {})
        if "variant_a_events_vs_time_ref" in events_only:
            variants["events_variant_a_events_vs_time_ref"] = events_only["variant_a_events_vs_time_ref"]
        if "variant_b_events_vs_gt_events" in events_only:
            variants["events_variant_b_events_vs_gt_events"] = events_only["variant_b_events_vs_gt_events"]
    
    elif mode in (6, 7):
        # Combined: frames + events
        frames_only = eval_entry.get("frames_only", {})
        events_only = eval_entry.get("events_only", {})
        
        if "variant_a_frames_vs_gt_events" in frames_only:
            variants["frames_variant_a_frames_vs_gt_events"] = frames_only["variant_a_frames_vs_gt_events"]
        if "variant_b_frames_events_vs_time_ref" in frames_only:
            variants["frames_variant_b_frames_events_vs_time_ref"] = frames_only["variant_b_frames_events_vs_time_ref"]
        if "variant_c_frames_events_vs_gt_events" in frames_only:
            variants["frames_variant_c_frames_events_vs_gt_events"] = frames_only["variant_c_frames_events_vs_gt_events"]
        
        if "variant_a_events_vs_time_ref" in events_only:
            variants["events_variant_a_events_vs_time_ref"] = events_only["variant_a_events_vs_time_ref"]
        if "variant_b_events_vs_gt_events" in events_only:
            variants["events_variant_b_events_vs_gt_events"] = events_only["variant_b_events_vs_gt_events"]
    
    return variants


def has_overlap(eval_entry: Dict[str, Any], mode: int) -> bool:
    """
    Check if ANY variant has overlap (binary_hit=1.0 or binary_overlap=1.0).
    If has_time_reference is false/null, treat as has_overlap (wrong_due_to_content).
    """
    # If no time reference, treat as wrong_due_to_content
    if not eval_entry.get("has_time_reference", True):
        return True
    
    variants = extract_overlap_variants(eval_entry, mode)
    
    for variant_name, variant_data in variants.items():
        if isinstance(variant_data, dict):
            # Check binary_hit (for variant_a_frames_vs_gt_events)
            if "binary_hit" in variant_data:
                if variant_data.get("binary_hit", 0.0) == 1.0:
                    return True
            # Check binary_overlap (for other variants)
            if "binary_overlap" in variant_data:
                if variant_data.get("binary_overlap", 0.0) == 1.0:
                    return True
    
    return False


def analyze_mode(
    mode: int,
    query_entries: Dict[QuestionKey, Dict[str, Any]],
    eval_entries: Dict[QuestionKey, Dict[str, Any]],
    require_answer_format: bool,
    include_full_response: bool,
) -> Dict[str, Any]:
    """
    Analyze wrong questions for a single mode and categorize by overlap.
    Returns dict with wrong_due_to_retrieval and wrong_due_to_content lists.
    """
    wrong_due_to_retrieval: List[Dict[str, Any]] = []
    wrong_due_to_content: List[Dict[str, Any]] = []
    
    stats = {
        "total_questions": 0,
        "gt_missing_or_invalid": 0,
        "pred_missing_or_invalid": 0,
        "correct": 0,
        "wrong": 0,
        "wrong_due_to_retrieval": 0,
        "wrong_due_to_content": 0,
        "eval_missing": 0,
    }
    
    for key, entry in query_entries.items():
        stats["total_questions"] += 1
        
        gt = normalize_gt_answer(entry.get("answer"))
        if gt is None:
            stats["gt_missing_or_invalid"] += 1
            continue
        
        pred = extract_predicted_answer(entry.get("response"))
        if pred is None:
            stats["pred_missing_or_invalid"] += 1
            # Count as wrong but need eval to categorize
            if key in eval_entries:
                eval_entry = eval_entries[key]
                has_ov = has_overlap(eval_entry, mode)
                variants = extract_overlap_variants(eval_entry, mode)
                question_record = {
                    "video_id": key[0],
                    "question_id": key[1],
                    "question": entry.get("question", ""),
                    "gt_answer": gt,
                    "pred": pred,
                    "correct": False,
                    "has_overlap": has_ov,
                    "overlap_variants": variants,
                    "response": entry.get("response", "") if include_full_response else (entry.get("response", "")[:500] + "..." if len(entry.get("response", "")) > 500 else entry.get("response", "")),
                }
                if has_ov:
                    wrong_due_to_content.append(question_record)
                    stats["wrong_due_to_content"] += 1
                else:
                    wrong_due_to_retrieval.append(question_record)
                    stats["wrong_due_to_retrieval"] += 1
                stats["wrong"] += 1
            else:
                stats["eval_missing"] += 1
            continue
        
        if require_answer_format and pred not in ("A", "B", "C", "D"):
            stats["pred_missing_or_invalid"] += 1
            # Count as wrong but need eval to categorize
            if key in eval_entries:
                eval_entry = eval_entries[key]
                has_ov = has_overlap(eval_entry, mode)
                variants = extract_overlap_variants(eval_entry, mode)
                question_record = {
                    "video_id": key[0],
                    "question_id": key[1],
                    "question": entry.get("question", ""),
                    "gt_answer": gt,
                    "pred": pred,
                    "correct": False,
                    "has_overlap": has_ov,
                    "overlap_variants": variants,
                    "response": entry.get("response", "") if include_full_response else (entry.get("response", "")[:500] + "..." if len(entry.get("response", "")) > 500 else entry.get("response", "")),
                }
                if has_ov:
                    wrong_due_to_content.append(question_record)
                    stats["wrong_due_to_content"] += 1
                else:
                    wrong_due_to_retrieval.append(question_record)
                    stats["wrong_due_to_retrieval"] += 1
                stats["wrong"] += 1
            else:
                stats["eval_missing"] += 1
            continue
        
        if pred == gt:
            stats["correct"] += 1
        else:
            stats["wrong"] += 1
            # Categorize based on overlap
            if key in eval_entries:
                eval_entry = eval_entries[key]
                has_ov = has_overlap(eval_entry, mode)
                variants = extract_overlap_variants(eval_entry, mode)
                question_record = {
                    "video_id": key[0],
                    "question_id": key[1],
                    "question": entry.get("question", ""),
                    "gt_answer": gt,
                    "pred": pred,
                    "correct": False,
                    "has_overlap": has_ov,
                    "overlap_variants": variants,
                    "response": entry.get("response", "") if include_full_response else (entry.get("response", "")[:500] + "..." if len(entry.get("response", "")) > 500 else entry.get("response", "")),
                }
                if has_ov:
                    wrong_due_to_content.append(question_record)
                    stats["wrong_due_to_content"] += 1
                else:
                    wrong_due_to_retrieval.append(question_record)
                    stats["wrong_due_to_retrieval"] += 1
            else:
                stats["eval_missing"] += 1
    
    return {
        "wrong_due_to_retrieval": sorted(wrong_due_to_retrieval, key=lambda x: (x["video_id"], x["question_id"])),
        "wrong_due_to_content": sorted(wrong_due_to_content, key=lambda x: (x["video_id"], x["question_id"])),
        "stats": stats,
    }


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze wrong questions per mode, categorized by retrieval vs reasoning failure."
    )
    parser.add_argument("--outputs_dir", default="outputs", help="Directory containing multimode output JSONs.")
    parser.add_argument("--dataset", required=True, help="Dataset name in filenames (e.g. ava100).")
    parser.add_argument("--model", default="qwenvl", help="Model name in filenames (e.g. qwenvl).")
    parser.add_argument(
        "--retrieval_mode",
        default="tri_view",
        help="retrieval_mode suffix in filenames (tri_view means no suffix).",
    )
    parser.add_argument("--processes", default="1", help="Comma-separated process numbers to merge, e.g. '1,2,3'.")
    parser.add_argument("--modes", default="1,2,3,4,5,6,7", help="Comma-separated config modes to include.")
    parser.add_argument(
        "--validate_answer_format",
        action="store_true",
        help="Require predicted Answer to be one of A/B/C/D; otherwise count as wrong.",
    )
    parser.add_argument(
        "--include_full_response",
        action="store_true",
        help="Include full response text in output JSON (can be huge).",
    )
    parser.add_argument(
        "--eval_dir_prefix",
        default="viz_eval",
        help="Prefix for evaluation results directories (default: viz_eval).",
    )

    args = parser.parse_args()

    outputs_dir = args.outputs_dir
    dataset = args.dataset
    model = args.model
    retrieval_mode = args.retrieval_mode
    processes = [int(x) for x in args.processes.split(",") if x.strip()]
    modes = [int(x) for x in args.modes.split(",") if x.strip()]

    # Process each mode
    for mode in modes:
        print(f"\n=== Processing Mode {mode} ===")
        
        # Load query results
        files = find_mode_files(outputs_dir, dataset, model, retrieval_mode, mode, processes)
        if not files:
            print(f"  WARNING: No query files found for mode {mode}, skipping...")
            continue
        
        print(f"  Loading query results from {len(files)} file(s)...")
        combined: List[Dict[str, Any]] = []
        for fp in files:
            combined.extend(load_json(fp))
        query_entries = index_mode_entries(combined)
        print(f"  Loaded {len(query_entries)} query entries")
        
        # Load evaluation results
        eval_dir = f"{args.eval_dir_prefix}_{dataset}_mode{mode}"
        print(f"  Loading evaluation results from {eval_dir}...")
        eval_entries = load_evaluation_results(eval_dir)
        print(f"  Loaded {len(eval_entries)} evaluation entries")
        
        # Analyze
        result = analyze_mode(
            mode,
            query_entries,
            eval_entries,
            require_answer_format=args.validate_answer_format,
            include_full_response=args.include_full_response,
        )
        
        # Prepare output
        suf = _mode_suffix(retrieval_mode)
        output_path = os.path.join(
            outputs_dir, f"wrong_questions_mode{mode}_{dataset}_{model}{suf}.json"
        )
        
        output_data = {
            "mode": mode,
            "dataset": dataset,
            "model": model,
            "retrieval_mode": retrieval_mode,
            "processes": processes,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "validate_answer_format": bool(args.validate_answer_format),
            "include_full_response": bool(args.include_full_response),
            "total_wrong_questions": len(result["wrong_due_to_retrieval"]) + len(result["wrong_due_to_content"]),
            "wrong_due_to_retrieval": len(result["wrong_due_to_retrieval"]),
            "wrong_due_to_content": len(result["wrong_due_to_content"]),
            "stats": result["stats"],
            "questions_wrong_due_to_retrieval": result["wrong_due_to_retrieval"],
            "questions_wrong_due_to_content": result["wrong_due_to_content"],
        }
        
        write_json(output_path, output_data)
        print(f"  ✓ Saved to {output_path}")
        print(f"    - Wrong due to retrieval: {len(result['wrong_due_to_retrieval'])}")
        print(f"    - Wrong due to content: {len(result['wrong_due_to_content'])}")
        print(f"    - Total wrong: {len(result['wrong_due_to_retrieval']) + len(result['wrong_due_to_content'])}")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
