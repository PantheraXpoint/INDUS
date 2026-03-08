#!/usr/bin/env python3
"""
Filter INDUS final output using VLM matrix:
- Remove an event ONLY if its VLM row contains NO "yes" values (i.e., all "no").
- Keep events not present in the VLM trace (unknown -> keep).
Then run accuracy using ECML-PKDD/calc_retrieval_accuracy.py (run_accuracy).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional

# Make sure we can import calc_retrieval_accuracy.py
SCRIPT_DIR = Path(__file__).resolve().parent
ECML_DIR = SCRIPT_DIR  # assume this script is placed under ECML-PKDD/
PROJECT_ROOT_DEFAULT = ECML_DIR.parent

sys.path.insert(0, str(ECML_DIR))
sys.path.insert(0, str(PROJECT_ROOT_DEFAULT))

try:
    from calc_retrieval_accuracy import run_accuracy  # type: ignore
except Exception as e:
    raise RuntimeError(
        f"Failed to import run_accuracy from calc_retrieval_accuracy.py. "
        f"Place this script in ECML-PKDD/ or fix sys.path. Error: {e}"
    )


def _event_id_from_event_obj(ev: Dict[str, Any]) -> Optional[str]:
    eid = ev.get("id") or ev.get("__id__")
    return str(eid) if eid is not None else None


def _build_all_no_removal_set(entry: Dict[str, Any]) -> Set[str]:
    """
    Build a set of event IDs to remove for this entry:
    - Only candidates that appear in vlm_verification_trace.batches
    - Remove if corresponding matrix row has NO 'yes'
    """
    trace = entry.get("vlm_verification_trace")
    if not isinstance(trace, dict):
        return set()

    batches = trace.get("batches")
    if not isinstance(batches, list):
        return set()

    # For each candidate that has a matrix row, mark if any yes occurred.
    has_any_yes: Dict[str, bool] = {}

    for b in batches:
        if not isinstance(b, dict):
            continue
        cand_ids = b.get("candidate_ids")
        matrix = b.get("matrix")
        if not isinstance(cand_ids, list) or not isinstance(matrix, list):
            continue

        # matrix should align with cand_ids by row index.
        # We'll iterate up to min length to be safe.
        n = min(len(cand_ids), len(matrix))
        for i in range(n):
            eid = cand_ids[i]
            if eid is None:
                continue
            eid = str(eid)

            row = matrix[i]
            if not isinstance(row, list):
                # If row is malformed, treat as unknown -> keep (do not remove)
                continue

            row_yes = any(str(x).strip().lower() == "yes" for x in row)
            # If candidate appears multiple times, keep if ANY batch row had yes
            if eid in has_any_yes:
                has_any_yes[eid] = has_any_yes[eid] or row_yes
            else:
                has_any_yes[eid] = row_yes

    # Remove only those we saw and that had no yes
    removal = {eid for eid, any_yes in has_any_yes.items() if not any_yes}
    return removal


def filter_indus_output(
    input_path: Path,
    output_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (filtered_data, stats)
    """
    data = json.loads(input_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {input_path}, got {type(data)}")

    total_entries = 0
    total_before = 0
    total_after = 0
    total_removed = 0
    entries_with_trace = 0

    filtered: List[Dict[str, Any]] = []

    for entry in data:
        if not isinstance(entry, dict):
            continue
        total_entries += 1

        seed_event_ids = entry.get("seed_event_ids") or []
        seed_events = entry.get("seed_events") or []

        if not isinstance(seed_event_ids, list):
            seed_event_ids = []
        if not isinstance(seed_events, list):
            seed_events = []

        before_ids = [str(x) for x in seed_event_ids if x is not None]
        total_before += len(before_ids)

        removal = _build_all_no_removal_set(entry)
        if removal:
            entries_with_trace += 1

        # Filter IDs: remove only if in removal set
        after_ids = [eid for eid in before_ids if eid not in removal]

        # Filter seed_events accordingly
        after_events: List[Dict[str, Any]] = []
        keep_set = set(after_ids)
        for ev in seed_events:
            if not isinstance(ev, dict):
                continue
            eid = _event_id_from_event_obj(ev)
            # If we cannot read an id, keep it (safe fallback)
            if eid is None or eid in keep_set:
                after_events.append(ev)

        removed_count = len(before_ids) - len(after_ids)
        total_removed += removed_count
        total_after += len(after_ids)

        # Update entry
        entry_out = dict(entry)
        entry_out["seed_event_ids"] = after_ids
        entry_out["seed_events"] = after_events

        # Update meta (optional, best-effort)
        meta = entry_out.get("indus_explore_meta")
        if isinstance(meta, dict):
            meta2 = dict(meta)
            meta2["final_event_count"] = len(after_ids)
            meta2["filtered_removed_all_no"] = removed_count
            entry_out["indus_explore_meta"] = meta2

        entry_out["post_filtering"] = {
            "rule": "remove_only_if_matrix_row_has_no_yes",
            "removed_event_ids": sorted(removal),
            "removed_count": removed_count,
            "before_count": len(before_ids),
            "after_count": len(after_ids),
        }

        filtered.append(entry_out)

    stats = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_entries": total_entries,
        "total_events_before": total_before,
        "total_events_after": total_after,
        "total_events_removed": total_removed,
        "entries_with_any_removal": entries_with_trace,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered, indent=2))
    return filtered, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter INDUS final output using VLM matrix (remove only all-no rows) and compute accuracy."
    )
    parser.add_argument("--dataset", required=True, choices=["AVA100", "LVBench"], help="Dataset name")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to seed_events_{dataset}_indus_explore_final.json",
    )
    parser.add_argument(
        "--output-filtered",
        type=Path,
        required=True,
        help="Where to write the filtered JSON",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root (default: parent of ECML-PKDD)",
    )
    parser.add_argument(
        "--accuracy-output",
        type=Path,
        default=None,
        help="Optional: write accuracy result JSON here",
    )

    args = parser.parse_args()
    project_root = args.project_root or PROJECT_ROOT_DEFAULT

    filtered_data, stats = filter_indus_output(args.input, args.output_filtered)

    print("Filtering done:")
    print(json.dumps(stats, indent=2))

    # Run accuracy on the filtered JSON file
    acc = run_accuracy(
        seed_events_path=args.output_filtered,
        dataset=args.dataset,
        project_root=project_root,
        output_path=args.accuracy_output,
    )

    print("\nAccuracy on filtered output:")
    print(f"  num_queries_total: {acc.get('num_queries_total')}")
    print(f"  num_queries_with_valid_time_gt: {acc.get('num_queries_with_valid_time_gt')}")
    print(f"  binary_overlap_accuracy: {acc.get('binary_overlap_accuracy')}")
    print(f"  percentage_overlap_accuracy: {acc.get('percentage_overlap_accuracy')}")

    if args.accuracy_output:
        print(f"\nWrote accuracy JSON to {args.accuracy_output}")


if __name__ == "__main__":
    main()