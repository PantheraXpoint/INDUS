#!/usr/bin/env python3
"""
Run ECML-PKDD/calc_retrieval_accuracy.py on top_k_events_retrieval output.

Events retrieval writes { "results": [...], "num_queries", "average_latency_seconds_per_query" }.
calc_retrieval_accuracy expects a JSON file whose root is a list of entries (video_key, question_id, seed_events).
This script extracts "results" and writes a list-only JSON, then invokes calc_retrieval_accuracy without modifying it.

The calc script reports: hit_rate, coverage_rate; event-level precision_evt, iou_evt, snr_evt;
second-level precision_sec, iou_sec, snr_sec; and per-query details. Use --output to write the full result JSON.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
script_dir = Path(__file__).resolve().parent


def _print_result_summary(result_path: Path) -> None:
    """Print a one-line summary of the written result (including new metrics)."""
    if not result_path.exists():
        return
    try:
        data = json.loads(result_path.read_text())
        n_valid = data.get("num_queries_with_valid_time_gt", 0)
        if n_valid == 0:
            print(f"Result written to {result_path} (no valid time GT)")
            return
        hit = data.get("hit_rate")
        cov = data.get("coverage_rate")
        pe = data.get("precision_evt")
        ie = data.get("iou_evt")
        ps = data.get("precision_sec")
        iis = data.get("iou_sec")
        parts = [f"hit_rate={hit:.4f}", f"coverage={cov:.4f}"]
        if pe is not None:
            parts.append(f"precision_evt={pe:.4f}")
        if ie is not None:
            parts.append(f"iou_evt={ie:.4f}")
        if ps is not None:
            parts.append(f"precision_sec={ps:.4f}")
        if iis is not None:
            parts.append(f"iou_sec={iis:.4f}")
        print(f"Result summary: {' | '.join(parts)} -> {result_path}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run retrieval accuracy on top_k_events_retrieval output (converts results wrapper to list)."
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
        help="Path to events retrieval JSON (default: script_dir/seed_events_{dataset}.json)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root (default: parent of top_k_events_retrieval)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write full accuracy result JSON from calc_retrieval_accuracy here",
    )
    parser.add_argument(
        "--keep-converted",
        action="store_true",
        help="Keep the converted list-only JSON file (default: write to temp and delete)",
    )
    args = parser.parse_args()

    seed_events_path = args.seed_events or (script_dir / f"seed_events_{args.dataset}.json")
    if not seed_events_path.exists():
        print(f"Error: {seed_events_path} not found")
        sys.exit(1)

    proot = args.project_root or project_root
    data = json.loads(seed_events_path.read_text())
    if isinstance(data, list):
        entries = data
    else:
        entries = data.get("results")
        if entries is None:
            print("Error: input JSON has no 'results' key and is not a list")
            sys.exit(1)

    out_list_path = script_dir / f"seed_events_{args.dataset}_for_calc.json"
    out_list_path.write_text(json.dumps(entries, indent=2))

    calc_script = proot / "ECML-PKDD" / "calc_retrieval_accuracy.py"
    if not calc_script.exists():
        print(f"Error: {calc_script} not found")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(calc_script),
        "--dataset",
        args.dataset,
        "--seed-events",
        str(out_list_path.resolve()),
        "--project-root",
        str(proot),
    ]
    if args.output:
        cmd.extend(["--output", str(args.output)])

    exit_code = subprocess.run(cmd, cwd=str(proot)).returncode
    if args.output and exit_code == 0:
        _print_result_summary(args.output)
    if not args.keep_converted and out_list_path.exists():
        out_list_path.unlink(missing_ok=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
