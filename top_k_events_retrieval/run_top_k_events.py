#!/usr/bin/env python3
"""
Run event retrieval for multiple top-k values. Edit TOP_K_VALUES and DATASETS below.
"""
import argparse
import subprocess
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Edit these: list of top-k values and datasets to run
# -----------------------------------------------------------------------------
TOP_K_VALUES = [20, 30, 40, 50, 60, 70, 80]
DATASETS = ["AVA100", "LVBench"]  # or ["AVA100"] / ["LVBench"] for a single dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def main():
    parser = argparse.ArgumentParser(description="Run events_retrieval.py for multiple top-k configs")
    parser.add_argument("--port", type=int, default=8000, help="LLM server port")
    parser.add_argument("--model", type=str, default=None, help="LLM model type")
    parser.add_argument("--top-k", type=int, nargs="+", default=None, help="Override TOP_K_VALUES from script (e.g. --top-k 5 10 20)")
    parser.add_argument("--dataset", type=str, nargs="+", default=None, help="Override DATASETS from script (e.g. --dataset AVA100)")
    args = parser.parse_args()

    top_k_list = args.top_k if args.top_k is not None else TOP_K_VALUES
    datasets = args.dataset if args.dataset is not None else DATASETS

    for dataset in datasets:
        for k in top_k_list:
            out_name = f"seed_events_{dataset}_k{k}.json"
            output_path = SCRIPT_DIR / out_name
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "events_retrieval.py"),
                "--dataset", dataset,
                "--top-k", str(k),
                "--output", str(output_path),
                "--port", str(args.port),
            ]
            if args.model:
                cmd.extend(["--model", args.model])
            print(f"Running: dataset={dataset} top_k={k} -> {output_path}")
            subprocess.run(cmd, check=True)
    print("Done.")


if __name__ == "__main__":
    main()
