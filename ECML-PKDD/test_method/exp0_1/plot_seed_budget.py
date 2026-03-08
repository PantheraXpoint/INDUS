#!/usr/bin/env python3
"""
Exp0_1: For each (top_k, expansion) variant, compute binary overlap, percentage overlap,
and total time (unique event duration). Plot accuracy vs number of seeds kept and
total time vs number of seeds kept.

Root path: ECML-PKDD/test_method/exp0_1
- Reads from exp0_1/{dataset}/top{N}/ (10 modes): 5 seed-only + 5 FB-expanded (emb_only_fb, etc.).
- Uses ECML-PKDD/calc_retrieval_accuracy.run_accuracy for overlap.
- Saves plots to exp0_1/plots/{dataset}/

Usage:
  python plot_seed_budget.py --dataset ava100
  python plot_seed_budget.py --dataset lvbench
  python plot_seed_budget.py --dataset both
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union

EXP0_1_ROOT = Path(__file__).resolve().parent
ECML_ROOT = EXP0_1_ROOT.parent.parent
PROJECT_ROOT = EXP0_1_ROOT.parent.parent.parent  # Project-Ava for run_accuracy

sys.path.insert(0, str(ECML_ROOT))
from calc_retrieval_accuracy import run_accuracy as calc_run_accuracy  # type: ignore[import-untyped]

TOP_K_DEFAULTS = [10, 20, 30, 40, 50, 60, 70, 80]
# Five FB-only modes: emb_only, temp_only, emb, temp, all
ALL_MODES = [
    "emb_only", "temp_only", "emb", "temp", "all",
    "emb_only_fb", "temp_only_fb", "emb_fb", "temp_fb", "all_fb",
]
EXPANSION_LABELS = {m: m for m in ALL_MODES}
REPORT_MODES = ALL_MODES
HOPS = 1

DATASET_CONFIG = {
    "ava100": {"name": "AVA100"},
    "lvbench": {"name": "LVBench"},
}


def _norm_key(video_key: Any, question_id: Any) -> Tuple[str, int]:
    vk = str(video_key) if video_key is not None else ""
    try:
        qid = int(question_id) if question_id is not None else -1
    except (TypeError, ValueError):
        qid = -1
    return (vk, qid)


def _load_merged_key_to_events(path_or_paths: Union[Path, List[Path]]) -> Dict[Tuple[str, int], List[Dict]]:
    """Load one or more JSONs and merge seed_events per (video_key, question_id)."""
    paths = [path_or_paths] if isinstance(path_or_paths, Path) else list(path_or_paths)
    key_to_events: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        if not isinstance(raw, list):
            continue
        for entry in raw:
            vk = entry.get("video_key")
            qid = entry.get("question_id")
            if vk is None or qid is None:
                continue
            key_to_events[_norm_key(vk, qid)].extend(entry.get("seed_events") or [])
    return key_to_events


def total_unique_event_duration_sec(path_or_paths: Union[Path, List[Path]]) -> Tuple[float, float]:
    """
    Sum of event durations with deduplication per question.
    Accepts a single path or list of paths (merged by query, like analyze_type fb_eoe).
    Returns (total_sec_all_queries, mean_sec_per_query).
    """
    key_to_events = _load_merged_key_to_events(path_or_paths)
    if not key_to_events:
        return 0.0, 0.0
    total_sec = 0.0
    for _key, events in key_to_events.items():
        unique_duration: Dict[str, float] = {}
        for ev in events:
            dur = ev.get("duration")
            if not isinstance(dur, (list, tuple)) or len(dur) < 2:
                continue
            a, b = float(dur[0]), float(dur[1])
            if b <= a:
                continue
            sec = b - a
            ev_id = ev.get("id") or ev.get("__id__")
            if ev_id:
                unique_duration[ev_id] = sec
        total_sec += sum(unique_duration.values())
    n_keys = len(key_to_events)
    mean_sec = total_sec / n_keys if n_keys else 0.0
    return total_sec, mean_sec


def mean_events_per_query(path_or_paths: Union[Path, List[Path]]) -> float:
    """Average number of unique retrieved events per query (dedup by event id per query). Accepts Path or list of paths (merged)."""
    key_to_events = _load_merged_key_to_events(path_or_paths)
    if not key_to_events:
        return 0.0
    total_events = 0
    for _key, events in key_to_events.items():
        unique_ids = set()
        for ev in events:
            ev_id = ev.get("id") or ev.get("__id__")
            if ev_id:
                unique_ids.add(ev_id)
        total_events += len(unique_ids)
    n_keys = len(key_to_events)
    return total_events / n_keys if n_keys else 0.0


def collect_metrics_for_dataset(dataset_key: str, top_k_list: List[int]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    For each (top_k, mode) load seed-only or expanded file, run run_accuracy, compute mean sec per query.
    Returns nested: result[top_k][mode] = {binary_acc, pct_acc, total_sec, mean_sec_per_query, n_valid}.
    """
    cfg = DATASET_CONFIG[dataset_key]
    dataset_name = cfg["name"]
    base_dir = EXP0_1_ROOT / dataset_key
    results: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    # File names: 5 seed-only (no expanded), 5 FB-expanded
    mode_files = {
        "emb_only": f"seed_events_{dataset_name}_emb_only.json",
        "temp_only": f"seed_events_{dataset_name}_temp_only.json",
        "emb": f"seed_events_{dataset_name}_emb.json",
        "temp": f"seed_events_{dataset_name}_temp.json",
        "all": f"seed_events_{dataset_name}_all.json",
        "emb_only_fb": f"seed_events_{dataset_name}_expanded_emb_only_{HOPS}.json",
        "temp_only_fb": f"seed_events_{dataset_name}_expanded_temp_only_{HOPS}.json",
        "emb_fb": f"seed_events_{dataset_name}_expanded_emb_{HOPS}.json",
        "temp_fb": f"seed_events_{dataset_name}_expanded_temp_{HOPS}.json",
        "all_fb": f"seed_events_{dataset_name}_expanded_all_{HOPS}.json",
    }

    for top_k in top_k_list:
        top_dir = base_dir / f"top{top_k}"
        for mode in ALL_MODES:
            path = top_dir / mode_files[mode]
            if not path.exists():
                print(f"  [Warning] Missing {path}")
                continue
            total_sec, mean_sec = total_unique_event_duration_sec(path)
            mean_ev = mean_events_per_query(path)
            try:
                acc_result = calc_run_accuracy(
                    seed_events_path=path,
                    dataset=dataset_name,
                    project_root=PROJECT_ROOT,
                    output_path=None,
                )
            except Exception as e:
                print(f"  [Warning] run_accuracy failed for {mode} {path}: {e}")
                continue
            n_valid = acc_result.get("num_queries_with_valid_time_gt") or 0
            results[top_k][mode] = {
                "binary_overlap_accuracy": acc_result.get("binary_overlap_accuracy"),
                "percentage_overlap_accuracy": acc_result.get("percentage_overlap_accuracy"),
                "total_sec": total_sec,
                "mean_sec_per_query": mean_sec,
                "mean_events_per_query": mean_ev,
                "n_valid": n_valid,
            }

    return results


def plot_dataset(dataset_key: str, results: Dict[int, Dict[str, Dict[str, Any]]], top_k_list: List[int]) -> None:
    """Generate plots: binary overlap vs N, percentage overlap vs N, total time vs N."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not installed; skipping plots.")
        return

    cfg = DATASET_CONFIG[dataset_key]
    dataset_name = cfg["name"]
    plot_dir = EXP0_1_ROOT / "plots" / dataset_key
    plot_dir.mkdir(parents=True, exist_ok=True)

    x = np.array(top_k_list)
    modes = ALL_MODES
    labels = [EXPANSION_LABELS.get(m, m) for m in modes]

    # 1) Binary overlap accuracy vs seeds kept
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for mode, label in zip(modes, labels):
        ys = []
        for n in top_k_list:
            r = results.get(n, {}).get(mode, {})
            v = r.get("binary_overlap_accuracy")
            ys.append(float(v) if v is not None else float("nan"))
        ax1.plot(x, ys, marker="o", label=label)
    ax1.set_xlabel("Number of seeds kept (top-K)")
    ax1.set_ylabel("Binary overlap accuracy")
    ax1.set_title(f"Binary overlap accuracy vs seed budget ({dataset_name})")
    ax1.set_xticks(top_k_list)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(plot_dir / "exp0_1_binary_overlap_vs_seeds.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # 2) Percentage overlap accuracy vs seeds kept
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for mode, label in zip(modes, labels):
        ys = []
        for n in top_k_list:
            r = results.get(n, {}).get(mode, {})
            v = r.get("percentage_overlap_accuracy")
            ys.append(float(v) if v is not None else float("nan"))
        ax2.plot(x, ys, marker="s", label=label)
    ax2.set_xlabel("Number of seeds kept (top-K)")
    ax2.set_ylabel("Percentage overlap accuracy")
    ax2.set_title(f"Percentage overlap accuracy vs seed budget ({dataset_name})")
    ax2.set_xticks(top_k_list)
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(plot_dir / "exp0_1_percentage_overlap_vs_seeds.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # 3) Mean unique event duration (sec) per query vs seeds kept
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for mode, label in zip(modes, labels):
        ys = []
        for n in top_k_list:
            r = results.get(n, {}).get(mode, {})
            v = r.get("mean_sec_per_query")
            ys.append(float(v) if v is not None else float("nan"))
        ax3.plot(x, ys, marker="^", label=label)
    ax3.set_xlabel("Number of seeds kept (top-K)")
    ax3.set_ylabel("Mean unique event duration (sec) per query")
    ax3.set_title(f"Mean time per query vs seed budget ({dataset_name})")
    ax3.set_xticks(top_k_list)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(plot_dir / "exp0_1_total_time_vs_seeds.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # 4) Efficiency figure: one subplot per mode. Row 0 = accuracy vs mean time/query, row 1 = accuracy vs mean events/query.
    n_modes = len(ALL_MODES)
    fig4, axes = plt.subplots(2, n_modes, figsize=(4 * n_modes, 8))
    for col, mode in enumerate(ALL_MODES):
        label = EXPANSION_LABELS.get(mode, mode)
        # Row 0: accuracy vs mean time per query
        ax0 = axes[0, col]
        x_time = []
        y_bin = []
        y_pct = []
        k_vals = []
        for n in top_k_list:
            r = results.get(n, {}).get(mode, {})
            if not r:
                continue
            t = r.get("mean_sec_per_query")
            b = r.get("binary_overlap_accuracy")
            p = r.get("percentage_overlap_accuracy")
            if t is not None and b is not None:
                x_time.append(float(t))
                y_bin.append(float(b))
                y_pct.append(float(p) if p is not None else float("nan"))
                k_vals.append(n)
        if x_time:
            ax0.plot(x_time, y_bin, "o-", label="Binary", color="C0")
            ax0.plot(x_time, y_pct, "s-", label="Percentage", color="C1")
            for i, k in enumerate(k_vals):
                ax0.annotate(str(k), (x_time[i], y_bin[i]), fontsize=7, alpha=0.8)
        ax0.set_xlabel("Mean time per query (sec)")
        ax0.set_ylabel("Accuracy")
        ax0.set_title(f"{label}")
        ax0.legend(loc="lower right", fontsize=8)
        ax0.set_ylim(0, 1.05)
        ax0.grid(True, alpha=0.3)

        # Row 1: accuracy vs mean events per query
        ax1 = axes[1, col]
        x_ev = []
        y_bin2 = []
        y_pct2 = []
        k_vals2 = []
        for n in top_k_list:
            r = results.get(n, {}).get(mode, {})
            if not r:
                continue
            e = r.get("mean_events_per_query")
            b = r.get("binary_overlap_accuracy")
            p = r.get("percentage_overlap_accuracy")
            if e is not None and b is not None:
                x_ev.append(float(e))
                y_bin2.append(float(b))
                y_pct2.append(float(p) if p is not None else float("nan"))
                k_vals2.append(n)
        if x_ev:
            ax1.plot(x_ev, y_bin2, "o-", label="Binary", color="C0")
            ax1.plot(x_ev, y_pct2, "s-", label="Percentage", color="C1")
            for i, k in enumerate(k_vals2):
                ax1.annotate(str(k), (x_ev[i], y_bin2[i]), fontsize=7, alpha=0.8)
        ax1.set_xlabel("Mean events per query")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(f"{label}")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylabel("Accuracy")
    fig4.suptitle(f"Efficiency: accuracy vs cost per query ({dataset_name})\nRow 1: cost = time (sec). Row 2: cost = # events. Annotations = top-K.", fontsize=11)
    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    fig4.savefig(plot_dir / "exp0_1_efficiency_by_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # 5) Combined comparison: 2x2 — compare all modes, both cost axes and both accuracy metrics in one figure
    # Row 0: binary accuracy. Row 1: percentage accuracy. Col 0: vs time/query. Col 1: vs events/query.
    fig5, axes5 = plt.subplots(2, 2, figsize=(12, 10))
    mode_colors = {
        "emb_only": "C0", "temp_only": "C1", "emb": "C2", "temp": "C3", "all": "C4",
        "emb_only_fb": "C0", "temp_only_fb": "C1", "emb_fb": "C2", "temp_fb": "C3", "all_fb": "C4",
    }
    markers = {
        "emb_only": "o", "temp_only": "s", "emb": "^", "temp": "X", "all": "P",
        "emb_only_fb": "v", "temp_only_fb": "v", "emb_fb": "v", "temp_fb": "v", "all_fb": "v",
    }

    for mode in ALL_MODES:
        label = EXPANSION_LABELS.get(mode, mode)
        c = mode_colors.get(mode, "gray")
        m = markers.get(mode, "o")
        x_time, y_bin_t, y_pct_t, x_ev, y_bin_e, y_pct_e = [], [], [], [], [], []
        k_vals = []
        for n in top_k_list:
            r = results.get(n, {}).get(mode, {})
            if not r:
                continue
            t = r.get("mean_sec_per_query")
            e = r.get("mean_events_per_query")
            b = r.get("binary_overlap_accuracy")
            p = r.get("percentage_overlap_accuracy")
            if t is not None and b is not None:
                x_time.append(float(t))
                y_bin_t.append(float(b))
                y_pct_t.append(float(p) if p is not None else float("nan"))
                k_vals.append(n)
            if e is not None and b is not None:
                x_ev.append(float(e))
                y_bin_e.append(float(b))
                y_pct_e.append(float(p) if p is not None else float("nan"))

        # Top-left: binary accuracy vs time
        if x_time:
            axes5[0, 0].plot(x_time, y_bin_t, marker=m, color=c, label=label)
            for i, k in enumerate(k_vals):
                axes5[0, 0].annotate(str(k), (x_time[i], y_bin_t[i]), fontsize=6, alpha=0.85)
        # Top-right: binary accuracy vs events
        if x_ev:
            axes5[0, 1].plot(x_ev, y_bin_e, marker=m, color=c, label=label)
            for i, k in enumerate(k_vals):
                axes5[0, 1].annotate(str(k), (x_ev[i], y_bin_e[i]), fontsize=6, alpha=0.85)
        # Bottom-left: percentage accuracy vs time
        if x_time and y_pct_t:
            axes5[1, 0].plot(x_time, y_pct_t, marker=m, color=c, label=label)
            for i, k in enumerate(k_vals):
                axes5[1, 0].annotate(str(k), (x_time[i], y_pct_t[i]), fontsize=6, alpha=0.85)
        # Bottom-right: percentage accuracy vs events
        if x_ev and y_pct_e:
            axes5[1, 1].plot(x_ev, y_pct_e, marker=m, color=c, label=label)
            for i, k in enumerate(k_vals):
                axes5[1, 1].annotate(str(k), (x_ev[i], y_pct_e[i]), fontsize=6, alpha=0.85)

    axes5[0, 0].set_xlabel("Mean time per query (sec)")
    axes5[0, 0].set_ylabel("Binary overlap accuracy")
    axes5[0, 0].set_title("Binary accuracy vs time")
    axes5[0, 0].legend(loc="lower right", fontsize=8)
    axes5[0, 0].set_ylim(0, 1.05)
    axes5[0, 0].grid(True, alpha=0.3)

    axes5[0, 1].set_xlabel("Mean events per query")
    axes5[0, 1].set_ylabel("Binary overlap accuracy")
    axes5[0, 1].set_title("Binary accuracy vs events")
    axes5[0, 1].legend(loc="lower right", fontsize=8)
    axes5[0, 1].set_ylim(0, 1.05)
    axes5[0, 1].grid(True, alpha=0.3)

    axes5[1, 0].set_xlabel("Mean time per query (sec)")
    axes5[1, 0].set_ylabel("Percentage overlap accuracy")
    axes5[1, 0].set_title("Percentage accuracy vs time")
    axes5[1, 0].legend(loc="lower right", fontsize=8)
    axes5[1, 0].set_ylim(0, 1.05)
    axes5[1, 0].grid(True, alpha=0.3)

    axes5[1, 1].set_xlabel("Mean events per query")
    axes5[1, 1].set_ylabel("Percentage overlap accuracy")
    axes5[1, 1].set_title("Percentage accuracy vs events")
    axes5[1, 1].legend(loc="lower right", fontsize=8)
    axes5[1, 1].set_ylim(0, 1.05)
    axes5[1, 1].grid(True, alpha=0.3)

    fig5.suptitle(f"Compare expansion approaches: accuracy vs cost ({dataset_name})\nAnnotations = top-K. Same color/marker = same mode in all panels.", fontsize=11)
    fig5.tight_layout(rect=[0, 0, 1, 0.96])
    fig5.savefig(plot_dir / "exp0_1_efficiency_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)

    print(f"  Plots saved to {plot_dir}")


def print_report(dataset_key: str, results: Dict[int, Dict[str, Dict[str, Any]]], top_k_list: List[int]) -> None:
    """Print and save a text report (binary, percentage, mean sec per query per variant)."""
    cfg = DATASET_CONFIG[dataset_key]
    dataset_name = cfg["name"]
    lines = [
        "=" * 80,
        f"Exp0_1 report: Seed budget vs accuracy and mean time per query ({dataset_name})",
        "=" * 80,
        "",
    ]
    for mode in REPORT_MODES:
        label = EXPANSION_LABELS.get(mode, mode)
        lines.append(f"--- {label} ({mode}) ---")
        for n in top_k_list:
            r = results.get(n, {}).get(mode, {})
            if not r:
                lines.append(f"  top-{n}: (no data)")
                continue
            bin_acc = r.get("binary_overlap_accuracy")
            pct_acc = r.get("percentage_overlap_accuracy")
            mean_sec = r.get("mean_sec_per_query") or 0
            mean_ev = r.get("mean_events_per_query") or 0
            n_valid = r.get("n_valid") or 0
            bin_s = f"{bin_acc:.4f}" if bin_acc is not None else "N/A"
            pct_s = f"{pct_acc:.4f}" if pct_acc is not None else "N/A"
            lines.append(f"  top-{n}: binary={bin_s}, pct={pct_s}, mean_sec/query={mean_sec:.0f}, mean_events/query={mean_ev:.1f}, n_valid={n_valid}")
        lines.append("")
    report_text = "\n".join(lines)
    print(report_text)
    report_path = EXP0_1_ROOT / "plots" / dataset_key / "exp0_1_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    print(f"  Report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Exp0_1: Plot accuracy and total time vs seed budget.")
    parser.add_argument("--dataset", choices=["ava100", "lvbench", "both"], required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=TOP_K_DEFAULTS)
    args = parser.parse_args()

    datasets = ["ava100", "lvbench"] if args.dataset == "both" else [args.dataset]
    for dataset_key in datasets:
        print(f"\n=== Dataset: {dataset_key} ===")
        results = collect_metrics_for_dataset(dataset_key, args.top_k)
        if not results:
            print("  No data found. Run run_seed_budget.py first.")
            continue
        print_report(dataset_key, results, args.top_k)
        plot_dataset(dataset_key, results, args.top_k)

    print("\nDone.")


if __name__ == "__main__":
    main()
