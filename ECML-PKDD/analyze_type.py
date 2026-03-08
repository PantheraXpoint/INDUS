#!/usr/bin/env python3
"""
Analyze question types from query_type_classification_AVA100.json and report
which expansion method (forward_backward, evt_obj_evt) is most effective for
overlap accuracy per type.

Uses:
- ECML-PKDD/indus_outputs/query_type_classification_AVA100.json (llm_output -> 5 types)
- ECML-PKDD/previous_run/seed_events_AVA100_expansion_trace_forward_backward_1.json
- ECML-PKDD/previous_run/seed_events_AVA100_expansion_trace_evt_obj_evt_1.json
- Overlap accuracy is recalculated from expanded seed_events JSON per method (run_accuracy).
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Import run_accuracy from ECML-PKDD (path set at runtime)
sys.path.insert(0, str(Path(__file__).resolve().parent / "ECML-PKDD"))
from calc_retrieval_accuracy import run_accuracy as calc_run_accuracy  # type: ignore[import-untyped]

# -----------------------------------------------------------------------------
# Paths (project root = parent of script). Set from --dataset in main().
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ECML = PROJECT_ROOT / "ECML-PKDD"

# These are set in main() from --dataset (default ava100).
DATASET: str = "AVA100"
INDUS_OUT: Path = ECML / "ava100_analysis"
PREVIOUS_RUN: Path = ECML / "ava100_retrieval"
CLASSIFICATION_PATH: Path = INDUS_OUT / "query_type_classification_AVA100.json"
TRACE_FORWARD_BACKWARD: Path = PREVIOUS_RUN / "seed_events_AVA100_expansion_trace_forward_backward_1.json"
TRACE_EVT_OBJ_EVT: Path = PREVIOUS_RUN / "seed_events_AVA100_expansion_trace_evt_obj_evt_1.json"
EXPANDED_SEED_PATHS: Dict[str, Path] = {}
EXPANDED_SEED_PATHS_LIST: Dict[str, List[Path]] = {}

EXPANSION_METHODS = ["base", "fb", "eoe", "fb_eoe"]
TYPE_NAMES = ["counting", "temporal", "spatial", "content_text"]

DATASET_CHOICES = ["ava100", "lvbench"]


def _norm_key(video_key: Any, question_id: Any) -> Tuple[str, int]:
    """Normalize (video_key, question_id) to (str, int) for consistent dict lookups across JSON sources."""
    vk = str(video_key) if video_key is not None else ""
    try:
        qid = int(question_id) if question_id is not None else -1
    except (TypeError, ValueError):
        qid = -1
    return (vk, qid)


def set_dataset_paths(dataset_key: str) -> None:
    """Set global path variables from --dataset (ava100 or lvbench)."""
    global DATASET, INDUS_OUT, PREVIOUS_RUN, CLASSIFICATION_PATH
    global TRACE_FORWARD_BACKWARD, TRACE_EVT_OBJ_EVT
    global EXPANDED_SEED_PATHS, EXPANDED_SEED_PATHS_LIST
    DATASET = "AVA100" if dataset_key == "ava100" else "LVBench"
    INDUS_OUT = ECML / f"{dataset_key}_analysis"
    PREVIOUS_RUN = ECML / f"{dataset_key}_retrieval"
    CLASSIFICATION_PATH = INDUS_OUT / f"query_type_classification_{DATASET}.json"
    TRACE_FORWARD_BACKWARD = PREVIOUS_RUN / f"seed_events_{DATASET}_expansion_trace_forward_backward_1.json"
    TRACE_EVT_OBJ_EVT = PREVIOUS_RUN / f"seed_events_{DATASET}_expansion_trace_evt_obj_evt_1.json"
    EXPANDED_SEED_PATHS = {
        "base": PREVIOUS_RUN / f"seed_events_{DATASET}.json",
        "fb": PREVIOUS_RUN / f"seed_events_{DATASET}_expanded_forward_backward_1.json",
        "eoe": PREVIOUS_RUN / f"seed_events_{DATASET}_expanded_evt_obj_evt_1.json",
    }
    EXPANDED_SEED_PATHS_LIST = {
        "fb_eoe": [EXPANDED_SEED_PATHS["fb"], EXPANDED_SEED_PATHS["eoe"]],
    }


def parse_llm_output(llm_output: str) -> Dict[str, Any]:
    """Parse llm_output JSON string; return dict with needs_counting, needs_temporal_direction, needs_spatial_positions, has_content_text."""
    try:
        return json.loads(llm_output)
    except (json.JSONDecodeError, TypeError):
        return {}


def _temporal_has_time_content(temporal_parsed: Dict[str, Any]) -> bool:
    """True if localization_time.exists or content_time.exists in parsed temporal llm_output."""
    if not temporal_parsed:
        return False
    loc = temporal_parsed.get("localization_time") or {}
    content = temporal_parsed.get("content_time") or {}
    return loc.get("exists") is True or content.get("exists") is True


def assign_types(parsed: Dict[str, Any], temporal_parsed: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Return list of type labels for this question (can be multiple).
    Types: counting, temporal, spatial, content_text.
    content_text is True if has_content_text (from classification) OR
    localization_time.exists or content_time.exists (from temporal_analysis).
    """
    types = []
    if parsed.get("needs_counting") is True:
        types.append("counting")
    if (parsed.get("needs_temporal_direction") or "").strip().lower() not in ("none", ""):
        types.append("temporal")
    if parsed.get("needs_spatial_positions") is True:
        types.append("spatial")
    if parsed.get("has_content_text") is True or _temporal_has_time_content(temporal_parsed or {}):
        types.append("content_text")
    return types if types else ["other"]


def load_classification(path: Path) -> Tuple[Dict[str, List[Tuple[str, int]]], List[Tuple[str, int]]]:
    """
    Load query_type_classification_AVA100.json and build:
    - type_to_items: each of the 4 types -> list of (video_key, question_id)
    - other_items: list of (video_key, question_id) for questions with no type
    Uses temporal_analysis_AVA100.json to also treat content_text as True when
    localization_time.exists or content_time.exists in that file.
    """
    data = json.loads(path.read_text())
    temporal_path = INDUS_OUT / f"temporal_analysis_{DATASET}.json"
    temporal_type_data = json.loads(temporal_path.read_text()) if temporal_path.exists() else []
    # Index by (video_key, question_id) -> parsed temporal llm_output (normalized key for consistent lookup)
    temporal_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for t_item in temporal_type_data:
        key = _norm_key(t_item.get("video_key"), t_item.get("question_id"))
        temporal_by_key[key] = parse_llm_output(t_item.get("llm_output") or "{}")

    type_to_items: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    other_items: List[Tuple[str, int]] = []

    for item in data:
        video_key = item["video_key"]
        question_id = item["question_id"]
        key = _norm_key(video_key, question_id)
        parsed = parse_llm_output(item.get("llm_output") or "{}")
        temporal_parsed = temporal_by_key.get(key)
        types = assign_types(parsed, temporal_parsed)
        if "other" in types and types == ["other"]:
            other_items.append(key)
        else:
            for t in types:
                if t != "other":
                    type_to_items[t].append(key)

    return dict(type_to_items), other_items


def load_trace(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Load expansion trace JSON; return (video_key, question_id) -> {expansion_mode, expanded_event_count, total_event_count, ...}."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    key_to_stats = {}
    for entry in data:
        key = (entry["video_key"], entry["question_id"])
        key_to_stats[key] = {
            "expansion_mode": entry.get("expansion_mode"),
            "expanded_event_count": entry.get("expanded_event_count"),
            "total_event_count": entry.get("total_event_count"),
            "seed_event_count": entry.get("seed_event_count"),
        }
    return key_to_stats


def total_unique_event_duration_sec(expanded_path_or_paths: Path | List[Path]) -> float:
    """
    Sum of event durations with deduplication per question (video_key, question_id).
    Within each question, each event is counted once; the same event in different
    questions is counted once per question. Handles Path or List[Path]; when
    multiple paths are given, entries are merged by (video_key, question_id).
    """
    if isinstance(expanded_path_or_paths, Path):
        paths = [expanded_path_or_paths]
    else:
        paths = list(expanded_path_or_paths)
    if not paths:
        return 0.0

    # (video_key, question_id) -> list of seed_events (merged across paths)
    key_to_events: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
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
            key = _norm_key(vk, qid)
            key_to_events[key].extend(entry.get("seed_events") or [])

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
    return total_sec / n_keys if n_keys else 0.0


def load_expanded_per_query_stats(
    expanded_path_or_paths: Path | List[Path],
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Load expanded seed_events and return per-query stats: (video_key, question_id) ->
    {total_unique_duration_sec, num_events}. num_events = count of unique events (by id).
    Denominator for later means: all queries in the returned dict (same as expanded file).
    """
    if isinstance(expanded_path_or_paths, Path):
        paths = [expanded_path_or_paths]
    else:
        paths = list(expanded_path_or_paths)
    key_to_events: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
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
            key = _norm_key(vk, qid)
            key_to_events[key].extend(entry.get("seed_events") or [])

    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for key, events in key_to_events.items():
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
        total_sec = sum(unique_duration.values())
        num_events = len(unique_duration)
        out[key] = {"total_unique_duration_sec": total_sec, "num_events": num_events}
    return out


def _overlap_for_items(
    items: List[Tuple[str, int]],
    per_query: Dict[Tuple[str, int], Dict[str, float]],
) -> Optional[Dict[str, float]]:
    """Helper: compute binary/percentage overlap accuracy for a list of (video_key, question_id)."""
    if not per_query or not items:
        return None
    subset = [per_query[k] for k in items if k in per_query]
    if not subset:
        return None
    binary = sum(p.get("binary_overlap") or 0 for p in subset) / len(subset)
    pct = sum(p.get("percentage_overlap") or 0 for p in subset) / len(subset)
    return {"binary_overlap_accuracy": binary, "percentage_overlap_accuracy": pct, "n_queries": len(subset)}


def recalculate_retrieval_per_query(expanded_seed_events_path: Path | List[Path]) -> Dict[Tuple[str, int], Dict[str, float]]:
    """
    Recalculate overlap accuracy from expanded seed_events JSON; return
    (video_key, question_id) -> {binary_overlap, percentage_overlap}.
    Uses calc_retrieval_accuracy.run_accuracy (no output file written).
    """
    # if not expanded_seed_events_path.exists():
    #     return {}
    result = calc_run_accuracy(
        seed_events_path=expanded_seed_events_path,
        dataset=DATASET,
        project_root=PROJECT_ROOT,
        output_path=None,
    )
    per_query = result.get("per_query") or []
    out = {}
    for p in per_query:
        key = _norm_key(p.get("video_key"), p.get("question_id"))
        out[key] = {
            "binary_overlap": p.get("binary_overlap"),
            "percentage_overlap": p.get("percentage_overlap"),
            "total_event_duration_sec": p.get("total_event_duration_sec"),
        }
    return out


def visualize_report(
    type_to_items: Dict[str, List[Tuple[str, int]]],
    retrieval_per_method: Dict[str, Dict[Tuple[str, int], Dict[str, float]]],
    type_names: Optional[List[str]] = None,
    methods_to_plot: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    total_time_per_method: Optional[Dict[str, float]] = None,
    dataset_label: Optional[str] = None,
    per_query_stats_per_method: Optional[Dict[str, Dict[Tuple[str, int], Dict[str, Any]]]] = None,
) -> None:
    """
    Create visualizations for the analyze_type report. Does not modify any inputs.
    Saves figures to output_dir (default: ECML-PKDD/indus_outputs).
    total_time_per_method: optional dict method -> total unique event duration (sec); if set, adds total-time bar chart.
    per_query_stats_per_method: optional; if set, adds efficiency scatter chart.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [visualize_report] matplotlib not installed; skipping visualizations.")
        return

    type_names = type_names or TYPE_NAMES
    methods_to_plot = methods_to_plot or ["base", "fb", "eoe"]
    out_dir = output_dir or INDUS_OUT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title_label = dataset_label or DATASET

    # Build matrix: type_name -> { method -> binary_overlap }; matrix_pct: type_name -> { method -> percentage_overlap }
    matrix: Dict[str, Dict[str, float]] = {}
    matrix_pct: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for t in type_names:
        items = type_to_items.get(t, [])
        counts[t] = len(items)
        matrix[t] = {}
        matrix_pct[t] = {}
        for method in methods_to_plot:
            per_query = retrieval_per_method.get(method, {})
            ov = _overlap_for_items(items, per_query)
            matrix[t][method] = (ov["binary_overlap_accuracy"] if ov else 0.0)
            matrix_pct[t][method] = (ov["percentage_overlap_accuracy"] if ov else 0.0)

    # Short labels for methods
    method_labels = {
        "base": "base",
        "fb": "fb",
        "eoe": "eoe",
    }
    labels = [method_labels.get(m, m) for m in methods_to_plot]

    # 1) Grouped bar chart: overlap accuracy by type and method (legend outside to avoid overlap)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(type_names))
    n_methods = len(methods_to_plot)
    width = 0.8 / n_methods
    for i, method in enumerate(methods_to_plot):
        vals = [matrix[t].get(method, 0.0) for t in type_names]
        ax.bar(x + i * width - (n_methods - 1) * width / 2, vals, width, label=labels[i])
    ax.set_ylabel("Binary overlap accuracy")
    ax.set_xlabel("Question type")
    ax.set_title(f"Overlap accuracy by question type and expansion method ({title_label})")
    ax.set_xticks(x)
    ax.set_xticklabels(type_names)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    ax.set_ylim(0, 1.05)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(out_dir / "analyze_type_overlap_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) Heatmap: type x method (binary) — x-axis = column index, legend lists method names
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    data = np.array([[matrix[t].get(m, 0.0) for m in methods_to_plot] for t in type_names])
    im = ax2.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_xticks(np.arange(len(methods_to_plot)))
    ax2.set_xticklabels([str(j + 1) for j in range(len(methods_to_plot))], fontsize=10)
    ax2.set_xlabel("Method (see legend)")
    ax2.set_yticks(np.arange(len(type_names)))
    ax2.set_yticklabels(type_names)
    for i in range(len(type_names)):
        for j in range(len(methods_to_plot)):
            ax2.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
    ax2.set_title("Binary overlap accuracy by question type and expansion method (heatmap)")
    plt.colorbar(im, ax=ax2, label="Binary overlap")
    # Method legend: column index -> method name (outside right)
    try:
        from matplotlib.lines import Line2D as L2D
        method_legend = [L2D([0], [0], marker="none", linestyle="none", label=f"{j+1}: {methods_to_plot[j]}") for j in range(len(methods_to_plot))]
        ax2.legend(handles=method_legend, title="Method", loc="center left", bbox_to_anchor=(1.28, 0.5), frameon=True, fontsize=8)
    except Exception:
        pass
    fig2.tight_layout(rect=[0, 0, 0.78, 1])
    fig2.savefig(out_dir / "analyze_type_overlap_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # 2b) Grouped bar chart: percentage overlap accuracy (legend outside to avoid overlap)
    fig2b, ax2b = plt.subplots(figsize=(10, 5))
    data_pct = np.array([[matrix_pct[t].get(m, 0.0) for m in methods_to_plot] for t in type_names])
    for i, method in enumerate(methods_to_plot):
        vals = [matrix_pct[t].get(method, 0.0) for t in type_names]
        ax2b.bar(x + i * width - (n_methods - 1) * width / 2, vals, width, label=labels[i])
    ax2b.set_ylabel("Percentage overlap accuracy")
    ax2b.set_xlabel("Question type")
    ax2b.set_title(f"Percentage overlap accuracy by question type and expansion method ({title_label})")
    ax2b.set_xticks(x)
    ax2b.set_xticklabels(type_names)
    ax2b.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    ax2b.set_ylim(0, 1.05)
    fig2b.tight_layout(rect=[0, 0, 0.85, 1])
    fig2b.savefig(out_dir / "analyze_type_percentage_overlap_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig2b)

    # 2c) Heatmap: type x method (percentage) — x-axis = column index, legend lists method names
    fig2c, ax2c = plt.subplots(figsize=(8, 5))
    im2c = ax2c.imshow(data_pct, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax2c.set_xticks(np.arange(len(methods_to_plot)))
    ax2c.set_xticklabels([str(j + 1) for j in range(len(methods_to_plot))], fontsize=10)
    ax2c.set_xlabel("Method (see legend)")
    ax2c.set_yticks(np.arange(len(type_names)))
    ax2c.set_yticklabels(type_names)
    for i in range(len(type_names)):
        for j in range(len(methods_to_plot)):
            ax2c.text(j, i, f"{data_pct[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
    ax2c.set_title("Percentage overlap accuracy by question type and expansion method (heatmap)")
    plt.colorbar(im2c, ax=ax2c, label="Percentage overlap")
    try:
        from matplotlib.lines import Line2D as _L2D
        method_legend2 = [_L2D([0], [0], marker="none", linestyle="none", label=f"{j+1}: {methods_to_plot[j]}") for j in range(len(methods_to_plot))]
        ax2c.legend(handles=method_legend2, title="Method", loc="center left", bbox_to_anchor=(1.28, 0.5), frameon=True, fontsize=8)
    except Exception:
        pass
    fig2c.tight_layout(rect=[0, 0, 0.78, 1])
    fig2c.savefig(out_dir / "analyze_type_percentage_overlap_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig2c)

    # 3) Question type counts
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.bar(type_names, [counts.get(t, 0) for t in type_names], color="steelblue", edgecolor="navy")
    ax3.set_ylabel("Number of questions")
    ax3.set_xlabel("Question type")
    ax3.set_title(f"Question type distribution ({title_label})")
    fig3.tight_layout()
    fig3.savefig(out_dir / "analyze_type_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # 4) Total time (unique event duration) per method
    if total_time_per_method:
        method_labels_long = {
            "base": "Base (no expansion)",
            "fb": "Fwd/Bwd",
            "eoe": "Evt-Obj-Evt",
            "fb_eoe": "Fwd/Bwd+Evt-Obj",
        }
        methods_tt = [m for m in total_time_per_method if total_time_per_method.get(m, 0) > 0]
        if methods_tt:
            fig4, ax4 = plt.subplots(figsize=(max(8, len(methods_tt) * 1.2), 5))
            total_min = [total_time_per_method[m] / 60 for m in methods_tt]
            labels_tt = [method_labels_long.get(m, m) for m in methods_tt]
            bars = ax4.bar(range(len(methods_tt)), total_min, color="coral", edgecolor="darkred")
            ax4.set_ylabel("Total time (minutes)")
            ax4.set_xlabel("Expansion method")
            ax4.set_title(f"Total unique event duration per method ({title_label}, duplicates removed)")
            ax4.set_xticks(range(len(methods_tt)))
            ax4.set_xticklabels(labels_tt, rotation=25, ha="right")
            for bar, min in zip(bars, total_min):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(total_min) * 0.01, f"{min:.1f}m", ha="center", va="bottom", fontsize=9)
            fig4.tight_layout()
            fig4.savefig(out_dir / "analyze_type_total_time.png", dpi=150, bbox_inches="tight")
            plt.close(fig4)

    # 5) Efficiency scatter: Total Duration (sec) vs Hit Rate (%) — color = question type, shape = expansion method
    if per_query_stats_per_method is not None:
        try:
            from matplotlib.lines import Line2D
        except ImportError:
            Line2D = None
        eff_methods = [m for m in methods_to_plot]
        eff_types = [t for t in type_names if type_to_items.get(t)]
        type_colors = {"counting": "C0", "temporal": "C1", "spatial": "C2", "content_text": "C3", "other": "C4"}
        # Distinct markers for expansion methods (cycle if more than 8)
        markers = ["o", "s", "^", "v", "D", "P", "X", "*"]
        method_markers = {m: markers[i % len(markers)] for i, m in enumerate(eff_methods)}
        data = []  # list of (x, y, type, method)
        for t in eff_types:
            items = type_to_items.get(t, [])
            if not items:
                continue
            n_type = len(items)
            for method in eff_methods:
                stats = per_query_stats_per_method.get(method, {})
                total_sec_sum = sum((stats.get(k) or {}).get("total_unique_duration_sec") or 0 for k in items)
                avg_sec = total_sec_sum / n_type
                hit_rate_pct = (matrix.get(t, {}).get(method, 0.0) or 0.0) * 100.0
                data.append((avg_sec, hit_rate_pct, t, method))
        if data:
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            for (x, y, t, m) in data:
                ax5.scatter(
                    x, y,
                    c=type_colors.get(t, "gray"),
                    marker=method_markers.get(m, "o"),
                    s=80,
                    alpha=0.85,
                    edgecolors="black",
                    linewidths=0.5,
                )
            ax5.set_xlabel("Total duration (seconds)")
            ax5.set_ylabel("Hit rate (%)")
            ax5.set_title(f"Efficiency vs accuracy ({title_label}) — color = question type, shape = method")
            ax5.grid(True, alpha=0.3)
            # Legend 1: question type (color) — add with add_artist so it is not replaced by legend 2
            legend_type = [Line2D([0], [0], marker="o", color="w", markerfacecolor=type_colors.get(t, "gray"), markeredgecolor="black", markersize=9, label=t) for t in eff_types] if Line2D else []
            legend_method = [Line2D([0], [0], marker=method_markers.get(m, "o"), color="w", markerfacecolor="gray", markeredgecolor="black", markersize=9, label=m) for m in eff_methods] if Line2D else []
            if legend_type:
                leg1 = ax5.legend(handles=legend_type, title="Question type", loc="upper left", framealpha=0.9)
                ax5.add_artist(leg1)
            if legend_method:
                ax5.legend(handles=legend_method, title="Expansion method", loc="lower right", framealpha=0.9)
            fig5.tight_layout()
            fig5.savefig(out_dir / "analyze_type_efficiency_scatter.png", dpi=150, bbox_inches="tight")
            plt.close(fig5)

    extra_viz = []
    if total_time_per_method:
        extra_viz.append("analyze_type_total_time.png")
    if per_query_stats_per_method is not None:
        extra_viz.append("analyze_type_efficiency_scatter.png")
    print(f"Visualizations saved to: {out_dir} (analyze_type_overlap_bars.png, analyze_type_overlap_heatmap.png, analyze_type_percentage_overlap_bars.png, analyze_type_percentage_overlap_heatmap.png, analyze_type_counts.png" + ("".join(", " + v for v in extra_viz) if extra_viz else "") + ")")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze question types vs expansion method overlap accuracy (AVA100 or LVBench)."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="ava100",
        help="Dataset to use: ava100 or lvbench (default: ava100)",
    )
    args = parser.parse_args()
    set_dataset_paths(args.dataset)
    print(f"Using dataset: {args.dataset} ({DATASET})")
    print("Loading classification...")
    type_to_items, other_items = load_classification(CLASSIFICATION_PATH)
    # Ensure all 4 types exist (possibly empty); add "other" so visualizations include it
    for t in TYPE_NAMES:
        type_to_items.setdefault(t, [])
    type_to_items["other"] = other_items

    print("Loading expansion traces...")
    traces = {
        "fb": load_trace(TRACE_FORWARD_BACKWARD),
        "eoe": load_trace(TRACE_EVT_OBJ_EVT),
    }

    print("Recalculating retrieval accuracy (per method from expanded seed_events)...")
    retrieval_per_method: Dict[str, Dict[Tuple[str, int], Dict[str, float]]] = {}
    for method in EXPANSION_METHODS:
        retrieval_per_method[method] = recalculate_retrieval_per_query(EXPANDED_SEED_PATHS.get(method, EXPANDED_SEED_PATHS_LIST.get(method, [])))

    # Total time (sum of unique event durations, no duplicate events) per method
    total_time_per_method: Dict[str, float] = {}
    for method in EXPANSION_METHODS:
        path_or_paths = EXPANDED_SEED_PATHS.get(method) or EXPANDED_SEED_PATHS_LIST.get(method, [])
        total_time_per_method[method] = total_unique_event_duration_sec(path_or_paths)
    print("Total time (sum of unique event duration, duplicates removed) per method:")
    for method in EXPANSION_METHODS:
        total_sec = total_time_per_method.get(method, 0)
        print(f"  {method}: {total_sec:,.0f} sec  ({total_sec/60:,.1f} min, {total_sec/3600:.2f} hr)")

    # Per-query stats (total_unique_duration_sec, num_events) per method for overall and per-type stats
    per_query_stats_per_method: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    for method in EXPANSION_METHODS:
        path_or_paths = EXPANDED_SEED_PATHS.get(method) or EXPANDED_SEED_PATHS_LIST.get(method, [])
        per_query_stats_per_method[method] = load_expanded_per_query_stats(path_or_paths)

    # Average number of retrieved events (mean over all queries) per method
    avg_num_events_per_method: Dict[str, float] = {}
    for method in EXPANSION_METHODS:
        stats = per_query_stats_per_method.get(method, {})
        if not stats:
            avg_num_events_per_method[method] = 0.0
            continue
        n = len(stats)
        avg_num_events_per_method[method] = sum(s.get("num_events") or 0 for s in stats.values()) / n
    print("Average number of retrieved events (mean over all queries) per method:")
    for method in EXPANSION_METHODS:
        print(f"  {method}: {avg_num_events_per_method.get(method, 0):.2f}")

    # Overall accuracy (mean over all queries with valid time GT) per method
    overall_accuracy_per_method: Dict[str, Dict[str, Any]] = {}
    for method in EXPANSION_METHODS:
        per_query = retrieval_per_method.get(method, {})
        if not per_query:
            overall_accuracy_per_method[method] = {"binary_overlap_accuracy": None, "percentage_overlap_accuracy": None, "n_queries": 0}
            continue
        n = len(per_query)
        binary_avg = sum(p.get("binary_overlap") or 0 for p in per_query.values()) / n
        pct_avg = sum(p.get("percentage_overlap") or 0 for p in per_query.values()) / n
        overall_accuracy_per_method[method] = {
            "binary_overlap_accuracy": binary_avg,
            "percentage_overlap_accuracy": pct_avg,
            "n_queries": n,
        }
    print("Overall accuracy (mean over queries with valid time GT) per method:")
    for method in EXPANSION_METHODS:
        oa = overall_accuracy_per_method.get(method, {})
        n = oa.get("n_queries", 0)
        if n > 0:
            print(f"  {method}: binary = {oa['binary_overlap_accuracy']:.4f}, percentage = {oa['percentage_overlap_accuracy']:.4f}  (n={n})")
        else:
            print(f"  {method}: (no queries with valid time GT)")

    # Build per-type, per-method overlap stats (when retrieval available)
    def overlap_for_items(
        items: List[Tuple[str, int]],
        per_query: Dict[Tuple[str, int], Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        if not per_query or not items:
            return None
        subset = [per_query[k] for k in items if k in per_query]
        if not subset:
            return None
        binary = sum(p.get("binary_overlap") or 0 for p in subset) / len(subset)
        pct = sum(p.get("percentage_overlap") or 0 for p in subset) / len(subset)
        return {"binary_overlap_accuracy": binary, "percentage_overlap_accuracy": pct, "n_queries": len(subset)}

    def expansion_for_items(
        items: List[Tuple[str, int]],
        trace: Dict[Tuple[str, int], Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not trace or not items:
            return None
        subset = [trace[k] for k in items if k in trace]
        if not subset:
            return None
        total = sum(s.get("total_event_count") or 0 for s in subset)
        expanded = sum(s.get("expanded_event_count") or 0 for s in subset)
        return {"avg_total_events": total / len(subset), "avg_expanded_events": expanded / len(subset), "n_queries": len(subset)}

    # Report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"QUESTION TYPE vs EXPANSION METHOD: OVERLAP ACCURACY REPORT ({DATASET})")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Question types from llm_output: counting, temporal, spatial, content_text.")
    report_lines.append("Baseline: base (seed, no expansion). Expansion methods: forward_backward, evt_obj_evt.")
    report_lines.append("")
    report_lines.append("Total time (mean over all queries: sum of unique event duration per query, duplicates removed), per method:")
    report_lines.append("  (Per-type total time below uses the same definition but over queries of that type only. Types overlap, so per-type means can be above or below the overall mean.)")
    for method in EXPANSION_METHODS:
        total_sec = total_time_per_method.get(method, 0)
        total_min = total_sec / 60
        total_hr = total_sec / 3600
        report_lines.append(f"  {method}: {total_sec:,.0f} sec  ({total_min:,.1f} min, {total_hr:.2f} hr)")
    report_lines.append("")
    report_lines.append("Average number of retrieved events (mean over all queries), per method:")
    for method in EXPANSION_METHODS:
        report_lines.append(f"  {method}: {avg_num_events_per_method.get(method, 0):.2f}")
    report_lines.append("")
    report_lines.append("Overall accuracy (mean over queries with valid time GT), per method:")
    for method in EXPANSION_METHODS:
        oa = overall_accuracy_per_method.get(method, {})
        n = oa.get("n_queries", 0)
        if n > 0:
            report_lines.append(f"  {method}: binary_overlap_accuracy = {oa['binary_overlap_accuracy']:.4f}, percentage_overlap_accuracy = {oa['percentage_overlap_accuracy']:.4f}  (n={n})")
        else:
            report_lines.append(f"  {method}: (no queries with valid time GT)")
    report_lines.append("")

    for type_name in TYPE_NAMES:
        items = type_to_items.get(type_name, [])
        report_lines.append("-" * 80)
        report_lines.append(f"TYPE: {type_name.upper()}  (count = {len(items)})")
        report_lines.append("-" * 80)
        report_lines.append(f"  (video_key, question_id) for this type (for lookup in trace files):")
        def _item_sort_key(item):
            vk, qid = item
            try:
                q = int(qid) if qid is not None else -1
            except (TypeError, ValueError):
                q = -1
            return (str(vk), q)
        for (vk, qid) in sorted(items, key=_item_sort_key)[:30]:
            report_lines.append(f"    ({vk}, {qid})")
        if len(items) > 30:
            report_lines.append(f"    ... and {len(items) - 30} more.")
        report_lines.append("")

        # Total time and average num events per method for this type (mean over all queries of this type)
        report_lines.append("  Total time (mean over all queries of this type) and avg num retrieved events (mean over all queries of this type), per method:")
        for method in EXPANSION_METHODS:
            stats = per_query_stats_per_method.get(method, {})
            if not items:
                report_lines.append(f"    {method}: (no queries of this type)")
                continue
            total_sec_sum = sum((stats.get(k) or {}).get("total_unique_duration_sec") or 0 for k in items)
            num_events_sum = sum((stats.get(k) or {}).get("num_events") or 0 for k in items)
            n_type = len(items)
            avg_sec = total_sec_sum / n_type
            avg_ev = num_events_sum / n_type
            report_lines.append(f"    {method}: total_time = {avg_sec:,.0f} sec ({avg_sec/60:.1f} min), avg_num_events = {avg_ev:.2f}  (n={n_type})")
        report_lines.append("")

        # Overlap accuracy per method (recalculated from expanded seed_events)
        report_lines.append("  Overlap accuracy (recalculated per method):")
        best_method = None
        best_binary = -1.0
        best_pct = -1.0
        for method in EXPANSION_METHODS:
            ov = overlap_for_items(items, retrieval_per_method[method])
            if ov is not None:
                report_lines.append(f"    {method}: binary_overlap_accuracy = {ov['binary_overlap_accuracy']:.4f}, percentage_overlap_accuracy = {ov['percentage_overlap_accuracy']:.4f}  (n={ov['n_queries']})")
                if ov["binary_overlap_accuracy"] > best_binary:
                    best_binary = ov["binary_overlap_accuracy"]
                    best_pct = ov["percentage_overlap_accuracy"]
                    best_method = method
            else:
                report_lines.append(f"    {method}: (no expanded seed_events file or no matching queries)")
        if best_method is not None:
            report_lines.append(f"  --> Best method for this type (by binary overlap): {best_method} (binary={best_binary:.4f}, pct={best_pct:.4f})")
        report_lines.append("")

        # Expansion stats from traces (all 3 methods)
        # report_lines.append("  Expansion stats from trace files (avg total events / avg expanded events):")
        # for method in EXPANSION_METHODS:
        #     ex = expansion_for_items(items, traces[method])
        #     if ex is not None:
        #         report_lines.append(f"    {method}: avg_total_events = {ex['avg_total_events']:.1f}, avg_expanded_events = {ex['avg_expanded_events']:.1f}  (n={ex['n_queries']})")
        #     else:
        #         report_lines.append(f"    {method}: (no trace or no matching keys)")
        # report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("OTHER (no type flag)  (count = {})".format(len(other_items)))
    report_lines.append("-" * 80)
    for (vk, qid) in other_items[:20]:
        report_lines.append(f"  ({vk}, {qid})")
    if len(other_items) > 20:
        report_lines.append(f"  ... and {len(other_items) - 20} more.")
    report_lines.append("")

    # Total time and average num events per method for OTHER (mean over all queries of this type)
    report_lines.append("  Total time (mean over all queries of this type) and avg num retrieved events (mean over all queries of this type), per method:")
    for method in EXPANSION_METHODS:
        stats = per_query_stats_per_method.get(method, {})
        if not other_items:
            report_lines.append(f"    {method}: (no queries of this type)")
            continue
        n_type = len(other_items)
        total_sec_sum = sum((stats.get(k) or {}).get("total_unique_duration_sec") or 0 for k in other_items)
        num_events_sum = sum((stats.get(k) or {}).get("num_events") or 0 for k in other_items)
        avg_sec = total_sec_sum / n_type
        avg_ev = num_events_sum / n_type
        report_lines.append(f"    {method}: total_time = {avg_sec:,.0f} sec ({avg_sec/60:.1f} min), avg_num_events = {avg_ev:.2f}  (n={n_type})")
    report_lines.append("")

    # Overlap accuracy per method for OTHER
    report_lines.append("  Overlap accuracy (recalculated per method):")
    best_method_other = None
    best_binary_other = -1.0
    best_pct_other = -1.0
    for method in EXPANSION_METHODS:
        ov = overlap_for_items(other_items, retrieval_per_method[method])
        if ov is not None:
            report_lines.append(f"    {method}: binary_overlap_accuracy = {ov['binary_overlap_accuracy']:.4f}, percentage_overlap_accuracy = {ov['percentage_overlap_accuracy']:.4f}  (n={ov['n_queries']})")
            if ov["binary_overlap_accuracy"] > best_binary_other:
                best_binary_other = ov["binary_overlap_accuracy"]
                best_pct_other = ov["percentage_overlap_accuracy"]
                best_method_other = method
        else:
            report_lines.append(f"    {method}: (no expanded seed_events file or no matching queries)")
    if best_method_other is not None:
        report_lines.append(f"  --> Best method for this type (by binary overlap): {best_method_other} (binary={best_binary_other:.4f}, pct={best_pct_other:.4f})")
    report_lines.append("")

    # Summary table: best method per type (when we have retrieval for at least one method)
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY: Best expansion method per question type (by binary overlap accuracy)")
    report_lines.append("=" * 80)
    for type_name in TYPE_NAMES:
        items = type_to_items.get(type_name, [])
        best_method = None
        best_binary = -1.0
        for method in EXPANSION_METHODS:
            ov = overlap_for_items(items, retrieval_per_method[method])
            if ov and ov["binary_overlap_accuracy"] > best_binary:
                best_binary = ov["binary_overlap_accuracy"]
                best_method = method
        if best_method is not None:
            report_lines.append(f"  {type_name}: {best_method} (binary_overlap = {best_binary:.4f})")
        else:
            report_lines.append(f"  {type_name}: (no retrieval data)")
    # OTHER
    best_method = None
    best_binary = -1.0
    for method in EXPANSION_METHODS:
        ov = overlap_for_items(other_items, retrieval_per_method[method])
        if ov and ov["binary_overlap_accuracy"] > best_binary:
            best_binary = ov["binary_overlap_accuracy"]
            best_method = method
    if best_method is not None:
        report_lines.append(f"  other: {best_method} (binary_overlap = {best_binary:.4f})")
    else:
        report_lines.append(f"  other: (no retrieval data)")
    report_lines.append("")
    report_lines.append("Note: Overlap accuracy is recalculated from expanded seed_events JSON for each method")
    report_lines.append("      (see EXPANDED_SEED_PATHS in analyze_type.py).")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    out_path = INDUS_OUT / "analyze_type_report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text)
    print(f"Report written to: {out_path}")

    # Visualizations (include base + all expansion methods; type_names includes "other")
    visualize_report(
        type_to_items=type_to_items,
        retrieval_per_method=retrieval_per_method,
        type_names=TYPE_NAMES + ["other"],
        methods_to_plot=["base", "fb", "eoe", "fb_eoe"],
        output_dir=INDUS_OUT,
        total_time_per_method=total_time_per_method,
        dataset_label=DATASET,
        per_query_stats_per_method=per_query_stats_per_method,
    )


if __name__ == "__main__":
    main()
