"""Print tables and save JSON."""
import json
from pathlib import Path
from typing import Dict, Tuple, Any

from .config import SCRIPT_DIR, TOP_K, THRESHOLD_T


def _table_grid(title: str, results: Dict[Tuple[int, int], Dict[str, Any]], key: str, fmt: str) -> None:
    """Print one T \\ k table (hit rate, total time, or avg time)."""
    print(f"{title}:")
    print("T \\ k    " + "".join(f"  k={k:<6}" for k in TOP_K))
    for T in THRESHOLD_T:
        row = "  " + f"T={T:<4}"
        for k in TOP_K:
            v = results.get((k, T), {}).get(key)
            row += f"  {(fmt.format(v) if v is not None else 'N/A'):<8}"
        print(row)
    print()


def print_expansion_grid_report(
    results_fb: Dict[Tuple[int, int], Dict[str, Any]],
    results_evo: Dict[Tuple[int, int], Dict[str, Any]],
    n_queries: int,
    n_with_gt: int,
    run_fb: bool,
    run_evo: bool,
    gt_available: bool,
) -> Path:
    """Print hit rate / total time / avg time tables for fb and evo (same style as clustering). Save JSON."""
    for res in (results_fb, results_evo):
        for kt in res:
            r = res[kt]
            n_gt = r["total_queries_with_gt"]
            r["hit_rate"] = (r["hit_count"] / n_gt) if n_gt else None
            n_q = r["n_queries_processed"]
            r["avg_time_per_query_sec"] = (r["total_time_sec"] / n_q) if n_q else 0.0

    print("\n" + "=" * 80)
    print("Expansion grid: hit rate and total time (top_k, threshold T) — forward_backward")
    print("=" * 80)
    print(f"Queries: {n_queries}. With GT: {n_with_gt}" + ("" if gt_available else " (GT not found)"))
    print()
    if run_fb:
        _table_grid("Hit rate", results_fb, "hit_rate", "{:.3f}")
        _table_grid("Total time (sec)", results_fb, "total_time_sec", "{:.0f}")
        _table_grid("Avg time/query (sec)", results_fb, "avg_time_per_query_sec", "{:.1f}")

    print("=" * 80)
    print("Expansion grid: hit rate and total time (top_k, threshold T) — evt_obj_evt")
    print("=" * 80)
    print(f"Queries: {n_queries}. With GT: {n_with_gt}" + ("" if gt_available else " (GT not found)"))
    print()
    if run_evo:
        _table_grid("Hit rate", results_evo, "hit_rate", "{:.3f}")
        _table_grid("Total time (sec)", results_evo, "total_time_sec", "{:.0f}")
        _table_grid("Avg time/query (sec)", results_evo, "avg_time_per_query_sec", "{:.1f}")

    out = SCRIPT_DIR / "expansion_grid_results.json"
    out.write_text(json.dumps({
        "n_queries": n_queries,
        "n_queries_with_gt": n_with_gt,
        "top_k": TOP_K,
        "threshold_T": THRESHOLD_T,
        "run_fb": run_fb,
        "run_evo": run_evo,
        "results_fb": {f"{k}_{T}": results_fb.get((k, T), {}) for k in TOP_K for T in THRESHOLD_T},
        "results_evo": {f"{k}_{T}": results_evo.get((k, T), {}) for k in TOP_K for T in THRESHOLD_T},
    }, indent=2))
    print(f"Expansion grid results saved to: {out}")
    return out


def print_and_save(
    results: Dict[Tuple[int, int], Dict[str, Any]],
    n_queries: int,
    n_with_gt: int,
    gt_available: bool,
) -> Path:
    for kt in results:
        r = results[kt]
        n_gt = r["total_queries_with_gt"]
        r["hit_rate"] = (r["hit_count"] / n_gt) if n_gt else None
        n_q = r["n_queries_processed"]
        r["avg_time_per_query_sec"] = (r["total_time_sec"] / n_q) if n_q else 0.0

    print("\n" + "=" * 80)
    print("Temporal clustering: hit rate and total time (top_k, threshold T)")
    print("=" * 80)
    print(f"Queries: {n_queries}. With GT: {n_with_gt}" + ("" if gt_available else " (datas/AVA100 not found)"))
    print()

    def table(title: str, key: str, fmt: str):
        print(f"{title}:")
        print("T \\ k    " + "".join(f"  k={k:<6}" for k in TOP_K))
        for T in THRESHOLD_T:
            row = "  " + f"T={T:<4}"
            for k in TOP_K:
                v = results.get((k, T), {}).get(key)
                row += f"  {(fmt.format(v) if v is not None else 'N/A'):<8}"
            print(row)
        print()

    table("Hit rate", "hit_rate", "{:.3f}")
    table("Total time (sec)", "total_time_sec", "{:.0f}")
    table("Avg time/query (sec)", "avg_time_per_query_sec", "{:.1f}")

    def table_lines(title: str, key: str, fmt: str) -> list:
        buf = [f"{title}:", "T \\ k    " + "".join(f"  k={k:<6}" for k in TOP_K)]
        for T in THRESHOLD_T:
            row = "  " + f"T={T:<4}"
            for k in TOP_K:
                v = results.get((k, T), {}).get(key)
                row += f"  {(fmt.format(v) if v is not None else 'N/A'):<8}"
            buf.append(row)
        buf.append("")
        return buf

    seed_grid_lines = [
        "=" * 80,
        "Seed-only (before expansion): hit rate and total time (top_k, threshold T)",
        "=" * 80,
        f"Queries: {n_queries}. With GT: {n_with_gt}" + ("" if gt_available else " (datas/AVA100 not found)"),
        "",
    ]
    seed_grid_lines += table_lines("Hit rate", "hit_rate", "{:.3f}")
    seed_grid_lines += table_lines("Total time (sec)", "total_time_sec", "{:.0f}")
    seed_grid_lines += table_lines("Avg time/query (sec)", "avg_time_per_query_sec", "{:.1f}")
    seed_grid_path = SCRIPT_DIR / "seed_grid_tables.txt"
    seed_grid_path.write_text("\n".join(seed_grid_lines))
    print(f"Seed (before expansion) tables saved to: {seed_grid_path}")

    out = SCRIPT_DIR / "temporal_cluster_results.json"
    out.write_text(json.dumps({
        "n_queries": n_queries,
        "n_queries_with_gt": n_with_gt,
        "top_k": TOP_K,
        "threshold_T": THRESHOLD_T,
        "results": {f"{k}_{T}": results.get((k, T), {}) for k in TOP_K for T in THRESHOLD_T},
    }, indent=2))
    print(f"Results saved to: {out}")
    return out
