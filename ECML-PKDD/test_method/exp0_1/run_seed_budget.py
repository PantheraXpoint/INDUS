#!/usr/bin/env python3
"""
Exp0_1: Ten modes — 5 seed-only (emb_only, temp_only, emb, temp, all) + 5 FB-expanded (*_fb).
Seed-only = truncated seed set per mode. Expanded = FB expansion from that seed set.
- all = top-K Borda + all temporal (varies with top-K). Other modes as before.
Reads seed file + temporal_retrieval_log and performs live forward-backward expansion (same VDB
and functions as indus_explore_clean), so *_fb modes are directly comparable to indus.

Usage:
  python run_seed_budget.py --dataset ava100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple

# Root of this script -> ECML-PKDD/test_method/exp0_1
EXP0_1_ROOT = Path(__file__).resolve().parent
ECML_ROOT = EXP0_1_ROOT.parent.parent  # ECML-PKDD
PROJECT_ROOT = ECML_ROOT.parent  # Project-Ava
sys.path.insert(0, str(ECML_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

TOP_K_DEFAULTS = [10, 20, 30, 40, 50, 60, 70, 80]
HOPS = 1

DATASET_CONFIG = {
    "ava100": {"name": "AVA100", "retrieval_dir": "ava100_retrieval", "seed_file": "seed_events_AVA100.json", "analysis_dir": "ava100_analysis"},
    "lvbench": {"name": "LVBench", "retrieval_dir": "lvbench_retrieval", "seed_file": "seed_events_LVBench.json", "analysis_dir": "lvbench_analysis"},
}


def _sort_key_event(e):
    """Sort seed events: by borda_score descending; None last."""
    score = e.get("borda_score")
    if score is None:
        return (1, 0.0)  # None last
    return (0, -float(score))


def _norm_key(video_key: Any, question_id: Any) -> Tuple[str, int]:
    vk = str(video_key) if video_key is not None else ""
    try:
        qid = int(question_id) if question_id is not None else -1
    except (TypeError, ValueError):
        qid = -1
    return (vk, qid)


def get_base_seed_ids_per_query(entries: list, top_k: int) -> Dict[Tuple[str, int], Set[str]]:
    """Top-k Borda only (no temporal). Dedup by event id."""
    key_to_ids: Dict[Tuple[str, int], Set[str]] = {}
    for entry in entries:
        seed_events = entry.get("seed_events") or []
        key = _norm_key(entry.get("video_key"), entry.get("question_id"))
        if not seed_events:
            key_to_ids[key] = set()
            continue
        with_score = [e for e in seed_events if e.get("borda_score") is not None]
        sorted_scored = sorted(with_score, key=_sort_key_event)
        seen: Set[str] = set()
        ids: Set[str] = set()
        for ev in sorted_scored:
            eid = ev.get("id") or ev.get("__id__")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            ids.add(eid)
            if len(ids) >= top_k:
                break
        key_to_ids[key] = ids
    return key_to_ids


def get_temporal_only_seed_ids_per_query(entries: list) -> Dict[Tuple[str, int], Set[str]]:
    """Events with borda_score null only (temporal without embedding). No top-K."""
    key_to_ids: Dict[Tuple[str, int], Set[str]] = {}
    for entry in entries:
        seed_events = entry.get("seed_events") or []
        key = _norm_key(entry.get("video_key"), entry.get("question_id"))
        if not seed_events:
            key_to_ids[key] = set()
            continue
        no_score = [e for e in seed_events if e.get("borda_score") is None]
        ids = {ev.get("id") or ev.get("__id__") for ev in no_score if ev.get("id") or ev.get("__id__")}
        key_to_ids[key] = ids
    return key_to_ids


def load_temporal_log_lookup(dataset_key: str) -> Dict[Tuple[str, int], Set[str]]:
    """Load temporal_retrieval_log_{dataset}.json and return (video_key, question_id) -> set of event_ids."""
    cfg = DATASET_CONFIG.get(dataset_key)
    if not cfg:
        return {}
    analysis_dir = cfg.get("analysis_dir", f"{dataset_key}_analysis")
    dataset_name = cfg["name"]
    log_path = ECML_ROOT / analysis_dir / f"temporal_retrieval_log_{dataset_name}.json"
    if not log_path.exists():
        print(f"  [Warning] Temporal log not found: {log_path}")
        return {}
    log_data = json.loads(log_path.read_text())
    if not isinstance(log_data, list):
        return {}
    lookup: Dict[Tuple[str, int], Set[str]] = {}
    for entry in log_data:
        key = _norm_key(entry.get("video_key"), entry.get("question_id"))
        ids: Set[str] = set()
        for seg in entry.get("temporal_segments") or []:
            for ev in seg.get("events_retrieved") or []:
                eid = ev.get("event_id")
                if eid:
                    ids.add(eid)
        lookup[key] = ids
    return lookup


def get_emb_only_seed_ids_per_query(
    entries: list, top_k: int, temporal_log_lookup: Dict[Tuple[str, int], Set[str]]
) -> Dict[Tuple[str, int], Set[str]]:
    """Embedding without temporal: borda not null AND id not in temporal_retrieval_log; top-K by score, dedup by id."""
    key_to_ids: Dict[Tuple[str, int], Set[str]] = {}
    for entry in entries:
        seed_events = entry.get("seed_events") or []
        key = _norm_key(entry.get("video_key"), entry.get("question_id"))
        if not seed_events:
            key_to_ids[key] = set()
            continue
        in_log = temporal_log_lookup.get(key, set())
        with_score = [e for e in seed_events if e.get("borda_score") is not None]
        emb_only = [e for e in with_score if (e.get("id") or e.get("__id__")) not in in_log]
        sorted_scored = sorted(emb_only, key=_sort_key_event)
        seen: Set[str] = set()
        top_list: List[Any] = []
        for ev in sorted_scored:
            eid = ev.get("id") or ev.get("__id__")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            top_list.append(ev)
            if len(top_list) >= top_k:
                break
        key_to_ids[key] = {ev.get("id") or ev.get("__id__") for ev in top_list if ev.get("id") or ev.get("__id__")}
    return key_to_ids


def get_temp_ids_from_log(temporal_log_lookup: Dict[Tuple[str, int], Set[str]]) -> Dict[Tuple[str, int], Set[str]]:
    """Temporal from log: same as temporal_log_lookup (event_ids per query from temporal_retrieval_log)."""
    return dict(temporal_log_lookup)


def get_base_temporal_seed_ids_per_query(entries: list, top_k: int) -> Dict[Tuple[str, int], Set[str]]:
    """Top-k Borda + all temporal (borda null). For 'all' mode — varies with top-K."""
    key_to_ids: Dict[Tuple[str, int], Set[str]] = {}
    for entry in entries:
        seed_events = entry.get("seed_events") or []
        key = _norm_key(entry.get("video_key"), entry.get("question_id"))
        if not seed_events:
            key_to_ids[key] = set()
            continue
        no_score = [e for e in seed_events if e.get("borda_score") is None]
        with_score = [e for e in seed_events if e.get("borda_score") is not None]
        sorted_scored = sorted(with_score, key=_sort_key_event)
        seen: Set[str] = set()
        combined: List[Any] = []
        for ev in sorted_scored:
            eid = ev.get("id") or ev.get("__id__")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            combined.append(ev)
            if len(combined) >= top_k:
                break
        for ev in no_score:
            eid = ev.get("id") or ev.get("__id__")
            if eid and eid not in seen:
                seen.add(eid)
                combined.append(ev)
        key_to_ids[key] = {ev.get("id") or ev.get("__id__") for ev in combined if ev.get("id") or ev.get("__id__")}
    return key_to_ids


def build_seed_only_entries(
    entries: list, key_to_ids: Dict[Tuple[str, int], Set[str]]
) -> list:
    """Build one entry per query with seed_events filtered to ids in key_to_ids; dedup by id."""
    out = []
    for entry in entries:
        key = _norm_key(entry.get("video_key"), entry.get("question_id"))
        allow_ids = key_to_ids.get(key, set())
        seed_events = entry.get("seed_events") or []
        seen: Set[str] = set()
        filtered_events = []
        for ev in seed_events:
            eid = ev.get("id") or ev.get("__id__")
            if not eid or eid not in allow_ids or eid in seen:
                continue
            seen.add(eid)
            filtered_events.append(ev)
        filtered_ids = [ev.get("id") or ev.get("__id__") for ev in filtered_events]
        out.append({
            **entry,
            "seed_event_ids": filtered_ids,
            "seed_events": filtered_events,
        })
    return out


def _live_fb_expand_and_fetch(
    seed_ids: Set[str],
    events_vdb,
    valid_event_ids: Set[str],
):
    """Same as indus: expand_forward_backward(seeds) & valid, then seeds ∪ result, fetch. Returns (events, ids)."""
    if not seed_ids:
        return [], []
    fb_all = set()
    try:
        from expand_indus_seed_events import expand_forward_backward
        fb_all = expand_forward_backward(seed_ids, events_vdb) & valid_event_ids
    except Exception as e:
        print(f"    [Warning] expand_forward_backward failed: {e}")
    all_ids = seed_ids | fb_all
    try:
        from expand_indus_seed_events import fetch_event_data
        events = fetch_event_data(all_ids, events_vdb)
    except Exception as e:
        print(f"    [Warning] fetch_event_data failed: {e}")
        events = []
    ids_list = [ev.get("id") or ev.get("__id__") for ev in events if ev.get("id") or ev.get("__id__")]
    return events, ids_list


def process_dataset_live_fb(
    dataset_key: str,
    data: list,
    top_k_list: list,
    temporal_log_lookup: Dict[Tuple[str, int], Set[str]],
    dataset_name: str,
    out_base: Path,
    skip_existing: bool,
) -> None:
    """Produce *_fb outputs using live expand_forward_backward (same VDB as indus). Load VDB per video."""
    from expand_indus_seed_events import (
        initialize_vdbs,
        resolve_kg_dir,
        _event_ids_in_vdb,
    )
    from embeddings.JinaCLIP import JinaCLIP

    embedding_model = JinaCLIP("jinaai/jina-clip-v1")
    current_video_key = None
    events_vdb = None
    valid_event_ids = set()

    for top_k in top_k_list:
        top_dir = out_base / f"top{top_k}"
        top_dir.mkdir(parents=True, exist_ok=True)

        emb_only_ids = get_emb_only_seed_ids_per_query(data, top_k, temporal_log_lookup)
        temp_only_ids = get_temporal_only_seed_ids_per_query(data)
        emb_ids = get_base_seed_ids_per_query(data, top_k)
        temp_ids = get_temp_ids_from_log(temporal_log_lookup)
        all_ids_map = get_base_temporal_seed_ids_per_query(data, top_k)

        mode_seed_maps = [
            ("emb_only_fb", emb_only_ids),
            ("temp_only_fb", temp_only_ids),
            ("emb_fb", emb_ids),
            ("temp_fb", temp_ids),
            ("all_fb", all_ids_map),
        ]
        out_lists = {name: [] for name, _ in mode_seed_maps}

        for entry in data:
            video_key = entry.get("video_key")
            key = _norm_key(video_key, entry.get("question_id"))
            if video_key != current_video_key:
                current_video_key = video_key
                kg_dir = resolve_kg_dir(PROJECT_ROOT, str(video_key), dataset_name)
                if not kg_dir:
                    for name in out_lists:
                        out_lists[name].append({**entry, "seed_events": [], "seed_event_ids": []})
                    continue
                events_vdb, _, _ = initialize_vdbs(kg_dir, embedding_model)
                valid_event_ids = _event_ids_in_vdb(events_vdb) if events_vdb else set()

            if not events_vdb:
                for name in out_lists:
                    out_lists[name].append({**entry, "seed_events": [], "seed_event_ids": []})
                continue

            for mode_name, key_to_ids in mode_seed_maps:
                seed_ids = key_to_ids.get(key, set())
                events_list, ids_ordered = _live_fb_expand_and_fetch(seed_ids, events_vdb, valid_event_ids)
                out_lists[mode_name].append({
                    **entry,
                    "seed_event_ids": ids_ordered,
                    "seed_events": events_list,
                })

        fnames = {
            "emb_only_fb": f"seed_events_{dataset_name}_expanded_emb_only_{HOPS}.json",
            "temp_only_fb": f"seed_events_{dataset_name}_expanded_temp_only_{HOPS}.json",
            "emb_fb": f"seed_events_{dataset_name}_expanded_emb_{HOPS}.json",
            "temp_fb": f"seed_events_{dataset_name}_expanded_temp_{HOPS}.json",
            "all_fb": f"seed_events_{dataset_name}_expanded_all_{HOPS}.json",
        }
        for mode_name, _ in mode_seed_maps:
            dst = top_dir / fnames[mode_name]
            if not skip_existing or not dst.exists():
                dst.write_text(json.dumps(out_lists[mode_name], indent=2))
                print(f"  Wrote {dst} (top-{top_k} {mode_name}, live FB)")


def process_dataset(dataset_key: str, top_k_list: list, skip_existing: bool) -> None:
    assert dataset_key in DATASET_CONFIG
    cfg = DATASET_CONFIG[dataset_key]
    dataset_name = cfg["name"]
    retrieval_dir = ECML_ROOT / cfg["retrieval_dir"]
    seed_path = retrieval_dir / cfg["seed_file"]
    if not seed_path.exists():
        print(f"[Error] Seed file not found: {seed_path}")
        return

    data = json.loads(seed_path.read_text())
    if not isinstance(data, list):
        print(f"[Error] Expected list in {seed_path}")
        return

    temporal_log_lookup = load_temporal_log_lookup(dataset_key)
    out_base = EXP0_1_ROOT / dataset_key
    out_base.mkdir(parents=True, exist_ok=True)

    # Seed-only outputs
    for top_k in top_k_list:
        top_dir = out_base / f"top{top_k}"
        top_dir.mkdir(parents=True, exist_ok=True)

        emb_only_ids = get_emb_only_seed_ids_per_query(data, top_k, temporal_log_lookup)
        temp_only_ids = get_temporal_only_seed_ids_per_query(data)
        emb_ids = get_base_seed_ids_per_query(data, top_k)
        temp_ids = get_temp_ids_from_log(temporal_log_lookup)
        all_ids = get_base_temporal_seed_ids_per_query(data, top_k)  # top-K Borda + all temporal

        # --- Seed-only (5 modes) ---
        for mode, key_to_ids, fname in [
            ("emb_only", emb_only_ids, f"seed_events_{dataset_name}_emb_only.json"),
            ("temp_only", temp_only_ids, f"seed_events_{dataset_name}_temp_only.json"),
            ("emb", emb_ids, f"seed_events_{dataset_name}_emb.json"),
            ("temp", temp_ids, f"seed_events_{dataset_name}_temp.json"),
            ("all", all_ids, f"seed_events_{dataset_name}_all.json"),
        ]:
            dst = top_dir / fname
            if not skip_existing or not dst.exists():
                seed_only = build_seed_only_entries(data, key_to_ids)
                dst.write_text(json.dumps(seed_only, indent=2))
                print(f"  Wrote {dst} (top-{top_k} {mode}, seed-only)")

    # Live FB-expanded outputs (5 modes)
    process_dataset_live_fb(dataset_key, data, top_k_list, temporal_log_lookup, dataset_name, out_base, skip_existing)


def main():
    parser = argparse.ArgumentParser(
        description="Exp0_1: Truncate seeds to top-N and compute FB-expanded sets live (same as indus_explore_clean)."
    )
    parser.add_argument("--dataset", choices=["ava100", "lvbench", "both"], required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=TOP_K_DEFAULTS,
                        help=f"List of N (default: {TOP_K_DEFAULTS})")
    parser.add_argument("--skip-existing", action="store_true", help="Skip writing if file already exists")
    args = parser.parse_args()

    if args.dataset == "both":
        for d in ["ava100", "lvbench"]:
            print(f"\n=== Dataset: {d} ===")
            process_dataset(d, args.top_k, args.skip_existing)
    else:
        print(f"\n=== Dataset: {args.dataset} ===")
        process_dataset(args.dataset, args.top_k, args.skip_existing)

    print("\nDone. Run plot_seed_budget.py to compute metrics and plot.")


if __name__ == "__main__":
    main()
