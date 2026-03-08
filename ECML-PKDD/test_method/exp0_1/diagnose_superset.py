#!/usr/bin/env python3
"""
Diagnose why indus has events not in exp0_1 all_fb.
For one failing query: check if missing events are (A) in the raw precomputed expanded file
or (B) not in it. A => filtering/reachable logic issue. B => precomputed expansion never had them (VDB/expansion difference).
"""
import json
import re
import sys
from pathlib import Path

EXP0_1_ROOT = Path(__file__).resolve().parent
ECML_ROOT = EXP0_1_ROOT.parent.parent


def _norm_key(video_key, question_id):
    vk = str(video_key) if video_key is not None else ""
    try:
        qid = int(question_id) if question_id is not None else -1
    except (TypeError, ValueError):
        qid = -1
    return (vk, qid)


def load_event_ids_by_key(path: Path):
    out = {}
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        return out
    for entry in data:
        key = _norm_key(entry.get("video_key"), entry.get("question_id"))
        ids = set()
        for eid in entry.get("seed_event_ids") or []:
            if eid:
                ids.add(str(eid))
        for ev in entry.get("seed_events") or []:
            eid = ev.get("id") or ev.get("__id__")
            if eid:
                ids.add(str(eid))
        out[key] = ids
    return out


def collect_event_ids_from_trace_node(d: dict, skip_hop_keys: bool = True) -> set:
    hop_key_re = re.compile(r"^hop_\d+$") if skip_hop_keys else None
    out = set()
    for k, v in d.items():
        if hop_key_re and hop_key_re.match(k):
            pass
        else:
            out.add(k)
        if isinstance(v, dict) and v:
            out |= collect_event_ids_from_trace_node(v, skip_hop_keys)
    return out


def reachable_from_seeds(trace: dict, seed_ids: set) -> set:
    reachable = set(seed_ids)
    for sid in seed_ids:
        if sid not in trace:
            continue
        reachable |= collect_event_ids_from_trace_node(trace[sid])
    return reachable


def main():
    dataset = "AVA100"
    top_k = 20
    # First failing query
    test_key = ("citytour1", 0)

    seed_path = ECML_ROOT / "ava100_retrieval" / "seed_events_AVA100.json"
    raw_expanded_path = ECML_ROOT / "ava100_retrieval" / f"seed_events_{dataset}_expanded_forward_backward_1.json"
    raw_trace_path = ECML_ROOT / "ava100_retrieval" / f"seed_events_{dataset}_expansion_trace_forward_backward_1.json"
    indus_path = ECML_ROOT / "ava100_retrieval" / "indus_explore" / f"seed_events_{dataset}_indus_explore_final.json"
    exp0_1_path = EXP0_1_ROOT / "ava100" / f"top{top_k}" / f"seed_events_{dataset}_expanded_all_1.json"

    for p, name in [
        (seed_path, "seed file"),
        (raw_expanded_path, "raw expanded"),
        (raw_trace_path, "trace"),
        (indus_path, "indus"),
        (exp0_1_path, "exp0_1 all_fb"),
    ]:
        if not p.exists():
            print(f"Missing {name}: {p}")
            sys.exit(1)

    seed_data = json.loads(seed_path.read_text())
    raw_expanded = json.loads(raw_expanded_path.read_text())
    raw_trace_list = json.loads(raw_trace_path.read_text())
    indus_by_key = load_event_ids_by_key(indus_path)
    exp0_1_by_key = load_event_ids_by_key(exp0_1_path)

    # Find entry index for test_key
    entry_idx = None
    for i, e in enumerate(seed_data):
        if _norm_key(e.get("video_key"), e.get("question_id")) == test_key:
            entry_idx = i
            break
    if entry_idx is None:
        print(f"Key {test_key} not in seed file")
        sys.exit(1)

    indus_ids = indus_by_key.get(test_key, set())
    exp0_1_ids = exp0_1_by_key.get(test_key, set())
    missing = indus_ids - exp0_1_ids
    if not missing:
        print(f"No missing events for {test_key}. Run verify_superset to find a failing query.")
        return

    # Raw expanded file events for this query
    raw_entry = raw_expanded[entry_idx]
    raw_file_ids = set()
    for ev in raw_entry.get("seed_events") or []:
        eid = ev.get("id") or ev.get("__id__")
        if eid:
            raw_file_ids.add(str(eid))
    for eid in raw_entry.get("seed_event_ids") or []:
        if eid:
            raw_file_ids.add(str(eid))

    # Trace for this query: reachable from indus seeds
    trace_entry = raw_trace_list[entry_idx]
    trace = trace_entry.get("trace") or {}
    # Indus seeds = top-20 null-first from seed file (same logic as indus_explore_clean)
    seed_events = seed_data[entry_idx].get("seed_events") or []

    def _borda_max_none(ev):
        s = ev.get("borda_score")
        if s is None:
            return float("inf")
        try:
            return float(s)
        except Exception:
            return float("-inf")

    def _sort_key(ev):
        s = ev.get("borda_score")
        if s is None:
            return (1, 0.0)
        try:
            return (0, -float(s))
        except Exception:
            return (1, 0.0)

    sorted_ev = sorted(seed_events, key=lambda ev: (-_borda_max_none(ev), _sort_key(ev)))
    indus_seeds = []
    seen = set()
    for ev in sorted_ev:
        eid = ev.get("id") or ev.get("__id__")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        indus_seeds.append(eid)
        if len(indus_seeds) >= top_k:
            break
    indus_seed_set = set(indus_seeds)

    reachable_ids = reachable_from_seeds(trace, indus_seed_set)

    # Classify missing events
    in_raw_not_reachable = set()   # in raw file but not in reachable(indus_seeds) -> trace/seed mismatch
    not_in_raw = set()             # not in raw file -> precomputed expansion never had them
    in_reachable_not_in_raw = set() # in reachable but not in raw (shouldn't happen if file is complete)

    for eid in missing:
        eid = str(eid)
        in_raw = eid in raw_file_ids
        in_reach = eid in reachable_ids
        if in_raw and not in_reach:
            in_raw_not_reachable.add(eid)
        elif not in_raw:
            not_in_raw.add(eid)
        elif in_reach and not in_raw:
            in_reachable_not_in_raw.add(eid)

    print(f"Query {test_key} (indus top-{top_k} null-first seeds, then FB)")
    print(f"  Indus events: {len(indus_ids)}, exp0_1 all_fb: {len(exp0_1_ids)}, missing: {len(missing)}")
    print(f"  Raw precomputed expanded file events for this query: {len(raw_file_ids)}")
    print(f"  Reachable from indus seeds (via trace): {len(reachable_ids)}")
    print()
    print("Missing events (in indus but not in exp0_1 all_fb):")
    print(f"  A) In raw file but NOT in reachable(indus_seeds): {len(in_raw_not_reachable)} -> trace does not link these to indus seeds (trace key / structure issue)")
    print(f"  B) NOT in raw precomputed file: {len(not_in_raw)} -> precomputed expansion never included them (VDB/expansion different from indus)")
    print(f"  C) In reachable but not in raw file: {len(in_reachable_not_in_raw)} -> file missing events that trace says are reachable")
    if in_raw_not_reachable:
        print(f"  Sample A: {list(in_raw_not_reachable)[:3]}")
    if not_in_raw:
        print(f"  Sample B: {list(not_in_raw)[:3]}")
    if in_reachable_not_in_raw:
        print(f"  Sample C: {list(in_reachable_not_in_raw)[:3]}")


if __name__ == "__main__":
    main()
