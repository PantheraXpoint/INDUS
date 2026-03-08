#!/usr/bin/env python3
"""
Verify that exp0_1 all_fb (per query) is a superset of indus_explore_clean output.
If not, reports which queries violate and which event IDs are in indus but not in exp0_1.
Run from repo root or ECML-PKDD.
"""
import json
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
    """Return dict (video_key, question_id) -> set of event ids."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        return {}
    out = {}
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


def main():
    dataset = "AVA100"
    top_k = 20

    # indus_explore_clean output
    indus_path = ECML_ROOT / f"{dataset.lower()}_retrieval" / "indus_explore" / f"seed_events_{dataset}_indus_explore_final.json"
    # exp0_1 all_fb output
    exp0_1_path = EXP0_1_ROOT / "ava100" / f"top{top_k}" / f"seed_events_{dataset}_expanded_all_1.json"

    if not indus_path.exists():
        print(f"Missing indus output: {indus_path}")
        sys.exit(1)
    if not exp0_1_path.exists():
        print(f"Missing exp0_1 all_fb output: {exp0_1_path}")
        sys.exit(1)

    indus_by_key = load_event_ids_by_key(indus_path)
    exp0_1_by_key = load_event_ids_by_key(exp0_1_path)

    all_keys = sorted(set(indus_by_key) | set(exp0_1_by_key))
    violations = []
    for key in all_keys:
        indus_ids = indus_by_key.get(key, set())
        exp_ids = exp0_1_by_key.get(key, set())
        missing = indus_ids - exp_ids
        if missing:
            violations.append((key, indus_ids, exp_ids, missing))

    if not violations:
        print("OK: For every query, indus event set ⊆ exp0_1 all_fb. (exp0_1 is superset.)")
        return

    print(f"Superset VIOLATED: {len(violations)} query/queries have events in indus not in exp0_1 all_fb.\n")
    for key, indus_ids, exp_ids, missing in violations[:10]:
        print(f"  key={key}: indus has {len(indus_ids)} events, exp0_1 has {len(exp_ids)}; {len(missing)} in indus but not exp0_1")
        print(f"    sample missing ids: {list(missing)[:5]}")
    if len(violations) > 10:
        print(f"  ... and {len(violations) - 10} more.")
    sys.exit(1)


if __name__ == "__main__":
    main()
