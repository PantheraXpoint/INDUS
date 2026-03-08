#!/usr/bin/env python3
"""
Temporal clustering -> boundary expansion (forward_backward only) -> SAVE ONLY final_items.

Changes vs your original:
- Removed --grid
- Removed evt_obj_evt / evo / eoe entirely (no mode switch)
- Removed all reports/tables/CSVs/seed-only evaluation output
- Only writes final_items JSON (kept events after FB expansion + always include top-k seeds)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from expansion_od.expansion import initialize_vdbs
from llms.QwenLM import QwenLM
from expand_indus_seed_events import fetch_event_data

from expansion_od.config import (
    SEED_PATHS,
    PROJECT_ROOT,
    EXPAND_K,
    EXPAND_T,
    DATASET,
    SCRIPT_DIR,
)

from expansion_od.gt import load_gt, load_questions
from expansion_od.clustering import (
    events_to_id_intervals,
    top_k_id_intervals,
    merge_gap,
    cluster_boundary_event_ids,
)


def _jsonify(x):
    """Recursively convert non-JSON types (set, tuple, Path, numpy types) into JSON-safe ones."""
    from pathlib import Path

    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonify(v) for v in x]
    if isinstance(x, set):
        return sorted(_jsonify(v) for v in x)  # deterministic
    if isinstance(x, tuple):
        return [_jsonify(v) for v in x]
    if isinstance(x, Path):
        return str(x)

    try:
        import numpy as np
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass

    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
    except Exception:
        pass

    return str(x)


def run_fb_final_items_only(
    data: List[Dict],
    gt: Dict,  # kept only because expand_and_metrics expects gt_seg
    embedding_model,
    fb_mode: str = "full",
    questions: Optional[Dict] = None,
    grounding_detector: Any = None,
    grounding_threshold: float = 0.1,
    llm: Optional[QwenLM] = None,
) -> List[Dict[str, Any]]:
    """
    For each query:
      - top-k seeds (EXPAND_K)
      - merge with EXPAND_T
      - boundary ids
      - expand with forward_backward (fb), optionally selective grounding filter
      - build final_items:
            {"dataset", "video_key", "question_id", "seed_events": <fetched kept events>}
        where kept events are:
            - fb_events_kept from expand_and_metrics
            - union with top-k seed event ids (always)
    """
    from expansion_od.expansion import resolve_kg_dir, get_video_path, expand_and_metrics
    from expansion_od.grounding import extract_query_objects

    final_items: List[Dict[str, Any]] = []

    prev_kg_dir = None
    events_vdb = None
    entities_vdb = None

    for idx, entry in enumerate(data):
        vk, qid = entry.get("video_key"), entry.get("question_id")
        if vk is None or qid is None:
            continue

        events = entry.get("seed_events") or []
        id_intervals = events_to_id_intervals(events)

        kept_ids: Set[Any] = set()

        if id_intervals:
            kg_dir = resolve_kg_dir(vk, DATASET)
            breakpoint()
            if kg_dir:
                if (prev_kg_dir is None) or (kg_dir != prev_kg_dir):
                    prev_kg_dir = kg_dir
                    events_vdb, entities_vdb, _ = initialize_vdbs(kg_dir, embedding_model)

                # cluster + boundary
                top = top_k_id_intervals(id_intervals, EXPAND_K, events)
                merged = merge_gap([x[1] for x in top], EXPAND_T)
                boundary_ids = cluster_boundary_event_ids(top, merged)

                # always keep top seeds
                kept_ids |= set([x[0] for x in top])

                if boundary_ids:
                    gt_seg = gt.get((vk, str(qid))) if gt else None

                    fb_selective = (fb_mode == "selective")
                    video_path = get_video_path(vk, DATASET) if fb_selective else None

                    qinfo = (questions or {}).get((vk, str(qid))) or {}
                    query_objects = (
                        extract_query_objects(
                            qinfo.get("question", ""),
                            qinfo.get("answer_statements", []),
                            llm,
                        )
                        if fb_selective
                        else []
                    )

                    try:
                        # We ignore metrics; only use fb_events_kept
                        _hit_fb, _time_fb, _hit_evo, _time_evo, fb_events_kept = expand_and_metrics(
                            boundary_ids,
                            gt_seg,
                            events_vdb,
                            entities_vdb,
                            run_fb=True,
                            run_evo=False,  # hard-disable evo/eoe
                            fb_selective=fb_selective,
                            video_path=video_path,
                            query_objects=query_objects or None,
                            grounding_detector=grounding_detector,
                            grounding_threshold=grounding_threshold,
                            # remove this line if your expand_and_metrics doesn't accept it:
                            original_merged=merged,
                        )
                        if fb_events_kept:
                            kept_ids |= set(fb_events_kept)
                    except Exception:
                        # if expansion fails, keep top seeds only
                        pass

        # fetch kept events from VDB (if available)
        seed_events_out = []
        if kept_ids and events_vdb is not None:
            seed_events_out = fetch_event_data(kept_ids, events_vdb)

        final_items.append(
            {
                "dataset": DATASET,
                "video_key": vk,
                "question_id": qid,
                "seed_events": seed_events_out,
            }
        )

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(data)}")

    return final_items


def main():
    parser = argparse.ArgumentParser(
        description="Temporal clustering + FB expansion (SAVE ONLY final_items)"
    )
    parser.add_argument(
        "--fb-mode",
        choices=["full", "selective"],
        default="full",
        help="FB expansion: full (all prev/next events) or selective (keep only events that see query object via grounding).",
    )
    args = parser.parse_args()
    fb_mode = args.fb_mode

    path = next((p for p in SEED_PATHS if p.exists()), None)
    if not path:
        raise FileNotFoundError(f"No seed file in {SEED_PATHS}")
    print(f"Loading seeds: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected list")

    # Keep gt load because expand_and_metrics expects gt_seg (even if you ignore metrics)
    gt = load_gt()
    print(f"GT loaded: {len(gt)} queries")

    print("Loading embedding model for expansion...")
    from embeddings.JinaCLIP import JinaCLIP
    embedding_model = JinaCLIP("jinaai/jina-clip-v1")

    questions = None
    grounding_detector = None
    llm = None
    if fb_mode == "selective":
        print("FB selective: loading questions + grounding detector + LLM...")
        questions = load_questions()
        from expansion_od.grounding import GroundingDetector
        grounding_detector = GroundingDetector()
        llm = QwenLM()

    print(f"Running FB expansion (fb_mode={fb_mode}) ...")
    final_items = run_fb_final_items_only(
        data=data,
        gt=gt,
        embedding_model=embedding_model,
        fb_mode=fb_mode,
        questions=questions,
        grounding_detector=grounding_detector,
        llm=llm,
    )

    out_path = SCRIPT_DIR / f"expansion_final_items_{DATASET}_{fb_mode}.json"
    out_path.write_text(json.dumps(_jsonify(final_items), indent=2, ensure_ascii=False))
    print(f"Saved final_items only to: {out_path}")


if __name__ == "__main__":
    # if str(PROJECT_ROOT) not in sys.path:
    #     sys.path.insert(0, str(PROJECT_ROOT))
    main()