"""Expand cluster boundary events via forward_backward and evt_obj_evt (uses expand_indus_seed_events)."""
import json
import sys
from pathlib import Path
from typing import Set, Tuple, List, Optional, Dict, Any

# Add ECML-PKDD so we can import expand_indus_seed_events
ECML = Path(__file__).resolve().parent.parent
if str(ECML) not in sys.path:
    sys.path.insert(0, str(ECML))
PROJECT_ROOT = ECML.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expand_indus_seed_events import (
    initialize_vdbs,
    expand_forward_backward,
    expand_evt_obj_evt,
    _build_event_to_entities,
    fetch_event_data,
    _event_ids_in_vdb,
)
from .clustering import merge_gap


def resolve_kg_dir(video_key: str, dataset: str = "AVA100") -> Optional[str]:
    """Resolve kg directory for video_key (same logic as expand_indus_seed_events)."""
    kg_dir = None
    if dataset == "AVA100":
        base_db = PROJECT_ROOT / "AVA_cache" / "AVA100"
        for video_index in range(1, 9):
            config_path = base_db / str(video_index) / "config.json"
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text())
                    source_path = config.get("source_path", "")
                    if source_path and Path(source_path).stem == video_key:
                        kg_dir = str(base_db / str(video_index) / "kg")
                        break
                except Exception:
                    pass
    elif dataset == "LVBench":
        base_db = PROJECT_ROOT / "AVA_cache" / "LVBench"
        if base_db.exists():
            for folder in base_db.iterdir():
                if not folder.is_dir():
                    continue
                config_path = folder / "config.json"
                if not config_path.exists():
                    continue
                try:
                    config = json.loads(config_path.read_text())
                    source_path = config.get("source_path") or config.get("video_path") or ""
                    stem = Path(source_path).stem if source_path else ""
                    if stem == video_key or folder.name == video_key:
                        kg_dir = str(folder / "kg")
                        break
                except Exception:
                    pass
    return kg_dir if (kg_dir and Path(kg_dir).exists()) else None


def get_video_path(video_key: str, dataset: str = "AVA100") -> Optional[str]:
    """Resolve video file path for video_key from AVA_cache config (source_path)."""
    if dataset == "AVA100":
        base_db = PROJECT_ROOT / "AVA_cache" / "AVA100"
        for video_index in range(1, 9):
            config_path = base_db / str(video_index) / "config.json"
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text())
                    source_path = config.get("source_path", "")
                    if source_path and Path(source_path).stem == video_key:
                        return source_path
                except Exception:
                    pass
    elif dataset == "LVBench":
        base_db = PROJECT_ROOT / "AVA_cache" / "LVBench"
        if base_db.exists():
            for folder in base_db.iterdir():
                if not folder.is_dir():
                    continue
                config_path = folder / "config.json"
                if not config_path.exists():
                    continue
                try:
                    config = json.loads(config_path.read_text())
                    source_path = config.get("source_path") or config.get("video_path") or ""
                    stem = Path(source_path).stem if source_path else ""
                    if stem == video_key or folder.name == video_key:
                        return source_path
                except Exception:
                    pass
    return None


def expand_forward_backward_selective(
    seed_event_ids: Set[str],
    events_vdb,
    valid_ids: Set[str],
    video_path: Optional[str],
    query_objects: List[str],
    grounding_detector: Any,
    threshold: float = 0.1,
) -> Set[str]:
    """
    Expand forward/backward then keep only events that see a query object (open-world detection).
    Returns seed_event_ids | filtered_expanded (only expanded events that pass grounding).
    """
    exp_fb = expand_forward_backward(seed_event_ids, events_vdb) & valid_ids
    if not exp_fb or not video_path or not query_objects or grounding_detector is None:
        return seed_event_ids | exp_fb
    from .grounding import filter_events_by_grounding

    event_list = fetch_event_data(exp_fb, events_vdb)
    if not event_list:
        return seed_event_ids | exp_fb
    # Normalize id field for filter_events_by_grounding
    for ev in event_list:
        if "__id__" in ev and "id" not in ev:
            ev["id"] = ev["__id__"]
    kept = filter_events_by_grounding(
        video_path, event_list, query_objects, grounding_detector, threshold=threshold
    )
    return seed_event_ids | kept


def _intervals_from_event_ids(eids: Set[str], vdb) -> List[Tuple[int, int]]:
    evs = fetch_event_data(eids, vdb)
    out = []
    for e in evs:
        d = e.get("duration")
        if isinstance(d, (list, tuple)) and len(d) >= 2:
            a, b = int(d[0]), int(d[1])
            if b > a:
                out.append((a, b))
    return out


def _hit(intervals: List[Tuple[int, int]], gt_segments: Optional[List[Tuple[int, int]]]) -> bool:
    if not gt_segments:
        return False
    for (a, b) in intervals:
        if b <= a:
            continue
        for (gs, ge) in gt_segments:
            if ge < gs:
                continue
            if gs == ge:
                if a <= gs < b:
                    return True
            elif max(a, gs) < min(b, ge):
                return True
    return False


def expand_and_metrics(
    seed_event_ids: Set[str],
    gt_segments: Optional[List[Tuple[int, int]]],
    events_vdb,
    entities_vdb,
    run_fb: bool = True,
    run_evo: bool = True,
    fb_selective: bool = False,
    video_path: Optional[str] = None,
    query_objects: Optional[List[str]] = None,
    grounding_detector: Any = None,
    grounding_threshold: float = 0.1,
    original_merged: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[bool, float, bool, float, Set[str]]:
    """
    Expand from seed_event_ids using forward_backward and/or evt_obj_evt (1 hop each).
    run_fb/run_evo: whether to run each method.
    fb_selective: if True and video_path/query_objects/grounding_detector set, only keep FB-expanded
      events that see a query object (open-world detection) to lower time while keeping hit rate.
    Returns (hit_fb, time_fb_sec, hit_evo, time_evo_sec).
    """
    valid_ids = _event_ids_in_vdb(events_vdb)
    event_to_entities = _build_event_to_entities(entities_vdb)
    seed_event_ids = seed_event_ids & valid_ids
    if not seed_event_ids:
        return False, 0.0, False, 0.0, set()

    hit_fb, time_fb = False, 0.0
    fb_events_kept = set()
    if run_fb:
        if fb_selective and video_path and query_objects and grounding_detector is not None:
            ids_fb = expand_forward_backward_selective(
                seed_event_ids, events_vdb, valid_ids,
                video_path, query_objects, grounding_detector, threshold=grounding_threshold,
            )
        else:
            exp_fb = expand_forward_backward(seed_event_ids, events_vdb) & valid_ids
            ids_fb = seed_event_ids | exp_fb
        intervals_fb = _intervals_from_event_ids(ids_fb, events_vdb)
        if original_merged:
            intervals_fb = merge_gap(intervals_fb + original_merged, T=1)
        hit_fb = _hit(intervals_fb, gt_segments)
        time_fb = sum(e - s for s, e in intervals_fb if e > s)
        fb_events_kept = ids_fb

    hit_evo, time_evo = False, 0.0
    if run_evo:
        exp_evo = expand_evt_obj_evt(seed_event_ids, entities_vdb, event_to_entities) & valid_ids
        ids_evo = seed_event_ids | exp_evo
        intervals_evo = _intervals_from_event_ids(ids_evo, events_vdb)
        if original_merged:
            intervals_evo = merge_gap(intervals_evo + original_merged, T=1)
        hit_evo = _hit(intervals_evo, gt_segments)
        time_evo = sum(e - s for s, e in intervals_evo if e > s)
        fb_events_kept |= ids_evo

    return hit_fb, time_fb, hit_evo, time_evo, fb_events_kept
