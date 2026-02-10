#!/usr/bin/env python3
"""
Phase 1: Run INDUS prompts (temporal_analysis, keyword_strategy, query_type_classification),
         save LLM outputs to 3 JSONs in ECML-PKDD/indus_outputs/.
Phase 2: Read those JSONs, run event/entity/visual extraction, tri-view retrieval (Borda),
         add temporal-overlap events, output seed_events_{dataset}.json.

Phase 2 LLM/VLM call sites:
  (1) Text-only LLM: run_phase2 — batch of 3 (event_view, entity_view, visual_view extraction).
      Input: {"text": prompt} x3. Use --port / --model (e.g. Qwen 14B AWQ).
  (2) VLM (vision): _ava100_build_ref — extract time from first frame (ref). Once per AVA100 video with temporal (cached).
  (3) VLM (vision): _ava100_vlm_time_at_video_sec — one call per time point/range in raw_localization_value or raw_content_values,
      to record what time the VLM sees at the converted frame (for log verification only).
  Use --vlm-port and --vlm-model to run (2) and (3) on a separate VLM server.
"""

import sys
import json
import re
import shutil
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms.init_model import init_model
import indus_prompts

OUTPUT_DIR = Path(__file__).resolve().parent / "indus_outputs"
FRAMES_CHECK_DIR = Path(__file__).resolve().parent / "temporal_verification_frames"
PHASE1_PROMPT_KEYS = ["temporal_analysis", "keyword_strategy", "query_type_classification"]
PHASE2_PROMPT_KEYS = ["event_view_extraction", "entity_view_extraction", "visual_view_extraction"]
TOP_K_PER_VIEW = 50
BORDA_TOP_K = 20
S = 1


def parse_lvbench_question_options(full_question: str):
    full_question = (full_question or "").strip()
    if not full_question:
        return "", ""
    lines = full_question.split("\n")
    question_parts = []
    options_parts = []
    in_options = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("(A)") or stripped.startswith("(a)"):
            in_options = True
        if in_options:
            options_parts.append(stripped)
        else:
            question_parts.append(stripped)
    return "\n".join(question_parts).strip(), "\n".join(options_parts).strip()


def load_ava100_queries(base_dir: Path):
    for name in ["citytour", "ego", "traffic", "wildlife"]:
        path = base_dir / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for video in data:
            video_key = video.get("video_key", "")
            for qa in video.get("qa", []):
                question = qa.get("query", "")
                opts = qa.get("options", [])
                options_str = "\n".join(opts) if opts else ""
                question_id = qa.get("question_id")
                yield video_key, question_id, question, options_str


def load_lvbench_queries(base_dir: Path):
    path = base_dir / "LVBench.json"
    data = json.loads(path.read_text())
    for video in data:
        video_key = video.get("key") or video.get("video_key", "")
        for qa in video.get("qa", []):
            full = qa.get("question") or qa.get("query", "")
            question, options_str = parse_lvbench_question_options(full)
            question_id = qa.get("uid") or qa.get("question_id", "")
            yield video_key, question_id, question, options_str


def _time_str_to_seconds(s: str) -> int:
    """Parse MM:SS or HH:MM:SS to seconds. E.g. '2:03' -> 123, '00:1:20' -> 80."""
    if not s or not isinstance(s, str):
        return 0
    s = s.strip()
    parts = re.split(r"[:.]", s)
    parts = [p for p in parts if p.isdigit()]
    if not parts:
        return 0
    if len(parts) == 2:  # MM:SS
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) >= 3:  # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(parts[0])


def parse_localization_time_seconds(llm_output: str):
    """Parse temporal_analysis LLM JSON. localization_time can be range (value={start,end}) or point (value=string).
    Returns [start_sec, end_sec] (range) or [sec, sec] (point), or None if no localization."""
    out = llm_output.strip()
    if not out:
        return None
    if out.startswith("```"):
        out = re.sub(r"^```\w*\n?", "", out).strip()
        out = re.sub(r"\n?```$", "", out).strip()
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    loc = data.get("localization_time") or {}
    if not loc.get("exists"):
        return None
    val = loc.get("value")
    if val is None:
        return None
    # Range: value = {"start": "HH:MM", "end": "HH:MM"} (or HH:MM:SS)
    if isinstance(val, dict) and "start" in val and "end" in val:
        return [_time_str_to_seconds(val["start"]), _time_str_to_seconds(val["end"])]
    # Point: value = "HH:MM" or "HH:MM:SS"
    if isinstance(val, str):
        sec = _time_str_to_seconds(val)
        return [sec, sec]
    return None


def parse_query_type_classification(llm_output: str) -> dict:
    """Parse query_type_classification LLM JSON. Returns dict with needs_counting, needs_temporal_direction, needs_spatial_positions, has_content_text (safe defaults)."""
    out = (llm_output or "").strip()
    if out.startswith("```"):
        out = re.sub(r"^```\w*\n?", "", out).strip()
        out = re.sub(r"\n?```$", "", out).strip()
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return {"needs_counting": False, "needs_temporal_direction": "none", "needs_spatial_positions": False, "has_content_text": False}
    return {
        "needs_counting": bool(data.get("needs_counting", False)),
        "needs_temporal_direction": (data.get("needs_temporal_direction") or "none").strip().lower(),
        "needs_spatial_positions": bool(data.get("needs_spatial_positions", False)),
        "has_content_text": bool(data.get("has_content_text", False)),
    }


def parse_keyword_strategy(llm_output: str) -> str:
    out = llm_output.strip()
    if not out:
        return "both"
    if out.startswith("```"):
        out = re.sub(r"^```\w*\n?", "", out).strip()
        out = re.sub(r"\n?```$", "", out).strip()
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return "both"
    s = (data.get("strategy") or "both").strip().lower()
    if s in ("question_only", "options_only", "both", "neither"):
        return s
    return "both"


def run_phase1(dataset: str, port: int, model: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_dir = project_root / "datas" / dataset
    rows = list(load_ava100_queries(base_dir) if dataset == "AVA100" else load_lvbench_queries(base_dir))
    total = len(rows)
    print(f"Phase 1 | Dataset: {dataset}, total queries: {total}")
    llm = init_model("qwenvl_vllm", num_gpus=1, model_type=model, port=port)
    results = {k: [] for k in PHASE1_PROMPT_KEYS}
    for idx, (video_key, question_id, question, options_str) in enumerate(rows):
        print(f"[{idx + 1}/{total}] {dataset} video={video_key} question_id={question_id}")
        entry_meta = {"dataset": dataset, "video_key": video_key, "question_id": question_id, "question": question, "options": options_str}
        for key in PHASE1_PROMPT_KEYS:
            prompt = indus_prompts.INDUS_PROMPT[key].format(question=question, options=options_str)
            out = llm.batch_generate_response([{"text": prompt}])[0]
            results[key].append({**entry_meta, "llm_output": out})
    for key in PHASE1_PROMPT_KEYS:
        out_path = OUTPUT_DIR / f"{key}_{dataset}.json"
        out_path.write_text(json.dumps(results[key], indent=2))
        print(f"Wrote {out_path} ({len(results[key])} entries)")
    print(f"Phase 1 done. Processed {total} queries for {dataset}.")


def _parse_temporal_analysis_raw(llm_output: str) -> dict:
    """Parse temporal_analysis LLM JSON for logging and content_time.
    Both localization_time and content_time can be range or point (in LVBench and AVA100).
    Returns dict with has_localization, has_content, raw values, parsed_*_seconds ([start,end] or [sec,sec])."""
    out = (llm_output or "").strip()
    if out.startswith("```"):
        out = re.sub(r"^```\w*\n?", "", out).strip()
        out = re.sub(r"\n?```$", "", out).strip()
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return {"has_localization": False, "has_content": False, "raw_localization_value": None, "raw_content_values": None, "parsed_localization_seconds": None, "parsed_content_seconds": None}
    loc = data.get("localization_time") or {}
    content = data.get("content_time") or {}
    has_loc = bool(loc.get("exists"))
    has_content = bool(content.get("exists"))
    raw_loc = loc.get("value")
    raw_content = content.get("values") or []
    parsed_loc = parse_localization_time_seconds(llm_output)
    # content_time: values is a list; one value = point [sec,sec], multiple = range [min,max]
    parsed_content = None
    if raw_content and isinstance(raw_content, list):
        secs = []
        for s in raw_content:
            if isinstance(s, str):
                t = re.sub(r"\s*(AM|PM|am|pm)\s*", "", s).strip()
                secs.append(_time_str_to_seconds(t))
        if secs:
            parsed_content = [min(secs), max(secs)]
    return {
        "has_localization": has_loc,
        "has_content": has_content,
        "raw_localization_value": raw_loc,
        "raw_content_values": raw_content if raw_content else None,
        "parsed_localization_seconds": parsed_loc,
        "parsed_content_seconds": parsed_content,
    }


def _build_phase1_lookup(dataset: str):
    """Load phase 1 JSONs and return (video_key, question_id) -> (strategy, localization_time_seconds, temporal_llm_output, query_type_dict)."""
    temporal_path = OUTPUT_DIR / f"temporal_analysis_{dataset}.json"
    keyword_path = OUTPUT_DIR / f"keyword_strategy_{dataset}.json"
    query_type_path = OUTPUT_DIR / f"query_type_classification_{dataset}.json"
    if not temporal_path.exists() or not keyword_path.exists():
        raise FileNotFoundError(f"Phase 1 files not found for {dataset}. Run phase 1 first.")
    if not query_type_path.exists():
        raise FileNotFoundError(f"Phase 1 query_type_classification not found for {dataset}. Run phase 1 first.")
    temporal_list = json.loads(temporal_path.read_text())
    keyword_list = json.loads(keyword_path.read_text())
    query_type_list = json.loads(query_type_path.read_text())
    lookup = {}
    for t, k, qt in zip(temporal_list, keyword_list, query_type_list):
        key = (t["video_key"], t["question_id"])
        temporal_out = t.get("llm_output", "")
        query_type = parse_query_type_classification(qt.get("llm_output", ""))
        lookup[key] = (
            parse_keyword_strategy(k.get("llm_output", "")),
            parse_localization_time_seconds(temporal_out),
            temporal_out,
            query_type,
        )
    return lookup


def _build_ava100_kg_mapping():
    base_db = project_root / "AVA_cache" / "AVA100"
    mapping = {}
    for video_index in range(1, 9):
        config_path = base_db / str(video_index) / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
            source_path = config.get("source_path", "")
            if source_path:
                video_key = Path(source_path).stem
                kg_dir = base_db / str(video_index) / "kg"
                if (kg_dir / "vdb_events.json").exists() and (kg_dir / "vdb_entities.json").exists() and (kg_dir / "vdb_features.json").exists():
                    mapping[video_key] = str(kg_dir)
        except Exception:
            pass
    return mapping


def _build_lvbench_kg_mapping():
    base_db = project_root / "AVA_cache" / "LVBench"
    mapping = {}
    if not base_db.exists():
        return mapping
    for folder in base_db.iterdir():
        if not folder.is_dir():
            continue
        config_path = folder / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
            source_path = config.get("source_path", "")
            if source_path:
                video_key = Path(source_path).stem
                kg_dir = folder / "kg"
                if (kg_dir / "vdb_events.json").exists() and (kg_dir / "vdb_entities.json").exists() and (kg_dir / "vdb_features.json").exists():
                    mapping[video_key] = str(kg_dir)
        except Exception:
            pass
    return mapping


def _load_vdbs(kg_dir: str, embedding_model, embedding_dim: int = 768):
    from AVA.storage import TextNanoVectorDBStorage, ImageNanoVectorDBStorage
    global_config = {
        "working_dir": kg_dir,
        "embedding_batch_num": 64,
        "cosine_better_than_threshold": 0.1,
    }
    events_vdb = TextNanoVectorDBStorage(
        namespace="events",
        global_config=global_config,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        meta_fields={"id", "name", "description", "duration"},
    )
    entities_vdb = TextNanoVectorDBStorage(
        namespace="entities",
        global_config=global_config,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        meta_fields={"id", "descriptions", "timestamps", "frame_indices", "durations", "events"},
    )
    features_vdb = ImageNanoVectorDBStorage(
        namespace="features",
        global_config=global_config,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        meta_fields={"id", "frame_dir", "event"},
    )
    return events_vdb, entities_vdb, features_vdb


def _tri_view_borda_top_k(events_vdb, entities_vdb, features_vdb, event_text: str, entity_text: str, visual_text: str, top_k_per_view: int, top_k_final: int):
    """Run tri-view retrieval, Borda merge, return list of (event_id, score) for top_k_final events."""
    er = events_vdb.query(event_text, top_k=top_k_per_view)
    entr = entities_vdb.query(entity_text, top_k=top_k_per_view)
    fr = features_vdb.query(visual_text, top_k=top_k_per_view)
    events_from_events = {e["id"]: e["__metrics__"] for e in er}
    events_from_entities = {}
    for ent in entr:
        for eid in ent.get("events", []):
            events_from_entities[eid] = events_from_entities.get(eid, 0) + ent["__metrics__"]
    events_from_features = {}
    for f in fr:
        eid = f.get("event")
        if eid:
            events_from_features[eid] = events_from_features.get(eid, 0) + f["__metrics__"]
    e_sorted = sorted(events_from_events.items(), key=lambda x: x[1], reverse=True)
    ent_sorted = sorted(events_from_entities.items(), key=lambda x: x[1], reverse=True)
    f_sorted = sorted(events_from_features.items(), key=lambda x: x[1], reverse=True)
    sum_e = sum(s for _, s in e_sorted) or 1
    sum_ent = sum(s for _, s in ent_sorted) or 1
    sum_f = sum(s for _, s in f_sorted) or 1
    event_scores = {}
    for eid, s in e_sorted:
        event_scores[eid] = s / sum_e * S
    for eid, s in ent_sorted:
        event_scores[eid] = event_scores.get(eid, 0) + s / sum_ent * S
    for eid, s in f_sorted:
        event_scores[eid] = event_scores.get(eid, 0) + s / sum_f * S
    ordered = sorted(event_scores.items(), key=lambda x: x[1], reverse=True)
    return [(eid, event_scores[eid]) for eid, _ in ordered[:top_k_final]]


def _parse_time_from_vlm_output(text: str) -> Optional[int]:
    """
    Parse HH:MM:SS or HH:MM from VLM output or raw content (may have extra text).
    - Returns None if text is empty or VLM answered 'none' (no time visible).
    - Strips leading prefixes (Around, At, About, ~) so time can be found in text.
    - Applies 12-to-24 hour: 1:00 PM -> 13:00, 12:00 AM -> 0:00. Leaves 12:00 PM as 12:00.
    - Returns seconds since midnight or None.
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if text.lower() == "none":
        return None
    # Strip only leading prefixes so we keep AM/PM for conversion
    working = re.sub(r"^\s*(around|at|about|approximately|~)\s*", "", text, flags=re.IGNORECASE).strip()
    m = re.search(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", working)
    if not m:
        return None
    h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else 0
    # 12-hour to 24-hour using original text (AM/PM may be after the time)
    s_lower = text.lower()
    if "pm" in s_lower:
        if 1 <= h <= 11:
            h += 12
        # 12:xx PM stays 12
    elif "am" in s_lower:
        if h == 12:
            h = 0
    return h * 3600 + mm * 60 + ss


# Per-video handling for AVA100: time jumps and day wrap (hardcoded from observed behavior).
# If query display time is after the threshold, use the post-jump ref instead of frame 0.
AVA100_REF_JUMP_OVERRIDES = {
    "traffic1": {
        "jump_after_display_sec": 14 * 3600 + 2 * 60 + 1,  # 14:02:01
        "ref_video_sec": 10 * 3600 + 53 * 60 + 29,  # 10:53:29
        "ref_display_sec": 14 * 3600 + 8 * 60 + 31,  # 14:08:31
        "ref_frame_llm_output": "14:08:31",
    },
    "traffic2": {
        "jump_after_display_sec": 14 * 3600 + 2 * 60 + 1,  # 14:02:01
        "ref_video_sec": 10 * 3600 + 53 * 60 + 29,  # 10:53:29
        "ref_display_sec": 14 * 3600 + 8 * 60 + 31,  # 14:08:31
        "ref_frame_llm_output": "14:08:31",
    },
    "wildlife1": {
        "jump_after_display_sec": 11 * 3600 + 3 * 60,  # 11:03
        "ref_video_sec": 6 * 3600 + 1,  # 6:00:01
        "ref_display_sec": 17 * 3600 + 49 * 60,  # 17:49
        "ref_frame_llm_output": "17:49",
    },
}
# wildlife2: no jump; displayed time resets to 00:00:00 (new day) after 20:10:29. Treat small display_sec as next day.
AVA100_DAY_WRAP_VIDEOS = {"wildlife2"}
SECONDS_HALF_DAY = 12 * 3600
SECONDS_PER_DAY = 24 * 3600


def _ava100_display_sec_to_video_sec(
    display_sec: float,
    ref_video_sec: float,
    ref_display_sec: float,
    video_key: Optional[str] = None,
) -> int:
    """Convert display time (seconds since midnight) to video second using ref. Clamp to >= 0.
    For wildlife2 (day wrap): if ref is late in day and display_sec is early, treat display as next day (+86400)."""
    d_sec = display_sec
    if video_key in AVA100_DAY_WRAP_VIDEOS and ref_display_sec > SECONDS_HALF_DAY and d_sec < SECONDS_HALF_DAY:
        d_sec = d_sec + SECONDS_PER_DAY
    return max(0, int(ref_video_sec + (d_sec - ref_display_sec)))


def _ava100_display_time_segments_from_raw(
    raw_localization_value, raw_content_values,
) -> List[Tuple[int, int, str, Optional[int]]]:
    """
    Parse raw localization/content into display-time segments with source labels.
    Times are HH:MM or HH:MM:SS (hours). Returns list of (display_start_sec, display_end_sec, source_label, target_display_sec).
    - For a range (dict start/end): target_display_sec = None. Segment is [start, end].
    - For a single time (string or raw_content item): segment has same start and end (time point), target_display_sec is that time.
    """
    segments: List[Tuple[int, int, str, Optional[int]]] = []

    if raw_localization_value is not None:
        if isinstance(raw_localization_value, dict) and "start" in raw_localization_value and "end" in raw_localization_value:
            start_str = str(raw_localization_value["start"]).strip()
            end_str = str(raw_localization_value["end"]).strip()
            s = _parse_time_from_vlm_output(start_str)
            e = _parse_time_from_vlm_output(end_str)
            if s is not None and e is not None:
                label = f"raw_localization_value (start: {start_str}, end: {end_str})"
                segments.append((min(s, e), max(s, e), label, None))
        elif isinstance(raw_localization_value, str):
            raw_str = raw_localization_value.strip()
            t = _parse_time_from_vlm_output(raw_str)
            if t is not None:
                label = f"raw_localization_value (single: {raw_str})"
                segments.append((t, t, label, t))

    if raw_content_values and isinstance(raw_content_values, list):
        for i, v in enumerate(raw_content_values):
            if not isinstance(v, str):
                continue
            raw_str = v.strip()
            t = _parse_time_from_vlm_output(raw_str)
            if t is not None:
                label = f"raw_content_values[{i}]: {raw_str}"
                segments.append((t, t, label, t))

    return segments


def _ava100_build_ref(video_key: str, kg_dir: str, vlm) -> tuple | None:
    """Smallest frame as reference. Returns (ref_video_sec, ref_display_sec, fps, frames_dir, ref_frame_llm_output, min_frame_index, max_frame_index) or None."""
    work_dir = Path(kg_dir).parent
    config_path = work_dir / "config.json"
    frames_dir = work_dir / "frames"
    if not config_path.exists() or not frames_dir.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
        fps = float(config.get("fps", 30))
    except Exception:
        return None
    jpgs = list(frames_dir.glob("*.jpg"))
    if not jpgs:
        return None
    frame_indices = []
    for f in jpgs:
        try:
            frame_indices.append(int(f.stem))
        except ValueError:
            continue
    if not frame_indices:
        return None
    min_idx = min(frame_indices)
    max_idx = max(frame_indices)
    ref_video_sec = min_idx / fps
    ref_frame_path = frames_dir / f"{min_idx}.jpg"
    if not ref_frame_path.exists():
        return None
    try:
        from PIL import Image
        img = Image.open(ref_frame_path).convert("RGB")
    except Exception:
        return None
    prompt = indus_prompts.INDUS_PROMPT["time_from_frame_extraction"].strip()
    out = vlm.batch_generate_response([{"video": [img], "text": prompt}])[0]
    ref_display_sec = _parse_time_from_vlm_output(out)
    if ref_display_sec is None:
        return None
    return (ref_video_sec, ref_display_sec, fps, str(frames_dir), out, min_idx, max_idx)


def _ava100_vlm_time_at_video_sec(
    frames_dir: str,
    video_sec: float,
    fps: float,
    min_frame_index: int,
    max_frame_index: int,
    vlm,
) -> Optional[str]:
    """Load frame at video_sec (frame_index = video_sec * fps), run time-from-frame VLM prompt, return raw VLM output string."""
    frame_index = int(round(video_sec * fps))
    frame_index = max(min_frame_index, min(max_frame_index, frame_index))
    frame_path = Path(frames_dir) / f"{frame_index}.jpg"
    if not frame_path.exists():
        return None
    try:
        from PIL import Image
        img = Image.open(frame_path).convert("RGB")
    except Exception:
        return None
    prompt = indus_prompts.INDUS_PROMPT["time_from_frame_extraction"].strip()
    out = vlm.batch_generate_response([{"video": [img], "text": prompt}])[0]
    return out.strip() if out else None


def _event_video_and_display_sec(
    events_vdb, event_id: str, ref_video_sec: float, ref_display_sec: float
) -> Optional[Tuple[List[float], List[float]]]:
    """Get event's video_sec [start, end] and computed display_sec [start, end]. Returns (video_sec, display_sec) or None."""
    try:
        raw = events_vdb.get_data(event_id)
    except (IndexError, KeyError):
        return None
    dur = raw.get("duration")
    if not isinstance(dur, (list, tuple)) or len(dur) < 2:
        return None
    e_start, e_end = float(dur[0]), float(dur[1])
    video_sec = [e_start, e_end]
    d_start = ref_display_sec + (e_start - ref_video_sec)
    d_end = ref_display_sec + (e_end - ref_video_sec)
    display_sec = [d_start, d_end]
    return (video_sec, display_sec)


def _events_overlapping_segment(events_vdb, q_start: int, q_end: int):
    """Return list of event ids whose duration overlaps [q_start, q_end]."""
    datas = events_vdb.get_datas()
    out = []
    for d in datas:
        dur = d.get("duration")
        if not isinstance(dur, (list, tuple)) or len(dur) < 2:
            continue
        e_start, e_end = int(dur[0]), int(dur[1])
        if e_start <= q_end and e_end >= q_start:
            out.append(d.get("id"))
    return out


def _event_payload_serializable(events_vdb, event_id: str) -> dict:
    try:
        raw = events_vdb.get_data(event_id)
    except (IndexError, KeyError):
        return {"id": event_id}
    payload = {}
    for k, v in raw.items():
        if k in ("__vector__", "__metrics__", "content"):
            continue
        if hasattr(v, "tolist"):
            payload[k] = v.tolist()
        else:
            payload[k] = v
    return payload


def run_phase2(dataset: str, port: int, model: str, vlm_port: Optional[int] = None, vlm_model: Optional[str] = None):
    assert dataset in ("AVA100", "LVBench")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_dir = project_root / "datas" / dataset
    rows = list(load_ava100_queries(base_dir) if dataset == "AVA100" else load_lvbench_queries(base_dir))
    phase1_lookup = _build_phase1_lookup(dataset)
    kg_mapping = _build_ava100_kg_mapping() if dataset == "AVA100" else _build_lvbench_kg_mapping()

    from embeddings.JinaCLIP import JinaCLIP
    embedding_model = JinaCLIP("jinaai/jina-clip-v1")
    # Text-only: extraction prompts (event/entity/visual view)
    llm = init_model("qwenvl_vllm", num_gpus=1, model_type=model, port=port)
    # Vision: AVA100 time-from-frame and verify (optional separate server)
    if vlm_port is not None and vlm_model is not None:
        vlm = init_model("qwenvl_vllm", num_gpus=1, model_type=vlm_model, port=vlm_port)
    else:
        vlm = llm

    results = []
    temporal_retrieval_log = []
    total = len(rows)
    print(f"Phase 2 | Dataset: {dataset}, total queries: {total}")

    current_kg_dir = None
    events_vdb = entities_vdb = features_vdb = None
    ava100_ref_cache = {}

    for idx, (video_key, question_id, question, options_str) in enumerate(rows):
        print(f"[{idx + 1}/{total}] {dataset} video={video_key} question_id={question_id}")
        key = (video_key, question_id)
        lookup_val = phase1_lookup.get(key, ("both", None, "", {}))
        strategy = lookup_val[0]
        loc_seconds = lookup_val[1]
        temporal_llm_output = lookup_val[2] if len(lookup_val) > 2 else ""
        query_type = lookup_val[3] if len(lookup_val) > 3 else {}
        temporal_parsed = _parse_temporal_analysis_raw(temporal_llm_output) if temporal_llm_output else {}
        has_time = temporal_parsed.get("has_localization") or temporal_parsed.get("has_content")
        has_counting = query_type.get("needs_counting", False)
        has_temporal_direction = (query_type.get("needs_temporal_direction") or "none") in ("after", "before")
        has_spatial_positions = query_type.get("needs_spatial_positions", False)
        has_content_text = query_type.get("has_content_text", False)

        kg_dir = kg_mapping.get(video_key)
        if not kg_dir:
            results.append({
                "dataset": dataset, "video_key": video_key, "question_id": question_id,
                "question": question, "options": options_str,
                "localization_time_seconds": None,
                "seed_event_ids": [], "seed_events": [],
                "skip_reason": "no_kg",
            })
            temporal_retrieval_log.append({
                "dataset": dataset, "video_key": video_key, "question_id": question_id,
                "question": question,
                "has_localization_time": temporal_parsed.get("has_localization", False),
                "has_content_time": temporal_parsed.get("has_content", False),
                "raw_localization_value": temporal_parsed.get("raw_localization_value"),
                "raw_content_values": temporal_parsed.get("raw_content_values"),
                "parsed_time_seconds": None,
                "reference_used": None,
                "temporal_segments": None,
                "skip_reason": "no_kg",
            })
            continue

        if kg_dir != current_kg_dir:
            current_kg_dir = kg_dir
            events_vdb, entities_vdb, features_vdb = _load_vdbs(kg_dir, embedding_model)

        # Use "Yes"/"No" to match the wording in the prompt examples
        yn = lambda b: "Yes" if b else "No"
        batch_inputs = []
        for pk in PHASE2_PROMPT_KEYS:
            prompt = indus_prompts.INDUS_PROMPT[pk].format(
                strategy=strategy,
                question=question,
                options=options_str,
                has_time=yn(has_time),
                has_counting=yn(has_counting),
                has_temporal_direction=yn(has_temporal_direction),
                has_spatial_positions=yn(has_spatial_positions),
                has_content_text=yn(has_content_text),
            )
            batch_inputs.append({"text": prompt})
        batch_outputs = llm.batch_generate_response(batch_inputs)
        event_text = batch_outputs[0].strip()
        entity_text = batch_outputs[1].strip()
        visual_text = batch_outputs[2].strip()

        borda_list = _tri_view_borda_top_k(events_vdb, entities_vdb, features_vdb, event_text, entity_text, visual_text, TOP_K_PER_VIEW, BORDA_TOP_K)
        seed_ids = []
        seed_scores = []
        for eid, score in borda_list:
            seed_ids.append(eid)
            seed_scores.append(score)
        seen = set(seed_ids)

        # LVBench: use parsed time directly as video timeline. AVA100: uses ref + raw (HH:MM/HH:MM:SS), not parsed_time_seconds.
        parsed_time_seconds = None
        if dataset == "LVBench" and temporal_parsed.get("has_localization"):
            parsed_time_seconds = temporal_parsed.get("parsed_localization_seconds")

        # AVA100: one VLM call on first frame gives ref; convert query times (HH:MM/HH:MM:SS) to video sec by offset.
        if dataset == "AVA100" and (temporal_parsed.get("has_localization") or temporal_parsed.get("has_content")) and video_key not in ava100_ref_cache:
            ava100_ref_cache[video_key] = _ava100_build_ref(video_key, kg_dir, vlm)

        q_start = q_end = None
        reference_used = None
        ref_frame_llm_output = None
        content_start = None
        content_end = None
        display_time_segments = []
        video_time_segments = []
        temporal_segments_log = []
        event_to_temporal_sources = {}

        if dataset == "AVA100" and (temporal_parsed.get("has_localization") or temporal_parsed.get("has_content")):
            ref = ava100_ref_cache.get(video_key)
            if ref is not None:
                ref_video_sec, ref_display_sec, fps, frames_dir, ref_frame_llm_output = ref[0], ref[1], ref[2], ref[3], ref[4]
                display_segments_with_source = _ava100_display_time_segments_from_raw(
                    temporal_parsed.get("raw_localization_value"),
                    temporal_parsed.get("raw_content_values"),
                )
                if display_segments_with_source:
                    # Apply per-video ref override when query time is after the visualized time jump
                    ref_override_applied = False
                    max_display_sec = max(
                        max(s, e) for s, e, _, _ in display_segments_with_source
                    )
                    override = AVA100_REF_JUMP_OVERRIDES.get(video_key)
                    if override and max_display_sec > override["jump_after_display_sec"]:
                        ref_video_sec = override["ref_video_sec"]
                        ref_display_sec = override["ref_display_sec"]
                        ref_frame_llm_output = override["ref_frame_llm_output"]
                        ref_override_applied = True
                    all_video_starts = []
                    all_video_ends = []
                    min_frame_index, max_frame_index = ref[5], ref[6]
                    FRAMES_CHECK_DIR.mkdir(parents=True, exist_ok=True)
                    for seg_i, (d_start, d_end, source_label, target_display_sec) in enumerate(display_segments_with_source):
                        video_start = _ava100_display_sec_to_video_sec(d_start, ref_video_sec, ref_display_sec, video_key)
                        video_end = _ava100_display_sec_to_video_sec(d_end, ref_video_sec, ref_display_sec, video_key)
                        display_time_segments.append({"display_sec": [d_start, d_end], "source": source_label})
                        video_time_segments.append({"video_sec": [video_start, video_end], "source": source_label})
                        overlap_ids_seg = _events_overlapping_segment(events_vdb, video_start, video_end)
                        frame_index_used = int(round(video_start * fps))
                        frame_index_used = max(min_frame_index, min(max_frame_index, frame_index_used))
                        src_frame = Path(frames_dir) / f"{frame_index_used}.jpg"
                        if src_frame.exists():
                            safe_key = str(video_key).replace("/", "_")
                            safe_qid = str(question_id).replace("/", "_")
                            dst_name = f"{safe_key}_q{safe_qid}_seg{seg_i}_idx{frame_index_used}.jpg"
                            shutil.copy2(src_frame, FRAMES_CHECK_DIR / dst_name)
                        vlm_time_at_frame = _ava100_vlm_time_at_video_sec(
                            frames_dir, video_start, fps, min_frame_index, max_frame_index, vlm
                        )
                        events_retrieved = []
                        for eid in overlap_ids_seg:
                            ev_secs = _event_video_and_display_sec(events_vdb, eid, ref_video_sec, ref_display_sec)
                            if ev_secs:
                                events_retrieved.append({
                                    "event_id": eid,
                                    "video_sec": ev_secs[0],
                                    "display_sec": ev_secs[1],
                                })
                            else:
                                events_retrieved.append({"event_id": eid, "video_sec": None, "display_sec": None})
                        seg_log = {
                            "source": source_label,
                            "converted_display_sec": [d_start, d_end],
                            "converted_video_sec": [video_start, video_end],
                            "frame_index_used": frame_index_used,
                            "vlm_time_at_frame": vlm_time_at_frame,
                            "events_retrieved": events_retrieved,
                        }
                        if target_display_sec is not None:
                            seg_log["target_display_sec"] = target_display_sec
                            seg_log["target_video_sec"] = _ava100_display_sec_to_video_sec(
                                target_display_sec, ref_video_sec, ref_display_sec, video_key
                            )
                            seg_log["note"] = "Single time point; segment has same start and end. Frame used for VLM = target_video_sec."
                        temporal_segments_log.append(seg_log)
                        for eid in overlap_ids_seg:
                            event_to_temporal_sources.setdefault(eid, []).append(source_label)
                        all_video_starts.append(video_start)
                        all_video_ends.append(video_end)
                    if all_video_starts and all_video_ends:
                        q_start = min(all_video_starts)
                        q_end = max(all_video_ends)
                        content_start = min(s for s, e, _, _ in display_segments_with_source)
                        content_end = max(e for s, e, _, _ in display_segments_with_source)
                        reference_used = {
                            "ref_video_sec": ref_video_sec,
                            "ref_display_sec": ref_display_sec,
                            "fps": fps,
                            "ref_frame_llm_output": ref_frame_llm_output,
                        }
                        if ref_override_applied:
                            reference_used["ref_override_applied"] = True
                            reference_used["ref_override_video"] = video_key
        elif dataset == "LVBench" and parsed_time_seconds is not None and len(parsed_time_seconds) >= 2:
            q_start, q_end = parsed_time_seconds[0], parsed_time_seconds[1]

        if event_to_temporal_sources:
            overlap_ids = list(dict.fromkeys(event_to_temporal_sources.keys()))
        elif q_start is not None and q_end is not None:
            overlap_ids = _events_overlapping_segment(events_vdb, q_start, q_end)
        else:
            overlap_ids = []

        if q_start is not None and q_end is not None:
            for eid in overlap_ids:
                if eid in seen:
                    continue
                seen.add(eid)
                seed_ids.append(eid)
                seed_scores.append(None)

        seed_events = []
        for i, eid in enumerate(seed_ids):
            payload = _event_payload_serializable(events_vdb, eid)
            payload["borda_score"] = seed_scores[i]
            seed_events.append(payload)

        results.append({
            "dataset": dataset,
            "video_key": video_key,
            "question_id": question_id,
            "question": question,
            "options": options_str,
            "localization_time_seconds": loc_seconds,
            "llm_outputs": {
                "event_view_extraction": batch_outputs[0],
                "entity_view_extraction": batch_outputs[1],
                "visual_view_extraction": batch_outputs[2],
            },
            "seed_event_ids": seed_ids,
            "seed_events": seed_events,
        })

        log_entry = {
            "dataset": dataset,
            "video_key": video_key,
            "question_id": question_id,
            "question": question,
            "has_localization_time": temporal_parsed.get("has_localization", False),
            "has_content_time": temporal_parsed.get("has_content", False),
            "raw_localization_value": temporal_parsed.get("raw_localization_value"),
            "raw_content_values": temporal_parsed.get("raw_content_values"),
            "parsed_time_seconds": parsed_time_seconds if dataset == "LVBench" else None,
            "reference_used": reference_used,
            "temporal_segments": temporal_segments_log if temporal_segments_log else None,
        }
        temporal_retrieval_log.append(log_entry)

    out_path = OUTPUT_DIR / f"seed_events_{dataset}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path} ({len(results)} entries)")
    log_path = OUTPUT_DIR / f"temporal_retrieval_log_{dataset}.json"
    log_path.write_text(json.dumps(temporal_retrieval_log, indent=2))
    print(f"Wrote {log_path} ({len(temporal_retrieval_log)} entries)")
    if dataset == "AVA100" and FRAMES_CHECK_DIR.exists():
        n_frames = len(list(FRAMES_CHECK_DIR.glob("*.jpg")))
        if n_frames:
            print(f"Verification frames for manual check: {FRAMES_CHECK_DIR} ({n_frames} frames)")
    print(f"Phase 2 done for {dataset}.")


def main():
    parser = argparse.ArgumentParser(description="INDUS prompts: phase 1 (3 JSONs) or phase 2 (seed events).")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True, help="1: generate 3 prompt JSONs; 2: use them and output seed_events")
    parser.add_argument("--dataset", type=str, choices=["AVA100", "LVBench"], required=True)
    parser.add_argument("--port", type=int, default=8000, help="LLM server port (phase 1 and phase 2 text extraction)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct-AWQ", help="LLM model (text-only)")
    parser.add_argument("--vlm-port", type=int, default=8001, help="VLM server port for AVA100 time-from-frame and verify (phase 2 only)")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct-AWQ", help="VLM model for vision (e.g. Qwen2.5-VL); if set, --vlm-port required")
    args = parser.parse_args()
    if args.phase == 1:
        run_phase1(args.dataset, args.port, args.model)
    else:
        if (args.vlm_port is None) != (args.vlm_model is None):
            parser.error("--vlm-port and --vlm-model must be set together or both omitted")
        run_phase2(args.dataset, args.port, args.model, vlm_port=args.vlm_port, vlm_model=args.vlm_model)


if __name__ == "__main__":
    main()
