#!/usr/bin/env python3
"""
Build AVA-style question → entity/event mappings (ava.json)
and ICDCS-style mappings from tracking databases (icdcs.json).

This script:
- Reads QA files: datas/AVA100/{citytour,ego,traffic,wildlife}.json
- For each question with time_reference != "N/A":
  - For ava.json: finds the event containing the time point from vdb_events.json,
    then gets all entities for that event from entities.json
  - For icdcs.json: finds the event containing the time point from event_embeddings.db,
    then gets all objects for that event from tracked_objects.db
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sqlite3

ROOT = Path(__file__).resolve().parent


def parse_time_str_to_seconds(time_str: str) -> float:
    """
    Parse a time string like:
      - "HH:MM:SS"
      - "H:MM:SS"
      - "MM:SS"
      - "M:SS"
      - "SS"
    into seconds (float).
    """
    time_str = time_str.strip()
    if not time_str:
        return 0.0

    parts = time_str.split(":")
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return 0.0

    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return 0.0

    if len(nums) == 3:
        h, m, s = nums
    elif len(nums) == 2:
        h = 0.0
        m, s = nums
    elif len(nums) == 1:
        h = 0.0
        m = 0.0
        s = nums[0]
    else:
        h, m, s = nums[-3:]

    return h * 3600.0 + m * 60.0 + s


def parse_time_reference(ref: str) -> Optional[Any]:
    """
    Parse a time_reference string. Returns one of:
    - Single point: float (seconds)
    - Multiple points: List[float] (list of seconds)
    - Range: Tuple[float, float] (start, end in seconds)
    - None if invalid/N/A

    Examples:
      - "00:24:46"                    -> 24.0 (single point)
      - "00:24:46, 00:30:00"          -> [24.0, 1800.0] (multiple points)
      - "36:00-36:40"                 -> (2160.0, 2200.0) (range)
      - "1:03:00"                     -> 3780.0 (single point)
    """
    if not ref or ref.strip().upper() == "N/A":
        return None

    text = ref.strip()

    # Check if it's a range (contains "-")
    if "-" in text:
        # Range: "36:00-36:40"
        parts = text.split("-", 1)
        start_str = parts[0].strip()
        end_str = parts[1].strip()
        # Remove any trailing commas or extra text from end_str
        if "," in end_str:
            end_str = end_str.split(",", 1)[0].strip()
        start = parse_time_str_to_seconds(start_str)
        end = parse_time_str_to_seconds(end_str)
        if end < start:
            start, end = end, start
        return (start, end)

    # Check if it's multiple points (contains ",")
    if "," in text:
        # Multiple points: "00:24:46, 00:30:00, 00:45:20"
        points = []
        for part in text.split(","):
            part = part.strip()
            if part:
                point = parse_time_str_to_seconds(part)
                points.append(point)
        if len(points) == 1:
            return points[0]  # Single point after all
        return points if points else None

    # Single timestamp
    return parse_time_str_to_seconds(text)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_entities_for_event(
    entities: List[Dict[str, Any]],
    event_id: str,
    time_point: Optional[float],
    time_range: Optional[Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """
    Helper function to get entities for an event, clipping durations appropriately.
    
    Args:
        entities: List of all entities
        event_id: Event ID to match
        time_point: If provided, only keep durations containing this point
        time_range: If provided (and time_point is None), only keep durations overlapping this range
    
    Returns:
        List of entities with clipped durations
    """
    entities_for_event: List[Dict[str, Any]] = []
    for ent in entities:
        ent_event_ids = ent.get("events", [])
        if event_id not in ent_event_ids:
            continue
        
        durations = ent.get("durations", [])
        clipped_durs: List[List[float]] = []
        for dur_ent in durations:
            if not isinstance(dur_ent, (list, tuple)) or len(dur_ent) != 2:
                continue
            # Convert from milliseconds to seconds
            e_start_ms, e_end_ms = float(dur_ent[0]), float(dur_ent[1])
            e_start, e_end = e_start_ms / 1000.0, e_end_ms / 1000.0
            
            if time_point is not None:
                # Only keep if this duration contains the time point
                if e_start <= time_point <= e_end:
                    clipped_durs.append([e_start, e_end])
            elif time_range is not None:
                # Only keep if this duration overlaps the range
                range_start, range_end = time_range
                if e_end >= range_start and e_start <= range_end:
                    c_start = max(e_start, range_start)
                    c_end = min(e_end, range_end)
                    if c_end > c_start:
                        clipped_durs.append([c_start, c_end])
        
        if clipped_durs:
            ent_copy = dict(ent)
            ent_copy["durations_clipped_to_question"] = clipped_durs
            entities_for_event.append(ent_copy)
    
    return entities_for_event


def _get_all_objects_for_event(
    tracked_objects: List[Dict[str, Any]],
    event_idx: int,
    fps: float,
) -> List[Dict[str, Any]]:
    """Get ALL objects for an event (all frames, not clipped)."""
    objects_for_event: List[Dict[str, Any]] = []
    for obj in tracked_objects:
        event_ids = obj.get("event_id", [])
        if not isinstance(event_ids, list) or event_idx not in event_ids:
            continue
        
        frame_numbers = obj.get("frame_numbers", [])
        bbox_history = obj.get("bbox_history", [])
        confidence_history = obj.get("confidence_history", [])

        if not isinstance(frame_numbers, list):
            continue

        # Get all frames/timestamps for this object
        timestamps = [f / fps for f in frame_numbers]

        obj_data = {
            "id": obj.get("id"),
            "track_id": obj.get("track_id"),
            "class_id": obj.get("class_id"),
            "class_name": obj.get("class_name"),
            "first_frame": obj.get("first_frame"),
            "last_frame": obj.get("last_frame"),
            "total_frames": obj.get("total_frames"),
            "frame_numbers": frame_numbers,
            "timestamps": timestamps,
            "bbox_history": bbox_history if isinstance(bbox_history, list) else [],
            "confidence_history": confidence_history if isinstance(confidence_history, list) else [],
        }
        objects_for_event.append(obj_data)
    
    return objects_for_event


def _get_objects_for_event_in_range(
    tracked_objects: List[Dict[str, Any]],
    event_idx: int,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> List[Dict[str, Any]]:
    """Get objects for an event in a frame range, clipping bboxes to the range."""
    objects_for_event: List[Dict[str, Any]] = []
    for obj in tracked_objects:
        event_ids = obj.get("event_id", [])
        if not isinstance(event_ids, list) or event_idx not in event_ids:
            continue
        
        frame_numbers = obj.get("frame_numbers", [])
        bbox_history = obj.get("bbox_history", [])
        confidence_history = obj.get("confidence_history", [])

        if not isinstance(frame_numbers, list):
            continue

        # Find indices where frame is in the range
        matching_indices = [
            i
            for i, f in enumerate(frame_numbers)
            if isinstance(f, (int, float))
            and start_frame <= int(f) <= end_frame
        ]

        if not matching_indices:
            continue

        # Extract bboxes/frames in the range
        clipped_frames = [frame_numbers[i] for i in matching_indices]
        clipped_timestamps = [f / fps for f in clipped_frames]
        clipped_bboxes = (
            [bbox_history[i] for i in matching_indices]
            if isinstance(bbox_history, list)
            and len(bbox_history) == len(frame_numbers)
            else []
        )
        clipped_confidences = (
            [confidence_history[i] for i in matching_indices]
            if isinstance(confidence_history, list)
            and len(confidence_history) == len(frame_numbers)
            else []
        )

        obj_data = {
            "id": obj.get("id"),
            "track_id": obj.get("track_id"),
            "class_id": obj.get("class_id"),
            "class_name": obj.get("class_name"),
            "first_frame": obj.get("first_frame"),
            "last_frame": obj.get("last_frame"),
            "total_frames": obj.get("total_frames"),
            "clipped": {
                "frame_numbers": clipped_frames,
                "timestamps": clipped_timestamps,
                "bbox_history": clipped_bboxes,
                "confidence_history": clipped_confidences,
            },
        }
        objects_for_event.append(obj_data)
    
    return objects_for_event


def build_ava_json(output_path: Path) -> None:
    """
    Build ava.json:
    1. Find the event in vdb_events.json that contains the time_reference point
    2. Use that event ID to find all entities in entities.json that reference it
    3. Store the event and its entities
    """
    qa_files = [
        ("citytour", ROOT / "datas" / "AVA100" / "citytour.json"),
        ("ego", ROOT / "datas" / "AVA100" / "ego.json"),
        ("traffic", ROOT / "datas" / "AVA100" / "traffic.json"),
        ("wildlife", ROOT / "datas" / "AVA100" / "wildlife.json"),
    ]

    results: List[Dict[str, Any]] = []

    for split_name, qa_path in qa_files:
        if not qa_path.exists():
            print(f"[AVA] QA file missing, skipping: {qa_path}")
            continue

        qa_data = load_json(qa_path)

        for video_entry in qa_data:
            video_id = video_entry.get("video_id")
            video_key = video_entry.get("video_key")
            qa_list = video_entry.get("qa", [])

            # Skip citytour1 as requested (missing data)
            if video_key == "citytour1":
                continue

            base_dir = (
                ROOT
                / "AVA_cache"
                / "AVA100"
                / str(video_id)
                / "kg"
            )
            entities_path = base_dir / "entities" / "entities.json"
            vdb_events_path = base_dir / "vdb_events.json"

            # Load entities and events if available, but continue processing even if missing
            entities = []
            vdb_events = []
            if entities_path.exists() and vdb_events_path.exists():
                entities = load_json(entities_path)
                vdb_events_data = load_json(vdb_events_path)
                # Extract events from vdb_events.json
                vdb_events = vdb_events_data.get("data", [])
                if not vdb_events:
                    print(
                        f"[AVA] No events in vdb_events.json for video_id={video_id}, "
                        f"video_key={video_key} (will still process questions with empty events)"
                    )
            else:
                print(
                    f"[AVA] Missing entities/vdb_events for video_id={video_id}, "
                    f"video_key={video_key}, base={base_dir} (will still process questions with empty events)"
                )

            for q in qa_list:
                time_ref = q.get("time_reference")
                time_ref_parsed = parse_time_reference(time_ref)
                if time_ref_parsed is None:
                    continue

                event_entries: List[Dict[str, Any]] = []

                # Case 1: Single point (float)
                if isinstance(time_ref_parsed, float):
                    time_point = time_ref_parsed
                    # Find the event that contains this time point
                    matched_event = None
                    for ev in vdb_events:
                        dur = ev.get("duration")
                        if not isinstance(dur, (list, tuple)) or len(dur) != 2:
                            continue
                        ev_start, ev_end = float(dur[0]), float(dur[1])
                        if ev_start <= time_point <= ev_end:
                            matched_event = ev
                            break

                    if matched_event:
                        event_id = matched_event.get("id") or matched_event.get("__id__")
                        if event_id:
                            ev_dur = matched_event.get("duration", [0, 0])
                            ev_start, ev_end = float(ev_dur[0]), float(ev_dur[1])
                            
                            # Find all entities that reference this event ID
                            entities_for_event = _get_entities_for_event(
                                entities, event_id, time_point, None
                            )
                            
                            event_entries.append({
                                "event_id": event_id,
                                "event_duration_seconds": {
                                    "start": ev_start,
                                    "end": ev_end,
                                },
                                "num_entities": len(entities_for_event),
                                "entities": entities_for_event,
                            })

                # Case 2: Multiple points (List[float])
                elif isinstance(time_ref_parsed, list):
                    for time_point in time_ref_parsed:
                        # Find the event that contains this time point
                        matched_event = None
                        for ev in vdb_events:
                            dur = ev.get("duration")
                            if not isinstance(dur, (list, tuple)) or len(dur) != 2:
                                continue
                            ev_start, ev_end = float(dur[0]), float(dur[1])
                            if ev_start <= time_point <= ev_end:
                                matched_event = ev
                                break

                        if matched_event:
                            event_id = matched_event.get("id") or matched_event.get("__id__")
                            if event_id:
                                ev_dur = matched_event.get("duration", [0, 0])
                                ev_start, ev_end = float(ev_dur[0]), float(ev_dur[1])
                                
                                # Find all entities that reference this event ID
                                entities_for_event = _get_entities_for_event(
                                    entities, event_id, time_point, None
                                )
                                
                                event_entries.append({
                                    "event_id": event_id,
                                    "event_duration_seconds": {
                                        "start": ev_start,
                                        "end": ev_end,
                                    },
                                    "num_entities": len(entities_for_event),
                                    "entities": entities_for_event,
                                })

                # Case 3: Range (Tuple[float, float])
                elif isinstance(time_ref_parsed, tuple):
                    range_start, range_end = time_ref_parsed
                    # Find all events that overlap with this range
                    overlapping_events: List[Dict[str, Any]] = []
                    for ev in vdb_events:
                        dur = ev.get("duration")
                        if not isinstance(dur, (list, tuple)) or len(dur) != 2:
                            continue
                        ev_start, ev_end = float(dur[0]), float(dur[1])
                        if ev_end >= range_start and ev_start <= range_end:
                            overlapping_events.append(ev)

                    # For each overlapping event, get entities
                    for ev in overlapping_events:
                        event_id = ev.get("id") or ev.get("__id__")
                        if not event_id:
                            continue
                        
                        ev_dur = ev.get("duration", [0, 0])
                        ev_start, ev_end = float(ev_dur[0]), float(ev_dur[1])
                        
                        # Find all entities that reference this event ID and overlap the range
                        entities_for_event = _get_entities_for_event(
                            entities, event_id, None, (range_start, range_end)
                        )
                        
                        event_entries.append({
                            "event_id": event_id,
                            "event_duration_seconds": {
                                "start": ev_start,
                                "end": ev_end,
                            },
                            "event_window_seconds": {
                                "start": max(ev_start, range_start),
                                "end": min(ev_end, range_end),
                            },
                            "num_entities": len(entities_for_event),
                            "entities": entities_for_event,
                        })

                results.append(
                    {
                        "split": split_name,
                        "video_id": video_id,
                        "video_key": video_key,
                        "question_id": q.get("question_id"),
                        "query": q.get("query"),
                        "time_reference": time_ref,
                        "options": q.get("options"),
                        "answer": q.get("answer"),
                        "sub_video": q.get("sub_video"),
                        "time_reference_parsed": time_ref_parsed,
                        "num_events": len(event_entries),
                        "events": event_entries,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[AVA] Saved {len(results)} entries to {output_path}")


def load_tracked_objects(db_path: Path) -> Tuple[List[Dict[str, Any]], float]:
    """
    Load all tracked_objects rows and FPS from a tracked_objects.db.

    Returns:
        (objects, fps)
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get fps from video_info (assume one row or take first)
    cur.execute("SELECT fps FROM video_info LIMIT 1;")
    row = cur.fetchone()
    fps = float(row["fps"]) if row and row["fps"] is not None else 30.0

    cur.execute("SELECT * FROM tracked_objects;")
    rows = cur.fetchall()
    objects: List[Dict[str, Any]] = []
    for r in rows:
        obj = dict(r)
        # frame_numbers, bbox_history, confidence_history, event_id are stored as JSON strings
        for key in ("frame_numbers", "bbox_history", "confidence_history", "event_id"):
            val = obj.get(key)
            if isinstance(val, str):
                try:
                    obj[key] = json.loads(val)
                except Exception:
                    # Keep raw string on failure
                    pass
        objects.append(obj)

    conn.close()
    return objects, fps


def build_icdcs_json(output_path: Path) -> None:
    """
    Build icdcs.json:
    1. Find the event in event_embeddings.db that contains the time_reference point
    2. Use that event index to find all objects in tracked_objects.db that have that event_id
    3. Store the event and its objects (with bboxes clipped to the time point)
    """
    qa_files = [
        ("citytour", ROOT / "datas" / "AVA100" / "citytour.json"),
        ("ego", ROOT / "datas" / "AVA100" / "ego.json"),
        ("traffic", ROOT / "datas" / "AVA100" / "traffic.json"),
        ("wildlife", ROOT / "datas" / "AVA100" / "wildlife.json"),
    ]

    results: List[Dict[str, Any]] = []

    for split_name, qa_path in qa_files:
        if not qa_path.exists():
            print(f"[ICDCS] QA file missing, skipping: {qa_path}")
            continue

        qa_data = load_json(qa_path)

        for video_entry in qa_data:
            video_id = video_entry.get("video_id")
            video_key = video_entry.get("video_key")
            qa_list = video_entry.get("qa", [])

            # Skip citytour1 as requested (missing data)
            if video_key == "citytour1":
                continue

            scenario_dir = ROOT / "database_og" / str(video_key)
            tracked_db = scenario_dir / "tracked_objects.db"
            event_emb_db = scenario_dir / "event_embeddings.db"

            if not (tracked_db.exists() and event_emb_db.exists()):
                print(
                    f"[ICDCS] Missing tracked_objects.db or event_embeddings.db for "
                    f"video_key={video_key} at {scenario_dir}"
                )
                continue

            # Load all tracked objects and fps
            tracked_objects, fps = load_tracked_objects(tracked_db)

            # Count total events with embeddings
            conn = sqlite3.connect(str(event_emb_db))
            cur = conn.cursor()
            try:
                cur.execute("SELECT COUNT(*) FROM event_embeddings;")
                total_events_with_embeddings = int(cur.fetchone()[0])
            except Exception as exc:
                print(
                    f"[ICDCS] Warning: failed to count events in {event_emb_db}: {exc}"
                )
                total_events_with_embeddings = 0
            finally:
                conn.close()

            # We need to find which event contains the time point
            # Since event_embeddings.db doesn't have time info, we need to use events.json
            # or infer from tracked_objects' event_id and frame_numbers
            # Actually, we can use tracked_objects to find which event_index contains the time point
            # by checking which objects have frames at that time point, then get their event_ids

            for q in qa_list:
                time_ref = q.get("time_reference")
                time_ref_parsed = parse_time_reference(time_ref)
                if time_ref_parsed is None:
                    continue

                event_entries: List[Dict[str, Any]] = []

                # Case 1: Single point (float) - select 1 event
                if isinstance(time_ref_parsed, float):
                    time_point = time_ref_parsed
                    target_frame = int(time_point * fps)
                    
                    # Find which event indices have objects at this frame and count objects per event
                    event_object_counts: Dict[int, int] = {}
                    for obj in tracked_objects:
                        frame_numbers = obj.get("frame_numbers", [])
                        if isinstance(frame_numbers, list) and target_frame in frame_numbers:
                            event_ids = obj.get("event_id", [])
                            if isinstance(event_ids, list):
                                for ev_idx in event_ids:
                                    if isinstance(ev_idx, (int, float)):
                                        ev_idx_int = int(ev_idx)
                                        event_object_counts[ev_idx_int] = event_object_counts.get(ev_idx_int, 0) + 1

                    # For single point, select the ONE event with the most objects at this frame
                    if event_object_counts:
                        # Sort by object count (descending), then by event index (ascending) for tie-breaking
                        best_event_idx = max(event_object_counts.items(), key=lambda x: (x[1], -x[0]))[0]
                        # Get ALL objects for this event (not just at the time point)
                        objects_for_event = _get_all_objects_for_event(
                            tracked_objects, best_event_idx, fps
                        )
                        if objects_for_event:
                            event_entries.append({
                                "event_index": best_event_idx,
                                "has_embedding": best_event_idx < total_events_with_embeddings,
                                "num_objects": len(objects_for_event),
                                "objects": objects_for_event,
                            })

                # Case 2: Multiple points (List[float]) - select 1 event per point
                elif isinstance(time_ref_parsed, list):
                    for time_point in time_ref_parsed:
                        target_frame = int(time_point * fps)
                        
                        # Find which event indices have objects at this frame and count objects per event
                        event_object_counts: Dict[int, int] = {}
                        for obj in tracked_objects:
                            frame_numbers = obj.get("frame_numbers", [])
                            if isinstance(frame_numbers, list) and target_frame in frame_numbers:
                                event_ids = obj.get("event_id", [])
                                if isinstance(event_ids, list):
                                    for ev_idx in event_ids:
                                        if isinstance(ev_idx, (int, float)):
                                            ev_idx_int = int(ev_idx)
                                            event_object_counts[ev_idx_int] = event_object_counts.get(ev_idx_int, 0) + 1

                        # For each point, select the ONE event with the most objects at this frame
                        if event_object_counts:
                            # Sort by object count (descending), then by event index (ascending) for tie-breaking
                            best_event_idx = max(event_object_counts.items(), key=lambda x: (x[1], -x[0]))[0]
                            # Get ALL objects for this event (not just at the time point)
                            objects_for_event = _get_all_objects_for_event(
                                tracked_objects, best_event_idx, fps
                            )
                            if objects_for_event:
                                event_entries.append({
                                    "event_index": best_event_idx,
                                    "has_embedding": best_event_idx < total_events_with_embeddings,
                                    "num_objects": len(objects_for_event),
                                    "objects": objects_for_event,
                                })

                # Case 3: Range (Tuple[float, float]) - can have multiple overlapping events
                elif isinstance(time_ref_parsed, tuple):
                    range_start, range_end = time_ref_parsed
                    start_frame = int(range_start * fps)
                    end_frame = int(range_end * fps)
                    
                    # Find all event indices that have objects in this frame range
                    event_indices_found: set = set()
                    for obj in tracked_objects:
                        frame_numbers = obj.get("frame_numbers", [])
                        if isinstance(frame_numbers, list):
                            # Check if any frame is in the range
                            if any(
                                isinstance(f, (int, float))
                                and start_frame <= int(f) <= end_frame
                                for f in frame_numbers
                            ):
                                event_ids = obj.get("event_id", [])
                                if isinstance(event_ids, list):
                                    for ev_idx in event_ids:
                                        if isinstance(ev_idx, (int, float)):
                                            event_indices_found.add(int(ev_idx))

                    # For each overlapping event, get ALL objects for that event (clipped to range)
                    for event_idx in sorted(event_indices_found):
                        objects_for_event = _get_objects_for_event_in_range(
                            tracked_objects, event_idx, start_frame, end_frame, fps
                        )
                        if objects_for_event:
                            event_entries.append({
                                "event_index": event_idx,
                                "has_embedding": event_idx < total_events_with_embeddings,
                                "num_objects": len(objects_for_event),
                                "objects": objects_for_event,
                            })

                results.append(
                    {
                        "split": split_name,
                        "video_id": video_id,
                        "video_key": video_key,
                        "question_id": q.get("question_id"),
                        "query": q.get("query"),
                        "time_reference": time_ref,
                        "options": q.get("options"),
                        "answer": q.get("answer"),
                        "sub_video": q.get("sub_video"),
                        "time_reference_parsed": time_ref_parsed,
                        "fps": fps,
                        "num_events": len(event_entries),
                        "events": event_entries,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[ICDCS] Saved {len(results)} entries to {output_path}")


def main() -> None:
    ava_out = ROOT / "datas" / "AVA100" / "ava.json"
    icdcs_out = ROOT / "datas" / "AVA100" / "icdcs.json"

    print("Building ava.json ...")
    build_ava_json(ava_out)

    print("\nBuilding icdcs.json ...")
    build_icdcs_json(icdcs_out)


if __name__ == "__main__":
    main()
