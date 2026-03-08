"""
Extract events and frames from sorted_SA_score_result.json for each query.

- Aggregates frame_durations from ALL nodes (ignores which node), finds events
  overlapping any range, sorts events and frames in chronological order.
- Events: seed_events-style (id, description, duration).
- Frames: uniform sampling 1 frame per 9 seconds per event (min 1 if event < 9s);
  saved as frame_outputs/1.jpg, 2.jpg, ... + frames_meta.json (rank, timestamp, event_id).
"""
import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
import sys
sys.path.insert(0, str(_project_root))

from video_utils import VideoRepresentation


def _events_overlap_range(event_start: float, event_end: float, range_start: float, range_end: float) -> bool:
    return event_start < range_end and event_end > range_start


def _collect_all_frame_durations(sa_results: List[dict]) -> List[List[float]]:
    """Collect and merge frame_durations from every node. Returns list of [start, end] (no dedup)."""
    out = []
    for node in sa_results:
        for d in node.get("frame_durations") or []:
            if isinstance(d, (list, tuple)) and len(d) >= 2:
                out.append([float(d[0]), float(d[1])])
    return out


def events_in_frame_durations(
    events: List[dict],
    frame_durations: List[List[float]],
) -> List[dict]:
    """Return events whose duration overlaps at least one range. Keeps original order; caller sorts."""
    selected = []
    for ev in events:
        dur = ev.get("duration") or [0, 0]
        if not isinstance(dur, (list, tuple)) or len(dur) < 2:
            continue
        es, ee = float(dur[0]), float(dur[1])
        for (rs, re) in frame_durations:
            if _events_overlap_range(es, ee, rs, re):
                selected.append(ev)
                break
    return selected


def uniform_sample_frames_for_event(
    video: VideoRepresentation,
    event_start: float,
    event_end: float,
    seconds_per_frame: float = 9.0,
) -> Tuple[List, List[float], List[int]]:
    """1 frame every seconds_per_frame; if event < seconds_per_frame, 1 frame. Returns (frames, timestamps, frame_indices)."""
    duration_sec = event_end - event_start
    if duration_sec <= 0:
        return [], [], []
    if duration_sec < seconds_per_frame:
        num_frames = 1
    else:
        num_frames = max(1, int(math.ceil(duration_sec / seconds_per_frame)))
    return video.get_frames_by_num(num_frames=num_frames, duration=(event_start, event_end))


def extract_events_and_frames_from_sorted_sa(
    sorted_sa_path: str | Path,
    kg_dir: str | Path,
    work_dir: str | Path,
    seconds_per_frame: float = 9.0,
    output_dir: str | Path | None = None,
    frame_output_dir: str | Path | None = None,
    write_events_json: bool = True,
    video_key: str | None = None,
    question_id: int | None = None,
    video_folder: str | None = None,
) -> dict:
    """
    For one query (one sorted_SA_score_result.json):
    - Collect frame_durations from ALL nodes.
    - Find events overlapping any range; sort events chronologically.
    - Build seed_events (id, description, duration) in chronological order.
    - Extract frames (1 per 9s per event, min 1); sort all frames by timestamp.
    - If write_events_json: write seed_events-style JSON to output_dir (or skip if aggregating).
    - Frames go to frame_output_dir if set, else output_dir / "frame_outputs".

    Returns:
        Result dict with seed_event_ids, seed_events, num_frames, frame_meta, and entry (for aggregation).
    """
    sorted_sa_path = Path(sorted_sa_path)
    kg_dir = Path(kg_dir)
    work_dir = Path(work_dir)
    output_dir = Path(output_dir) if output_dir else sorted_sa_path.parent
    frames_dir = Path(frame_output_dir) if frame_output_dir else (output_dir / "frame_outputs")

    with open(sorted_sa_path, "r") as f:
        sa_results = json.load(f)
    if not sa_results:
        entry = {
            "video_key": video_key or "",
            "question_id": question_id,
            "video_folder": video_folder or "",
            "question": "",
            "options": "",
            "seed_event_ids": [],
            "seed_events": [],
            "num_frames": 0,
        }
        return {"seed_event_ids": [], "seed_events": [], "num_frames": 0, "frame_meta": [], "entry": entry}

    frame_durations = _collect_all_frame_durations(sa_results)
    if not frame_durations:
        entry = {
            "video_key": video_key or "",
            "question_id": question_id,
            "video_folder": video_folder or "",
            "question": "",
            "options": "",
            "seed_event_ids": [],
            "seed_events": [],
            "num_frames": 0,
        }
        return {"seed_event_ids": [], "seed_events": [], "num_frames": 0, "frame_meta": [], "entry": entry}

    vdb_events_path = kg_dir / "vdb_events.json"
    if not vdb_events_path.exists():
        raise FileNotFoundError(f"vdb_events.json not found: {vdb_events_path}")
    with open(vdb_events_path, "r") as f:
        all_events = (json.load(f).get("data") or [])

    selected = events_in_frame_durations(all_events, frame_durations)
    # Sort events chronologically (by start time)
    selected.sort(key=lambda e: (e.get("duration") or [0, 0])[0])

    seed_events = []
    for ev in selected:
        eid = ev.get("id") or ev.get("__id__") or ""
        dur = ev.get("duration") or [0, 0]
        seed_events.append({
            "__id__": eid,
            "id": eid,
            "description": ev.get("description", ""),
            "duration": [float(dur[0]), float(dur[1])],
        })
    seed_event_ids = [e["id"] for e in seed_events]

    if not selected:
        entry = {
            "video_key": video_key or "",
            "question_id": question_id,
            "video_folder": video_folder or "",
            "question": "",
            "options": "",
            "seed_event_ids": [],
            "seed_events": [],
            "num_frames": 0,
        }
        if write_events_json:
            _write_seed_events_json(output_dir, video_key, question_id, [], [], [])
        return {"seed_event_ids": [], "seed_events": [], "num_frames": 0, "frame_meta": [], "entry": entry}

    config_path = work_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {work_dir}")
    config = json.load(open(config_path))
    source_path = config.get("source_path")
    if not source_path or not os.path.isabs(source_path):
        source_path = _project_root / source_path
    else:
        source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Video source not found: {source_path}")

    video = VideoRepresentation(str(source_path), str(work_dir))

    # Collect (timestamp, frame_idx, image, event_id) for each event
    all_frames_data = []
    for ev in selected:
        eid = ev.get("id") or ev.get("__id__") or ""
        dur = ev.get("duration") or [0, 0]
        es, ee = float(dur[0]), float(dur[1])
        frames, timestamps, frame_indices = uniform_sample_frames_for_event(
            video, es, ee, seconds_per_frame=seconds_per_frame
        )
        for t, idx, img in zip(timestamps, frame_indices, frames):
            all_frames_data.append((t, idx, img, eid))

    # Sort all frames by timestamp (chronological)
    all_frames_data.sort(key=lambda x: (x[0], x[1]))

    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_meta = []
    for rank, (t, idx, img, eid) in enumerate(all_frames_data, start=1):
        frame_meta.append({"rank": rank, "timestamp": t, "event_id": eid})
        img.save(frames_dir / f"{rank}.jpg")

    with open(frames_dir / "frames_meta.json", "w") as f:
        json.dump(frame_meta, f, indent=2)

    entry = {
        "video_key": video_key or "",
        "question_id": question_id,
        "video_folder": video_folder or "",
        "question": "",
        "options": "",
        "seed_event_ids": seed_event_ids,
        "seed_events": seed_events,
        "num_frames": len(frame_meta),
    }
    if write_events_json:
        _write_seed_events_json(output_dir, video_key, question_id, seed_event_ids, seed_events, frame_meta)

    return {
        "seed_event_ids": seed_event_ids,
        "seed_events": seed_events,
        "num_frames": len(all_frames_data),
        "frame_meta": frame_meta,
        "entry": entry,
    }


def _write_seed_events_json(
    output_dir: Path,
    video_key: str | None,
    question_id: int | None,
    seed_event_ids: List[str],
    seed_events: List[dict],
    frame_meta: List[dict],
) -> None:
    """Write seed_events-style JSON (like top_k_events_retrieval / top_k_frames_retrieval)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "video_key": video_key or "",
        "question_id": question_id,
        "question": "",
        "options": "",
        "seed_event_ids": seed_event_ids,
        "seed_events": seed_events,
        "num_frames": len(frame_meta),
    }
    out_path = output_dir / "seed_events_SA.json"
    with open(out_path, "w") as f:
        json.dump(entry, f, indent=2)


def _find_sa_result_paths(cache_root: Path, datasets: List[str]) -> List[Path]:
    """
    Find all sorted_SA_score_result.json paths under AVA_cache/<dataset>/*/questions/*/.
    Returns list of paths to sorted_SA_score_result.json files.
    """
    out = []
    for name in datasets:
        base = cache_root / name
        if not base.is_dir():
            continue
        for video_folder in base.iterdir():
            if not video_folder.is_dir():
                continue
            questions_dir = video_folder / "questions"
            if not questions_dir.is_dir():
                continue
            for q_dir in questions_dir.iterdir():
                if not q_dir.is_dir():
                    continue
                sa_path = q_dir / "sorted_SA_score_result.json"
                if sa_path.exists():
                    out.append(sa_path)
    return sorted(out)


def _path_to_dataset_and_query(path: Path, cache_root: Path) -> Tuple[str, str, int]:
    """From path like .../AVA_cache/AVA100/1/questions/0/sorted_SA_score_result.json return (dataset, video_folder, question_id)."""
    try:
        rel = path.parent.relative_to(cache_root)
        parts = rel.parts  # e.g. ("AVA100", "1", "questions", "0")
        if len(parts) >= 4 and parts[2] == "questions":
            return parts[0], parts[1], int(parts[3])
        if len(parts) >= 1:
            return parts[0], parts[1] if len(parts) >= 2 else "0", 0
    except ValueError:
        pass
    parts = path.parts
    if "questions" in parts:
        qidx = parts.index("questions")
        video_folder = parts[qidx - 1] if qidx > 0 else "0"
        question_id = int(parts[qidx + 1]) if qidx + 1 < len(parts) else 0
        dataset = "LVBench" if "LVBench" in parts else "AVA100"
        return dataset, video_folder, question_id
    return "AVA100", "0", 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract events (with descriptions) and frames from sorted_SA_score_result.json; store seed_events + frame_outputs (chronological)."
    )
    parser.add_argument(
        "sorted_sa_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to one sorted_SA_score_result.json (optional; if omitted, run batch on --dataset)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["AVA100", "LVBench"],
        help="Datasets to run in batch (default: AVA100 LVBench). Ignored if sorted_sa_path is given.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Cache root (default: project_root / AVA_cache)",
    )
    parser.add_argument(
        "--kg-dir",
        type=str,
        default=None,
        help="Path to kg dir (default: inferred from sorted_sa_path; single-query only)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Video work dir (default: parent of kg_dir; single-query only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output dir (default: AVA/extract_sa_output). Each dataset gets <output-dir>/<dataset>/ with seed_events.json and frames/<video>_<qid>/",
    )
    parser.add_argument(
        "--seconds-per-frame",
        type=float,
        default=9.0,
        help="1 frame per N seconds per event (default 9)",
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default=None,
        help="Video key for seed_events JSON (default: from config source_path stem; single-query only)",
    )
    parser.add_argument(
        "--question-id",
        type=int,
        default=None,
        help="Question ID for seed_events JSON (default: from path; single-query only)",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache_root) if args.cache_root else _project_root / "AVA_cache"
    base_out = Path(args.output_dir) if args.output_dir else (_script_dir / "extract_sa_output")

    if args.sorted_sa_path:
        # Single-query mode: one path given
        paths = [Path(args.sorted_sa_path)]
        if not paths[0].exists():
            raise FileNotFoundError(paths[0])
    else:
        # Batch mode: discover all sorted_SA_score_result.json for given datasets
        paths = _find_sa_result_paths(cache_root, args.dataset)
        if not paths:
            print(f"No sorted_SA_score_result.json found under {cache_root} for datasets {args.dataset}")
            return
        print(f"Batch: found {len(paths)} question folders for {args.dataset}")

    # Group paths by dataset for per-dataset events JSON + frames dir
    by_dataset: dict = {}
    for p in paths:
        path = Path(p)
        if path.parts[-1] != "sorted_SA_score_result.json":
            path = path / "sorted_SA_score_result.json"
        if not path.exists():
            continue
        dataset_name, video_folder, _ = _path_to_dataset_and_query(path, cache_root)
        by_dataset.setdefault(dataset_name, []).append((path, video_folder))

    for dataset_name, path_list in by_dataset.items():
        dataset_dir = base_out / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "frames").mkdir(parents=True, exist_ok=True)
        events_entries = []

        for path, video_folder in path_list:
            if path.parts[-1] != "sorted_SA_score_result.json":
                path = path / "sorted_SA_score_result.json"
            if not path.exists():
                continue

            if args.kg_dir:
                kg_dir = Path(args.kg_dir)
            else:
                parts = path.parts
                if "questions" in parts:
                    i = parts.index("questions")
                    kg_dir = Path(*parts[:i]) / "kg"
                else:
                    kg_dir = path.parent.parent / "kg"

            work_dir = Path(args.work_dir) if args.work_dir else kg_dir.parent
            _, _, question_id_from_path = _path_to_dataset_and_query(path, cache_root)
            question_id = args.question_id if args.question_id is not None else question_id_from_path
            video_key = args.video_key
            if video_key is None and work_dir:
                cfg_path = work_dir / "config.json"
                if cfg_path.exists():
                    try:
                        config = json.load(open(cfg_path))
                        src = config.get("source_path", "")
                        if src:
                            video_key = Path(src).stem
                    except Exception:
                        pass

            frame_query_dir = dataset_dir / "frames" / f"{video_folder}_{question_id}"

            try:
                result = extract_events_and_frames_from_sorted_sa(
                    sorted_sa_path=path,
                    kg_dir=kg_dir,
                    work_dir=work_dir,
                    seconds_per_frame=args.seconds_per_frame,
                    output_dir=frame_query_dir,
                    frame_output_dir=frame_query_dir,
                    write_events_json=False,
                    video_key=video_key,
                    question_id=question_id,
                    video_folder=video_folder,
                )
                events_entries.append(result["entry"])
                if len(paths) > 1:
                    print(f"  {path.parent}: events={len(result['seed_events'])}, frames={result['num_frames']}")
            except Exception as e:
                print(f"Error {path}: {e}")
                continue

        with open(dataset_dir / "seed_events.json", "w") as f:
            json.dump(events_entries, f, indent=2)

        if len(by_dataset) > 1 or len(path_list) > 1:
            print(f"[{dataset_name}] {len(events_entries)} queries -> {dataset_dir / 'seed_events.json'}, frames in {dataset_dir / 'frames'}/")
        else:
            print(f"Events: {len(events_entries[0]['seed_events']) if events_entries else 0}, Frames: see {dataset_dir / 'frames'}")
            print(f"Written: {dataset_dir / 'seed_events.json'}, {dataset_dir / 'frames'}")

    if len(paths) > 1:
        print(f"Done. Processed {len(paths)} question folders.")


if __name__ == "__main__":
    main()
