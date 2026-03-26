#!/usr/bin/env python3
"""
Unified generation + evaluation for three input types:
- events_retrieval: JSON from top_k_events_retrieval (events_only mode).
- frames_retrieval: output dir from top_k_frames_retrieval (frames_only mode).
- ava_extract: AVA/extract_sa_output/<dataset>/ (single query, --video-key + --question-id, --mode).

Outputs two files: (1) responses JSON, (2) accuracy/latency log (global metrics + per-query).
Timer: include load + prompt build + model call; exclude parsing, storing, writing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_script_dir = Path(__file__).resolve().parent
_ecml_dir = _script_dir.parent
_project_root = _ecml_dir.parent
sys.path.insert(0, str(_project_root))

from PIL import Image
from AVA.prompt import PROMPTS
from llms.init_model import init_model

# -----------------------------------------------------------------------------
# Ground truth (answer) loading — same locations as calc_retrieval_accuracy
# -----------------------------------------------------------------------------


def load_gt_answers(dataset: str, project_root: Path) -> Dict[Tuple[str, str], str]:
    """(video_key, str(question_id)) -> answer (A/B/C/D)."""
    cache: Dict[Tuple[str, str], str] = {}
    if dataset == "AVA100":
        base = project_root / "datas" / "AVA100"
        for name in ["citytour.json", "ego.json", "traffic.json", "wildlife.json"]:
            path = base / name
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
                if isinstance(data, list):
                    for video in data:
                        video_key = video.get("video_key")
                        if not video_key:
                            continue
                        for qa in video.get("qa", []):
                            qid = qa.get("question_id")
                            if qid is not None:
                                ans = (qa.get("answer") or "").strip().upper()
                                if ans in ("A", "B", "C", "D"):
                                    cache[(video_key, str(qid))] = ans
            except Exception as e:
                print(f"  Warning: failed to load {path}: {e}")
    elif dataset == "LVBench":
        path = project_root / "datas" / "LVBench" / "LVBench.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, list):
                    for video in data:
                        video_key = video.get("key") or video.get("video_key")
                        if not video_key:
                            continue
                        # Support both key styles so all inputs match:
                        # - (video_key, str(idx)) for extract-style (question_id 0, 1, 2, ...)
                        # - (video_key, str(uid)) for top_k_events_retrieval / exp0_1 (question_id "55", "56", ...)
                        for idx, qa in enumerate(video.get("qa", [])):
                            ans = (qa.get("answer") or "").strip().upper()
                            if ans not in ("A", "B", "C", "D"):
                                continue
                            cache[(video_key, str(idx))] = ans
                            uid = qa.get("uid")
                            if uid is not None:
                                cache[(video_key, str(uid))] = ans
            except Exception as e:
                print(f"  Warning: failed to load {path}: {e}")
    return cache


def extract_predicted_answer(response: str | None) -> str | None:
    """Extract predicted letter (A/B/C/D) from response. Same logic as evaluate_vlm_direct."""
    if not response or not isinstance(response, str):
        return None
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        ans = data.get("Answer") or data.get("answer")
        if ans and str(ans).upper() in ("A", "B", "C", "D"):
            return str(ans).upper()
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r'"Answer"\s*:\s*"([ABCD])"', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\s*[\.\)]\s*$", text)
    if m:
        return m.group(1).upper()
    return None


# -----------------------------------------------------------------------------
# Events-only helpers (events_retrieval path)
# -----------------------------------------------------------------------------


def _duration_to_str(duration) -> str:
    if not duration or not isinstance(duration, (list, tuple)) or len(duration) < 2:
        return ""
    a, b = duration[0], duration[1]
    try:
        a, b = float(a), float(b)
        if a > 10000 or b > 10000:
            a, b = a / 1000.0, b / 1000.0
        return f"{a:.2f}s - {b:.2f}s"
    except (TypeError, ValueError):
        return f"{a} - {b}"


def build_event_descriptions(seed_events: list) -> str:
    lines = []
    for i, ev in enumerate(seed_events, start=1):
        desc = ev.get("description", "").strip()
        duration = ev.get("duration", [])
        dur_str = _duration_to_str(duration)
        if dur_str:
            lines.append(f"{i}. [{dur_str}] {desc}")
        else:
            lines.append(f"{i}. {desc}")
    return "\n\n".join(lines) if lines else "(No events)"


# -----------------------------------------------------------------------------
# Frames-only helpers (frames_retrieval path)
# -----------------------------------------------------------------------------

MAX_FRAMES_BUDGET = 256


def load_frames_for_query(
    frame_outputs_dir: Path,
    video_key: str,
    question_id: Any,
    max_frames: int = MAX_FRAMES_BUDGET,
) -> Tuple[List[Any], str]:
    """Load images from frame_outputs/{video_key}_{question_id}/; return (images, frame_timestamps_str)."""
    slug = f"{video_key}_{question_id}"
    query_dir = frame_outputs_dir / slug
    if not query_dir.is_dir():
        return [], ""
    images = []
    for rank in range(1, max_frames + 1):
        p = query_dir / f"{rank}.jpg"
        if not p.exists():
            break
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception:
            break
    lines = [f"Image {i+1}" for i in range(len(images))]
    frame_timestamps_str = "\n".join(lines) if lines else "(No images)"
    return images, frame_timestamps_str


# -----------------------------------------------------------------------------
# AVA extract helpers (HH:MM:SS formatting, load one query)
# -----------------------------------------------------------------------------


def _seconds_to_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_frames_for_prompt_ava(frame_meta: List[Dict]) -> str:
    lines = []
    for m in frame_meta:
        rank = m.get("rank", 0)
        t = m.get("timestamp", 0.0)
        lines.append(f"Image {rank} @ {_seconds_to_timestamp(t)}")
    return "\n".join(lines)


def format_events_for_prompt_ava(events: List[Dict]) -> str:
    lines = []
    for i, ev in enumerate(events, start=1):
        dur = ev.get("duration") or [0, 0]
        start_ts = _seconds_to_timestamp(float(dur[0]))
        end_ts = _seconds_to_timestamp(float(dur[1]))
        desc = ev.get("description", "")
        lines.append(f"{i}. [{start_ts} - {end_ts}] {desc}")
    return "\n".join(lines)


def format_frames_and_events_ava(
    frame_meta: List[Dict],
    events: List[Dict],
) -> str:
    formatted_lines = []
    for i, event in enumerate(events, start=1):
        dur = event.get("duration") or [0, 0]
        event_start, event_end = float(dur[0]), float(dur[1])
        start_ts = _seconds_to_timestamp(event_start)
        end_ts = _seconds_to_timestamp(event_end)
        description = event.get("description", "")
        matching_frames = []
        for m in frame_meta:
            rank = m.get("rank", 0)
            t = m.get("timestamp", 0.0)
            if event_start <= t <= event_end:
                matching_frames.append(f"Image {rank} @ {_seconds_to_timestamp(t)}")
        formatted_lines.append(f"{i}. [{start_ts} - {end_ts}] {description}")
        if matching_frames:
            visual_evidence = ", ".join([f"[{f}]" for f in matching_frames])
            formatted_lines.append(f"   > Visual Evidence: {visual_evidence}")
        else:
            formatted_lines.append("   > Visual Evidence: None in this range")
    return "\n".join(formatted_lines)


def list_ava_queries(extract_base: Path, dataset: str) -> List[Tuple[str, int]]:
    """Return all (video_key, question_id) pairs from seed_events.json (one per entry)."""
    dataset_dir = extract_base / dataset
    seed_path = dataset_dir / "seed_events.json"
    if not seed_path.exists():
        raise FileNotFoundError(f"{seed_path} not found")
    with open(seed_path, "r") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("seed_events.json must be a list")
    return [(e["video_key"], e["question_id"]) for e in entries if e.get("video_key") is not None and e.get("question_id") is not None]


def load_ava_single_query(
    extract_base: Path,
    dataset: str,
    video_key: str,
    question_id: int,
) -> Tuple[Dict, List[Dict], List[Any], List[Dict]]:
    """Load one entry from seed_events.json and frames from frames/<video_folder>_<qid>/.
    Returns (seed_entry, frame_meta, frames_list, seed_events).
    """
    dataset_dir = extract_base / dataset
    seed_path = dataset_dir / "seed_events.json"
    if not seed_path.exists():
        raise FileNotFoundError(f"{seed_path} not found")
    with open(seed_path, "r") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("seed_events.json must be a list")
    entry = None
    for e in entries:
        if (e.get("video_key") == video_key and e.get("question_id") == question_id):
            entry = e
            break
    if not entry:
        raise ValueError(f"No entry for video_key={video_key} question_id={question_id}")
    video_folder = entry.get("video_folder", "")
    seed_events = entry.get("seed_events") or []
    frame_meta = []
    frames = []
    frames_subdir = f"{video_folder}_{question_id}" if video_folder else None
    if not frames_subdir:
        # Fallback for old extract without video_folder: find any dir ending with _<question_id>
        frames_parent = dataset_dir / "frames"
        if frames_parent.is_dir():
            for d in frames_parent.iterdir():
                if d.is_dir() and d.name.endswith(f"_{question_id}"):
                    frames_subdir = d.name
                    break
    frames_dir = dataset_dir / "frames" / (frames_subdir or f"_{question_id}")
    frame_meta_path = frames_dir / "frames_meta.json"
    if frame_meta_path.exists():
        with open(frame_meta_path, "r") as f:
            frame_meta = json.load(f)
        for m in frame_meta:
            rank = m.get("rank", 0)
            img_path = frames_dir / f"{rank}.jpg"
            if img_path.exists():
                frames.append(Image.open(img_path).convert("RGB"))
            else:
                frames.append(None)
        valid = [(m, img) for m, img in zip(frame_meta, frames) if img is not None]
        frame_meta = [x[0] for x in valid]
        frames = [x[1] for x in valid]
    return entry, frame_meta, frames, seed_events


# -----------------------------------------------------------------------------
# Run paths
# -----------------------------------------------------------------------------


def _call_model_get_usage(model, batch_inputs: List[Dict]) -> Tuple[str, Optional[Dict]]:
    """
    Call model.batch_generate_response; return (response_text, usage_dict).
    usage_dict has prompt_tokens, completion_tokens, total_tokens when the model supports return_usage=True.
    """
    # Unified decoding config for all models when supported.
    # do_sample=False is approximated by using a very low temperature.
    generation_kwargs = {
        "max_new_tokens": 350,
        "temperature": 0.4,
    }

    # Try progressively more general signatures to support different model wrappers.
    try:
        result = model.batch_generate_response(
            batch_inputs,
            return_usage=True,
            **generation_kwargs,
        )
    except TypeError:
        try:
            result = model.batch_generate_response(
                batch_inputs,
                **generation_kwargs,
            )
        except TypeError:
            try:
                result = model.batch_generate_response(
                    batch_inputs,
                    return_usage=True,
                )
            except TypeError:
                result = model.batch_generate_response(batch_inputs)
    if isinstance(result, tuple) and len(result) >= 2:
        texts, usages = result[0], result[1]
        response = texts[0] if texts else ""
        usage = usages[0] if usages else None
        return response, usage
    response = result[0] if result else ""
    return response, None


def run_events_retrieval(
    input_path: Path,
    dataset: str,
    project_root: Path,
    model: str,
    port: int,
    gpus: int,
    model_type: str | None = None,
    max_queries: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Returns (response_entries, eval_entries with latency_seconds, correct). Uses VLM for text-only prompts."""
    data = json.loads(input_path.read_text())
    results_list = data.get("results", [])
    if not results_list:
        return [], []
    if max_queries is not None:
        results_list = results_list[:max_queries]
        print(f"[profiling] Running first {len(results_list)} queries (max_queries={max_queries})")

    gt_cache = load_gt_answers(dataset, project_root)
    # Only pass model_type for qwenvl_vllm so init_model passes port (connect to hosted server, no local load)
    mt = model_type if model == "qwenvl_vllm" else None
    vlm = init_model(model, num_gpus=gpus, port=port, model_type=mt)

    response_entries = []
    eval_entries = []

    for idx, entry in enumerate(results_list):
        if entry.get("skip_reason"):
            continue
        video_key = entry.get("video_key", "")
        question_id = entry.get("question_id")
        question = entry.get("question", "")
        options = entry.get("options", "")
        user_query = f"{question}\n{options}".strip() if options else question
        seed_events = entry.get("seed_events") or []

        # Timer: prompt build + model call
        t0 = __import__("time").perf_counter()
        event_descriptions = build_event_descriptions(seed_events)
        prompt = PROMPTS["events_only"].format(
            event_descriptions=event_descriptions,
            user_query=user_query,
        )
        try:
            response, usage = _call_model_get_usage(vlm, [{"text": prompt}])
            # Surface first server/API error so user can fix connection or model name
            if response and response.strip().startswith("[ERROR]"):
                if idx == 0:
                    print("\n[SERVER ERROR] First request failed (see below). Fix connection, port, or model name and re-run.\n", file=sys.stderr)
                    print(f"[server error] {response.strip()}", file=sys.stderr)
        except Exception as e:
            response, usage = "", None
            print(f"[{idx+1}] {video_key} q{question_id}: {e}")
        elapsed = __import__("time").perf_counter() - t0

        response_entries.append({
            "video_key": video_key,
            "question_id": question_id,
            "question": question,
            "options": options,
            "response": (response or "").strip(),
        })

        gt = gt_cache.get((video_key, str(question_id)))
        pred = extract_predicted_answer(response) if response else None
        correct = (gt and pred and pred == gt)
        eval_entries.append({
            "video_key": video_key,
            "question_id": question_id,
            "latency_seconds": round(elapsed, 4),
            "correct": correct,
            "ground_truth": gt,
            "predicted": pred,
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "completion_tokens": usage.get("completion_tokens") if usage else None,
            "total_tokens": usage.get("total_tokens") if usage else None,
        })
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(results_list)}")

    return response_entries, eval_entries


def run_frames_retrieval(
    input_dir: Path,
    dataset: str,
    project_root: Path,
    model: str,
    port: int,
    gpus: int,
    max_frames: int,
    model_type: str | None = None,
    max_queries: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Returns (response_entries, eval_entries)."""
    candidates = list(input_dir.glob("seed_events_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No seed_events_*.json in {input_dir}")
    data = json.loads(candidates[0].read_text())
    results_list = data.get("results", [])
    if not results_list:
        return [], []
    if max_queries is not None:
        results_list = results_list[:max_queries]
        print(f"[profiling] Running first {len(results_list)} queries (max_queries={max_queries})")

    frame_outputs_dir = input_dir / "frame_outputs"
    gt_cache = load_gt_answers(dataset, project_root)
    mt = model_type if model == "qwenvl_vllm" else None
    vlm = init_model(model, num_gpus=gpus, port=port, model_type=mt)

    response_entries = []
    eval_entries = []

    for idx, entry in enumerate(results_list):
        if entry.get("skip_reason"):
            continue
        video_key = entry.get("video_key", "")
        question_id = entry.get("question_id")
        question = entry.get("question", "")
        options = entry.get("options", "")
        user_query = f"{question}\n{options}".strip() if options else question

        # Timer: image load + prompt build + model call
        t0 = __import__("time").perf_counter()
        images, frame_timestamps_str = load_frames_for_query(
            frame_outputs_dir, video_key, question_id, max_frames=max_frames
        )
        if not images:
            response_entries.append({
                "video_key": video_key,
                "question_id": question_id,
                "question": question,
                "options": options,
                "response": "",
                "error": "no_frames",
                "num_frames": 0,
            })
            gt = gt_cache.get((video_key, str(question_id)))
            eval_entries.append({
                "video_key": video_key,
                "question_id": question_id,
                "latency_seconds": 0.0,
                "correct": False,
                "ground_truth": gt,
                "predicted": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            })
            print(f"[{idx+1}] {video_key} q{question_id}: no frames")
            continue

        prompt = PROMPTS["frames_only"].format(
            frame_timestamps=frame_timestamps_str,
            user_query=user_query,
        )
        try:
            response, usage = _call_model_get_usage(vlm, [{"text": prompt, "video": images}])
            if response and response.strip().startswith("[ERROR]") and idx == 0:
                print("\n[SERVER ERROR] First request failed (see below). Fix connection, port, or model name and re-run.\n", file=sys.stderr)
                print(f"[server error] {response.strip()}", file=sys.stderr)
        except Exception as e:
            response, usage = "", None
            print(f"[{idx+1}] {video_key} q{question_id}: {e}")
        elapsed = __import__("time").perf_counter() - t0

        response_entries.append({
            "video_key": video_key,
            "question_id": question_id,
            "question": question,
            "options": options,
            "response": response or "",
            "num_frames": len(images),
        })

        gt = gt_cache.get((video_key, str(question_id)))
        pred = extract_predicted_answer(response) if response else None
        correct = (gt and pred and pred == gt)
        eval_entries.append({
            "video_key": video_key,
            "question_id": question_id,
            "latency_seconds": round(elapsed, 4),
            "correct": correct,
            "ground_truth": gt,
            "predicted": pred,
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "completion_tokens": usage.get("completion_tokens") if usage else None,
            "total_tokens": usage.get("total_tokens") if usage else None,
        })
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(results_list)}")

    return response_entries, eval_entries


def run_ava_extract(
    extract_base: Path,
    dataset: str,
    video_key: str,
    question_id: int,
    mode: str,
    project_root: Path,
    model: str,
    port: int,
    gpus: int,
    model_type: str | None = None,
) -> Tuple[Dict, Dict]:
    """Single query, one mode. Returns (response_entry, eval_entry). Uses same VLM for all modes (text-only, frames, or combined)."""
    # Timer: load + prompt build + model call (all inside)
    t0 = __import__("time").perf_counter()
    seed_entry, frame_meta, frames, seed_events = load_ava_single_query(
        extract_base, dataset, video_key, question_id
    )
    question = seed_entry.get("question", "")
    options = seed_entry.get("options", "")
    user_query = f"{question}\n{options}".strip() if options else question

    response = ""
    usage: Optional[Dict] = None
    mt = model_type if model == "qwenvl_vllm" else None
    vlm = init_model(model, num_gpus=gpus, port=port, model_type=mt)

    if mode == "frames_only":
        if not frames or not frame_meta:
            response = ""
        else:
            frame_timestamps = format_frames_for_prompt_ava(frame_meta)
            prompt = PROMPTS["frames_only"].format(
                frame_timestamps=frame_timestamps,
                user_query=user_query,
            )
            try:
                response, usage = _call_model_get_usage(vlm, [{"text": prompt, "video": frames}])
                response = response or ""
            except Exception as e:
                response, usage = "", None
                print(f"AVA frames_only: {e}")
    elif mode == "events_only":
        if not seed_events:
            response = ""
        else:
            event_descriptions = format_events_for_prompt_ava(seed_events)
            prompt = PROMPTS["events_only"].format(
                event_descriptions=event_descriptions,
                user_query=user_query,
            )
            try:
                response, usage = _call_model_get_usage(vlm, [{"text": prompt}])
                response = response or ""
            except Exception as e:
                response, usage = "", None
                print(f"AVA events_only: {e}")
    elif mode == "combined":
        if not frames or not frame_meta or not seed_events:
            response = ""
        else:
            event_descriptions_with_frames = format_frames_and_events_ava(frame_meta, seed_events)
            prompt = PROMPTS["frames_and_events_aligned"].format(
                event_descriptions_with_frames=event_descriptions_with_frames,
                user_query=user_query,
            )
            try:
                response, usage = _call_model_get_usage(vlm, [{"text": prompt, "video": frames}])
                response = response or ""
            except Exception as e:
                response, usage = "", None
                print(f"AVA combined: {e}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    elapsed = __import__("time").perf_counter() - t0

    gt_cache = load_gt_answers(dataset, project_root)
    gt = gt_cache.get((video_key, str(question_id)))
    pred = extract_predicted_answer(response) if response else None
    correct = (gt and pred and pred == gt)

    response_entry = {
        "video_key": video_key,
        "question_id": question_id,
        "question": question,
        "options": options,
        "mode": mode,
        "response": (response or "").strip(),
    }
    eval_entry = {
        "video_key": video_key,
        "question_id": question_id,
        "mode": mode,
        "latency_seconds": round(elapsed, 4),
        "correct": correct,
        "ground_truth": gt,
        "predicted": pred,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": usage.get("completion_tokens") if usage else None,
        "total_tokens": usage.get("total_tokens") if usage else None,
    }
    return response_entry, eval_entry


def run_ava_extract_all(
    extract_base: Path,
    dataset: str,
    mode: str,
    project_root: Path,
    model: str,
    port: int,
    gpus: int,
    model_type: str | None = None,
    max_queries: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Run ava_extract for every (video_key, question_id) in seed_events.json. Returns (response_entries, eval_entries)."""
    queries = list_ava_queries(extract_base, dataset)
    if not queries:
        return [], []
    if max_queries is not None:
        queries = queries[:max_queries]
        print(f"[profiling] Running first {len(queries)} queries (max_queries={max_queries})")
    response_entries = []
    eval_entries = []
    for idx, (video_key, question_id) in enumerate(queries):
        response_entry, eval_entry = run_ava_extract(
            extract_base,
            dataset,
            video_key,
            question_id,
            mode,
            project_root,
            model,
            port,
            gpus,
            model_type=model_type,
        )
        response_entries.append(response_entry)
        eval_entries.append(eval_entry)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(queries)}")
    return response_entries, eval_entries


# -----------------------------------------------------------------------------
# Build eval log (global + per_query) and write outputs
# -----------------------------------------------------------------------------


def eval_from_responses_file(
    responses_path: Path,
    dataset: str,
    project_root: Path,
) -> Tuple[List[Dict], List[Dict], str, str]:
    """
    Load existing responses JSON, recompute accuracy using current GT (e.g. fixed LVBench keys).
    Returns (response_entries, eval_entries, input_type, mode).
    """
    data = json.loads(responses_path.read_text())
    response_entries = data.get("results", [])
    if not response_entries:
        return [], [], data.get("input_type", "events_retrieval"), data.get("mode", "events_only")
    input_type = data.get("input_type", "events_retrieval")
    mode = data.get("mode", "events_only")
    gt_cache = load_gt_answers(dataset, project_root)
    eval_entries = []
    for r in response_entries:
        video_key = r.get("video_key")
        question_id = r.get("question_id")
        response = r.get("response") or ""
        pred = extract_predicted_answer(response)
        key = (video_key, str(question_id))
        gt = gt_cache.get(key)
        correct = (gt and pred and pred == gt)
        eval_entries.append({
            "video_key": video_key,
            "question_id": question_id,
            "latency_seconds": r.get("latency_seconds", 0),
            "correct": correct,
            "ground_truth": gt,
            "predicted": pred,
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "total_tokens": r.get("total_tokens"),
        })
    return response_entries, eval_entries, input_type, mode


def build_eval_log(
    eval_entries: List[Dict],
    dataset: str,
    input_type: str,
    mode: str | None = None,
) -> Dict[str, Any]:
    valid = [e for e in eval_entries if e.get("ground_truth")]
    total = len(valid)
    correct = sum(1 for e in valid if e.get("correct"))
    accuracy = (correct / total) if total else 0.0
    latencies = [e.get("latency_seconds", 0) for e in eval_entries]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    # Token usage: average over entries that have total_tokens
    with_usage = [e for e in eval_entries if e.get("total_tokens") is not None]
    n_usage = len(with_usage)
    if n_usage:
        avg_prompt = sum(e["prompt_tokens"] for e in with_usage) / n_usage
        avg_completion = sum(e["completion_tokens"] for e in with_usage) / n_usage
        avg_total = sum(e["total_tokens"] for e in with_usage) / n_usage
    else:
        avg_prompt = avg_completion = avg_total = None

    log = {
        "dataset": dataset,
        "input_type": input_type,
        "mode": mode,
        "num_queries": len(eval_entries),
        "num_with_gt": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "average_latency_seconds": round(avg_latency, 4),
        "average_prompt_tokens": round(avg_prompt, 2) if avg_prompt is not None else None,
        "average_completion_tokens": round(avg_completion, 2) if avg_completion is not None else None,
        "average_total_tokens": round(avg_total, 2) if avg_total is not None else None,
        "per_query": eval_entries,
    }
    return log


def main():
    parser = argparse.ArgumentParser(
        description="Unified generation + evaluation (events_retrieval, frames_retrieval, ava_extract)."
    )
    parser.add_argument(
        "--input-type",
        type=str,
        choices=["events_retrieval", "frames_retrieval", "ava_extract"],
        required=False,
        help="Source of input (not needed with --eval-from-responses)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path: events_retrieval=JSON file; frames_retrieval=output dir; ava_extract=extract base (default AVA/extract_sa_output)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["AVA100", "LVBench"],
        required=True,
        help="Dataset (GT location and for ava_extract data path)",
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default=None,
        help="AVA single-query: video_key (optional; if omitted with ava_extract, run all queries in dataset)",
    )
    parser.add_argument(
        "--question-id",
        type=int,
        default=None,
        help="AVA single-query: question_id (optional; if omitted with ava_extract, run all queries in dataset)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["frames_only", "events_only", "combined"],
        default=None,
        help="AVA only: frames_only | events_only | combined",
    )
    parser.add_argument(
        "--output-responses",
        type=Path,
        default=None,
        help="Path to write responses JSON",
    )
    parser.add_argument(
        "--output-eval",
        type=Path,
        default=None,
        help="Path to write accuracy/latency log JSON",
    )
    parser.add_argument(
        "--eval-from-responses",
        type=Path,
        default=None,
        help="Recompute eval only: load this responses JSON, use current GT, write --output-eval. Skips model calls.",
    )
    parser.add_argument("--model", type=str, default="qwenvl_vllm",
                        help="Model backend (e.g. qwenvl_vllm). Same VLM used for all modes (text-only, frames, or combined).")
    parser.add_argument("--model-type", type=str, default=None,
                        help="Model name on server (e.g. Qwen/Qwen2.5-VL-7B-Instruct-AWQ). Pass with qwenvl_vllm so port is used.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port of the hosted VLM server (used for all modes).")
    parser.add_argument("--vlm-model", type=str, default=None, help="(Alias for --model)")
    parser.add_argument("--vlm-port", type=int, default=None, help="(Alias for --port)")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES_BUDGET)
    parser.add_argument("--max-queries", type=int, default=None,
                        help="If set, run only this many queries (first N) for profiling; reports average_latency_seconds and average_total_tokens on that subset.")
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args()

    # Single VLM: allow deprecated/alias args to override
    if args.vlm_model is not None:
        args.model = args.vlm_model
    if args.vlm_port is not None:
        args.port = args.vlm_port

    project_root = args.project_root or _project_root

    if args.eval_from_responses is not None:
        if not args.eval_from_responses.exists():
            raise FileNotFoundError(f"--eval-from-responses file not found: {args.eval_from_responses}")
        if args.output_eval is None:
            raise ValueError("--eval-from-responses requires --output-eval")
        if args.dataset is None:
            raise ValueError("--eval-from-responses requires --dataset")
        response_entries, eval_entries, input_type, mode_label = eval_from_responses_file(
            args.eval_from_responses, args.dataset, project_root
        )
        eval_log = build_eval_log(eval_entries, args.dataset, input_type, mode_label)
        args.output_eval.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_eval, "w") as f:
            json.dump(eval_log, f, indent=2)
        print(f"Eval log (from existing responses): {args.output_eval}")
        print(f"Accuracy:  {eval_log['accuracy']}  ({eval_log['correct']}/{eval_log['num_with_gt']})")
        return

    if args.input_type is None:
        raise ValueError("--input-type is required (or use --eval-from-responses to recompute eval only)")
    if args.input is None and args.input_type != "ava_extract":
        raise ValueError("--input is required for events_retrieval and frames_retrieval")

    if args.input_type == "events_retrieval":
        if not args.input or not args.input.is_file():
            raise FileNotFoundError("--input must be path to events retrieval JSON")
        response_entries, eval_entries = run_events_retrieval(
            args.input,
            args.dataset,
            project_root,
            args.model,
            args.port,
            args.gpus,
            model_type=args.model_type,
            max_queries=args.max_queries,
        )
        out_resp = args.output_responses or (_script_dir / f"responses_events_{args.dataset}.json")
        out_eval = args.output_eval or (_script_dir / f"eval_events_{args.dataset}.json")
        mode_label = "events_only"
    elif args.input_type == "frames_retrieval":
        if not args.input or not args.input.is_dir():
            raise FileNotFoundError("--input must be path to frames retrieval output dir")
        response_entries, eval_entries = run_frames_retrieval(
            args.input,
            args.dataset,
            project_root,
            args.model,
            args.port,
            args.gpus,
            args.max_frames,
            model_type=args.model_type,
            max_queries=args.max_queries,
        )
        out_resp = args.output_responses or (_script_dir / f"responses_frames_{args.dataset}.json")
        out_eval = args.output_eval or (_script_dir / f"eval_frames_{args.dataset}.json")
        mode_label = "frames_only"
    elif args.input_type == "ava_extract":
        if not args.mode:
            raise ValueError("ava_extract requires --mode (frames_only | events_only | combined)")
        extract_base = args.input or (_project_root / "AVA" / "extract_sa_output")
        if args.video_key is not None and args.question_id is not None:
            response_entry, eval_entry = run_ava_extract(
                extract_base,
                args.dataset,
                args.video_key,
                args.question_id,
                args.mode,
                project_root,
                args.model,
                args.port,
                args.gpus,
                model_type=args.model_type,
            )
            response_entries = [response_entry]
            eval_entries = [eval_entry]
        else:
            response_entries, eval_entries = run_ava_extract_all(
                extract_base,
                args.dataset,
                args.mode,
                project_root,
                args.model,
                args.port,
                args.gpus,
                model_type=args.model_type,
                max_queries=args.max_queries,
            )
        out_resp = args.output_responses or (_script_dir / f"responses_ava_{args.dataset}_{args.mode}.json")
        out_eval = args.output_eval or (_script_dir / f"eval_ava_{args.dataset}_{args.mode}.json")
        mode_label = args.mode
    else:
        raise ValueError(f"Unknown input_type: {args.input_type}")

    # Write responses (no eval fields)
    out_resp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_resp, "w") as f:
        json.dump({"dataset": args.dataset, "input_type": args.input_type, "mode": mode_label, "results": response_entries}, f, indent=2)

    # Build and write eval log
    eval_log = build_eval_log(eval_entries, args.dataset, args.input_type, mode_label)
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    with open(out_eval, "w") as f:
        json.dump(eval_log, f, indent=2)

    print(f"Responses: {out_resp}")
    print(f"Eval log:  {out_eval}")
    print(f"Accuracy:  {eval_log['accuracy']}  ({eval_log['correct']}/{eval_log['num_with_gt']})")
    print(f"Avg latency: {eval_log['average_latency_seconds']} s")


if __name__ == "__main__":
    main()
