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
import glob
import os
import math
import cv2

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
                        for qa in video.get("qa", []):
                            uid = qa.get("uid") or qa.get("question_id")
                            if uid is not None:
                                ans = (qa.get("answer") or "").strip().upper()
                                if ans in ("A", "B", "C", "D"):
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


MAX_FRAMES_BUDGET = 256

# -----------------------------------------------------------------------------
# AVA extract helpers (HH:MM:SS formatting, load one query)
# -----------------------------------------------------------------------------


def _seconds_to_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_events_for_prompt(events: List[Dict]) -> str:
    lines = []
    for i, ev in enumerate(events, start=1):
        dur = ev.get("duration") or [0, 0]
        start_ts = _seconds_to_timestamp(float(dur[0]))
        end_ts = _seconds_to_timestamp(float(dur[1]))
        desc = ev.get("description", "")
        lines.append(f"{i}. [{start_ts} - {end_ts}] {desc}")
    return "\n".join(lines)


def format_frames_and_events(
    frames_data: List[Tuple[int, Image.Image]],
    events: List[Dict],
    fps: float,
) -> str:
    formatted_lines = []
    for i, event in enumerate(events, start=1):
        dur = event.get("duration") or [0, 0]
        event_start, event_end = float(dur[0]), float(dur[1])
        start_ts = _seconds_to_timestamp(event_start)
        end_ts = _seconds_to_timestamp(event_end)
        description = event.get("description", "")
        matching_frames = []
        for idx,(frame_number, _) in enumerate(frames_data):
            t = frame_number / fps
            if event_start <= t <= event_end:
                matching_frames.append(f"Image {idx+1} @ {_seconds_to_timestamp(t)}")
        formatted_lines.append(f"{i}. [{start_ts} - {end_ts}] {description}")
        if matching_frames:
            visual_evidence = ", ".join([f"[{f}]" for f in matching_frames])
            formatted_lines.append(f"   > Visual Evidence: {visual_evidence}")
        else:
            formatted_lines.append("   > Visual Evidence: None in this range")
    return "\n".join(formatted_lines)

def extract_frames_from_events(
    video_path: str,
    events: List[Dict],
    target_fps: float = 0.1,
    work_dir: str = None,
    max_frames: int = None
) -> Tuple[List[Tuple[int, Image.Image]], str, float]:
    """
    Extract frames from video at target_fps rate within event durations.
    
    Args:
        video_path: Path to video file
        events: List of event dictionaries with metadata.duration
        target_fps: Target frames per second to sample
        
    Returns:
        List of tuples (frame_number, PIL.Image)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frame_dir = os.path.join(work_dir, "frames")
    video_fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    frame_numbers = []
    frame_interval = max(1, int(video_fps / target_fps))  # Frames to skip between samples
    for event in events:
        duration = event.get('duration', event.get('metadata', {}).get('duration', None))  # [start_time_sec, end_time_sec]
        if duration is None:
            continue
        frame_numbers.extend(range(int(duration[0]*video_fps), int(duration[1]*video_fps) + 1, int(frame_interval)))
    extracted_frames = []
    for frame_number in frame_numbers:
        frame_path = os.path.join(frame_dir, f"{int(frame_number)}.jpg")
        if os.path.exists(frame_path):
            extracted_frames.append((frame_number, Image.open(frame_path)))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (540, 360))
                pil_image = Image.fromarray(frame_rgb)
                extracted_frames.append((frame_number, pil_image))
    cap.release()
    extracted_frames.sort(key=lambda x: x[0])
    frame_timestamps = [convert_frame_to_time(fn, video_fps) for fn, _ in extracted_frames]
    frame_timestamps_text = ""
    for idx, timestamp in enumerate(frame_timestamps):
        frame_timestamps_text += f"Image {idx+1} @ {timestamp}\n"
    return extracted_frames, frame_timestamps_text, video_fps

def convert_frame_to_time(frame_number, fps=30):
    # format the time as HH:MM:SS
    return f"{frame_number//fps//3600:02d}:{frame_number//fps%3600//60:02d}:{frame_number//fps%60:02d}"




# -----------------------------------------------------------------------------
# Run paths
# -----------------------------------------------------------------------------


def _call_model_get_usage(model, batch_inputs: List[Dict]) -> Tuple[str, Optional[Dict]]:
    """
    Call model.batch_generate_response; return (response_text, usage_dict).
    usage_dict has prompt_tokens, completion_tokens, total_tokens when the model supports return_usage=True.

    We standardize decoding hyperparameters here so that all backends
    (qwenlm, qwenvl, qwenvl_vllm, etc.) run with the same settings
    when their wrappers support these arguments.
    """
    generation_kwargs = {
        "max_new_tokens": 100,
        "temperature": 0.1,
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
    llm_model: str,
    port: int,
    gpus: int,
    model_type: str | None = None,
    mode: str = "events_only",
) -> Tuple[List[Dict], List[Dict]]:
    """Returns (response_entries, eval_entries with latency_seconds, correct)."""
    results_list = json.loads(input_path.read_text())
    if not results_list:
        return [], []

    gt_cache = load_gt_answers(dataset, project_root)
    # Only pass model_type for qwenvl_vllm so init_model passes port (connect to hosted server, no local load)
    mt = model_type if llm_model == "qwenvl_vllm" else None
    llm = init_model(llm_model, num_gpus=gpus, port=port, model_type=mt)

    response_entries = []
    eval_entries = []

    for idx, entry in enumerate(results_list):
        video_key = entry.get("video_key", "")
        question_id = entry.get("question_id")
        question = entry.get("question", "")
        options = entry.get("options", "")
        user_query = f"{question}\n{options}".strip() if options else question
        seed_events = entry.get("seed_events") or []
        video_path = f"{project_root}/datas/{dataset}/videos/{video_key}.mp4"
        work_dirs = f"{project_root}/AVA_cache/{dataset}/*"
        work_dir_path = None
        for work_dir in glob.glob(work_dirs):
            config_path = os.path.join(work_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                if config.get("source_path").split("/")[-1].split(".")[0] == video_key:
                    work_dir_path = work_dir
                    break
        if work_dir_path is None:
            raise ValueError(f"Work directory not found for video_key: {video_key}")
        # Timer: prompt build + model call
        t0 = __import__("time").perf_counter()
        response = ""
        usage: Optional[Dict] = None
        if mode == "frames_only":
            frames_data, frame_timestamps, _ = extract_frames_from_events(video_path, seed_events, target_fps=1/9.0, work_dir=work_dir_path, max_frames=MAX_FRAMES_BUDGET)
            prompt = PROMPTS["frames_only"].format(
                frame_timestamps=frame_timestamps,
                user_query=user_query,
            )
            frames = [img for _, img in frames_data]
            try:
                response, usage = _call_model_get_usage(llm, [{"text": prompt, "video": frames}])
                response = response or ""
            except Exception as e:
                response, usage = "", None
                print(f"AVA frames_only: {e}")
        elif mode == "events_only":
            if not seed_events:
                response = ""
            else:
                event_descriptions = format_events_for_prompt(seed_events)
                prompt = PROMPTS["events_only"].format(
                    event_descriptions=event_descriptions,
                    user_query=user_query,
                )
                try:
                    response, usage = _call_model_get_usage(llm, [{"text": prompt}])
                    response = response or ""
                except Exception as e:
                    response, usage = "", None
                    print(f"AVA events_only: {e}")
        elif mode == "combined":
            frames_data, frame_timestamps, video_fps = extract_frames_from_events(video_path, seed_events, target_fps=1/9.0, work_dir=work_dir_path, max_frames=MAX_FRAMES_BUDGET)
            event_descriptions_with_frames = format_frames_and_events(frames_data, seed_events, video_fps)
            prompt = PROMPTS["frames_and_events_aligned"].format(
                event_descriptions_with_frames=event_descriptions_with_frames,
                user_query=user_query,
            )
            frames = [img for _, img in frames_data]
            try:
                response, usage = _call_model_get_usage(llm, [{"text": prompt, "video": frames}])
                response = response or ""
            except Exception as e:
                response, usage = "", None
                print(f"AVA combined: {e}")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        elapsed = __import__("time").perf_counter() - t0

        response_entries.append({
            "video_key": video_key,
            "question_id": question_id,
            "question": question,
            "options": options,
            "response": (response or "").strip(),
            "mode": mode,
        })

        gt = gt_cache.get((video_key, str(question_id)))
        pred = extract_predicted_answer(response) if response else None
        correct = (gt and pred and pred == gt)
        eval_entries.append({
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
        })
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(results_list)}")

    return response_entries, eval_entries
# -----------------------------------------------------------------------------
# Build eval log (global + per_query) and write outputs
# -----------------------------------------------------------------------------


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
    latencies = [e["latency_seconds"] for e in eval_entries]
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
        "--input",
        type=Path,
        default=None,
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
    parser.add_argument("--llm-model", type=str, default="qwenvl_vllm",
                        help="Use qwenvl_vllm to connect to hosted server (no local load); use qwenlm to load locally")
    parser.add_argument("--model-type", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                        help="Model name on server (e.g. Qwen/Qwen2-VL-7B-Instruct). Pass this with qwenvl_vllm so port is used.")
    parser.add_argument("--llm-port", type=int, default=8000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES_BUDGET)
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root or _project_root
    if not args.input or not args.input.is_file():
        raise FileNotFoundError("--input must be path to events retrieval JSON")
    response_entries, eval_entries = run_events_retrieval(
        args.input,
        args.dataset,
        project_root,
        args.llm_model,
        args.llm_port,
        args.gpus,
        model_type=args.model_type,
        mode=args.mode,
    )
    out_resp = args.output_responses or (_script_dir / f"responses_events_{args.dataset}_{args.mode}.json")
    out_eval = args.output_eval or (_script_dir / f"eval_events_{args.dataset}_{args.mode}.json")
    mode_label = args.mode

    # Write responses (no eval fields)
    out_resp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_resp, "w") as f:
        json.dump({"dataset": args.dataset, "mode": mode_label, "results": response_entries}, f, indent=2)

    # Build and write eval log
    eval_log = build_eval_log(eval_entries, args.dataset, mode_label)
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    with open(out_eval, "w") as f:
        json.dump(eval_log, f, indent=2)

    print(f"Responses: {out_resp}")
    print(f"Eval log:  {out_eval}")
    print(f"Accuracy:  {eval_log['accuracy']}  ({eval_log['correct']}/{eval_log['num_with_gt']})")
    print(f"Avg latency: {eval_log['average_latency_seconds']} s")


if __name__ == "__main__":
    main()
