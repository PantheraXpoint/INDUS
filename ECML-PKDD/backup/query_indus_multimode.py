"""
Query VLM with IndusMultiMode (modes 3–7) using INDUS seed events.

This script is analogous to `query_vlm_multimode.py`, but:
    - Uses `ECML-PKDD/indus_outputs/seed_events_*.json` to obtain
      per-(video, question) seed events and time segments.
    - Uses `IndusMultiMode` (modes 3–7 only) instead of AVAMultiMode.
    - Writes outputs under `ECML-PKDD/outputs`.
"""

import os
import sys
import json
import re
import fcntl
import time
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure project root and ECML-PKDD directory are on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from AVA.utils import logger as ava_logger
from indus_multimode import IndusMultiMode
from dataset.init_dataset import init_dataset, get_video_idx
from llms.init_model import init_model


# ---------------------------------------------------------------------------
# Validation and deduplication helpers
# ---------------------------------------------------------------------------

def has_valid_answer_in_response(response_str, require_valid_format=False):
    """
    Check if response string contains a valid JSON with an "Answer" field
    
    Args:
        response_str: Response string that may contain JSON
        require_valid_format: If True, also check that Answer is A, B, C, or D
        
    Returns:
        bool: True if response has valid "Answer" field, False otherwise
    """
    if not response_str or response_str.strip() == "":
        return False
    
    # Check for error messages (case-insensitive)
    error_patterns = [
        r'internal error',
        r'error happened',
        r'status code',
        r'ResponseType\.',
        r'INPUT_LENGTH_ERROR',
        r'TOKEN_LIMIT',
        r'TIMEOUT',
        r'FAILED',
        r'exception',
        r'traceback'
    ]
    
    response_lower = response_str.lower()
    for pattern in error_patterns:
        if re.search(pattern, response_lower, re.IGNORECASE):
            return False
    
    try:
        # Try to extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return False
        
        # Parse JSON
        parsed = json.loads(json_str)
        
        # Check if "Answer" field exists and is not empty
        answer = parsed.get("Answer") or parsed.get("answer")
        if answer is None or (isinstance(answer, str) and answer.strip() == ""):
            return False
        
        # If require_valid_format is True, check that answer is A, B, C, or D
        if require_valid_format:
            answer_clean = str(answer).strip().upper()
            if answer_clean not in ['A', 'B', 'C', 'D']:
                return False
        
        return True
    except (json.JSONDecodeError, AttributeError, KeyError):
        return False


def deduplicate_results(results, require_valid_format=False):
    """
    Remove duplicate entries, keeping the most recent one for each (video_id, question_id) pair.
    If multiple entries exist, prefer the one with a valid Answer in response.
    
    Args:
        results: List of result dictionaries
        require_valid_format: If True, also check that Answer is A, B, C, or D
        
    Returns:
        Deduplicated list of results
    """
    # Use a dictionary to track the best entry for each (video_id, question_id) pair
    seen = {}
    
    for entry in results:
        video_id = entry.get("video_id")
        question_id = entry.get("question_id")
        
        if video_id is None or question_id is None:
            continue
        
        key = (video_id, question_id)
        
        if key not in seen:
            # First time seeing this pair, keep it
            seen[key] = entry
        else:
            # Already seen this pair, decide which one to keep
            existing = seen[key]
            existing_has_answer = has_valid_answer_in_response(existing.get("response", ""), require_valid_format=require_valid_format)
            new_has_answer = has_valid_answer_in_response(entry.get("response", ""), require_valid_format=require_valid_format)
            
            # Prefer entry with valid Answer, or the newer one if both have/don't have Answer
            if new_has_answer and not existing_has_answer:
                # New entry has Answer, existing doesn't - replace
                seen[key] = entry
            elif not new_has_answer and existing_has_answer:
                # Existing has Answer, new doesn't - keep existing
                pass
            else:
                # Both have or both don't have Answer - keep the newer one (later in list)
                seen[key] = entry
    
    # Return as list, maintaining order
    return list(seen.values())


def validate_and_clean_results(results, dataset=None, require_valid_format=False):
    """
    Validate and clean results to ensure:
    1. No duplicates for (video_id, question_id) pairs
    2. All questions for each video are present (if dataset is provided)
    
    Args:
        results: List of result dictionaries
        dataset: Optional dataset object to check for missing questions
        require_valid_format: If True, also check that Answer is A, B, C, or D
        
    Returns:
        Tuple of (cleaned_results, validation_report)
    """
    validation_report = {
        "original_count": len(results),
        "duplicates_removed": 0,
        "missing_questions": [],
        "final_count": 0
    }
    
    # Step 1: Remove duplicates
    original_count = len(results)
    cleaned_results = deduplicate_results(results, require_valid_format=require_valid_format)
    validation_report["duplicates_removed"] = original_count - len(cleaned_results)
    
    # Step 2: Check for missing questions if dataset is provided
    if dataset is not None:
        # Group by video_id
        video_questions = {}
        for entry in cleaned_results:
            video_id = entry.get("video_id")
            question_id = entry.get("question_id")
            if video_id is not None and question_id is not None:
                if video_id not in video_questions:
                    video_questions[video_id] = set()
                video_questions[video_id].add(question_id)
        
        # Check each video for missing questions
        for video_id, question_ids in video_questions.items():
            try:
                if hasattr(dataset, 'get_video_info'):
                    video_info = dataset.get_video_info(video_id=video_id)
                    qas = video_info.get("qa", [])
                    total_questions = len(qas)
                    present_questions = len(question_ids)
                    
                    if present_questions < total_questions:
                        missing = set(range(total_questions)) - question_ids
                        validation_report["missing_questions"].append({
                            "video_id": video_id,
                            "total_questions": total_questions,
                            "present_questions": present_questions,
                            "missing_question_ids": sorted(list(missing))
                        })
            except Exception as e:
                # Skip if we can't check this video
                pass
    
    validation_report["final_count"] = len(cleaned_results)
    return cleaned_results, validation_report


# ---------------------------------------------------------------------------
# JSON I/O helpers (copied from query_vlm_multimode with minimal changes)
# ---------------------------------------------------------------------------

def safe_write_json(file_path, data, max_retries=10, require_valid_format=False):
    """Safely write JSON data to file with file locking to prevent race conditions"""
    # Deduplicate before saving
    if isinstance(data, list):
        data = deduplicate_results(data, require_valid_format=require_valid_format)
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(random.uniform(0.1, 0.5))

            with open(file_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                json.dump(data, f, indent=4)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except (IOError, OSError) as e:
            if attempt == max_retries - 1:
                print(
                    f"Failed to write to {file_path} after "
                    f"{max_retries} attempts: {e}"
                )
                return False
            print(
                f"Attempt {attempt + 1} failed to acquire lock for write, "
                f"retrying..."
            )
    return False


def safe_read_json(file_path):
    """Safely read JSON data from file with file locking."""
    max_retries = 10
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(random.uniform(0.1, 0.5))

            with open(file_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
        except (IOError, OSError) as e:
            if attempt == max_retries - 1:
                print(
                    f"Failed to read from {file_path} after "
                    f"{max_retries} attempts: {e}"
                )
                return []
            print(
                f"Attempt {attempt + 1} failed to acquire lock for read, "
                f"retrying..."
            )
    return []


def setup_profiling_logger(output_folder, process_num, config_mode):
    """Setup detailed profiling logger for IndusMultiMode."""
    log_file = os.path.join(
        output_folder,
        f"profiling_indus_multimode_mode{config_mode}_process_"
        f"{process_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    plogger = logging.getLogger(
        f"profiling_indus_multimode_mode{config_mode}_process_{process_num}"
    )
    plogger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in plogger.handlers[:]:
        plogger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    plogger.addHandler(file_handler)

    # Also capture AVA logger output to the same file
    ava_logger.setLevel(logging.INFO)
    for handler in ava_logger.handlers[:]:
        ava_logger.removeHandler(handler)

    ava_file_handler = logging.FileHandler(log_file)
    ava_file_handler.setLevel(logging.INFO)
    ava_file_handler.setFormatter(formatter)
    ava_logger.addHandler(ava_file_handler)

    return plogger


def log_timing(plogger, step_name, start_time, end_time=None, additional_info=""):
    """Log timing information for a step."""
    if end_time is None:
        end_time = time.time()
    duration = end_time - start_time
    plogger.info(
        f"TIMING - {step_name}: {duration:.4f} seconds {additional_info}"
    )
    return end_time


# ---------------------------------------------------------------------------
# INDUS seed events loading / indexing
# ---------------------------------------------------------------------------

def _load_seed_events_for_dataset(
    dataset_name: str, indus_outputs_dir: str
) -> List[Dict[str, Any]]:
    """
    Load INDUS seed events JSON for the given dataset.

    Expects file names:
        - AVA100:  seed_events_AVA100.json
        - LVBench: seed_events_LVBench.json
    """
    upper_name = dataset_name.upper()
    if upper_name == "AVA100":
        fname = "seed_events_AVA100.json"
    elif upper_name == "LVBENCH":
        fname = "seed_events_LVBench.json"
    else:
        raise ValueError(
            f"Unsupported dataset for INDUS seed events: {dataset_name}"
        )

    path = os.path.join(indus_outputs_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"INDUS seed events file not found: {path}"
        )

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected list in INDUS seed events file: {path}"
        )
    return data


def _build_seed_index(
    seed_events: List[Dict[str, Any]]
) -> Dict[Tuple[str, str, int], List[Dict[str, Any]]]:
    """
    Build index: (dataset, video_key, question_id) -> list of seed events.

    The seed_events JSON from INDUS stores entries at the top level; each
    entry contains:
        - "dataset": "AVA100" or "LVBench"
        - "video_key": ...
        - "question_id": int
        - "seed_events": [...]
    """
    index: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for entry in seed_events:
        dataset = str(entry.get("dataset", "")).upper()
        video_key = str(entry.get("video_key", ""))
        qid = entry.get("question_id")
        seeds = entry.get("seed_events", [])
        if not video_key or qid is None:
            continue
        key = (dataset, video_key, int(qid))
        index[key] = seeds
    return index


def _get_seed_events_for_video_question(
    seed_index: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    dataset_name: str,
    video_key: str,
    question_id: int,
) -> List[Dict[str, Any]]:
    """Lookup INDUS seed events for (dataset, video_key, question_id)."""
    key = (dataset_name.upper(), video_key, int(question_id))
    return seed_index.get(key, [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "VLM IndusMultiMode Query - Uses INDUS seed events and "
            "supports configuration modes 3–7"
        )
    )
    parser.add_argument(
        "--model",
        default="qwenvl",
        help=(
            "Name of the VLM model to use "
            "(default: qwenvl, options: qwenvl, qwenlm, gemini, fastvlm, "
            "qwenvl_vllm)"
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Name of the dataset (ava100 or lvbench)",
    )
    parser.add_argument(
        "--config_mode",
        type=int,
        required=True,
        choices=[3, 4, 5, 6, 7],
        help=(
            "Configuration mode: "
            "3=Uniform→Events, 4=TopK→Events, "
            "5=All→Events, 6=Uniform→Frames+Events, "
            "7=TopK→Frames+Events"
        ),
    )
    parser.add_argument(
        "--video_id",
        type=int,
        default=-1,
        help="ID of the video to process (1-based); -1 for batch processing",
    )
    parser.add_argument(
        "--question_id",
        type=int,
        help="ID of the question to process (index in video_info['qa'])",
    )
    parser.add_argument(
        "--video_start",
        type=int,
        help="Start video ID for batch processing (used when video_id=-1)",
    )
    parser.add_argument(
        "--video_end",
        type=int,
        help="End video ID for batch processing (used when video_id=-1)",
    )
    parser.add_argument(
        "--process_num",
        type=int,
        choices=[1, 2, 3],
        help="Process number (1,2,3) for separate JSON files (batch only)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--retrieval_mode",
        default="tri_view",
        choices=["tri_view", "events_only", "entities_only", "features_only"],
        help="Retrieval mode (kept for logging compatibility)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=256,
        help="Maximum number of frames to sample (modes 3,4,6,7)",
    )
    parser.add_argument(
        "--batch_video_count",
        type=int,
        default=6,
        help="Number of videos to batch together (for VideoMME)",
    )
    parser.add_argument(
        "--validate_answer_format",
        action="store_true",
        help=(
            "If enabled, only accept answers that are A, B, C, or D. "
            "Invalid answers will be removed and re-processed."
        ),
    )

    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    if dataset_name not in {"ava100", "lvbench"}:
        raise ValueError(
            f"Unsupported dataset for IndusMultiMode: {args.dataset}. "
            f"Expected 'ava100' or 'lvbench'."
        )

    # Output directories
    base_output_folder = os.path.join("ECML-PKDD", "outputs")
    if not os.path.exists(base_output_folder):
        os.makedirs(base_output_folder)

    # Single-video mode
    if args.video_id != -1:
        profiler_logger = setup_profiling_logger(
            base_output_folder, 0, args.config_mode
        )

        overall_start = time.time()
        profiler_logger.info(
            f"=== STARTING SINGLE VIDEO PROCESSING "
            f"(IndusMultiMode - Mode {args.config_mode}) ==="
        )

        # Dataset
        dataset_start = time.time()
        dataset = init_dataset(dataset_name)
        dataset_end = log_timing(
            profiler_logger,
            "Dataset Initialization",
            dataset_start,
        )

        # Load INDUS seed events index
        indus_outputs_dir = os.path.join("ECML-PKDD", "lvbench_retrieval")
        seed_events_list = _load_seed_events_for_dataset(
            "AVA100" if dataset_name == "ava100" else "LVBench",
            indus_outputs_dir,
        )
        seed_index = _build_seed_index(seed_events_list)

        # Model
        model_start = time.time()
        vlm = init_model(args.model, args.gpus)
        model_end = log_timing(
            profiler_logger,
            "VLM Initialization",
            model_start,
        )

        # Video
        video_start = time.time()
        if dataset_name == "lvbench":
            video = dataset.get_video_for_vlm(args.video_id)
        else:
            video = dataset.get_video(args.video_id)
        video_info = dataset.get_video_info(video_id=args.video_id)
        video_end = log_timing(
            profiler_logger,
            "Video Loading",
            video_start,
        )

        qas = video_info["qa"]
        if args.question_id is None:
            raise ValueError(
                "In single-video mode, --question_id is required for IndusMultiMode."
            )
        if args.question_id < 0 or args.question_id >= len(qas):
            raise ValueError(
                f"Invalid question_id {args.question_id} for video "
                f"{args.video_id} (has {len(qas)} questions)"
            )

        # Determine video_key and seed question_id
        if dataset_name == "ava100":
            video_key = video_info.get("video_key", "")
            seed_qid = qas[args.question_id].get(
                "question_id", args.question_id
            )
            indus_dataset_tag = "AVA100"
        else:  # lvbench
            video_key = video_info.get("key") or video_info.get("video_key", "")
            seed_qid = qas[args.question_id].get(
                "uid", qas[args.question_id].get(
                    "question_id", args.question_id
                )
            )
            indus_dataset_tag = "LVBench"

        if not video_key:
            raise ValueError(
                f"Could not determine video_key for dataset={dataset_name}, "
                f"video_id={args.video_id}"
            )

        # Output file path (used for "already processed" skip logic)
        mode_suffix = (
            f"_{args.retrieval_mode}"
            if args.retrieval_mode != "tri_view"
            else ""
        )
        json_file = os.path.join(
            base_output_folder,
            f"query_IndusMultimode_{dataset_name}_{args.model}_"
            f"mode{args.config_mode}{mode_suffix}_single.json",
        )

        # Skip if this (video_id, question_id) is already processed with a valid Answer
        if os.path.exists(json_file):
            existing_results = safe_read_json(json_file)
            already_processed = any(
                entry.get("video_id") == args.video_id
                and entry.get("question_id") == args.question_id
                and has_valid_answer_in_response(
                    entry.get("response", ""),
                    require_valid_format=args.validate_answer_format,
                )
                for entry in existing_results
            )
            if already_processed:
                print(
                    f"Skipping video {args.video_id}, question {args.question_id} "
                    f"(already processed)"
                )
                raise SystemExit(0)

        # Build seed_events_by_question for this video
        seed_events_for_question = _get_seed_events_for_video_question(
            _build_seed_index(seed_events_list),
            indus_dataset_tag,
            video_key,
            seed_qid,
        )
        seed_events_by_question = {
            int(seed_qid): seed_events_for_question
        }

        # Create IndusMultiMode object
        ava_start = time.time()
        indus_multimode = IndusMultiMode(
            video=video,
            llm_model=vlm,
            seed_events_by_question=seed_events_by_question,
        )
        ava_end = log_timing(
            profiler_logger,
            "IndusMultiMode Object Creation",
            ava_start,
        )

        # Answer generation
        answer_start = time.time()
        qdata = qas[args.question_id]
        final_answer, metadata = indus_multimode.generate_answer(
            qdata["question"],
            int(seed_qid),
            config_mode=args.config_mode,
            retrieval_mode=args.retrieval_mode,
            max_frames=args.max_frames,
        )
        answer_end = log_timing(
            profiler_logger,
            "IndusMultiMode Answer Generation",
            answer_start,
        )

        overall_end = log_timing(
            profiler_logger,
            "TOTAL PROCESSING TIME",
            overall_start,
        )

        # Save result
        results = safe_read_json(json_file) if os.path.exists(json_file) else []

        result_entry = {
            "video_id": args.video_id,
            "question_id": args.question_id,
            "seed_question_id": int(seed_qid),
            "video_key": video_key,
            "question": qdata["question"],
            "answer": qdata.get("answer", ""),
            "response": final_answer,
            "question_type": qdata.get("question_type", ""),
            "retrieval_mode": args.retrieval_mode,
            "config_mode": args.config_mode,
            "metadata": metadata,
            "method": "indus_multimode",
            "time_reference": qdata.get("time_reference", "N/A"),
        }
        results.append(result_entry)

        if safe_write_json(json_file, results, require_valid_format=args.validate_answer_format):
            print(f"Results saved to: {json_file}")
        else:
            print(f"Warning: Failed to save results to {json_file}")

        profiler_logger.info("=== SINGLE VIDEO PROCESSING COMPLETED ===")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        print(f"Answer: {final_answer}")

    else:
        # Batch processing (simplified, single-process)
        if args.process_num is None:
            print(
                "Error: --process_num is required for batch processing. "
                "Use 1, 2, or 3."
            )
            raise SystemExit(1)

        profiler_logger = setup_profiling_logger(
            base_output_folder, args.process_num, args.config_mode
        )
        profiler_logger.info(
            f"=== STARTING BATCH PROCESSING (IndusMultiMode - Mode "
            f"{args.config_mode}) ==="
        )

        # Use per-process JSON output file
        mode_suffix = (
            f"_{args.retrieval_mode}"
            if args.retrieval_mode != "tri_view"
            else ""
        )
        json_file = os.path.join(
            base_output_folder,
            f"query_IndusMultimode_{dataset_name}_{args.model}_"
            f"mode{args.config_mode}{mode_suffix}_process{args.process_num}.json",
        )

        if os.path.exists(json_file):
            results = safe_read_json(json_file)
        else:
            results = []

        # Deduplicate on load and build a fast skip set for already-processed questions
        original_count = len(results)
        results = deduplicate_results(
            results, require_valid_format=args.validate_answer_format
        )
        if len(results) < original_count:
            print(
                f"Process {args.process_num} (Mode {args.config_mode}): "
                f"Removed {original_count - len(results)} duplicate entries on load"
            )
            safe_write_json(
                json_file,
                results,
                require_valid_format=args.validate_answer_format,
            )

        processed_valid = set()
        for entry in results:
            vid = entry.get("video_id")
            qid = entry.get("question_id")
            if vid is None or qid is None:
                continue
            if has_valid_answer_in_response(
                entry.get("response", ""),
                require_valid_format=args.validate_answer_format,
            ):
                processed_valid.add((int(vid), int(qid)))

        # Dataset initialization
        dataset_start = time.time()
        dataset = init_dataset(dataset_name)
        dataset_end = log_timing(
            profiler_logger,
            "Dataset Initialization",
            dataset_start,
        )

        # Load INDUS seed events index once
        indus_outputs_dir = os.path.join("ECML-PKDD", "lvbench_retrieval")
        seed_events_list = _load_seed_events_for_dataset(
            "AVA100" if dataset_name == "ava100" else "LVBench",
            indus_outputs_dir,
        )
        seed_index = _build_seed_index(seed_events_list)

        # Model initialization
        model_start = time.time()
        vlm = init_model(args.model, args.gpus)
        model_end = log_timing(
            profiler_logger,
            "VLM Initialization",
            model_start,
        )

        # Video index range
        video_idx = get_video_idx(dataset_name)
        if args.video_start is not None and args.video_end is not None:
            start_video = args.video_start
            end_video = args.video_end
        else:
            start_video = video_idx[0]
            end_video = video_idx[1]

        if start_video < video_idx[0] or end_video > video_idx[1]:
            print(
                f"Error: Video range {start_video}-{end_video} is outside "
                f"valid range {video_idx[0]}-{video_idx[1]}"
            )
            raise SystemExit(1)
        if start_video > end_video:
            print(
                f"Error: Start video {start_video} cannot be greater than "
                f"end video {end_video}"
            )
            raise SystemExit(1)

        total_videos = end_video - start_video + 1
        processed_videos = 0

        batch_start = time.time()
        profiler_logger.info(
            f"Starting batch processing: {total_videos} videos from "
            f"{start_video} to {end_video} "
            f"(config_mode: {args.config_mode}, retrieval_mode: "
            f"{args.retrieval_mode}, max_frames: {args.max_frames})"
        )

        for video_id in range(start_video, end_video + 1):
            video_start_time = time.time()
            profiler_logger.info(f"=== PROCESSING VIDEO {video_id} ===")

            try:
                load_start = time.time()
                if dataset_name == "lvbench":
                    video = dataset.get_video_for_vlm(video_id)
                else:
                    video = dataset.get_video(video_id)
                video_info = dataset.get_video_info(video_id=video_id)
                load_end = log_timing(
                    profiler_logger,
                    f"Video {video_id} Loading",
                    load_start,
                )

                qas = video_info["qa"]
                processed_videos += 1
                print(
                    f"Process {args.process_num} (Mode {args.config_mode}): "
                    f"Processing video {video_id} with {len(qas)} questions "
                    f"({processed_videos}/{total_videos})"
                )
                profiler_logger.info(
                    f"Video {video_id}: {len(qas)} questions to process"
                )
            except Exception as e:
                print(
                    f"Process {args.process_num}: Error loading video "
                    f"{video_id}: {e}"
                )
                profiler_logger.error(
                    f"Error loading video {video_id}: {e}"
                )
                continue

            # Determine video_key and dataset tag
            if dataset_name == "ava100":
                video_key = video_info.get("video_key", "")
                indus_dataset_tag = "AVA100"
            else:
                video_key = video_info.get("key") or video_info.get(
                    "video_key", ""
                )
                indus_dataset_tag = "LVBench"

            if not video_key:
                print(
                    f"Warning: Could not determine video_key for video "
                    f"{video_id}, skipping..."
                )
                profiler_logger.error(
                    f"Video {video_id}: Missing video_key, skipped"
                )
                continue

            # Prepare seed_events_by_question for this video
            seed_events_by_question: Dict[int, List[Dict[str, Any]]] = {}
            for local_qid, qdata in enumerate(qas):
                if dataset_name == "ava100":
                    seed_qid = qdata.get("question_id", local_qid)
                else:
                    seed_qid = qdata.get(
                        "uid",
                        qdata.get("question_id", local_qid),
                    )
                seeds = _get_seed_events_for_video_question(
                    seed_index,
                    indus_dataset_tag,
                    video_key,
                    seed_qid,
                )
                if seeds:
                    seed_events_by_question[int(seed_qid)] = seeds

            if not seed_events_by_question:
                print(
                    f"  No INDUS seed events for video {video_id}, skipping..."
                )
                profiler_logger.warning(
                    f"Video {video_id}: No INDUS seed events found"
                )
                continue

            # Create IndusMultiMode object once per video
            try:
                ava_start = time.time()
                indus_multimode = IndusMultiMode(
                    video=video,
                    llm_model=vlm,
                    seed_events_by_question=seed_events_by_question,
                )
                ava_end = log_timing(
                    profiler_logger,
                    f"Video {video_id} IndusMultiMode Creation",
                    ava_start,
                )
            except Exception as e:
                print(
                    f"  Error creating IndusMultiMode for video {video_id}: {e}"
                )
                profiler_logger.error(
                    f"Error creating IndusMultiMode for video {video_id}: {e}"
                )
                continue

            # Process all questions
            for local_qid, qdata in enumerate(qas):
                # Skip if already processed with a valid Answer in existing results
                if (int(video_id), int(local_qid)) in processed_valid:
                    print(
                        f"    Skipping question {local_qid} (already processed)"
                    )
                    profiler_logger.info(
                        f"Video {video_id} Q{local_qid}: Skipped (already processed)"
                    )
                    continue

                if dataset_name == "ava100":
                    seed_qid = qdata.get("question_id", local_qid)
                else:
                    seed_qid = qdata.get(
                        "uid",
                        qdata.get("question_id", local_qid),
                    )

                if int(seed_qid) not in seed_events_by_question:
                    print(
                        f"    Skipping question {local_qid} "
                        f"(no INDUS seeds for seed_qid={seed_qid})"
                    )
                    profiler_logger.info(
                        f"Video {video_id} Q{local_qid}: Skipped "
                        f"(no INDUS seeds)"
                    )
                    continue

                profiler_logger.info(
                    f"--- Processing Video {video_id}, Question {local_qid} "
                    f"(seed_qid={seed_qid}) ---"
                )

                try:
                    gen_start = time.time()
                    response, metadata = indus_multimode.generate_answer(
                        qdata["question"],
                        int(seed_qid),
                        config_mode=args.config_mode,
                        retrieval_mode=args.retrieval_mode,
                        max_frames=args.max_frames,
                    )
                    gen_end = log_timing(
                        profiler_logger,
                        f"Video {video_id} Q{local_qid} Answer Generation",
                        gen_start,
                    )

                    if response is None:
                        print(
                            f"    Warning: No response for question "
                            f"{local_qid}"
                        )
                        profiler_logger.warning(
                            f"Video {video_id} Q{local_qid}: "
                            f"No response generated"
                        )
                        continue

                    result_entry = {
                        "video_id": video_id,
                        "question_id": local_qid,
                        "seed_question_id": int(seed_qid),
                        "video_key": video_key,
                        "question": qdata["question"],
                        "answer": qdata.get("answer", ""),
                        "response": response,
                        "question_type": qdata.get(
                            "question_type",
                            qdata.get("task_type", ""),
                        ),
                        "retrieval_mode": args.retrieval_mode,
                        "config_mode": args.config_mode,
                        "metadata": metadata,
                        "method": "indus_multimode",
                        "time_reference": qdata.get(
                            "time_reference", "N/A"
                        ),
                    }
                    results.append(result_entry)
                    # Mark as processed for this run as well (avoid duplicates if rerun logic hits)
                    if has_valid_answer_in_response(
                        response,
                        require_valid_format=args.validate_answer_format,
                    ):
                        processed_valid.add((int(video_id), int(local_qid)))
                    print(
                        f"    Completed question {local_qid} "
                        f"(seed_qid={seed_qid})"
                    )
                except Exception as e:
                    print(
                        f"    Error processing question {local_qid}: {e}"
                    )
                    profiler_logger.error(
                        f"Video {video_id} Q{local_qid} processing error: {e}"
                    )
                    continue

            # Save results after each video
            save_start = time.time()
            if not safe_write_json(json_file, results, require_valid_format=args.validate_answer_format):
                print(
                    f"Warning: Failed to save results for video {video_id}"
                )
                profiler_logger.error(
                    f"Failed to save results for video {video_id}"
                )
                break
            save_end = log_timing(
                profiler_logger,
                f"Video {video_id} Save Results",
                save_start,
            )

            video_end_time = log_timing(
                profiler_logger,
                f"Video {video_id} TOTAL",
                video_start_time,
            )
            print(
                f"Process {args.process_num} (Mode {args.config_mode}): "
                f"Completed video {video_id}"
            )

        # Final validation and cleanup
        validation_start = time.time()
        print(
            f"\nProcess {args.process_num} (Mode {args.config_mode}): "
            f"Running final validation and cleanup..."
        )
        profiler_logger.info("=== FINAL VALIDATION AND CLEANUP ===")
        
        # Load final results
        final_results = safe_read_json(json_file)
        
        # Validate and clean (also remove entries with invalid answer format if flag is set)
        if args.validate_answer_format:
            # Remove entries with invalid answer format
            original_count = len(final_results)
            final_results = [
                entry for entry in final_results
                if has_valid_answer_in_response(
                    entry.get("response", ""), require_valid_format=True
                )
            ]
            removed_count = original_count - len(final_results)
            if removed_count > 0:
                print(
                    f"Process {args.process_num} (Mode {args.config_mode}): "
                    f"Removed {removed_count} entries with invalid answer format "
                    f"(not A/B/C/D)"
                )
                profiler_logger.info(
                    f"Removed {removed_count} entries with invalid answer format"
                )
        
        cleaned_results, validation_report = validate_and_clean_results(
            final_results, dataset, require_valid_format=args.validate_answer_format
        )
        
        # Save cleaned results
        if safe_write_json(json_file, cleaned_results, require_valid_format=args.validate_answer_format):
            print(f"Process {args.process_num} (Mode {args.config_mode}): Validation complete")
            print(f"  - Original entries: {validation_report['original_count']}")
            print(f"  - Duplicates removed: {validation_report['duplicates_removed']}")
            print(f"  - Final entries: {validation_report['final_count']}")
            if validation_report['missing_questions']:
                print(
                    f"  - Videos with missing questions: "
                    f"{len(validation_report['missing_questions'])}"
                )
                for missing_info in validation_report['missing_questions'][:5]:  # Show first 5
                    print(
                        f"    Video {missing_info['video_id']}: "
                        f"{missing_info['present_questions']}/"
                        f"{missing_info['total_questions']} questions "
                        f"(missing: {missing_info['missing_question_ids'][:5]}...)"
                    )
            profiler_logger.info(
                f"Validation report: {json.dumps(validation_report, indent=2)}"
            )
        else:
            print(f"Warning: Failed to save cleaned results")
            profiler_logger.error("Failed to save cleaned results after validation")
        
        validation_end = log_timing(
            profiler_logger,
            "Final Validation and Cleanup",
            validation_start,
        )
        
        batch_end = log_timing(
            profiler_logger,
            "BATCH PROCESSING TOTAL",
            batch_start,
        )
        profiler_logger.info("=== BATCH PROCESSING COMPLETED ===")


