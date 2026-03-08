"""
VLM MultiMode Query - Inference using 7 configuration modes
Based on query_vlm_direct.py but supports all 7 modes
"""
from AVA.ava_multimode import AVAMultiMode
import os
import json
import fcntl
import time
import random
import re
from dataset.init_dataset import init_dataset, get_video_idx
from llms.init_model import init_model
import argparse
import logging
from datetime import datetime


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


def validate_and_clean_results(results, dataset=None):
    """
    Validate and clean results to ensure:
    1. No duplicates for (video_id, question_id) pairs
    2. All questions for each video are present (if dataset is provided)
    
    Args:
        results: List of result dictionaries
        dataset: Optional dataset object to check for missing questions
        
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
    cleaned_results = deduplicate_results(results)
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


def safe_write_json(file_path, data, max_retries=10, require_valid_format=False):
    """Safely write JSON data to file with file locking to prevent race conditions"""
    # Deduplicate before saving
    if isinstance(data, list):
        data = deduplicate_results(data, require_valid_format=require_valid_format)
    
    for attempt in range(max_retries):
        try:
            # Add random delay to reduce collision probability
            if attempt > 0:
                time.sleep(random.uniform(0.1, 0.5))
            
            with open(file_path, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # Non-blocking exclusive lock
                json.dump(data, f, indent=4)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
            return True
        except (IOError, OSError) as e:
            if attempt == max_retries - 1:
                print(f"Failed to write to {file_path} after {max_retries} attempts: {e}")
                return False
            print(f"Attempt {attempt + 1} failed to acquire lock, retrying...")
            continue
    return False


def safe_read_json(file_path):
    """Safely read JSON data from file with file locking"""
    max_retries = 10
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(random.uniform(0.1, 0.5))
            
            with open(file_path, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
            return data
        except (IOError, OSError) as e:
            if attempt == max_retries - 1:
                print(f"Failed to read from {file_path} after {max_retries} attempts: {e}")
                return []
            print(f"Attempt {attempt + 1} failed to acquire lock for reading, retrying...")
            continue
    return []


def load_all_processed_entries(output_folder, dataset, model, retrieval_mode, config_mode):
    """Load all processed entries from the main JSON file to check for duplicates"""
    mode_suffix = f"_{retrieval_mode}" if retrieval_mode != "tri_view" else ""
    main_file = f"{output_folder}/query_VLM_multimode_{dataset}_{model}_mode{config_mode}{mode_suffix}.json"
    if os.path.exists(main_file):
        return safe_read_json(main_file)
    return []


def setup_profiling_logger(output_folder, process_num, config_mode):
    """Setup detailed profiling logger"""
    log_file = os.path.join(output_folder, f"profiling_vlm_multimode_mode{config_mode}_process_{process_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(f'profiling_vlm_multimode_mode{config_mode}_process_{process_num}')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Also capture AVA logger output to the same file
    ava_logger = logging.getLogger('videorag')
    ava_logger.setLevel(logging.INFO)
    
    # Remove existing handlers from AVA logger
    for handler in ava_logger.handlers[:]:
        ava_logger.removeHandler(handler)
    
    # Add the same file handler to AVA logger
    ava_file_handler = logging.FileHandler(log_file)
    ava_file_handler.setLevel(logging.INFO)
    ava_file_handler.setFormatter(formatter)
    ava_logger.addHandler(ava_file_handler)
    
    return logger


def log_timing(logger, step_name, start_time, end_time=None, additional_info=""):
    """Log timing information for a step"""
    if end_time is None:
        end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"TIMING - {step_name}: {duration:.4f} seconds {additional_info}")
    return end_time


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM MultiMode Query - Uses 7 configuration modes")
    parser.add_argument("--model", default="qwenvl", help="Name of the VLM model to use (default: qwenvl, options: qwenvl, qwenlm, gemini, fastvlm, qwenvl_vllm)")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--config_mode", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7],
                       help="Configuration mode: 1=Uniform→Frames, 2=TopK→Frames, 3=Uniform→Events, "
                            "4=TopK→Events, 5=All→Events, 6=Uniform→Frames+Events, 7=TopK→Frames+Events")
    parser.add_argument("--video_id", type=int, default=-1, help="ID of the video to process")
    parser.add_argument("--question_id", type=int, help="ID of the question to process")
    parser.add_argument("--video_start", type=int, help="Start video ID for batch processing (used when video_id=-1)")
    parser.add_argument("--video_end", type=int, help="End video ID for batch processing (used when video_id=-1)")
    parser.add_argument("--process_num", type=int, choices=[1, 2, 3], help="Process number (1, 2, or 3) for separate JSON files")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--retrieval_mode", default="tri_view", choices=["tri_view", "events_only", "entities_only", "features_only"], 
                       help="Retrieval mode: tri_view, events_only, entities_only, or features_only")
    parser.add_argument("--max_frames", type=int, default=256, help="Maximum number of frames to sample (modes 1-4, 6-7)")
    parser.add_argument("--batch_video_count", type=int, default=6, help="Number of videos to batch together (for VideoMME)")
    parser.add_argument("--validate_answer_format", action="store_true", 
                       help="If enabled, only accept answers that are A, B, C, or D. Invalid answers will be removed and re-processed.")
    
    args = parser.parse_args()
    
    # Mode 5 doesn't use max_frames, but we still allow the argument
    if args.config_mode == 5:
        print(f"Note: Mode 5 (All→Events) does not use --max_frames parameter")
    
    if args.video_id != -1:
        # Single video processing
        output_folder = "./outputs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        profiler_logger = setup_profiling_logger(output_folder, 0, args.config_mode)
        
        # Start overall timing
        overall_start = time.time()
        profiler_logger.info(f"=== STARTING SINGLE VIDEO PROCESSING (VLM MultiMode - Mode {args.config_mode}) ===")
        
        # Dataset initialization
        dataset_start = time.time()
        dataset = init_dataset(args.dataset)
        dataset_end = log_timing(profiler_logger, "Dataset Initialization", dataset_start)
        
        # Model initialization
        model_start = time.time()
        vlm = init_model(args.model, args.gpus)
        model_end = log_timing(profiler_logger, "VLM Initialization", model_start)
        
        # Video loading
        video_start = time.time()
        # Use get_video_for_vlm for LVBench and VideoMME, get_video for others
        if args.dataset in ["lvbench", "videomme"]:
            video = dataset.get_video_for_vlm(args.video_id)
        else:
            video = dataset.get_video(args.video_id)
        video_info = dataset.get_video_info(video_id=args.video_id)
        video_end = log_timing(profiler_logger, "Video Loading", video_start)
        
        qas = video_info["qa"]
        
        # AVAMultiMode object creation
        ava_start = time.time()
        ava_multimode = AVAMultiMode(
            video=video,
            llm_model=vlm,
        )
        ava_end = log_timing(profiler_logger, "AVAMultiMode Object Creation", ava_start)
        
        # Answer generation
        answer_start = time.time()
        final_answer, metadata = ava_multimode.generate_answer(
            qas[args.question_id]["question"], 
            args.question_id,
            config_mode=args.config_mode,
            retrieval_mode=args.retrieval_mode,
            max_frames=args.max_frames
        )
        answer_end = log_timing(profiler_logger, "MultiMode Answer Generation", answer_start)
        
        # Overall timing
        overall_end = log_timing(profiler_logger, "TOTAL PROCESSING TIME", overall_start)
        
        # Save result to JSON file
        mode_suffix = f"_{args.retrieval_mode}" if args.retrieval_mode != "tri_view" else ""
        json_file = f"{output_folder}/query_VLM_multimode_{args.dataset}_{args.model}_mode{args.config_mode}{mode_suffix}_single.json"
        
        # Load existing results if file exists
        if os.path.exists(json_file):
            results = safe_read_json(json_file)
        else:
            results = []
        
        # Create result entry (include time_reference for visualization and evaluation)
        qdata = qas[args.question_id]
        result = {
            "video_id": args.video_id,
            "question_id": args.question_id,
            "question": qdata["question"],
            "answer": qdata.get("answer", ""),
            "response": final_answer,
            "question_type": qdata.get("question_type", ""),
            "retrieval_mode": args.retrieval_mode,
            "config_mode": args.config_mode,
            "metadata": metadata,
            "method": "vlm_multimode",
            "time_reference": qdata.get("time_reference", "N/A"),
        }
        results.append(result)
        
        # Save results
        if safe_write_json(json_file, results, require_valid_format=args.validate_answer_format):
            print(f"Results saved to: {json_file}")
        else:
            print(f"Warning: Failed to save results to {json_file}")
        
        profiler_logger.info("=== SINGLE VIDEO PROCESSING COMPLETED ===")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        print(f"Answer: {final_answer}")
    
    else:
        # Batch processing
        # Validate process number
        if args.process_num is None:
            print("Error: --process_num is required for batch processing. Use 1, 2, or 3.")
            exit(1)
        
        output_folder = "./outputs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Setup profiling logger
        profiler_logger = setup_profiling_logger(output_folder, args.process_num, args.config_mode)
        profiler_logger.info(f"=== STARTING BATCH PROCESSING (VLM MultiMode - Mode {args.config_mode}) ===")
        
        # Use separate JSON file for each process, retrieval mode, and config mode
        mode_suffix = f"_{args.retrieval_mode}" if args.retrieval_mode != "tri_view" else ""
        json_file = f"{output_folder}/query_VLM_multimode_{args.dataset}_{args.model}_mode{args.config_mode}{mode_suffix}_process{args.process_num}.json"
        
        # Load current process results
        if os.path.exists(json_file):
            results = safe_read_json(json_file)
            # Deduplicate on load to clean up any existing duplicates
            original_count = len(results)
            results = deduplicate_results(results, require_valid_format=args.validate_answer_format)
            if len(results) < original_count:
                print(f"Process {args.process_num} (Mode {args.config_mode}): Removed {original_count - len(results)} duplicate entries on load")
                # Save deduplicated results back
                safe_write_json(json_file, results, require_valid_format=args.validate_answer_format)
        else:
            results = []
        
        # Load all processed entries from main file to check for duplicates
        profiler_logger.info("STEP - Duplicate Check Loading Start")
        duplicate_check_start = time.time()
        all_processed = load_all_processed_entries(output_folder, args.dataset, args.model, args.retrieval_mode, args.config_mode)
        duplicate_check_end = log_timing(profiler_logger, "Duplicate Check Loading", duplicate_check_start, additional_info=f"({len(all_processed)} entries)")
        print(f"Process {args.process_num} (Mode {args.config_mode}): Loaded {len(all_processed)} entries from main file for duplicate checking")
        
        # Dataset initialization
        profiler_logger.info("STEP - Dataset Initialization Start")
        dataset_start = time.time()
        dataset = init_dataset(args.dataset)
        dataset_end = log_timing(profiler_logger, "Dataset Initialization", dataset_start)
        
        # Model initialization
        profiler_logger.info("STEP - VLM Initialization Start")
        model_start = time.time()
        vlm = init_model(args.model, args.gpus)
        model_end = log_timing(profiler_logger, "VLM Initialization", model_start)
        
        # Video index loading
        profiler_logger.info("STEP - Video Index Loading Start")
        video_idx_start = time.time()
        video_idx = get_video_idx(args.dataset)
        video_idx_end = log_timing(profiler_logger, "Video Index Loading", video_idx_start)
        
        # Determine video range
        if args.video_start is not None and args.video_end is not None:
            # Use custom range
            start_video = args.video_start
            end_video = args.video_end
            
            # Validate range
            if start_video < video_idx[0] or end_video > video_idx[1]:
                print(f"Error: Video range {start_video}-{end_video} is outside valid range {video_idx[0]}-{video_idx[1]}")
                exit(1)
            if start_video > end_video:
                print(f"Error: Start video {start_video} cannot be greater than end video {end_video}")
                exit(1)
                
            print(f"Process {args.process_num} (Mode {args.config_mode}): Using custom video range: {start_video} to {end_video}")
        else:
            # Use default range - always start from beginning and let duplicate detection handle skipping
            # This ensures we catch any manually deleted questions in the middle of the file
            start_video = video_idx[0]
            end_video = video_idx[1]
            print(f"Process {args.process_num} (Mode {args.config_mode}): Processing videos {start_video} to {end_video}")
            if results:
                print(f"Process {args.process_num} (Mode {args.config_mode}): Found {len(results)} existing results - will skip already processed questions")
        
        total_videos = end_video - start_video + 1
        processed_videos = 0
        
        # Start batch processing timing
        batch_start = time.time()
        mode_info = f" (config_mode: {args.config_mode}, retrieval_mode: {args.retrieval_mode}, max_frames: {args.max_frames})"
        profiler_logger.info(f"Starting batch processing: {total_videos} videos from {start_video} to {end_video}{mode_info}")
        
        # VideoMME uses multi-video batching (batch multiple videos together)
        if args.dataset == "videomme":
            # Process videos in batches of batch_video_count
            video_ids = list(range(start_video, end_video + 1))
            batch_size = args.batch_video_count
            
            for batch_start_idx in range(0, len(video_ids), batch_size):
                batch_video_ids = video_ids[batch_start_idx:batch_start_idx + batch_size]
                batch_start_time = time.time()
                profiler_logger.info(f"=== PROCESSING VIDEO BATCH {batch_video_ids} ===")
                print(f"Process {args.process_num} (Mode {args.config_mode}): Processing video batch {batch_video_ids} ({len(batch_video_ids)} videos)")
                
                # Collect all questions from all videos in this batch
                batch_videos = []
                batch_ava_multimodes = []
                batch_video_infos = []
                
                # Load all videos in this batch
                for video_id in batch_video_ids:
                    try:
                        profiler_logger.info(f"STEP - Video {video_id} Loading Start")
                        video_load_start = time.time()
                        video = dataset.get_video_for_vlm(video_id)
                        video_info = dataset.get_video_info(video_id=video_id)
                        video_load_end = log_timing(profiler_logger, f"Video {video_id} Loading", video_load_start)
                        
                        batch_videos.append((video_id, video))
                        batch_video_infos.append((video_id, video_info))
                        
                        print(f"  Loaded video {video_id} with {len(video_info['qa'])} questions")
                        profiler_logger.info(f"Video {video_id}: {len(video_info['qa'])} questions")
                        
                    except Exception as e:
                        print(f"  Error loading video {video_id}: {e}")
                        profiler_logger.error(f"Error loading video {video_id}: {e}")
                        continue
                
                if not batch_videos:
                    print(f"  No valid videos in batch {batch_video_ids}, skipping...")
                    continue
                
                # Create AVAMultiMode objects for all videos
                ava_creation_start = time.time()
                for video_id, video in batch_videos:
                    try:
                        ava_multimode = AVAMultiMode(video=video, llm_model=vlm)
                        batch_ava_multimodes.append((video_id, ava_multimode))
                    except Exception as e:
                        print(f"  Error creating AVAMultiMode for video {video_id}: {e}")
                        profiler_logger.error(f"Error creating AVAMultiMode for video {video_id}: {e}")
                        continue
                ava_creation_end = log_timing(profiler_logger, f"Batch AVAMultiMode Creation", ava_creation_start, additional_info=f"({len(batch_ava_multimodes)} videos)")
                
                # Prepare and process all questions from all videos
                for (video_id, ava_multimode), (vid, video_info) in zip(batch_ava_multimodes, batch_video_infos):
                    if video_id != vid:
                        print(f"Warning: Video ID mismatch {video_id} != {vid}")
                        continue
                    
                    qas = video_info["qa"]
                    
                    for question_id in range(len(qas)):
                        # Check if already processed (with valid Answer in response)
                        already_processed_main = any(
                            entry.get("video_id") == video_id and 
                            entry.get("question_id") == question_id and
                            entry.get("response") is not None and
                            has_valid_answer_in_response(entry.get("response", ""))
                            for entry in all_processed
                        )
                        already_processed_current = any(
                            entry.get("video_id") == video_id and 
                            entry.get("question_id") == question_id and
                            entry.get("response") is not None and
                            has_valid_answer_in_response(entry.get("response", ""))
                            for entry in results
                        )
                        
                        if already_processed_main or already_processed_current:
                            print(f"  Skipping video {video_id}, question {question_id} (already processed)")
                            profiler_logger.info(f"Video {video_id} Q{question_id}: Skipped (already processed)")
                            continue
                        
                        try:
                            # Generate answer
                            gen_start = time.time()
                            question_data = qas[question_id]
                            response, metadata = ava_multimode.generate_answer(
                                question_data["question"],
                                question_id,
                                config_mode=args.config_mode,
                                retrieval_mode=args.retrieval_mode,
                                max_frames=args.max_frames
                            )
                            gen_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Answer Generation", gen_start)
                            
                            if response is None:
                                print(f"  Warning: No response for video {video_id}, question {question_id}")
                                profiler_logger.warning(f"Video {video_id} Q{question_id}: No response generated")
                                continue
                            
                            # Save result (include time_reference for visualization and evaluation)
                            result = {
                                "video_id": video_id,
                                "question_id": question_id,
                                "question": question_data["question"],
                                "answer": question_data["answer"],
                                "response": response,
                                "question_type": question_data.get("task_type", ""),
                                "retrieval_mode": args.retrieval_mode,
                                "config_mode": args.config_mode,
                                "metadata": metadata,
                                "method": "vlm_multimode",
                                "time_reference": question_data.get("time_reference", "N/A"),
                            }
                            results.append(result)
                            print(f"  Completed video {video_id}, question {question_id}")
                            
                        except Exception as e:
                            print(f"  Error processing video {video_id}, question {question_id}: {e}")
                            profiler_logger.error(f"Video {video_id} Q{question_id} processing error: {e}")
                            continue
                
                # Save results after processing this batch of videos
                save_start = time.time()
                if not safe_write_json(json_file, results, require_valid_format=args.validate_answer_format):
                    print(f"Warning: Failed to save results for batch {batch_video_ids}")
                    profiler_logger.error(f"Failed to save results for batch {batch_video_ids}")
                    break
                save_end = log_timing(profiler_logger, f"Batch Save Results", save_start)
                
                processed_videos += len(batch_video_ids)
                batch_total_end = log_timing(profiler_logger, f"Video Batch {batch_video_ids} TOTAL", batch_start_time)
                print(f"Process {args.process_num} (Mode {args.config_mode}): Completed batch {batch_video_ids} ({processed_videos}/{total_videos} videos)")
        
        else:
            # Original per-video processing for AVA100/LVBench
            for video_id in range(start_video, end_video + 1):
                video_start_time = time.time()
                profiler_logger.info(f"=== PROCESSING VIDEO {video_id} ===")
            
                try:
                    # Video loading
                    profiler_logger.info(f"STEP - Video {video_id} Loading Start")
                    video_load_start = time.time()
                    # Use get_video_for_vlm for LVBench, get_video for others
                    if args.dataset == "lvbench":
                        video = dataset.get_video_for_vlm(video_id)
                    else:
                        video = dataset.get_video(video_id)
                    video_info = dataset.get_video_info(video_id=video_id)
                    video_load_end = log_timing(profiler_logger, f"Video {video_id} Loading", video_load_start)
                
                    qas = video_info["qa"]
                    
                    processed_videos += 1
                    print(f"Process {args.process_num} (Mode {args.config_mode}): Processing video {video_id} with {len(qas)} questions ({processed_videos}/{total_videos})")
                    profiler_logger.info(f"Video {video_id}: {len(qas)} questions to process")
                    
                except Exception as e:
                    print(f"Process {args.process_num}: Error loading video {video_id}: {e}")
                    profiler_logger.error(f"Error loading video {video_id}: {e}")
                    continue
                
                # Check for duplicates and collect questions to process
                batch_duplicate_check_start = time.time()
                questions_to_process = []
                for question_id in range(len(qas)):
                    # Check if already processed in main file (with valid Answer in response)
                    already_processed_main = any(
                        entry.get("video_id") == video_id and 
                        entry.get("question_id") == question_id and
                        entry.get("response") is not None and
                        has_valid_answer_in_response(entry.get("response", ""), require_valid_format=args.validate_answer_format)
                        for entry in all_processed
                    )
                    
                    # Check if already processed in current process file (with valid Answer in response)
                    already_processed_current = any(
                        entry.get("video_id") == video_id and 
                        entry.get("question_id") == question_id and
                        entry.get("response") is not None and
                        has_valid_answer_in_response(entry.get("response", ""), require_valid_format=args.validate_answer_format)
                        for entry in results
                    )
                    
                    if not (already_processed_main or already_processed_current):
                        questions_to_process.append((question_id, qas[question_id]))
                    else:
                        print(f"    Skipping question {question_id} (already processed)")
                        profiler_logger.info(f"Video {video_id} Q{question_id}: Skipped (already processed)")
                
                batch_duplicate_check_end = log_timing(profiler_logger, f"Video {video_id} Batch Duplicate Check", batch_duplicate_check_start)
                
                if not questions_to_process:
                    print(f"  All questions for video {video_id} already processed, skipping...")
                    continue
                
                print(f"  Processing {len(questions_to_process)} questions for video {video_id}")
                
                try:
                    # Create AVAMultiMode object once for this video
                    ava_creation_start = time.time()
                    ava_multimode = AVAMultiMode(video=video, llm_model=vlm)
                    ava_creation_end = log_timing(profiler_logger, f"Video {video_id} AVAMultiMode Creation", ava_creation_start)
                    
                    # Process all questions for this video
                    for question_id, question_data in questions_to_process:
                        profiler_logger.info(f"--- Processing Video {video_id}, Question {question_id} ---")
                        
                        try:
                            # Generate answer
                            gen_start = time.time()
                            response, metadata = ava_multimode.generate_answer(
                                question_data["question"],
                                question_id,
                                config_mode=args.config_mode,
                                retrieval_mode=args.retrieval_mode,
                                max_frames=args.max_frames
                            )
                            gen_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Answer Generation", gen_start)
                            
                            if response is None:
                                print(f"    Warning: No response for question {question_id}")
                                profiler_logger.warning(f"Video {video_id} Q{question_id}: No response generated")
                                continue
                            
                            # Save result (include time_reference for visualization and evaluation)
                            result = {
                                "video_id": video_id,
                                "question_id": question_id,
                                "question": question_data["question"],
                                "answer": question_data["answer"],
                                "response": response,
                                "question_type": question_data.get("question_type", ""),
                                "retrieval_mode": args.retrieval_mode,
                                "config_mode": args.config_mode,
                                "metadata": metadata,
                                "method": "vlm_multimode",
                                "time_reference": question_data.get("time_reference", "N/A"),
                            }
                            results.append(result)
                            print(f"    Completed question {question_id}")
                            
                        except Exception as e:
                            print(f"    Error processing question {question_id}: {e}")
                            profiler_logger.error(f"Video {video_id} Q{question_id} processing error: {e}")
                            continue
                    
                    # Save results after processing all questions in this video
                    save_start = time.time()
                    if not safe_write_json(json_file, results, require_valid_format=args.validate_answer_format):
                        print(f"Warning: Failed to save results for video {video_id}")
                        profiler_logger.error(f"Failed to save results for video {video_id}")
                        break
                    save_end = log_timing(profiler_logger, f"Video {video_id} Save Results", save_start)
                    
                except Exception as e:
                    print(f"Error processing video {video_id}: {e}")
                    profiler_logger.error(f"Error processing video {video_id}: {e}")
                    continue
                
                # Video completion timing
                video_total_end = log_timing(profiler_logger, f"Video {video_id} TOTAL", video_start_time)
                print(f"Process {args.process_num} (Mode {args.config_mode}): Completed video {video_id}")
        
        # Final validation and cleanup
        validation_start = time.time()
        print(f"\nProcess {args.process_num} (Mode {args.config_mode}): Running final validation and cleanup...")
        profiler_logger.info("=== FINAL VALIDATION AND CLEANUP ===")
        
        # Load final results
        final_results = safe_read_json(json_file)
        
        # Validate and clean (also remove entries with invalid answer format if flag is set)
        if args.validate_answer_format:
            # Remove entries with invalid answer format
            original_count = len(final_results)
            final_results = [entry for entry in final_results 
                           if has_valid_answer_in_response(entry.get("response", ""), require_valid_format=True)]
            removed_count = original_count - len(final_results)
            if removed_count > 0:
                print(f"Process {args.process_num} (Mode {args.config_mode}): Removed {removed_count} entries with invalid answer format (not A/B/C/D)")
                profiler_logger.info(f"Removed {removed_count} entries with invalid answer format")
        
        cleaned_results, validation_report = validate_and_clean_results(final_results, dataset)
        
        # Save cleaned results
        if safe_write_json(json_file, cleaned_results, require_valid_format=args.validate_answer_format):
            print(f"Process {args.process_num} (Mode {args.config_mode}): Validation complete")
            print(f"  - Original entries: {validation_report['original_count']}")
            print(f"  - Duplicates removed: {validation_report['duplicates_removed']}")
            print(f"  - Final entries: {validation_report['final_count']}")
            if validation_report['missing_questions']:
                print(f"  - Videos with missing questions: {len(validation_report['missing_questions'])}")
                for missing_info in validation_report['missing_questions'][:5]:  # Show first 5
                    print(f"    Video {missing_info['video_id']}: {missing_info['present_questions']}/{missing_info['total_questions']} questions (missing: {missing_info['missing_question_ids'][:5]}...)")
            profiler_logger.info(f"Validation report: {json.dumps(validation_report, indent=2)}")
        else:
            print(f"Warning: Failed to save cleaned results")
            profiler_logger.error("Failed to save cleaned results after validation")
        
        validation_end = log_timing(profiler_logger, "Final Validation and Cleanup", validation_start)
        
        # Batch completion timing
        batch_total_end = log_timing(profiler_logger, "BATCH PROCESSING TOTAL", batch_start)
        profiler_logger.info("=== BATCH PROCESSING COMPLETED ===")

