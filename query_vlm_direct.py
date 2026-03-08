"""
VLM Direct Query - Simplified inference using cached search results
Based on query_SA.py but uses pre-computed sorted_SA_score_result.json
"""
from AVA.ava_direct import AVADirect
import os
import json
import fcntl
import time
import random
from dataset.init_dataset import init_dataset, get_video_idx
from llms.init_model import init_model
import argparse
import logging
from datetime import datetime


def safe_write_json(file_path, data, max_retries=10):
    """Safely write JSON data to file with file locking to prevent race conditions"""
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


def load_all_processed_entries(output_folder, dataset, model, retrieval_mode):
    """Load all processed entries from the main JSON file to check for duplicates"""
    mode_suffix = f"_{retrieval_mode}" if retrieval_mode != "tri_view" else ""
    main_file = f"{output_folder}/query_VLM_direct_{dataset}_{model}{mode_suffix}.json"
    if os.path.exists(main_file):
        return safe_read_json(main_file)
    return []


def setup_profiling_logger(output_folder, process_num):
    """Setup detailed profiling logger"""
    log_file = os.path.join(output_folder, f"profiling_vlm_direct_process_{process_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(f'profiling_vlm_direct_process_{process_num}')
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Direct Query - Uses cached SA results for inference")
    parser.add_argument("--model", required=True, help="Name of the VLM model to use")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--video_id", type=int, default=-1, help="ID of the video to process")
    parser.add_argument("--question_id", type=int, help="ID of the question to process")
    parser.add_argument("--video_start", type=int, help="Start video ID for batch processing (used when video_id=-1)")
    parser.add_argument("--video_end", type=int, help="End video ID for batch processing (used when video_id=-1)")
    parser.add_argument("--process_num", type=int, choices=[1, 2, 3], help="Process number (1, 2, or 3) for separate JSON files")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--retrieval_mode", default="tri_view", choices=["tri_view", "events_only", "entities_only", "features_only"], 
                       help="Retrieval mode: tri_view, events_only, entities_only, or features_only")
    parser.add_argument("--max_frames", type=int, default=256, help="Maximum number of frames to sample")
    parser.add_argument("--batch_video_count", type=int, default=6, help="Number of videos to batch together (for VideoMME)")
    
    args = parser.parse_args()
    
    if args.video_id != -1:
        # Single video processing
        output_folder = "./outputs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        profiler_logger = setup_profiling_logger(output_folder, 0)
        
        # Start overall timing
        overall_start = time.time()
        profiler_logger.info("=== STARTING SINGLE VIDEO PROCESSING (VLM Direct) ===")
        
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
        
        # AVADirect object creation
        ava_start = time.time()
        ava_direct = AVADirect(
            video=video,
            llm_model=vlm,
        )
        ava_end = log_timing(profiler_logger, "AVADirect Object Creation", ava_start)
        
        # Answer generation
        answer_start = time.time()
        final_answer = ava_direct.generate_direct_answer(
            qas[args.question_id]["question"], 
            args.question_id, 
            retrieval_mode=args.retrieval_mode,
            max_frames=args.max_frames
        )
        answer_end = log_timing(profiler_logger, "Direct Answer Generation", answer_start)
        
        # Overall timing
        overall_end = log_timing(profiler_logger, "TOTAL PROCESSING TIME", overall_start)
        
        profiler_logger.info("=== SINGLE VIDEO PROCESSING COMPLETED ===")
        print(final_answer)
    
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
        profiler_logger = setup_profiling_logger(output_folder, args.process_num)
        profiler_logger.info("=== STARTING BATCH PROCESSING (VLM Direct) ===")
        
        # Use separate JSON file for each process and retrieval mode
        mode_suffix = f"_{args.retrieval_mode}" if args.retrieval_mode != "tri_view" else ""
        json_file = f"{output_folder}/query_VLM_direct_{args.dataset}_{args.model}{mode_suffix}_process{args.process_num}.json"
        
        # Load current process results
        if os.path.exists(json_file):
            results = safe_read_json(json_file)
        else:
            results = []
        
        # Load all processed entries from main file to check for duplicates
        profiler_logger.info("STEP - Duplicate Check Loading Start")
        duplicate_check_start = time.time()
        all_processed = load_all_processed_entries(output_folder, args.dataset, args.model, args.retrieval_mode)
        duplicate_check_end = log_timing(profiler_logger, "Duplicate Check Loading", duplicate_check_start, additional_info=f"({len(all_processed)} entries)")
        print(f"Process {args.process_num}: Loaded {len(all_processed)} entries from main file for duplicate checking")
        
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
                
            print(f"Process {args.process_num}: Using custom video range: {start_video} to {end_video}")
        else:
            # Use default range (resume from last processed)
            start_video = results[-1]["video_id"] if results else video_idx[0]
            end_video = video_idx[1]
            print(f"Process {args.process_num}: Using default range: {start_video} to {end_video}")
        
        total_videos = end_video - start_video + 1
        processed_videos = 0
        
        # Start batch processing timing
        batch_start = time.time()
        mode_info = f" (mode: {args.retrieval_mode}, max_frames: {args.max_frames})"
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
                print(f"Process {args.process_num}: Processing video batch {batch_video_ids} ({len(batch_video_ids)} videos)")
                
                # Collect all questions from all videos in this batch
                batch_videos = []
                batch_ava_directs = []
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
                
                # Create AVADirect objects for all videos
                ava_creation_start = time.time()
                for video_id, video in batch_videos:
                    try:
                        ava_direct = AVADirect(video=video, llm_model=vlm)
                        batch_ava_directs.append((video_id, ava_direct))
                    except Exception as e:
                        print(f"  Error creating AVADirect for video {video_id}: {e}")
                        profiler_logger.error(f"Error creating AVADirect for video {video_id}: {e}")
                        continue
                ava_creation_end = log_timing(profiler_logger, f"Batch AVADirect Creation", ava_creation_start, additional_info=f"({len(batch_ava_directs)} videos)")
                
                # Prepare all questions from all videos
                batch_prep_start = time.time()
                batch_inputs = []
                batch_question_metadata = []  # (video_id, question_id, question_data)
                
                for (video_id, ava_direct), (vid, video_info) in zip(batch_ava_directs, batch_video_infos):
                    if video_id != vid:
                        print(f"Warning: Video ID mismatch {video_id} != {vid}")
                        continue
                    
                    qas = video_info["qa"]
                    
                    for question_id in range(len(qas)):
                        # Check if already processed
                        already_processed_main = any(
                            entry["video_id"] == video_id and entry["question_id"] == question_id 
                            for entry in all_processed
                        )
                        already_processed_current = any(
                            entry["video_id"] == video_id and entry["question_id"] == question_id 
                            for entry in results
                        )
                        
                        if already_processed_main or already_processed_current:
                            print(f"  Skipping video {video_id}, question {question_id} (already processed)")
                            profiler_logger.info(f"Video {video_id} Q{question_id}: Skipped (already processed)")
                            continue
                        
                        try:
                            # Prepare data for this question
                            prep_start = time.time()
                            frames, event_descriptions = ava_direct.prepare_question_data(
                                question_id, 
                                retrieval_mode=args.retrieval_mode,
                                max_frames=args.max_frames
                            )
                            prep_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Data Preparation", prep_start)
                            
                            if not frames:
                                print(f"  Warning: No frames for video {video_id}, question {question_id}")
                                profiler_logger.warning(f"Video {video_id} Q{question_id}: No frames extracted")
                                continue
                            
                            # Create prompt
                            question_data = qas[question_id]
                            prompt = ava_direct.create_prompt(question_data["question"], event_descriptions)
                            
                            # Add to batch
                            batch_inputs.append({
                                "text": prompt,
                                "video": frames
                            })
                            batch_question_metadata.append((video_id, question_id, question_data))
                            
                        except Exception as e:
                            print(f"  Error preparing video {video_id}, question {question_id}: {e}")
                            profiler_logger.error(f"Video {video_id} Q{question_id} preparation error: {e}")
                            continue
                
                batch_prep_end = log_timing(profiler_logger, f"Batch Preparation", batch_prep_start, additional_info=f"({len(batch_inputs)} questions)")
                
                if not batch_inputs:
                    print(f"  No valid questions in batch {batch_video_ids}")
                    continue
                
                # Single batched VLM call for all questions from all videos
                vlm_call_start = time.time()
                print(f"  Calling VLM with batch of {len(batch_inputs)} questions from {len(batch_ava_directs)} videos...")
                profiler_logger.info(f"Batch VLM Call: {len(batch_inputs)} questions from {len(batch_ava_directs)} videos")
                
                responses = vlm.batch_generate_response(batch_inputs)
                vlm_call_end = log_timing(profiler_logger, f"Batch VLM Call", vlm_call_start, additional_info=f"({len(batch_inputs)} questions)")
                
                # Process and save responses
                save_start = time.time()
                for (video_id, question_id, question_data), response in zip(batch_question_metadata, responses):
                    result = {
                        "video_id": video_id,
                        "question_id": question_id,
                        "question": question_data["question"],
                        "answer": question_data["answer"],
                        "response": response,
                        "question_type": question_data.get("task_type", ""),
                        "retrieval_mode": args.retrieval_mode,
                        "method": "vlm_direct"
                    }
                    results.append(result)
                    print(f"  Completed video {video_id}, question {question_id}")
                
                # Save results after processing this batch of videos
                if not safe_write_json(json_file, results):
                    print(f"Warning: Failed to save results for batch {batch_video_ids}")
                    profiler_logger.error(f"Failed to save results for batch {batch_video_ids}")
                    break
                
                save_end = log_timing(profiler_logger, f"Batch Save Results", save_start)
                
                processed_videos += len(batch_video_ids)
                batch_total_end = log_timing(profiler_logger, f"Video Batch {batch_video_ids} TOTAL", batch_start_time)
                print(f"Process {args.process_num}: Completed batch {batch_video_ids} ({processed_videos}/{total_videos} videos)")
        
        else:
            # Original per-video batching for AVA100/LVBench
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
                    print(f"Process {args.process_num}: Processing video {video_id} with {len(qas)} questions ({processed_videos}/{total_videos})")
                    profiler_logger.info(f"Video {video_id}: {len(qas)} questions to process")
                    
                except Exception as e:
                    print(f"Process {args.process_num}: Error loading video {video_id}: {e}")
                    profiler_logger.error(f"Error loading video {video_id}: {e}")
                    continue
                
                # STRATEGY A: Batch all questions of this video together
                print(f"  Processing video {video_id} in BATCH MODE ({len(qas)} questions)")
                profiler_logger.info(f"Video {video_id}: Processing {len(qas)} questions in batch mode")
                
                # Check for duplicates and collect questions to process
                batch_duplicate_check_start = time.time()
                questions_to_process = []
                for question_id in range(len(qas)):
                    # Check if already processed in main file
                    already_processed_main = any(
                        entry["video_id"] == video_id and entry["question_id"] == question_id 
                        for entry in all_processed
                    )
                    
                    # Check if already processed in current process file
                    already_processed_current = any(
                        entry["video_id"] == video_id and entry["question_id"] == question_id 
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
                
                print(f"  Processing {len(questions_to_process)} questions in batch")
                
                try:
                    # Create AVADirect object once for this video
                    ava_creation_start = time.time()
                    ava_direct = AVADirect(video=video, llm_model=vlm)
                    ava_creation_end = log_timing(profiler_logger, f"Video {video_id} AVADirect Creation", ava_creation_start)
                    
                    # Prepare batch inputs for all questions
                    batch_prep_start = time.time()
                    batch_inputs = []
                    batch_question_data = []
                    
                    for question_id, question_data in questions_to_process:
                        profiler_logger.info(f"--- Preparing Video {video_id}, Question {question_id} ---")
                        
                        try:
                            # Prepare data for this question
                            prep_start = time.time()
                            frames, event_descriptions = ava_direct.prepare_question_data(
                                question_id, 
                                retrieval_mode=args.retrieval_mode,
                                max_frames=args.max_frames
                            )
                            prep_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Data Preparation", prep_start)
                            
                            if not frames:
                                print(f"    Warning: No frames for question {question_id}, skipping")
                                profiler_logger.warning(f"Video {video_id} Q{question_id}: No frames extracted")
                                continue
                            
                            # Create prompt
                            prompt = ava_direct.create_prompt(question_data["question"], event_descriptions)
                            
                            # Add to batch
                            batch_inputs.append({
                                "text": prompt,
                                "video": frames
                            })
                            batch_question_data.append((question_id, question_data))
                            
                        except Exception as e:
                            print(f"    Error preparing question {question_id}: {e}")
                            profiler_logger.error(f"Video {video_id} Q{question_id} preparation error: {e}")
                            continue
                    
                    batch_prep_end = log_timing(profiler_logger, f"Video {video_id} Batch Preparation", batch_prep_start, additional_info=f"({len(batch_inputs)} questions)")
                    
                    if not batch_inputs:
                        print(f"  No valid questions to process for video {video_id}")
                        continue
                    
                    # Single batched VLM call for all questions
                    vlm_call_start = time.time()
                    print(f"  Calling VLM with batch of {len(batch_inputs)} questions...")
                    profiler_logger.info(f"Video {video_id}: Calling VLM with batch of {len(batch_inputs)} questions")
                    
                    responses = vlm.batch_generate_response(batch_inputs)
                    vlm_call_end = log_timing(profiler_logger, f"Video {video_id} Batch VLM Call", vlm_call_start, additional_info=f"({len(batch_inputs)} questions)")
                    
                    # Process and save responses
                    save_start = time.time()
                    for (question_id, question_data), response in zip(batch_question_data, responses):
                        result = {
                            "video_id": video_id,
                            "question_id": question_id,
                            "question": question_data["question"],
                            "answer": question_data["answer"],
                            "response": response,
                            "question_type": question_data.get("question_type", ""),
                            "retrieval_mode": args.retrieval_mode,
                            "method": "vlm_direct"
                        }
                        results.append(result)
                        print(f"    Completed question {question_id}")
                    
                    # Save results after processing all questions in this video
                    if not safe_write_json(json_file, results):
                        print(f"Warning: Failed to save results for video {video_id}")
                        profiler_logger.error(f"Failed to save results for video {video_id}")
                        break
                    
                    save_end = log_timing(profiler_logger, f"Video {video_id} Save Results", save_start)
                    
                except Exception as e:
                    print(f"Error processing video {video_id} in batch mode: {e}")
                    profiler_logger.error(f"Error processing video {video_id} in batch mode: {e}")
                    continue
                
                # Video completion timing
                video_total_end = log_timing(profiler_logger, f"Video {video_id} TOTAL", video_start_time)
                print(f"Process {args.process_num}: Completed video {video_id}")
        
        # Batch completion timing
        batch_total_end = log_timing(profiler_logger, "BATCH PROCESSING TOTAL", batch_start)
        profiler_logger.info("=== BATCH PROCESSING COMPLETED ===")

