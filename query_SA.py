from AVA.ava import AVA
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


def load_all_processed_entries(output_folder, dataset, model):
    """Load all processed entries from the main JSON file to check for duplicates"""
    main_file = f"{output_folder}/query_SA_{dataset}_{model}.json"
    if os.path.exists(main_file):
        return safe_read_json(main_file)
    return []


def setup_profiling_logger(output_folder, process_num):
    """Setup detailed profiling logger"""
    log_file = os.path.join(output_folder, f"profiling_process_{process_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(f'profiling_process_{process_num}')
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

def get_questions_folder_name(retrieval_mode):
    """Get the appropriate questions folder name based on retrieval mode"""
    folder_mapping = {
        "tri_view": "questions",
        "events_only": "questions_events_only", 
        "entities_only": "questions_entities_only",
        "features_only": "questions_features_only"
    }
    return folder_mapping.get(retrieval_mode, "questions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Name of the LLM model to use")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--video_id", type=int, default=-1, help="ID of the video to process")
    parser.add_argument("--video_path", type=str, help="ID of the video to process")
    parser.add_argument("--question_id", type=int, help="ID of the question to process")
    parser.add_argument("--video_start", type=int, help="Start video ID for batch processing (used when video_id=-1)")
    parser.add_argument("--video_end", type=int, help="End video ID for batch processing (used when video_id=-1)")
    parser.add_argument("--process_num", type=int, choices=[1, 2, 3], help="Process number (1, 2, or 3) for separate JSON files")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--retrieval_mode", default="tri_view", choices=["tri_view", "events_only", "entities_only", "features_only"], 
                       help="Retrieval mode: tri_view, events_only, entities_only, or features_only")
    parser.add_argument("--batch_mode", action="store_true", help="Enable batching mode (process all questions of a video together)")
    
    args = parser.parse_args()
    
    if args.video_id != -1:
        # Setup profiling logger
        output_folder = "./outputs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        profiler_logger = setup_profiling_logger(output_folder, 0)
        
        # Start overall timing
        overall_start = time.time()
        profiler_logger.info("=== STARTING SINGLE VIDEO PROCESSING ===")
        
        # Dataset initialization
        dataset_start = time.time()
        dataset = init_dataset(args.dataset)
        dataset_end = log_timing(profiler_logger, "Dataset Initialization", dataset_start)
        
        # Model initialization
        model_start = time.time()
        llm = init_model(args.model, args.gpus)
        model_end = log_timing(profiler_logger, "Model Initialization", model_start)
        
        # Video loading
        video_start = time.time()
        video = dataset.get_video(args.video_id)
        video_info = dataset.get_video_info(video_id = args.video_id)
        video_end = log_timing(profiler_logger, "Video Loading", video_start)
        
        qas = video_info["qa"]
        
        # AVA object creation
        ava_start = time.time()
        ava = AVA(
            video=video,
            llm_model=llm,
        )
        ava_end = log_timing(profiler_logger, "AVA Object Creation", ava_start)
        
        # Tree search
        tree_search_start = time.time()
        ava.query_tree_search(qas[args.question_id]["question"], args.question_id, retrieval_mode=args.retrieval_mode)
        tree_search_end = log_timing(profiler_logger, "Tree Search", tree_search_start)
        
        # Answer generation
        answer_start = time.time()
        final_sa_answer = ava.generate_SA_answer(qas[args.question_id]["question"], args.question_id, retrieval_mode=args.retrieval_mode)
        answer_end = log_timing(profiler_logger, "Answer Generation", answer_start)
        
        # Overall timing
        overall_end = log_timing(profiler_logger, "TOTAL PROCESSING TIME", overall_start)
        
        profiler_logger.info("=== SINGLE VIDEO PROCESSING COMPLETED ===")
        print(final_sa_answer)
    elif args.question_id is not None:
        dataset = init_dataset(args.dataset, args.video_path)
        llm = init_model(args.model, args.gpus)
        
        video = dataset.get_video()
        
        ava = AVA(
            video=video,
            llm_model=llm,
        )
        
        ava.query_tree_search(args.question)
        
        final_sa_answer = ava.generate_SA_answer(args.question)
        print(final_sa_answer)
    else:
        # Validate process number
        if args.process_num is None:
            print("Error: --process_num is required for batch processing. Use 1, 2, or 3.")
            exit(1)
        
        output_folder = "./outputs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Setup profiling logger
        profiler_logger = setup_profiling_logger(output_folder, args.process_num)
        profiler_logger.info("=== STARTING BATCH PROCESSING ===")
        
        # Use separate JSON file for each process and retrieval mode
        mode_suffix = f"_{args.retrieval_mode}" if args.retrieval_mode != "tri_view" else ""
        batch_suffix = "_batch" if args.batch_mode else ""
        json_file = f"{output_folder}/query_SA_{args.dataset}_{args.model}{mode_suffix}{batch_suffix}_process{args.process_num}.json"
        
        # Load current process results
        if os.path.exists(json_file):
            results = safe_read_json(json_file)
        else:
            results = []
        
        # Load all processed entries from main file to check for duplicates
        profiler_logger.info("STEP - Duplicate Check Loading Start")
        duplicate_check_start = time.time()
        all_processed = load_all_processed_entries(output_folder, args.dataset, args.model)
        duplicate_check_end = log_timing(profiler_logger, "Duplicate Check Loading", duplicate_check_start, additional_info=f"({len(all_processed)} entries)")
        print(f"Process {args.process_num}: Loaded {len(all_processed)} entries from main file for duplicate checking")
        
        # Dataset initialization
        profiler_logger.info("STEP - Dataset Initialization Start")
        dataset_start = time.time()
        dataset = init_dataset(args.dataset)
        dataset_end = log_timing(profiler_logger, "Dataset Initialization", dataset_start)
        
        # Model initialization
        profiler_logger.info("STEP - Model Initialization Start")
        model_start = time.time()
        llm = init_model(args.model, args.gpus)
        model_end = log_timing(profiler_logger, "Model Initialization", model_start)
        
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
        mode_info = f" (mode: {args.retrieval_mode}, batching: {'ON' if args.batch_mode else 'OFF'})"
        profiler_logger.info(f"Starting batch processing: {total_videos} videos from {start_video} to {end_video}{mode_info}")
        
        for video_id in range(start_video, end_video + 1):
            video_start_time = time.time()
            profiler_logger.info(f"=== PROCESSING VIDEO {video_id} ===")
            
            try:
                # Video loading
                profiler_logger.info(f"STEP - Video {video_id} Loading Start")
                video_load_start = time.time()
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
            
            if args.batch_mode:
                # BATCH MODE: Process all questions of this video together
                print(f"  Processing video {video_id} in BATCH MODE ({len(qas)} questions)")
                profiler_logger.info(f"Video {video_id}: Processing {len(qas)} questions in batch mode")
                
                # Check for duplicates in this batch
                batch_duplicate_check_start = time.time()
                batch_to_process = []
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
                        batch_to_process.append((question_id, qas[question_id]))
                    else:
                        print(f"    Skipping question {question_id} (already processed)")
                        profiler_logger.info(f"Video {video_id} Q{question_id}: Skipped (already processed)")
                
                batch_duplicate_check_end = log_timing(profiler_logger, f"Video {video_id} Batch Duplicate Check", batch_duplicate_check_start)
                
                if not batch_to_process:
                    print(f"  All questions for video {video_id} already processed, skipping...")
                    continue
                
                print(f"  Processing {len(batch_to_process)} questions in batch")
                
                try:
                    # Create AVA object for this batch
                    ava_creation_start = time.time()
                    ava = AVA(video=video, llm_model=llm)
                    ava_creation_end = log_timing(profiler_logger, f"Video {video_id} Batch AVA Creation", ava_creation_start)
                    
                    # Process each question in the batch
                    for question_id, question_data in batch_to_process:
                        question_start = time.time()
                        profiler_logger.info(f"--- Processing Video {video_id}, Question {question_id} ---")
                        
                        try:
                            # Tree search with specific retrieval mode
                            tree_search_start = time.time()
                            ava.query_tree_search(question_data["question"], question_id, retrieval_mode=args.retrieval_mode)
                            tree_search_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Tree Search", tree_search_start)
                            
                            # Answer generation
                            answer_generation_start = time.time()
                            final_sa_answer = ava.generate_SA_answer(question_data["question"], question_id, retrieval_mode=args.retrieval_mode)
                            answer_generation_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Answer Generation", answer_generation_start)
                            
                            # Result preparation
                            result_prep_start = time.time()
                            result = {
                                "video_id": video_id,
                                "question_id": question_id,
                                "question": question_data["question"],
                                "answer": question_data["answer"],
                                "response": final_sa_answer,
                                "question_type": question_data["question_type"],
                                "retrieval_mode": args.retrieval_mode
                            }
                            results.append(result)
                            result_prep_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Result Preparation", result_prep_start)
                            
                            # Total question processing time
                            question_total_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} TOTAL", question_start)
                            
                        except Exception as e:
                            print(f"Error processing video {video_id}, question {question_id}: {e}")
                            profiler_logger.error(f"Error processing video {video_id}, question {question_id}: {e}")
                            # Add error result
                            results.append({
                                "video_id": video_id,
                                "question_id": question_id,
                                "question": question_data["question"],
                                "answer": question_data["answer"],
                                "response": None,
                                "question_type": question_data["question_type"],
                                "retrieval_mode": args.retrieval_mode,
                                "error": str(e)
                            })
                    
                    # Save results after processing all questions in this video
                    save_start = time.time()
                    if not safe_write_json(json_file, results):
                        print(f"Warning: Failed to save results for video {video_id}")
                        profiler_logger.error(f"Failed to save results for video {video_id}")
                        break
                    save_end = log_timing(profiler_logger, f"Video {video_id} Batch Save Results", save_start)
                    
                except Exception as e:
                    print(f"Error processing video {video_id} in batch mode: {e}")
                    profiler_logger.error(f"Error processing video {video_id} in batch mode: {e}")
                    continue
                    
            else:
                # ORIGINAL MODE: Process questions one by one
                for question_id in range(len(qas)):
                    question_start_time = time.time()
                    profiler_logger.info(f"--- Processing Video {video_id}, Question {question_id} ---")
                    
                    # Check if this video/question combination is already processed in main file
                    duplicate_check_start = time.time()
                    already_processed_main = any(
                        entry["video_id"] == video_id and entry["question_id"] == question_id 
                        for entry in all_processed
                    )
                    
                    # Check if this video/question combination is already processed in current process file
                    already_processed_current = any(
                        entry["video_id"] == video_id and entry["question_id"] == question_id 
                        for entry in results
                    )
                    duplicate_check_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Duplicate Check", duplicate_check_start)
                    
                    if already_processed_main or already_processed_current:
                        print(f"  Skipping video {video_id}, question {question_id} (already processed)")
                        profiler_logger.info(f"Video {video_id} Q{question_id}: Skipped (already processed)")
                        continue
                    
                    print(f"  Processing video {video_id}, question {question_id}")
                    
                    try:
                        # AVA object creation
                        profiler_logger.info(f"STEP - Video {video_id} Q{question_id} AVA Creation Start")
                        ava_creation_start = time.time()
                        ava = AVA(
                            video=video,
                            llm_model=llm,
                        )
                        ava_creation_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} AVA Creation", ava_creation_start)
                        
                        # Tree search
                        profiler_logger.info(f"STEP - Video {video_id} Q{question_id} Tree Search Start")
                        tree_search_start = time.time()
                        ava.query_tree_search(qas[question_id]["question"], question_id, retrieval_mode=args.retrieval_mode)
                        tree_search_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Tree Search", tree_search_start)
                        
                        # Answer generation
                        profiler_logger.info(f"STEP - Video {video_id} Q{question_id} Answer Generation Start")
                        answer_generation_start = time.time()
                        final_sa_answer = ava.generate_SA_answer(qas[question_id]["question"], question_id, retrieval_mode=args.retrieval_mode)
                        answer_generation_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Answer Generation", answer_generation_start)
                        
                        # Result preparation
                        profiler_logger.info(f"STEP - Video {video_id} Q{question_id} Result Preparation Start")
                        result_prep_start = time.time()
                        results.append({
                            "video_id": video_id,
                            "question_id": question_id,
                            "question": qas[question_id]["question"],
                            "answer": qas[question_id]["answer"],
                            "response": final_sa_answer,
                            "question_type": qas[question_id]["question_type"],
                            "retrieval_mode": args.retrieval_mode
                        })
                        result_prep_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Result Preparation", result_prep_start)
                        
                        # Save results
                        profiler_logger.info(f"STEP - Video {video_id} Q{question_id} Save Results Start")
                        save_start = time.time()
                        if not safe_write_json(json_file, results):
                            print(f"Warning: Failed to save results for video {video_id}, question {question_id}")
                            profiler_logger.error(f"Failed to save results for video {video_id}, question {question_id}")
                            break
                        save_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Save Results", save_start)
                        
                        # Total question processing time
                        question_total_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} TOTAL", question_start_time)
                            
                    except Exception as e:
                        print(f"Error processing video {video_id}, question {question_id}: {e}")
                        profiler_logger.error(f"Error processing video {video_id}, question {question_id}: {e}")
                        continue
            
            # Video completion timing
            video_total_end = log_timing(profiler_logger, f"Video {video_id} TOTAL", video_start_time)
            print(f"Process {args.process_num}: Completed video {video_id}")
        
        # Batch completion timing
        batch_total_end = log_timing(profiler_logger, "BATCH PROCESSING TOTAL", batch_start)
        profiler_logger.info("=== BATCH PROCESSING COMPLETED ===")
                
                