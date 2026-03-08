"""
Vectorized Retrieval Query - Frame-level semantic retrieval baseline
Retrieves top-k=256 frames based on query similarity
"""
from AVA.vectorized_retrieval import VectorizedRetrieval
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
    """Safely write JSON data to file with file locking"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(random.uniform(0.1, 0.5))
            
            with open(file_path, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                json.dump(data, f, indent=4)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except (IOError, OSError) as e:
            if attempt == max_retries - 1:
                print(f"Failed to write to {file_path} after {max_retries} attempts: {e}")
                return False
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
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
        except (IOError, OSError) as e:
            if attempt == max_retries - 1:
                print(f"Failed to read from {file_path} after {max_retries} attempts: {e}")
                return []
            continue
    return []


def setup_profiling_logger(output_folder, process_num):
    """Setup detailed profiling logger"""
    log_file = os.path.join(output_folder, f"profiling_vectorized_retrieval_process_{process_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(f'profiling_vectorized_retrieval_process_{process_num}')
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_timing(logger, step_name, start_time, end_time=None, additional_info=""):
    """Log timing information for a step"""
    if end_time is None:
        end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"TIMING - {step_name}: {duration:.4f} seconds {additional_info}")
    return end_time


def get_time_reference(dataset, video_id, question_id):
    """Extract time_reference from dataset QA data"""
    try:
        video_info = dataset.get_video_info(video_id)
        qas = video_info['qa']
        if question_id < len(qas):
            return qas[question_id].get('time_reference', 'N/A')
    except Exception as e:
        print(f"Error getting time_reference for video {video_id}, question {question_id}: {e}")
    return 'N/A'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorized Retrieval Query - Frame-level semantic retrieval")
    parser.add_argument("--model", required=True, help="Name of the VLM model to use")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--video_id", type=int, default=-1, help="ID of the video to process")
    parser.add_argument("--question_id", type=int, help="ID of the question to process")
    parser.add_argument("--video_start", type=int, help="Start video ID for batch processing")
    parser.add_argument("--video_end", type=int, help="End video ID for batch processing")
    parser.add_argument("--process_num", type=int, choices=[1, 2, 3], help="Process number for separate JSON files")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    
    # Retrieval parameter (fixed k=256)
    parser.add_argument("--k", type=int, default=256, help="Number of frames to retrieve (default 256)")
    
    # Batching parameter
    parser.add_argument("--batch_size", type=int, default=4, help="Number of questions per VLM batch")
    
    args = parser.parse_args()
    
    if args.video_id != -1:
        # Single video processing
        output_folder = "./outputs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        profiler_logger = setup_profiling_logger(output_folder, 0)
        
        overall_start = time.time()
        profiler_logger.info("=== STARTING SINGLE VIDEO PROCESSING (Vectorized Retrieval) ===")
        profiler_logger.info(f"Retrieval k = {args.k} frames")
        
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
        if args.dataset in ["lvbench", "videomme"]:
            video = dataset.get_video_for_vlm(args.video_id)
        else:
            video = dataset.get_video(args.video_id)
        video_info = dataset.get_video_info(video_id=args.video_id)
        video_end = log_timing(profiler_logger, "Video Loading", video_start)
        
        qas = video_info["qa"]
        question = qas[args.question_id]["question"]
        
        # Get time reference
        time_reference = get_time_reference(dataset, args.video_id, args.question_id)
        
        # VectorizedRetrieval object creation
        vectorized_start = time.time()
        vectorized_retrieval = VectorizedRetrieval(video=video, llm_model=vlm)
        vectorized_end = log_timing(profiler_logger, "VectorizedRetrieval Object Creation", vectorized_start)
        
        # Answer generation
        answer_start = time.time()
        result = vectorized_retrieval.generate_answer(
            question=question,
            time_reference=time_reference,
            k=args.k
        )
        answer_end = log_timing(profiler_logger, "Answer Generation", answer_start)
        
        # Overall timing
        overall_end = log_timing(profiler_logger, "TOTAL PROCESSING TIME", overall_start)
        
        profiler_logger.info("=== SINGLE VIDEO PROCESSING COMPLETED ===")
        print(json.dumps(result, indent=2))
    
    else:
        # Batch processing
        if args.process_num is None:
            print("Error: --process_num is required for batch processing")
            exit(1)
        
        output_folder = "./outputs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Setup profiling logger
        profiler_logger = setup_profiling_logger(output_folder, args.process_num)
        profiler_logger.info("=== STARTING BATCH PROCESSING (Vectorized Retrieval) ===")
        profiler_logger.info(f"Retrieval k = {args.k} frames")
        profiler_logger.info(f"Batch size: {args.batch_size} questions per batch")
        
        # Output file
        json_file = f"{output_folder}/query_vectorized_retrieval_{args.dataset}_{args.model}_k{args.k}_process{args.process_num}.json"
        
        # Load current results
        if os.path.exists(json_file):
            results = safe_read_json(json_file)
        else:
            results = []
        
        profiler_logger.info(f"Loaded {len(results)} existing results")
        
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
            start_video = args.video_start
            end_video = args.video_end
            
            if start_video < video_idx[0] or end_video > video_idx[1]:
                print(f"Error: Video range {start_video}-{end_video} is outside valid range {video_idx[0]}-{video_idx[1]}")
                exit(1)
            if start_video > end_video:
                print(f"Error: Start video {start_video} cannot be greater than end video {end_video}")
                exit(1)
            
            print(f"Process {args.process_num}: Using custom video range: {start_video} to {end_video}")
        else:
            start_video = results[-1]["video_id"] if results else video_idx[0]
            end_video = video_idx[1]
            print(f"Process {args.process_num}: Using default range: {start_video} to {end_video}")
        
        total_videos = end_video - start_video + 1
        processed_questions = 0
        
        # Start batch processing
        batch_start = time.time()
        profiler_logger.info(f"Starting batch processing: {total_videos} videos from {start_video} to {end_video}")
        
        # Collect all questions into batches
        current_batch = []
        current_batch_metadata = []  # (video_id, question_id, question_data, retrieval_info, overlap_info)
        
        for video_id in range(start_video, end_video + 1):
            video_start_time = time.time()
            profiler_logger.info(f"=== PROCESSING VIDEO {video_id} ===")
            
            try:
                # Video loading
                profiler_logger.info(f"STEP - Video {video_id} Loading Start")
                video_load_start = time.time()
                if args.dataset in ["lvbench", "videomme"]:
                    video = dataset.get_video_for_vlm(video_id)
                else:
                    video = dataset.get_video(video_id)
                video_info = dataset.get_video_info(video_id=video_id)
                video_load_end = log_timing(profiler_logger, f"Video {video_id} Loading", video_load_start)
                
                qas = video_info["qa"]
                print(f"Process {args.process_num}: Processing video {video_id} with {len(qas)} questions")
                profiler_logger.info(f"Video {video_id}: {len(qas)} questions to process")
                
                # Create VectorizedRetrieval object once per video
                vectorized_creation_start = time.time()
                vectorized_retrieval = VectorizedRetrieval(video=video, llm_model=vlm)
                vectorized_creation_end = log_timing(profiler_logger, f"Video {video_id} VectorizedRetrieval Creation", vectorized_creation_start)
                
            except Exception as e:
                print(f"Process {args.process_num}: Error loading video {video_id}: {e}")
                profiler_logger.error(f"Error loading video {video_id}: {e}")
                continue
            
            # Process each question
            for question_id in range(len(qas)):
                # Check if already processed
                already_processed = any(
                    entry["video_id"] == video_id and entry["question_id"] == question_id
                    for entry in results
                )
                
                if already_processed:
                    print(f"  Skipping video {video_id}, question {question_id} (already processed)")
                    profiler_logger.info(f"Video {video_id} Q{question_id}: Skipped (already processed)")
                    continue
                
                try:
                    question_data = qas[question_id]
                    time_reference = get_time_reference(dataset, video_id, question_id)
                    
                    # Prepare question data
                    prep_start = time.time()
                    frames, retrieval_info, overlap_info = vectorized_retrieval.prepare_question_data(
                        question=question_data["question"],
                        time_reference=time_reference,
                        k=args.k
                    )
                    prep_end = log_timing(profiler_logger, f"Video {video_id} Q{question_id} Data Preparation", prep_start)
                    
                    if not frames:
                        print(f"  Warning: No frames for video {video_id}, question {question_id}")
                        profiler_logger.warning(f"Video {video_id} Q{question_id}: No frames retrieved")
                        continue
                    
                    # Create prompt
                    prompt = vectorized_retrieval.create_prompt(question_data["question"])
                    
                    # Add to batch
                    current_batch.append({
                        "text": prompt,
                        "video": frames
                    })
                    current_batch_metadata.append((
                        video_id, question_id, question_data, 
                        retrieval_info, overlap_info
                    ))
                    
                    print(f"  Added video {video_id}, question {question_id} to batch (batch size: {len(current_batch)})")
                    
                    # Process batch if it reaches batch_size
                    if len(current_batch) >= args.batch_size:
                        vlm_call_start = time.time()
                        print(f"  Calling VLM with batch of {len(current_batch)} questions...")
                        profiler_logger.info(f"Batch VLM Call: {len(current_batch)} questions")
                        
                        responses = vlm.batch_generate_response(current_batch)
                        vlm_call_end = log_timing(profiler_logger, f"Batch VLM Call", vlm_call_start, additional_info=f"({len(current_batch)} questions)")
                        
                        # Save responses
                        save_start = time.time()
                        for (vid, qid, qdata, retr_info, overlap_info), response in zip(current_batch_metadata, responses):
                            result = {
                                "video_id": vid,
                                "question_id": qid,
                                "question": qdata["question"],
                                "answer": qdata["answer"],
                                "response": response,
                                "question_type": qdata.get("question_type", qdata.get("task_type", "")),
                                "method": "vectorized_retrieval",
                                "retrieval_info": retr_info,
                                "ground_truth_overlap": overlap_info,
                                "time_reference": overlap_info.get("time_reference", "N/A") if overlap_info else "N/A",
                            }
                            results.append(result)
                            processed_questions += 1
                            print(f"  Completed video {vid}, question {qid}")
                        
                        # Save to file
                        if not safe_write_json(json_file, results):
                            print(f"Warning: Failed to save results")
                            profiler_logger.error(f"Failed to save results")
                        
                        save_end = log_timing(profiler_logger, f"Batch Save Results", save_start)
                        
                        # Clear batch
                        current_batch = []
                        current_batch_metadata = []
                
                except Exception as e:
                    print(f"  Error preparing video {video_id}, question {question_id}: {e}")
                    profiler_logger.error(f"Video {video_id} Q{question_id} preparation error: {e}")
                    continue
            
            video_total_end = log_timing(profiler_logger, f"Video {video_id} TOTAL", video_start_time)
            print(f"Process {args.process_num}: Completed video {video_id}")
        
        # Process remaining batch
        if current_batch:
            vlm_call_start = time.time()
            print(f"Processing final batch of {len(current_batch)} questions...")
            profiler_logger.info(f"Final Batch VLM Call: {len(current_batch)} questions")
            
            responses = vlm.batch_generate_response(current_batch)
            vlm_call_end = log_timing(profiler_logger, f"Final Batch VLM Call", vlm_call_start, additional_info=f"({len(current_batch)} questions)")
            
            # Save responses
            save_start = time.time()
            for (vid, qid, qdata, retr_info, overlap_info), response in zip(current_batch_metadata, responses):
                result = {
                    "video_id": vid,
                    "question_id": qid,
                    "question": qdata["question"],
                    "answer": qdata["answer"],
                    "response": response,
                    "question_type": qdata.get("question_type", qdata.get("task_type", "")),
                    "method": "vectorized_retrieval",
                    "retrieval_info": retr_info,
                    "ground_truth_overlap": overlap_info,
                    "time_reference": overlap_info.get("time_reference", "N/A") if overlap_info else "N/A",
                }
                results.append(result)
                processed_questions += 1
                print(f"  Completed video {vid}, question {qid}")
            
            # Save to file
            if not safe_write_json(json_file, results):
                print(f"Warning: Failed to save final results")
                profiler_logger.error(f"Failed to save final results")
            
            save_end = log_timing(profiler_logger, f"Final Batch Save Results", save_start)
        
        # Batch completion timing
        batch_total_end = log_timing(profiler_logger, "BATCH PROCESSING TOTAL", batch_start)
        profiler_logger.info(f"=== BATCH PROCESSING COMPLETED: {processed_questions} questions ===")
        print(f"Process {args.process_num}: Completed {processed_questions} questions")
