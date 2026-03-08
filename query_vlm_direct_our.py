"""
VLM Direct Query - Simplified inference using cached search results
Based on query_SA.py but uses pre-computed sorted_SA_score_result.json
"""
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
from utils.iterative_pruning import extract_frames_from_events
import re
import gc
from utils.grounding import GroundingDetector
from PIL import Image
from AVA.prompt import PROMPTS
from typing import List
import cv2
from utils.eva_vlm_direct import extract_predicted_answer

def create_prompt(question: str, event_descriptions: List[str], no_event_description: bool, frame_timestamps: List[str], event_timestamps: List[str]) -> str:
    """
    Create the prompt with event context
    
    Args:
        question: The user's question
        event_descriptions: List of event descriptions
        
    Returns:
        Formatted prompt string
    """
    if not no_event_description:
        # Format event descriptions as numbered list
        event_text = ""
        for i, (event_description, event_timestamp) in enumerate(zip(event_descriptions, event_timestamps)):
            event_text += f"{i+1}. [{event_timestamp[0]} - {event_timestamp[1]}] {event_description}\n"
        prompt = PROMPTS["events_only"].format(
            user_query=question,
            event_descriptions=event_text
        )

        return prompt

    else:
        frame_timestamps_text = ""
        for idx, timestamp in enumerate(frame_timestamps):
            frame_timestamps_text += f"Image {idx+1} at {timestamp}\n"
        prompt = PROMPTS["frames_only"].format(
            user_query=question,
            frame_timestamps=frame_timestamps_text
        )

        return prompt

def convert_sencond_to_time(seconds: float) -> str:
    # format the time as HH:MM:SS
    return f"{int(seconds//3600):02d}:{int(seconds%3600//60):02d}:{int(seconds%60):02d}"

def convert_frame_to_time(frame_number, fps=30):
    # format the time as HH:MM:SS
    return f"{frame_number//fps//3600:02d}:{frame_number//fps%3600//60:02d}:{frame_number//fps%60:02d}"

def final_frames(video_path, events, llm, question, answer_statements, detector, work_dir):
    # Extract frames at 2 fps from event durations
    print("Extracting frames at 2 fps from event durations...")
    extracted_frames = extract_frames_from_events(video_path, events, work_dir=work_dir, target_fps=0.1)
    print(f"Extracted {len(extracted_frames)} frames")
    
    if not extracted_frames:
        print("No frames extracted. Skipping...")
        return
    
    # Extract query objects using LLM
    print("Extracting objects using QwenLM...")
    # Create prompt to extract objects/entities from question and statements
    prompt = f"""Given the following question and answer statements, extract all relevant objects, entities, places, or things that can be searched by a open-world object detection model in a video.

Question: {question}

Answer Statements:
{chr(10).join([f"- {stmt}" for stmt in answer_statements])}

Please list all objects, entities, places, shops, landmarks, or things mentioned that can be searched by a open-world object detection model in a video. Return only a comma-separated list of objects, one per line. Be specific and do not include any explanatory text.

Objects to search for:"""
    
    llm_response = llm.generate_response({"text": prompt}, max_new_tokens=256, temperature=0.3)
    # Parse the response to extract objects
    # Split by newlines and commas, clean up
    query_objects = []
    for line in llm_response.strip().split('\n'):
        # Remove numbering, bullets, etc.
        line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
        line = re.sub(r'^[-•]\s*', '', line)
        # Split by comma
        objects_in_line = [obj.strip() for obj in line.split(',')]
        query_objects.extend([obj for obj in objects_in_line if obj and len(obj) > 1])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_objects = []
    for obj in query_objects:
        obj_lower = obj.lower()
        if obj_lower not in seen and obj_lower not in ['the', 'a', 'an', 'camera', 'wearer']:
            seen.add(obj_lower)
            unique_objects.append(obj)
    
    query_objects = unique_objects
    print(f"LLM extracted objects: {query_objects}")
    
    text_labels = query_objects
    
    # Run grounding detection on all frames
    print("Running grounding detection...")
    frame_numbers = [frame_num for frame_num, _ in extracted_frames]
    images = [img for _, img in extracted_frames]
    
    # Process in batches to avoid memory issues
    batch_size = 8
    all_results = []
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_results = detector.detect(
            batch_images,
            text_labels,
            threshold=0.1,
            verbose=False
        )
        all_results.extend(batch_results)

    
    # Calculate scores for each frame
    # Use maximum score from all detections in the frame
    frame_scores = []
    for result in all_results:
        if result['scores']:
            max_score = max(result['scores']) # we can use the sum, avg score instead of the max score
        else:
            max_score = 0.0
        frame_scores.append(max_score)
    return frame_scores, frame_numbers


def prepare_question_data(question_id, video_info, llm, detector,
                          max_frames, work_dir, question_data, uniform_sampling,
                          seed_events_path):

    # event_paths = [seed_events_path, "ECML-PKDD/ava100_retrieval/indus_explore/seed_events_AVA100_indus_explore_final.json"]
    event_paths = [seed_events_path]
    all_event_results = []
    for event_path in event_paths:
        if not os.path.exists(event_path):
            return None, None
        with open(event_path, "r") as f:
            all_event_results.extend(json.load(f))
    video_key = video_info.get("video_key", video_info.get("key"))
    event_results = []
    for event in all_event_results:
        if event["video_key"] == video_key and event["question_id"] == question_id:
            event_results.extend(event["seed_events"])
            break
    if event_results == []:
        print(f"No event results found for video {video_info['video_key']} question {question_id}")
        return None, None

    if not uniform_sampling:
        frame_scores, frame_numbers = final_frames(video_info["video_path"], event_results, llm, video_info["qa"][question_id]["question"],
                                                    video_info["qa"][question_id]["answer"], detector, work_dir)
        sorted_scores = sorted(frame_scores, reverse=True)
        extracted_frames = []
        count = 0
        threshold = 0
        if len(frame_numbers) >= max_frames:
            threshold = sorted_scores[max_frames-1]
        for frame_number, score in zip(frame_numbers, frame_scores):
            if score > threshold:
                if not os.path.exists(os.path.join(work_dir, "frames", f"{frame_number}.jpg")):
                    continue
                extracted_frames.append((frame_number, Image.open(os.path.join(work_dir, "frames", f"{frame_number}.jpg"))))
                count += 1
            if count >= max_frames:
                break
        if len(extracted_frames) < max_frames:
            # uniform sampling the rest of the frames
            uniform_extracted_frames = extract_frames_from_events(video_info["video_path"], event_results, target_fps=0.1, work_dir=work_dir, max_frames=max_frames-len(extracted_frames))
            for frame_number, img in uniform_extracted_frames:
                if frame_number not in [frame_number for frame_number, _ in extracted_frames]:
                    extracted_frames.append((frame_number, img))
    else:
        extracted_frames = extract_frames_from_events(video_info["video_path"], event_results, target_fps=0.1, work_dir=work_dir, max_frames=max_frames)
        for frame_number, img in extracted_frames:
            if frame_number not in [frame_number for frame_number, _ in extracted_frames]:
                extracted_frames.append((frame_number, img))
    
    events_descriptions = []
    cap = cv2.VideoCapture(video_info["video_path"])
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    existing_events = set()
    for event in event_results:
        for frame_number, _ in extracted_frames:
            timestamp = int(frame_number/video_fps) # convert frame number to timestamp in seconds
            duration = event.get('duration', event.get('metadata', {}).get('duration', None))  # [start_time_sec, end_time_sec]
            if duration is None:
                continue
            if timestamp >= duration[0] and timestamp <= duration[1]:
                if event["id"] not in existing_events:
                    events_descriptions.append((duration, event.get("description", event.get("content", "")))) 
                    existing_events.add(event["id"])
    print(f"Extracted {len(extracted_frames)} frames for question {question_id}")
    print(f"Extracted {len(events_descriptions)} events for question {question_id}")
    return extracted_frames, events_descriptions
    


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

def process_question(questions_to_process):
    remaining = []

    for q_idx, (question_id, question_data) in enumerate(questions_to_process):
        profiler_logger.info(f"--- Video {video_id}, Question {question_id} ({q_idx+1}/{len(questions_to_process)}) ---")
        try:
            prep_start = time.time()
            frames, event_descriptions = prepare_question_data(
                question_id, video_info, vlm, detector,
                args.max_frames, video.work_dir, question_data,
                args.uniform_sampling,
                args.seed_events_path
            )
            log_timing(profiler_logger, f"Video {video_id} Q{question_id} Data Preparation", prep_start)

            if not frames:
                profiler_logger.warning(f"Video {video_id} Q{question_id}: No frames extracted")
                # keep it if you want to retry; otherwise skip permanently
                # remaining.append((question_id, question_data))
                continue

            frames.sort(key=lambda x: x[0])
            frame_timestamps = [convert_frame_to_time(fn) for fn, _ in frames]
            frames_data = [img for _, img in frames]

            event_descriptions.sort(key=lambda x: x[0])
            event_timestamps = [(convert_sencond_to_time(e[0][0]), convert_sencond_to_time(e[0][1])) for e in event_descriptions]
            event_texts = [e[1] for e in event_descriptions]

            prompt = create_prompt(
                question_data["question"],
                event_texts,
                args.no_event_description,
                frame_timestamps,
                event_timestamps,
            )

            if args.no_event_description or args.use_frames:
                single_input = {"text": prompt, "video": frames_data}
            else:
                single_input = {"text": prompt}

            vlm_call_start = time.time()
            response = vlm.generate_response(single_input)
            log_timing(profiler_logger, f"Video {video_id} Q{question_id} VLM Call", vlm_call_start)

            if extract_predicted_answer(response) is None:
                profiler_logger.warning(f"Video {video_id} Q{question_id}: No valid prediction")
                remaining.append((question_id, question_data))  # retry later
                continue

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

            if not safe_write_json(json_file, results):
                profiler_logger.error(f"Failed to save results for video {video_id} Q{question_id}")
                remaining.append((question_id, question_data))  # retry if save failed

            del frames, frames_data, single_input, event_descriptions, frame_timestamps, prompt, response
            gc.collect()

        except Exception as e:
            profiler_logger.error(f"Video {video_id} Q{question_id} error: {e}")
            remaining.append((question_id, question_data))  # retry

    return remaining

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Direct Query - Uses cached SA results for inference")
    parser.add_argument("--model", required=True, help="Name of the VLM model to use")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--seed_events_path", type=str, help="Path to the seed events file")
    parser.add_argument("--video_id", type=int, default=-1, help="ID of the video to process")
    parser.add_argument("--question_id", type=int, help="ID of the question to process")
    parser.add_argument("--video_start", type=int, help="Start video ID for batch processing (used when video_id=-1)")
    parser.add_argument("--video_end", type=int, help="End video ID for batch processing (used when video_id=-1)")
    parser.add_argument("--process_num", type=int, choices=[1, 2, 3], help="Process number (1, 2, or 3) for separate JSON files")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--retrieval_mode", default="tri_view", choices=["tri_view", "events_only", "entities_only", "features_only"], 
                       help="Retrieval mode: tri_view, events_only, entities_only, or features_only")
    parser.add_argument("--max_frames", type=int, default=256, help="Maximum number of frames to sample")
    parser.add_argument("--no_event_description", action="store_true", help="Do not use event description")
    parser.add_argument("--use_frames", action="store_true", help="Use frames for answer generation")
    parser.add_argument("--uniform_sampling", action="store_true", help="Use uniform sampling for answer generation")
    parser.add_argument("--max_batch_size", type=int, default=64, help="Maximum batch size for VLM batch generation")
    
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
        video = dataset.get_video(args.video_id)
        video_info = dataset.get_video_info(video_id=args.video_id)
        video_end = log_timing(profiler_logger, "Video Loading", video_start)
        
        qas = video_info["qa"]
        
        # AVA object creation
        ava_start = time.time()
        ava_direct = AVA(
            video=video,
            llm_model=vlm,
        )
        ava_end = log_timing(profiler_logger, "AVA Object Creation", ava_start)
        
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
        json_file = f"{output_folder}/query_VLM_direct_{args.dataset}_{args.model}{mode_suffix}_process{args.process_num}_{'no_event_description' if args.no_event_description else ''}_{'uniform_sampling' if args.uniform_sampling else ''}_{'use_frames' if args.use_frames else ''}_{args.seed_events_path.split('/')[-1]}.json"
        
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
        detector = GroundingDetector()
        
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
            start_video = video_idx[0]
            end_video = video_idx[1]
            print(f"Process {args.process_num}: Using default range: {start_video} to {end_video}")
        
        total_videos = end_video - start_video + 1
        processed_videos = 0
        
        # Start batch processing timing
        batch_start = time.time()
        mode_info = f" (mode: {args.retrieval_mode}, max_frames: {args.max_frames})"
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
                none_response = [
                    entry
                    for entry in results
                    if entry["question_id"] == question_id and extract_predicted_answer(entry["response"]) is None
                ]
                if len(none_response) > 0:
                    print(none_response[0]["response"])
                if len(none_response) > 0 or not (already_processed_main or already_processed_current):
                    questions_to_process.append((question_id, qas[question_id]))
                else:
                    print(f"    Skipping question {question_id} (already processed)")
                    profiler_logger.info(f"Video {video_id} Q{question_id}: Skipped (already processed)")
            
            batch_duplicate_check_end = log_timing(profiler_logger, f"Video {video_id} Batch Duplicate Check", batch_duplicate_check_start)
            
            if not questions_to_process:
                print(f"  All questions for video {video_id} already processed, skipping...")
                continue
            
            print(f"  Processing {len(questions_to_process)} questions one-by-one (low RAM)")
            
            ### process the questions one-by-one
            try:
                patience = 5
                while len(questions_to_process) > 0:
                    questions_to_process = process_question(questions_to_process)
                    if len(questions_to_process) == 0 or patience == 0:
                        break
                    patience -= 1
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                profiler_logger.error(f"Error processing video {video_id}: {e}")
                continue
            
            # Video completion timing
            video_total_end = log_timing(profiler_logger, f"Video {video_id} TOTAL", video_start_time)
            print(f"Process {args.process_num}: Completed video {video_id}")
        
        # Batch completion timing
        batch_total_end = log_timing(profiler_logger, "BATCH PROCESSING TOTAL", batch_start)
        profiler_logger.info("=== BATCH PROCESSING COMPLETED ===")
