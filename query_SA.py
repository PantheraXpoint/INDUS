from AVA.ava import AVA
import os
import json
import fcntl
import time
import random
from dataset.init_dataset import init_dataset, get_video_idx
from llms.init_model import init_model
import argparse
import time


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
    
    args = parser.parse_args()
    
    if args.video_id != -1:
        dataset = init_dataset(args.dataset)
        llm = init_model(args.model, args.gpus)
        
        video = dataset.get_video(args.video_id)
        video_info = dataset.get_video_info(video_id = args.video_id)
        
        qas = video_info["qa"]
        
        ava = AVA(
            video=video,
            llm_model=llm,
        )
        
        ava.query_tree_search(qas[args.question_id]["question"], args.question_id)
        
        final_sa_answer = ava.generate_SA_answer(qas[args.question_id]["question"], args.question_id)
        print(final_sa_answer)
    else:
        # Validate process number
        if args.process_num is None:
            print("Error: --process_num is required for batch processing. Use 1, 2, or 3.")
            exit(1)
        
        output_folder = "./outputs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Use separate JSON file for each process
        json_file = f"{output_folder}/query_SA_{args.dataset}_{args.model}_process{args.process_num}.json"
        
        # Load current process results
        if os.path.exists(json_file):
            results = safe_read_json(json_file)
        else:
            results = []
        
        # Load all processed entries from main file to check for duplicates
        all_processed = load_all_processed_entries(output_folder, args.dataset, args.model)
        print(f"Process {args.process_num}: Loaded {len(all_processed)} entries from main file for duplicate checking")
        
        dataset = init_dataset(args.dataset)
        llm = init_model(args.model, args.gpus)
        video_idx = get_video_idx(args.dataset)
        
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
        
        for video_id in range(start_video, end_video + 1):
            try:
                video = dataset.get_video(video_id)
                video_info = dataset.get_video_info(video_id=video_id)
                qas = video_info["qa"]
                
                processed_videos += 1
                print(f"Process {args.process_num}: Processing video {video_id} with {len(qas)} questions ({processed_videos}/{total_videos})")
            except Exception as e:
                print(f"Process {args.process_num}: Error loading video {video_id}: {e}")
                continue
            
            for question_id in range(len(qas)):
                # Check if this video/question combination is already processed in main file
                already_processed_main = any(
                    entry["video_id"] == video_id and entry["question_id"] == question_id 
                    for entry in all_processed
                )
                
                # Check if this video/question combination is already processed in current process file
                already_processed_current = any(
                    entry["video_id"] == video_id and entry["question_id"] == question_id 
                    for entry in results
                )
                
                if already_processed_main or already_processed_current:
                    print(f"  Skipping video {video_id}, question {question_id} (already processed)")
                    continue
                
                print(f"  Processing video {video_id}, question {question_id}")
                
                try:
                    ava = AVA(
                        video=video,
                        llm_model=llm,
                    )
                    
                    ava.query_tree_search(qas[question_id]["question"], question_id)
                    
                    final_sa_answer = ava.generate_SA_answer(qas[question_id]["question"], question_id)
                    
                    results.append({
                        "video_id": video_id,
                        "question_id": question_id,
                        "question": qas[question_id]["question"],
                        "answer": qas[question_id]["answer"],
                        "response": final_sa_answer,
                        "question_type": qas[question_id]["question_type"],
                    })
                    
                    if not safe_write_json(json_file, results):
                        print(f"Warning: Failed to save results for video {video_id}, question {question_id}")
                        break
                        
                except Exception as e:
                    print(f"Error processing video {video_id}, question {question_id}: {e}")
                    continue
            
            print(f"Process {args.process_num}: Completed video {video_id}")
                
                