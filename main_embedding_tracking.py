#!/usr/bin/env python3
"""
Main script for embedding-based object tracking and search
"""

import os
import sys
import argparse
import cv2
import time
from llms.init_model import init_model
from llms.BaseModel import BaseLanguageModel
from PIL import Image
import concurrent.futures
from dataset.init_dataset import init_dataset, get_video_idx
import time
import json

# Add embeddings directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'embeddings'))

from JinaCLIP import JinaCLIP
from embeddings.object_search import SearchSystem, filter_and_extract_bounding_box
from AVA.object_detect import ObjectDetectorTracker
from AVA.event_tracker import EventTracker
from AVA.utils import tri_view_retrieval, filter_answer_generation

def process_video(video_path: str, object_faiss_db_path: str = "object_embeddings.faiss", 
                                 event_faiss_db_path: str = "event_embeddings.faiss", 
                                 object_sqlite_db_path: str = "tracked_objects.db",
                                 llm: BaseLanguageModel = None):
    """
    Process video with embedding generation for new track IDs
    
    Args:
        video_path: Path to input video
        object_faiss_db_path: Path to FAISS database
        event_faiss_db_path: Path to FAISS database
        object_sqlite_db_path: Path to SQLite database
    """
    print("Initializing JinaCLIP embedding model...")
    try:
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        print("JinaCLIP model initialized successfully")
    except Exception as e:
        print(f"Error initializing JinaCLIP model: {e}")
        return None
    
    print("Initializing object detector with embedding support...")
    detector = ObjectDetectorTracker(
        model_path="checkpoints/yolo11l.pt",
        conf_threshold=0.5,
        iou_threshold=0.5,
        tracker_config="config/tracker.yaml",
        embedding_model=embedding_model,
        faiss_db_path=object_faiss_db_path,
        sqlite_db_path=object_sqlite_db_path
    )
    print("Event generator initializing...")
    event_tracker = EventTracker(
        llm=llm,
        embedding_model=embedding_model,
        faiss_db_path=event_faiss_db_path
    )
    print("Event generator initialized successfully")
    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
        
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    # Get video properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set processing fps to 10fps
    tracking_processing_fps = 5
    event_processing_fps = 2
    chunk_duration = 3
    tracking_frame_skip = max(1, original_fps // tracking_processing_fps)  # Skip frames to achieve 10fps processing
    event_frame_skip = max(1, original_fps // event_processing_fps)  # Skip frames to achieve 3fps processing
    
    frame_count = 0
    start_time = time.time()
    # Save tracked objects to SQLite database
    if detector.sqlite_db is not None:
        # Add video info
        detector.sqlite_db.add_video_info(video_path, width, height, original_fps, frame_count, tracking_processing_fps)
    
    print(f"Processing video (tracking only): {video_path}")
    print(f"Resolution: {width}x{height}, Original FPS: {original_fps}")
    print(f"Processing at ~{tracking_processing_fps} FPS (every {tracking_frame_skip} frames)")
    print(f"Processing at ~{event_processing_fps} FPS (every {event_frame_skip} frames)")
    
    processed_frame_count = 0
    frame_indices = []
    frames = []
    detected_objects = []
    video_chunk_num_frames = int(event_processing_fps * chunk_duration)
    event_id = 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    future_chunk = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        # Only process every frame_skip frames for 10fps processing
        if frame_count % tracking_frame_skip == 0:
            # Process tracking
            detected_objects = detector.process_frame(frame, frame_count, event_id, detected_objects)
            processed_frame_count += 1
                    
        if frame_count % event_frame_skip == 0:
            # Process event
            frame_indices.append(frame_count)
            frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(frame, (540, 360)), cv2.COLOR_BGR2RGB)))
        
        if len(frame_indices) == video_chunk_num_frames:
            event_id = frame_indices[0]
            if future_chunk and not future_chunk.done():
                print("Waiting for previous task to complete...")
                future_chunk.result()
            future_chunk = executor.submit(event_tracker.process_chunk, frames, frame_indices,
                                            detected_objects, video_chunk_num_frames,
                                            event_frame_skip)
            frame_indices = []
            frames = []
            detected_objects = []
        
        frame_count += 1
    # Cleanup
    cap.release()
    executor.shutdown(wait=True)
    
    elapsed_time = time.time() - start_time
    processing_avg_fps = processed_frame_count / elapsed_time
    
    print(f"Processing complete!")
    print(f"Total input frames: {frame_count}")
    print(f"Processed frames: {processed_frame_count}")
    print(f"Processing speed: {processing_avg_fps:.1f} FPS")
    
    return embedding_model, detector

def search_video(query: str, video_path: str, output_dir: str,
                              object_faiss_db_path: str = "database/object_embeddings.faiss",
                              event_faiss_db_path: str = "database/event_embeddings.faiss",
                              object_sqlite_db_path: str = "database/tracked_objects.db",
                              k: int = 5, max_images: int = 10, llm: BaseLanguageModel = None,
                              question_id: int = -1):
    """
    Search for objects by description and extract bounding box images
    
    Args:
        query: Text description to search for
        video_path: Path to original video
        output_dir: Directory to save extracted images
        faiss_db_path: Path to FAISS database
        sqlite_db_path: Path to SQLite database
        k: Number of search results to return
        max_images: Maximum images to extract per track
        llm: LLM model for generating descriptions
    """
    print("Initializing JinaCLIP embedding model...")
    try:
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        print("JinaCLIP model initialized successfully")
    except Exception as e:
        print(f"Error initializing JinaCLIP model: {e}")
        return [], []
    
    print("Initializing object search system...")
    object_search_system = SearchSystem(object_faiss_db_path, object_sqlite_db_path, embedding_model)
    event_search_system = SearchSystem(event_faiss_db_path, None, embedding_model)
    search_results = tri_view_retrieval(query, event_search_system, object_search_system, llm, "both")
    filtered_search_results = filter_answer_generation(search_results, llm, video_path, question_id)
    saved_images = []
    for result, filtered_answer in zip(search_results, filtered_search_results):
        if result["event_description"] != "":
            print("In event ", result["event_id"] ,": ", result["event_description"])
            print("The confidence score is: ", result["score"])
        # filtered_answer = filter_answer_generation(search_results, llm)
        print(filtered_answer)
        # entities_result = [entity for entity in result["entities"] if int(entity["id"]) in filtered_answer["track_ids"]]
        entities_result = result["entities"]
        print("There are ", len(filtered_answer.get("track_ids", [])), " entities.")
        # saved_images = filter_and_extract_bounding_box(video_path, entities_result, output_dir, max_images)
    return search_results, saved_images

def get_database_statistics(faiss_db_path: str = "embeddings.faiss",
                           sqlite_db_path: str = "tracked_objects.db"):
    """
    Get statistics about the databases
    
    Args:
        faiss_db_path: Path to FAISS database
        sqlite_db_path: Path to SQLite database
    """
    print("Initializing JinaCLIP embedding model...")
    try:
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        print("JinaCLIP model initialized successfully")
    except Exception as e:
        print(f"Error initializing JinaCLIP model: {e}")
        return
    
    print("Initializing object search system...")
    search_system = ObjectSearch(faiss_db_path, sqlite_db_path, embedding_model)
    
    print("Getting database statistics...")
    stats = search_system.get_track_statistics()
    
    print("\n=== Database Statistics ===")
    print(f"Total tracks: {stats['total_tracks']}")
    print(f"Total frames tracked: {stats['total_frames_tracked']}")
    print(f"Average track length: {stats['average_track_length']:.2f} frames")
    print("\nClass distribution:")
    for class_name, count in stats['class_distribution'].items():
        print(f"  {class_name}: {count} tracks")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Embedding-based Object Tracking and Search')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset to use (AVA, LVbench, etc.)')
    parser.add_argument('--video_id', type=int, default=-1,
                       help='ID of the video to process')
    parser.add_argument('--question_id', type=int, default=-1,
                       help='ID of the question to process')
    parser.add_argument('--model', type=str, default='qwenvl',
                       help='Model to use')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to input video file')
    parser.add_argument('--description', type=str,
                       help='Text description to search for')
    parser.add_argument('--output-dir', type=str, default='extracted_objects',
                       help='Directory to save extracted images')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of search results to return')
    parser.add_argument('--max-images', type=int, default=1,
                       help='Maximum images to extract per track')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--process-only', action='store_true',
                       help='Only process video without searching')
    parser.add_argument('--video-range', type=str, default=None,
                       help='Range of videos to process (e.g. "1-10" for videos 1 to 10)')
    
    args = parser.parse_args()
    llm = init_model(args.model, 1)

    if args.dataset is None:
        base_path = os.path.join("database", os.path.basename(args.video)[:-4])
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        object_faiss_db_path = os.path.join(base_path, "object_embeddings.db")
        event_faiss_db_path = os.path.join(base_path, "event_embeddings.db")
        object_sqlite_db_path = os.path.join(base_path, "tracked_objects.db")
        # Process video with embeddings
        if args.process_only and not args.description:
            print("Processing video with embedding generation...")
            start_time = time.time()
            result = process_video(args.video, object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path, llm)
            end_time = time.time()
            print(f"Time taken for video processing: {end_time - start_time} seconds")
            if result is None:
                print("Failed to process video")
                return
            embedding_model, detector = result
            print("Video processing completed successfully!")
        
        # Show statistics
        if args.stats:
            get_database_statistics(object_faiss_db_path, object_sqlite_db_path)
        
        # Search and extract objects
        if args.description:
            print(f"\nSearching for objects matching: '{args.description}'")
            search_results, saved_images = search_video(
                args.description, args.video, args.output_dir,
                object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path,
                args.k, args.max_images, llm
            )
    else:
        dataset = init_dataset(args.dataset)
        # Process video with embeddings
        if args.process_only:
            if args.video_range is not None:
                start_video, end_video = map(int, args.video_range.split('-'))
                print(f"Processing videos {start_video} to {end_video}...")
                for video_id in range(start_video, end_video + 1):
                    video, object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path = dataset.get_video(video_id)
                    print(f"Processing video {video_id} with embedding generation...")
                    start_time = time.time()
                    result = process_video(video, object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path, llm)
                    end_time = time.time()
                    total_time = end_time - start_time
                    json.dump({
                        "video_id": video_id,
                        "time": total_time
                    }, open(f"results/{args.dataset}_{args.model}_{video_id}.json", "w"))
                    print(f"Time taken for video {video_id} processing: {total_time} seconds")
            elif args.video_id == -1:
                for video_id in range(get_video_idx(args.dataset)[0], get_video_idx(args.dataset)[1] + 1):
                    video, object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path = dataset.get_video(video_id)
                    print(f"Processing video {video_id} with embedding generation...")
                    start_time = time.time()
                    result = process_video(video, object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path, llm)
                    end_time = time.time()
                    print(f"Time taken for video processing: {end_time - start_time} seconds")
            else:
                video, object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path = dataset.get_video(args.video_id)
                print(f"Processing video {args.video_id} with embedding generation...")
                start_time = time.time()
                result = process_video(video, object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path, llm)
                end_time = time.time()
                print(f"Time taken for video processing: {end_time - start_time} seconds")
        if args.question_id == -1:
            for video_id in range(1, len(dataset) + 1):
                if args.video_id != -1 and video_id != args.video_id:
                    continue
                video, object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path = dataset.get_video(video_id)
                for question_id in range(len(dataset.get_video_info(video_id)["qa"])):
                    search_results, saved_images = search_video(
                        dataset.get_video_info(video_id)["qa"][question_id]["question"], video, args.output_dir,
                        object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path,
                        args.k, args.max_images, llm, 
                        question_id
                    )
        elif args.question_id != -1:
            video, object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path = dataset.get_video(args.video_id)
            question = dataset.get_video_info(args.video_id)["qa"][args.question_id]["question"]
            search_results, saved_images = search_video(
                question, video, args.output_dir,
                object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path,
                args.k, args.max_images, llm, 
                args.question_id
            )

if __name__ == "__main__":
    main()
