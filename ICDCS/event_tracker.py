import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import List, Tuple, Dict, Optional
import argparse
import os
import sys
from PIL import Image
from AVA.events import batch_generate_descriptions_external
from llms.init_model import init_model

# Add embeddings directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embeddings'))

from embeddings.FAISSDB import FAISSDB
from embeddings.Milvus import MilvusDB
from JinaCLIP import JinaCLIP
from llms.BaseModel import BaseVideoModel
import json
from AVA.utils import chunk_text


class EventTracker:
    """
    Event tracking
    """
    
    def __init__(self, llm: BaseVideoModel, embedding_model: Optional[JinaCLIP] = None, faiss_db_path: str = "event_embeddings.faiss"):
        """
        Initialize the event tracker
        
        Args:
            embedding_model: JinaCLIP embedding model for generating embeddings
            faiss_db_path: Path to FAISS database
            llm: LLM model for generating descriptions
        """
        # Initialize embedding system if model is provided
        self.embedding_model = embedding_model
        self.llm = llm
        self.faiss_db = None
        if self.embedding_model is not None:
            self.faiss_db = MilvusDB(faiss_db_path, self.embedding_model.embedding_dim)

    def process_chunk(self, frames: list, frame_indices: list, detected_objects: list, video_chunk_num_frames: int, frame_skip: int):
        """
        Process a chunk of frames
        """
        descriptions = batch_generate_descriptions_external(self.llm, batch_size=video_chunk_num_frames,
                                                video_chunk_num_frames=video_chunk_num_frames,
                                                frames=frames,
                                                frame_indices=frame_indices,
                                                frame_skip=frame_skip,
                                                detected_objects=detected_objects)
        # TODO: add semantic chunking
        
        self._add_descriptions(descriptions)

    def process_video(self, video_path: str):
        """
        Process a video file for event tracking
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return []
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set processing fps to 10fps
        processing_fps = 3
        chunk_duration = 3
        frame_skip = max(1, original_fps // processing_fps)  # Skip frames to achieve 10fps processing
        
        frame_count = 0
        start_time = time.time()
        
        print(f"Processing video (event tracking): {video_path}")
        print(f"Resolution: {width}x{height}, Original FPS: {original_fps}")
        print(f"Processing at ~{processing_fps} FPS (every {frame_skip} frames)")
        
        processed_frame_count = 0
        frame_indices = []
        frames = []
        video_chunk_num_frames = int(processing_fps * chunk_duration)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every frame_skip frames for 10fps processing
            if frame_count % frame_skip == 0:
                
                frame_indices.append(frame_count)
                frames.append(Image.fromarray(frame))
                
                if len(frame_indices) == video_chunk_num_frames:
                    self.process_chunk(frames, frame_indices, video_chunk_num_frames, frame_skip)
                    frame_indices = []
                    frames = []
                
                processed_frame_count += 1
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        
        elapsed_time = time.time() - start_time
        processing_avg_fps = processed_frame_count / elapsed_time
        
        print(f"Processing complete!")
        print(f"Total input frames: {frame_count}")
        print(f"Processed frames: {processed_frame_count}")
        print(f"Processing speed: {processing_avg_fps:.1f} FPS")
        
        return

    def _add_descriptions(self, descriptions: list):
        """
        Add descriptions to the event database
        """
        if self.faiss_db is not None:
            try:
                for description in descriptions:
                    chunks = chunk_text(description["description"])
                    embeddings = self.embedding_model.get_text_features(chunks)
                    for chunk_index, chunk in enumerate(chunks):
                        faiss_id = self.faiss_db.add_embedding(embeddings[chunk_index], str(description["duration"][0]) + "_" + str(chunk_index), {
                            'start_time': description["duration"][0],
                            'end_time': description["duration"][1],
                            'description': chunk,
                            'full_description': description["description"],
                            'objects': description["objects"]
                        })
                        print(f"Added description {chunk} to FAISS database with ID {faiss_id}")
            except Exception as e:
                print(f"Error adding descriptions to FAISS database: {e}")

def main():
    """Main function for testing the ObjectDetectorTracker"""
    parser = argparse.ArgumentParser(description='Object Detection and Tracking with YOLOv11')
    parser.add_argument('--faiss-db-path', type=str, default='database/event_embeddings.faiss',
                       help='Path to FAISS database file')
    parser.add_argument('--video', type=str, 
                       help='Path to input video file')
    
    args = parser.parse_args()
    
    # Check if output is required
    if not args.video:
        print("Error: --video is required")
        parser.print_help()
        return
    
    # Initialize embedding model if requested
    print("Initializing JinaCLIP embedding model...")
    try:
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        print("JinaCLIP model initialized successfully")
    except Exception as e:
        print(f"Error initializing JinaCLIP model: {e}")
        print("Continuing without embedding functionality...")

    llm = init_model("qwenvl", 1)
    
    # Initialize detector and tracker
    event_tracker = EventTracker(
        llm=llm,
        embedding_model=embedding_model,
        faiss_db_path=args.faiss_db_path
    )
    
    if args.video:
        event_tracker.process_video(args.video)
    else:
        print("Error: Please specify either --video or --image input file")
        parser.print_help()


if __name__ == "__main__":
    main()
