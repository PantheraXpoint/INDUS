import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
# FAISSDB import removed - not used, code uses MilvusDB instead
# Use proper relative imports for embeddings package
from .Milvus import MilvusDB
from .SQLiteDB import SQLiteDB
from .JinaCLIP import JinaCLIP

class SearchSystem:
    """
    Object search functionality using embeddings and databases
    """
    
    def __init__(self, faiss_db_path: str, sqlite_db_path: Optional[str], embedding_model: JinaCLIP):
        """
        Initialize object search system
        
        Args:
            faiss_db_path: Path to FAISS database
            sqlite_db_path: Path to SQLite database
            embedding_model: JinaCLIP embedding model
        """
        self.faiss_db = MilvusDB(faiss_db_path, embedding_model.embedding_dim)
        if sqlite_db_path is not None:
            self.sqlite_db = SQLiteDB(sqlite_db_path)
        else:
            self.sqlite_db = None
        self.embedding_model = embedding_model
    
    def search_by_description(self, description: str, k: Optional[int] = None, filter_expr: Optional[str] = None) -> List[Dict]:
        """
        Search for objects by text description
        
        Args:
            description: Text description of the object to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing search results with metadata
        """
        # Generate embedding for the description
        description_embedding = self.embedding_model.get_text_features([description])[0]
        
        # Search in FAISS database
        if k is None:
            k = len(self.faiss_db.get_all_ids())
        faiss_results = self.faiss_db.search(description_embedding, k, filter_expr)
        
        # Get additional information from SQLite database
        results = []
        for faiss_id, similarity_score, metadata in faiss_results:
            id = metadata['id']
            sqlite_info = None
            if self.sqlite_db is not None:
                sqlite_info = self.sqlite_db.get_tracked_object(id)
            
            if sqlite_info:
                result = {
                    'id': id,
                    'similarity_score': similarity_score,
                    'class_name': sqlite_info['class_name'],
                    'class_id': sqlite_info['class_id'],
                    'first_frame': sqlite_info['first_frame'],
                    'last_frame': sqlite_info['last_frame'],
                    'total_frames': sqlite_info['total_frames'],
                    'bbox_history': sqlite_info['bbox_history'],
                    'confidence_history': sqlite_info['confidence_history'],
                    'frame_numbers': sqlite_info['frame_numbers'],
                    'event_id': sqlite_info['event_id'],
                    'faiss_metadata': metadata
                }
                results.append(result)
            else:
                result = {
                    'id': id,
                    'similarity_score': similarity_score,
                    'faiss_metadata': metadata
                }
                results.append(result)
        return results
    
    def search_and_extract(self, description: str, video_path: str, output_dir: str, 
                          k: int = 5, max_images: int = 10) -> Tuple[List[Dict], List[str]]:
        """
        Complete search and extraction pipeline
        
        Args:
            description: Text description to search for
            video_path: Path to original video
            output_dir: Directory to save extracted images
            k: Number of search results to return
            max_images: Maximum images to extract per track
            
        Returns:
            Tuple of (search_results, saved_image_paths)
        """
        # Search for objects
        search_results = self.search_by_description(description, k)
        
        if not search_results:
            print(f"No objects found matching description: {description}")
            return [], []
        
        print(f"Found {len(search_results)} objects matching description: {description}")
        
        # Extract images
        saved_images = filter_and_extract_bounding_box(
            video_path, search_results, output_dir, max_images
        )
        
        print(f"Extracted {len(saved_images)} images to {output_dir}")
        
        return search_results, saved_images
    
    def get_track_statistics(self) -> Dict:
        """
        Get statistics about tracked objects
        
        Returns:
            Dictionary with statistics
        """
        all_objects = self.sqlite_db.get_all_tracked_objects()
        
        if not all_objects:
            return {
                'total_tracks': 0,
                'class_distribution': {},
                'total_frames_tracked': 0,
                'average_track_length': 0
            }
        
        class_distribution = {}
        total_frames = 0
        
        for obj in all_objects:
            class_name = obj['class_name']
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
            total_frames += obj['total_frames']
        
        return {
            'total_tracks': len(all_objects),
            'class_distribution': class_distribution,
            'total_frames_tracked': total_frames,
            'average_track_length': total_frames / len(all_objects) if all_objects else 0
        }

def filter_and_extract_bounding_box(video_path: str, search_results: List[Dict], 
                                  output_dir: str, max_images: int = 10) -> List[str]:
    """
    Extract and save bounding box images from video based on search results
    
    Args:
        video_path: Path to original video file
        search_results: Results from search_by_description
        output_dir: Directory to save extracted images
        max_images: Maximum number of images to extract per track
        
    Returns:
        List of paths to saved images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_images = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return saved_images
    
    for result in search_results:
        # print("Entity confidence score: ", result['similarity_score'])
        id = result['id']
        bbox_history = result.get('bbox_history', result.get('filtered_bboxes', []))
        frame_numbers = result.get('frame_numbers', result.get('filtered_frames', []))
        class_name = result['class_name']
        
        # Limit number of images to extract
        num_images = min(len(frame_numbers), max_images)
        step = max(1, len(frame_numbers) // num_images)
        
        for i in range(0, len(frame_numbers), step):
            frame_num = frame_numbers[i]
            bbox = bbox_history[i]
            
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Extract bounding box
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                frame_with_bbox = frame.copy()
                cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_with_bbox, f"ID:{id} {class_name}", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                bbox_filename = f"track_{id}_frame_{frame_num}_{class_name}_bbox.jpg"
                bbox_filepath = os.path.join(output_dir, bbox_filename)
                cv2.imwrite(bbox_filepath, frame_with_bbox)
                print(f"Saved image to {bbox_filepath}")
                saved_images.append(bbox_filepath)
    
    cap.release()
    return saved_images
