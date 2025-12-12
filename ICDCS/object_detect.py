import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import List, Tuple, Dict, Optional
import argparse
import os
import sys
from PIL import Image

# Add embeddings directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embeddings'))

from embeddings.FAISSDB import FAISSDB
from embeddings.Milvus import MilvusDB
from embeddings.SQLiteDB import SQLiteDB
from JinaCLIP import JinaCLIP
from ICDCS.tracker import CustomTracker
MAX_TRACKED_OBJECTS = 30


class ObjectDetectorTracker:
    """
    Object Detection and Tracking using YOLOv11 with built-in ultralytics tracking
    """
    
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.5, 
                 iou_threshold: float = 0.5, tracker_config: str = "config/tracker.yaml",
                 embedding_model: Optional[JinaCLIP] = None, faiss_db_path: str = "embeddings.faiss",
                 sqlite_db_path: str = "tracked_objects.db", use_persistent_sqlite: bool = False):
        """
        Initialize the object detector and tracker
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            tracker_config: Path to tracker configuration file
            embedding_model: JinaCLIP embedding model for generating embeddings
            faiss_db_path: Path to FAISS database
            sqlite_db_path: Path to SQLite database
            use_persistent_sqlite: If True, use persistent SQLite connection (faster but uses more resources).
                                   If False, use per-request connections (lighter, better for vLLM)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize custom tracker
        self.tracker = CustomTracker(tracker_config)
        
        # Initialize embedding system if model is provided
        self.embedding_model = embedding_model
        self.faiss_db = None
        self.sqlite_db = None
        
        if self.embedding_model is not None:
            self.faiss_db = MilvusDB(faiss_db_path, self.embedding_model.embedding_dim)
            self.sqlite_db = SQLiteDB(sqlite_db_path)
        
        # Colors for visualization (BGR format)
        self.colors = self._generate_colors(80)  # COCO dataset has 80 classes
        self.batch_embedding = []
        
        
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class"""
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def detect_and_track(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and track objects using custom tracking algorithm
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of tracked objects with bbox, confidence, class_id, class_name, track_id
        """
        # Detect objects
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        # Update tracking using custom tracker
        tracked_objects = self.tracker.update(detections)
        return tracked_objects
    
    def visualize_results(self, frame: np.ndarray, tracked_objects: List[Dict], 
                         show_confidence: bool = True, show_tracking_id: bool = True) -> np.ndarray:
        """
        Visualize detection and tracking results on the frame
        
        Args:
            frame: Input frame
            tracked_objects: List of tracked objects
            show_confidence: Whether to show confidence scores
            show_tracking_id: Whether to show tracking IDs
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        for obj in tracked_objects:
            bbox = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            track_id = obj.get('track_id')
            
            # Get color for this class
            color = self.colors[obj['class_id'] % len(self.colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_parts = [class_name]
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            if show_tracking_id and track_id is not None:
                label_parts.append(f"ID:{track_id}")
            
            label = " ".join(label_parts)
            
            # Get label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw label background
            cv2.rectangle(vis_frame, (x1, y1 - label_height - baseline), 
                         (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x1, y1 - baseline), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw centroid
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(vis_frame, centroid, 5, color, -1)
        
        return vis_frame
    
    def process_video(self, video_path: str, output_path: str, visualize: bool = True) -> List[List[Dict]]:
        """
        Process a video file for object detection and tracking
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            visualize: Whether to visualize results on output video
            
        Returns:
            List of tracked objects for each frame
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
        processing_fps = 10
        frame_skip = max(1, original_fps // processing_fps)  # Skip frames to achieve 10fps processing
        
        # Setup video writer (keep original fps for output)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        all_tracked_objects = {}  # Store all tracked objects for each frame
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, Original FPS: {original_fps}")
        print(f"Processing at ~{processing_fps} FPS (every {frame_skip} frames)")
        print(f"Output will be saved to: {output_path}")
        print(f"Visualization: {'Enabled' if visualize else 'Disabled'}")
        
        processed_frame_count = 0
        total_frames_written = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Only process every frame_skip frames for 10fps processing
            if frame_count % frame_skip == 0:
                # Detect and track objects
                tracked_objects = self.detect_and_track(frame)
                
                processed_frame_count += 1
                
                if visualize:
                    # Visualize results
                    vis_frame = self.visualize_results(frame, tracked_objects)
                    
                    # Add frame info
                    info_text = f"Frame: {frame_count} | Objects: {len(tracked_objects)}"
                    cv2.putText(vis_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Write visualized frame multiple times to maintain original fps
                    for _ in range(frame_skip):
                        writer.write(vis_frame)
                        total_frames_written += 1
                else:
                    # Write original frame multiple times to maintain original fps
                    for _ in range(frame_skip):
                        writer.write(frame)
                        total_frames_written += 1
            else:
                # For non-processed frames, use the last processed frame result
                if visualize and processed_frame_count > 0:
                    # Use the last visualized frame
                    for _ in range(frame_skip):
                        writer.write(vis_frame)
                        total_frames_written += 1
                else:
                    # Use original frame
                    for _ in range(frame_skip):
                        writer.write(frame)
                        total_frames_written += 1
            
            # Print progress every 100 processed frames
            if processed_frame_count % 100 == 0 and processed_frame_count > 0:
                elapsed_time = time.time() - start_time
                fps_current = processed_frame_count / elapsed_time
                print(f"Processed {processed_frame_count} frames at {fps_current:.1f} FPS")
        
        # Cleanup
        cap.release()
        writer.release()
        
        elapsed_time = time.time() - start_time
        processing_avg_fps = processed_frame_count / elapsed_time
        print(f"Processing complete!")
        print(f"Total input frames: {frame_count}")
        print(f"Processed frames: {processed_frame_count}")
        print(f"Output frames written: {total_frames_written}")
        print(f"Processing speed: {processing_avg_fps:.1f} FPS")
        print(f"Output video saved to: {output_path}")
        print(f"Total unique tracked objects: {len(all_tracked_objects)}")

        return all_tracked_objects
    
    def process_frame(self, frame: np.ndarray, frame_count: int, event_id: int = None, detected_objects: List = None):
        # Detect and track objects
        tracked_objects = self.detect_and_track(frame)
        
        # Store tracked objects with history
        for tracked_object in tracked_objects:
            tracked_object["event_id"] = event_id
            tracked_object["frame_numbers"] = frame_count

            existing = self._add_tracked_object(tracked_object)
            if not existing:
                self._generate_embedding(frame, tracked_object)
            if str(tracked_object['track_id']) not in detected_objects:
                detected_objects.append(str(tracked_object['track_id']))

        # vis_frame = self.visualize_results(frame, tracked_objects)
        # cv2.imwrite(f"debug/tracked_objects_{frame_count}.jpg", vis_frame)
        return detected_objects

    def process_video_tracking_only(self, video_path: str) -> List[List[Dict]]:
        """
        Process a video file for object detection and tracking without saving video or visualization
        
        Args:
            video_path: Path to input video
            
        Returns:
            List of tracked objects for each processed frame
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
        processing_fps = 10
        frame_skip = max(1, original_fps // processing_fps)  # Skip frames to achieve 10fps processing
        
        frame_count = 0
        start_time = time.time()
        # Save tracked objects to SQLite database
        if self.sqlite_db is not None:
            # Add video info
            self.sqlite_db.add_video_info(video_path, width, height, original_fps, frame_count, processing_fps)
        
        print(f"Processing video (tracking only): {video_path}")
        print(f"Resolution: {width}x{height}, Original FPS: {original_fps}")
        print(f"Processing at ~{processing_fps} FPS (every {frame_skip} frames)")
        
        processed_frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Only process every frame_skip frames for 10fps processing
            if frame_count % frame_skip == 0:
                
                self.process_frame(frame, frame_count)
                
                processed_frame_count += 1
                
                # Print progress every 100 processed frames
                if processed_frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    fps_current = processed_frame_count / elapsed_time
                    print(f"Processed {processed_frame_count} frames at {fps_current:.1f} FPS")
        
        # Cleanup
        cap.release()
        
        elapsed_time = time.time() - start_time
        processing_avg_fps = processed_frame_count / elapsed_time
        
        print(f"Processing complete!")
        print(f"Total input frames: {frame_count}")
        print(f"Processed frames: {processed_frame_count}")
        print(f"Processing speed: {processing_avg_fps:.1f} FPS")
    
    def _add_tracked_object(self, tracked_object: Dict):
        """
        Add tracked object to SQLite database - only insert new or update existing
        """
        if self.sqlite_db is not None:
            obj_data = tracked_object
            existing = self.sqlite_db.add_tracked_object(
                track_id=obj_data["track_id"],
                class_id=obj_data["class_id"],
                class_name=obj_data["class_name"],
                bbox_history=[int(x) for x in obj_data["bbox"]],
                confidence_history=obj_data["confidence"],
                frame_numbers=obj_data["frame_numbers"],
                event_id=obj_data["event_id"]
            )
            return existing
        return False
    
    def _generate_embedding(self, frame: np.ndarray, tracked_object: Dict):
        """
        Generate embedding for a bounding box in the frame
        """
        # Generate embedding for new track if embedding model is available
        if self.embedding_model is not None and self.faiss_db is not None:
            try:
                # Extract ROI from frame
                height, width = frame.shape[:2]  # OpenCV uses (height, width) format
                x1, y1, x2, y2 = map(int, tracked_object['bbox'])
                x1 = max(0, min(x1, width-1))
                y1 = max(0, min(y1, height-1))
                x2 = max(x1+1, min(x2, width))
                y2 = max(y1+1, min(y2, height))
                
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Convert BGR to RGB for JinaCLIP
                    # roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_rgb = roi
                    # Generate embedding
                    roi_pil = Image.fromarray(roi_rgb)

                    id = str(tracked_object["track_id"])
                    # Store in FAISS database
                    metadata = {
                        'track_id': tracked_object["track_id"],
                        'frame_number': tracked_object["frame_numbers"],
                        'bbox': tracked_object["bbox"],
                        'confidence': tracked_object["confidence"],
                        'class_id': tracked_object["class_id"],
                        'class_name': tracked_object["class_name"],
                        'event_id': tracked_object["event_id"],
                    }
                    self.batch_embedding.append((roi_pil, id, metadata))
                    if len(self.batch_embedding) == 64:
                        roi_pils, ids, metadatas = zip(*self.batch_embedding)    
                        embeddings = self.embedding_model.get_image_features(roi_pils)
                        self.batch_embedding = []
                        for embedding, id, metadata in zip(embeddings, ids, metadatas):
                            faiss_id = self.faiss_db.add_embedding(embedding, id, metadata)
                    
            except Exception as e:
                print(f"Error generating embedding for tracks {ids}: {e}")

def main():
    """Main function for testing the ObjectDetectorTracker"""
    parser = argparse.ArgumentParser(description='Object Detection and Tracking with YOLOv11')
    parser.add_argument('--model', type=str, default='checkpoints/yolo11l.pt', 
                       help='Path to YOLO model weights')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, 
                       help='IoU threshold for NMS')
    parser.add_argument('--tracker-config', type=str, default='config/tracker.yaml',
                       help='Path to tracker configuration file')
    parser.add_argument('--video', type=str, 
                       help='Path to input video file')
    parser.add_argument('--image', type=str,
                       help='Path to input image file')
    parser.add_argument('--output', type=str,
                       help='Path to output file (video or image)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization on output video')
    parser.add_argument('--tracking-only', action='store_true',
                       help='Process video for tracking only (no output video)')
    parser.add_argument('--enable-embeddings', action='store_true',
                       help='Enable embedding generation for new track IDs')
    parser.add_argument('--faiss-db', type=str, default='database/object_embeddings.faiss',
                       help='Path to FAISS database file')
    parser.add_argument('--sqlite-db', type=str, default='database/tracked_objects.db',
                       help='Path to SQLite database file')
    
    args = parser.parse_args()
    
    # Check if output is required
    if not args.tracking_only and not args.output:
        print("Error: --output is required when not using --tracking-only")
        parser.print_help()
        return
    
    # Initialize embedding model if requested
    embedding_model = None
    if args.enable_embeddings:
        print("Initializing JinaCLIP embedding model...")
        try:
            embedding_model = JinaCLIP("jinaai/jina-clip-v1")
            print("JinaCLIP model initialized successfully")
        except Exception as e:
            print(f"Error initializing JinaCLIP model: {e}")
            print("Continuing without embedding functionality...")
    
    # Initialize detector and tracker
    detector = ObjectDetectorTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        tracker_config=args.tracker_config,
        embedding_model=embedding_model,
        faiss_db_path=args.faiss_db,
        sqlite_db_path=args.sqlite_db
    )
    
    if args.video:
        if args.tracking_only:
            # Video file processing (tracking only)
            tracked_objects = detector.process_video_tracking_only(video_path=args.video)
            print(f"Returned tracking data for {len(tracked_objects)} frames")
        else:
            # Video file processing with output
            tracked_objects = detector.process_video(
                video_path=args.video, 
                output_path=args.output,
                visualize=not args.no_visualize
            )
            print(f"Returned tracking data for {len(tracked_objects)} objects")
    elif args.image:
        # Image file processing
        detector.process_image(image_path=args.image, output_path=args.output)
    else:
        print("Error: Please specify either --video or --image input file")
        parser.print_help()


if __name__ == "__main__":
    main()
