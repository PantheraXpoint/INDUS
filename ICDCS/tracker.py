import numpy as np
import yaml
from typing import List, Dict, Tuple


class CustomTracker:
    """
    Custom object tracker implementation
    """
    
    def __init__(self, config_path: str = "config/tracker.yaml"):
        """
        Initialize the custom tracker
        
        Args:
            config_path: Path to tracker configuration file
        """
        # Load tracker configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['tracker']
        
        # Tracking variables
        self.next_object_id = 0
        self.tracks = {}  # track_id -> track_data
        self.disappeared = {}  # track_id -> frames_missing
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance(self, box1: List[float], box2: List[float]) -> float:
        """Calculate distance between box centers"""
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def _create_track(self, detection: Dict) -> int:
        """Create a new track"""
        track_id = self.next_object_id
        self.next_object_id += 1
        
        self.tracks[track_id] = {
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'class_id': detection['class_id'],
            'class_name': detection['class_name'],
            'velocity': [0.0, 0.0]
        }
        self.disappeared[track_id] = 0
        return track_id
    
    def _update_track(self, track_id: int, detection: Dict):
        """Update existing track with new detection"""
        old_bbox = self.tracks[track_id]['bbox']
        new_bbox = detection['bbox']
        
        # Calculate velocity
        velocity = [
            (new_bbox[0] - old_bbox[0]) * self.config['velocity_smoothing'],
            (new_bbox[1] - old_bbox[1]) * self.config['velocity_smoothing']
        ]
        
        # Update track
        self.tracks[track_id].update({
            'bbox': new_bbox,
            'confidence': detection['confidence'],
            'velocity': velocity
        })
        self.disappeared[track_id] = 0
    
    def _match_detections_to_tracks(self, detections: List[Dict]):
        """Match detections to existing tracks using IoU and distance"""
        if not self.tracks or not detections:
            return
        
        # Calculate distance matrix
        track_ids = list(self.tracks.keys())
        distances = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracks[track_id]['bbox']
            for j, detection in enumerate(detections):
                distances[i, j] = self._calculate_distance(track_bbox, detection['bbox'])
        
        # Find best matches
        used_tracks = set()
        used_detections = set()
        
        # Sort by distance
        indices = np.unravel_index(np.argsort(distances.ravel()), distances.shape)
        
        for i, j in zip(indices[0], indices[1]):
            if i in used_tracks or j in used_detections:
                continue
            
            track_id = track_ids[i]
            detection = detections[j]
            distance = distances[i, j]
            
            # Check distance threshold
            if distance > self.config['max_distance']:
                continue
            
            # Check IoU threshold
            iou = self._calculate_iou(self.tracks[track_id]['bbox'], detection['bbox'])
            if iou < self.config['min_iou']:
                continue
            
            # Update track
            self._update_track(track_id, detection)
            used_tracks.add(i)
            used_detections.add(j)
        
        # Create new tracks for unmatched detections
        for j, detection in enumerate(detections):
            if j not in used_detections and detection['confidence'] >= self.config['min_confidence']:
                self._create_track(detection)
        
        # Handle disappeared tracks
        for i, track_id in enumerate(track_ids):
            if i not in used_tracks:
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.config['max_disappeared']:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracking with new detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of tracked objects with track_id
        """
        if len(detections) == 0:
            # No detections, increment disappeared count
            for track_id in list(self.tracks.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.config['max_disappeared']:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            return []
        
        if len(self.tracks) == 0:
            # No existing tracks, create new ones
            for detection in detections:
                if detection['confidence'] >= self.config['min_confidence']:
                    self._create_track(detection)
        else:
            # Match detections to existing tracks
            self._match_detections_to_tracks(detections)
        
        # Return current tracks
        tracked_objects = []
        for track_id, track_data in self.tracks.items():
            if self.disappeared.get(track_id, 0) == 0:  # Only active tracks
                tracked_objects.append({
                    'bbox': track_data['bbox'],
                    'confidence': track_data['confidence'],
                    'class_id': track_data['class_id'],
                    'class_name': track_data['class_name'],
                    'track_id': track_id
                })
        
        return tracked_objects
    
    def get_track_count(self) -> int:
        """Get number of active tracks"""
        return len([t for t in self.tracks.values() if self.disappeared.get(t, 0) == 0])
    
    def reset(self):
        """Reset tracker state"""
        self.next_object_id = 0
        self.tracks = {}
        self.disappeared = {}
