"""
Uniform Sampling Module for VLM Direct Inference
Samples frames uniformly from entire video without retrieval
"""
import os
import json
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import sys

# Add ICDCS to path for overlap functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AVA.prompt import PROMPTS

# Import overlap functions from ICDCS/calculate_accuracy.py
try:
    from ICDCS.calculate_accuracy import (
        has_binary_overlap_helper,
        overlap_reference_helper,
        time_to_seconds
    )
except ImportError:
    print("Warning: Could not import overlap functions from ICDCS/calculate_accuracy.py")
    has_binary_overlap_helper = None
    overlap_reference_helper = None


class UniformSampling:
    """
    Uniform sampling class for baseline VLM inference
    Samples frames uniformly from entire video without using retrieval
    """
    
    def __init__(self, video, llm_model):
        """
        Initialize UniformSampling
        
        Args:
            video: VideoRepresentation object
            llm_model: VLM model instance
        """
        self.video = video
        self.llm_model = llm_model
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load video configuration from config.json"""
        config_path = os.path.join(self.video.work_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def calculate_sampling(self, target_fps: Optional[float] = None, 
                          max_frames: Optional[int] = None) -> Tuple[np.ndarray, float, int]:
        """
        Calculate frame indices for uniform sampling
        
        Args:
            target_fps: Target FPS for sampling (mutually exclusive with max_frames)
            max_frames: Maximum number of frames to sample (mutually exclusive with target_fps)
        
        Returns:
            Tuple of (frame_indices, effective_fps, actual_frames)
        """
        if target_fps is None and max_frames is None:
            raise ValueError("Either target_fps or max_frames must be specified")
        
        if target_fps is not None and max_frames is not None:
            raise ValueError("Cannot specify both target_fps and max_frames")
        
        original_fps = self.config['fps']
        total_frames = self.config['frame_count']
        duration = self.config['duration']
        
        if target_fps is not None:
            # Calculate how many frames we want at target FPS
            desired_frames = int(duration * target_fps)
            # Uniformly sample from [0, total_frames-1]
            if desired_frames >= total_frames:
                # If target fps results in more frames than available, use all
                frame_indices = np.arange(total_frames, dtype=int)
            else:
                frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int)
            effective_fps = target_fps
            actual_frames = len(frame_indices)
        
        else:  # max_frames is not None
            # Directly sample max_frames uniformly from all original frames
            if max_frames >= total_frames:
                # If max_frames exceeds total, use all frames
                frame_indices = np.arange(total_frames, dtype=int)
            else:
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            effective_fps = max_frames / duration
            actual_frames = len(frame_indices)
        
        return frame_indices, effective_fps, actual_frames
    
    def _get_cache_dir(self, target_fps: Optional[float] = None, 
                       max_frames: Optional[int] = None) -> str:
        """Get cache directory path based on sampling parameters"""
        cache_base = os.path.join(self.video.work_dir, "uniform_sampling")
        
        if target_fps is not None:
            cache_key = f"{target_fps}fps"
        else:
            cache_key = f"{max_frames}frames"
        
        return os.path.join(cache_base, cache_key)
    
    def _save_frames_to_cache(self, frames: List, frame_indices: np.ndarray, cache_dir: str):
        """Save extracted frames to cache"""
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save frames
        for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
            frame_path = os.path.join(cache_dir, f"{int(frame_idx)}.jpg")
            if isinstance(frame, Image.Image):
                frame.save(frame_path)
            else:
                # If it's already a path string, copy the file
                import shutil
                shutil.copy(frame, frame_path)
        
        # Save metadata
        metadata = {
            'frame_indices': frame_indices.tolist(),
            'num_frames': len(frame_indices)
        }
        metadata_path = os.path.join(cache_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_frames_from_cache(self, cache_dir: str) -> Tuple[List, np.ndarray]:
        """Load frames from cache"""
        metadata_path = os.path.join(cache_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return None, None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        frame_indices = np.array(metadata['frame_indices'])
        frames = []
        
        for frame_idx in frame_indices:
            frame_path = os.path.join(cache_dir, f"{int(frame_idx)}.jpg")
            if not os.path.exists(frame_path):
                return None, None
            frames.append(Image.open(frame_path))
        
        return frames, frame_indices
    
    def extract_uniform_frames(self, target_fps: Optional[float] = None, 
                               max_frames: Optional[int] = None,
                               use_cache: bool = True) -> Tuple[List, Dict]:
        """
        Extract frames uniformly from entire video
        
        Args:
            target_fps: Target FPS for sampling
            max_frames: Maximum number of frames to sample
            use_cache: Whether to use cached frames
        
        Returns:
            Tuple of (frames, sampling_info_dict)
        """
        # Calculate sampling
        frame_indices, effective_fps, actual_frames = self.calculate_sampling(
            target_fps=target_fps, max_frames=max_frames
        )
        
        # Try to load from cache
        if use_cache:
            cache_dir = self._get_cache_dir(target_fps=target_fps, max_frames=max_frames)
            cached_frames, cached_indices = self._load_frames_from_cache(cache_dir)
            
            if cached_frames is not None and np.array_equal(cached_indices, frame_indices):
                print(f"Loaded {len(cached_frames)} frames from cache")
                sampling_info = {
                    'strategy': 'target_fps' if target_fps is not None else 'max_frames',
                    'target_fps': target_fps,
                    'max_frames': max_frames,
                    'effective_fps': effective_fps,
                    'actual_frames_used': actual_frames,
                    'video_duration': self.config['duration'],
                    'original_fps': self.config['fps'],
                    'frame_indices': frame_indices.tolist(),
                    'frame_indices_sample': frame_indices[:min(100, len(frame_indices))].tolist(),
                    'cached': True
                }
                return cached_frames, sampling_info
        
        # Extract frames from video
        print(f"Extracting {actual_frames} frames uniformly from video...")
        frames = self.video.get_frames_by_indices(frame_indices.tolist())
        
        # Save to cache if enabled
        if use_cache:
            cache_dir = self._get_cache_dir(target_fps=target_fps, max_frames=max_frames)
            self._save_frames_to_cache(frames, frame_indices, cache_dir)
            print(f"Saved {len(frames)} frames to cache: {cache_dir}")
        
        sampling_info = {
            'strategy': 'target_fps' if target_fps is not None else 'max_frames',
            'target_fps': target_fps,
            'max_frames': max_frames,
            'effective_fps': effective_fps,
            'actual_frames_used': actual_frames,
            'video_duration': self.config['duration'],
            'original_fps': self.config['fps'],
            'frame_indices': frame_indices.tolist(),
            'frame_indices_sample': frame_indices[:min(100, len(frame_indices))].tolist(),
            'cached': False
        }

        return frames, sampling_info
    
    def calculate_ground_truth_overlap(self, frame_indices: np.ndarray, 
                                       time_reference: str) -> Dict:
        """
        Calculate overlap between sampled frames and ground truth
        
        Args:
            frame_indices: Array of frame indices used
            time_reference: Ground truth time string (e.g., "00:1:20-00:5:30")
        
        Returns:
            Dict with overlap metrics
        """
        if not time_reference or time_reference in ["N/A", "", "None", "None-None"]:
            return {
                'has_ground_truth': False,
                'time_reference': time_reference if time_reference else 'N/A',
                'binary_overlap': None,
                'percentage_overlap': None,
                'num_frames_checked': len(frame_indices)
            }
        
        if has_binary_overlap_helper is None or overlap_reference_helper is None:
            print("Warning: Overlap functions not available, skipping overlap calculation")
            return {
                'has_ground_truth': True,
                'time_reference': time_reference,
                'binary_overlap': None,
                'percentage_overlap': None,
                'num_frames_checked': len(frame_indices),
                'error': 'overlap_functions_not_available'
            }
        
        # Convert frame indices to time intervals (in seconds)
        fps = self.config['fps']
        time_intervals = []
        
        for frame_idx in frame_indices:
            start_time = frame_idx / fps
            end_time = (frame_idx + 1) / fps
            time_intervals.append((start_time, end_time))
        
        # Calculate overlap using functions from calculate_accuracy.py
        try:
            binary_overlap = has_binary_overlap_helper(time_reference, time_intervals)
            percentage_overlap = overlap_reference_helper(time_reference, time_intervals)
            
            return {
                'has_ground_truth': True,
                'time_reference': time_reference,
                'binary_overlap': binary_overlap,      # 1.0 if ANY overlap, 0.0 otherwise
                'percentage_overlap': percentage_overlap,  # 0.0 to 1.0
                'num_frames_checked': len(frame_indices)
            }
        except Exception as e:
            print(f"Error calculating overlap: {e}")
            return {
                'has_ground_truth': True,
                'time_reference': time_reference,
                'binary_overlap': None,
                'percentage_overlap': None,
                'num_frames_checked': len(frame_indices),
                'error': str(e)
            }
    
    def create_prompt(self, question: str) -> str:
        """
        Create prompt for VLM using uniform sampling template
        
        Args:
            question: Question text (already includes options)
        
        Returns:
            Formatted prompt string
        """
        return PROMPTS["uniform_sampling_answer"].format(user_query=question)
    
    def prepare_question_data(self, question: str, time_reference: str,
                              target_fps: Optional[float] = None,
                              max_frames: Optional[int] = None,
                              use_cache: bool = True) -> Tuple[List, Dict, Dict]:
        """
        Prepare data for a single question (for batching)
        
        Args:
            question: Question text
            time_reference: Ground truth time reference
            target_fps: Target FPS for sampling
            max_frames: Maximum number of frames
            use_cache: Whether to use cache
        
        Returns:
            Tuple of (frames, sampling_info, overlap_info)
        """
        # Extract frames
        frames, sampling_info = self.extract_uniform_frames(
            target_fps=target_fps,
            max_frames=max_frames,
            use_cache=use_cache
        )
        
        # Calculate ground truth overlap (use full frame_indices when logged)
        frame_indices = np.array(
            sampling_info.get('frame_indices') or sampling_info['frame_indices_sample']
        )
        if sampling_info['actual_frames_used'] > len(frame_indices):
            frame_indices, _, _ = self.calculate_sampling(
                target_fps=target_fps, max_frames=max_frames
            )
        overlap_info = self.calculate_ground_truth_overlap(frame_indices, time_reference)
        
        return frames, sampling_info, overlap_info
    
    def generate_answer(self, question: str, time_reference: str,
                       target_fps: Optional[float] = None,
                       max_frames: Optional[int] = None,
                       use_cache: bool = True) -> Dict:
        """
        Generate answer for a single question
        
        Args:
            question: Question text
            time_reference: Ground truth time reference
            target_fps: Target FPS for sampling
            max_frames: Maximum number of frames
            use_cache: Whether to use cache
        
        Returns:
            Dict with answer and metadata
        """
        # Prepare data
        frames, sampling_info, overlap_info = self.prepare_question_data(
            question, time_reference, target_fps, max_frames, use_cache
        )
        
        # Create prompt
        prompt = self.create_prompt(question)
        
        # Call VLM
        vlm_input = {
            "text": prompt,
            "video": frames
        }
        
        response = self.llm_model.batch_generate_response([vlm_input])[0]
        
        return {
            'response': response,
            'sampling_info': sampling_info,
            'ground_truth_overlap': overlap_info
        }
