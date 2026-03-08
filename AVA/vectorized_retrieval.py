"""
Vectorized Retrieval Module for VLM Direct Inference
Retrieves top-k frames based on semantic similarity to query
"""
import os
import json
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import sys

# Add ICDCS to path for overlap functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AVA.prompt import PROMPTS
from AVA.storage import ImageNanoVectorDBStorage
from embeddings.JinaCLIP import JinaCLIP

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


class VectorizedRetrieval:
    """
    Vectorized retrieval class for frame-level semantic search
    Retrieves top-k=256 frames based on query similarity
    """
    
    def __init__(self, video, llm_model):
        """
        Initialize VectorizedRetrieval
        
        Args:
            video: VideoRepresentation object
            llm_model: VLM model instance
        """
        self.video = video
        self.llm_model = llm_model
        self.config = self._load_config()
        self.features_vdb = self._load_features_vdb()
    
    def _load_config(self) -> Dict:
        """Load video configuration from config.json"""
        config_path = os.path.join(self.video.work_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def _load_features_vdb(self) -> ImageNanoVectorDBStorage:
        """Load features vector database"""
        # The working_dir for VDB should be the kg directory (same as in AVA)
        kg_dir = os.path.join(self.video.work_dir, "kg")
        vdb_path = os.path.join(kg_dir, "vdb_features.json")
        
        if not os.path.exists(vdb_path):
            raise FileNotFoundError(f"Features VDB not found: {vdb_path}")
        
        # Initialize embedding model (same as in AVA)
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        embedding_dim = embedding_model.embedding_dim
        
        # Create global_config with kg directory as working_dir (same structure as AVA)
        global_config = {
            "video": self.video,
            "working_dir": kg_dir,  # Must be kg directory for VDB to find vdb_features.json
            "embedding_batch_num": 100,  # Required by NanoVectorDBStorage
        }
        
        features_vdb = ImageNanoVectorDBStorage(
            namespace="features",
            global_config=global_config,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            meta_fields={"id", "frame_dir", "event"},
        )
        
        return features_vdb
    
    def rewrite_query_for_visual(self, query: str) -> str:
        """
        Rewrite query for visual retrieval using LLM
        
        Args:
            query: Original question text
        
        Returns:
            Rewritten query optimized for visual search
        """
        rewrite_prompt = PROMPTS["query_rewrite_for_visual_retrieval"].format(
            input_text=query
        )
        
        rewritten_query = self.llm_model.generate_response({"text": rewrite_prompt})
        
        return rewritten_query
    
    def retrieve_top_k_frames(self, query: str, k: int = 256) -> List[Dict]:
        """
        Retrieve top-k frames based on semantic similarity to query
        
        Args:
            query: Original question text
            k: Number of frames to retrieve (default 256)
        
        Returns:
            List of frame data dictionaries with similarity scores
        """
        # Rewrite query for visual retrieval
        print(f"Rewriting query for visual retrieval...")
        rewritten_query = self.rewrite_query_for_visual(query)
        print(f"Rewritten query: {rewritten_query[:100]}...")
        
        # Query features vector database
        print(f"Querying features VDB for top {k} frames...")
        features_result = self.features_vdb.query(rewritten_query, top_k=k)
        
        print(f"Retrieved {len(features_result)} frames from VDB")
        
        return features_result
    
    def sort_frames_temporally(self, frames_data: List[Dict]) -> List[Dict]:
        """
        Sort retrieved frames by temporal order (frame index)
        
        Args:
            frames_data: List of frame dictionaries from VDB
        
        Returns:
            Temporally sorted list of frame dictionaries
        """
        # Extract frame index from frame_dir path (e.g., "AVA_cache/LVBench/1/frames/0.jpg" -> 0)
        def get_frame_index(frame_data):
            frame_dir = frame_data['frame_dir']
            # Get filename without extension
            filename = os.path.basename(frame_dir)
            frame_idx = int(os.path.splitext(filename)[0])
            return frame_idx
        
        sorted_frames = sorted(frames_data, key=get_frame_index)
        
        return sorted_frames
    
    def extract_frames_by_indices(self, frame_indices: List[int]) -> List:
        """
        Extract frames from video by frame indices
        
        Args:
            frame_indices: List of frame indices to extract
        
        Returns:
            List of PIL Image objects
        """
        frames = self.video.get_frames_by_indices(frame_indices)
        
        return frames
    
    def calculate_ground_truth_overlap(self, frame_indices: List[int], 
                                       time_reference: str) -> Dict:
        """
        Calculate overlap between retrieved frames and ground truth
        
        Args:
            frame_indices: List of frame indices used
            time_reference: Ground truth time string
        
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
                'binary_overlap': binary_overlap,
                'percentage_overlap': percentage_overlap,
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
                              k: int = 256) -> Tuple[List, Dict, Dict]:
        """
        Prepare data for a single question (for batching)
        
        Args:
            question: Question text
            time_reference: Ground truth time reference
            k: Number of frames to retrieve (default 256)
        
        Returns:
            Tuple of (frames, retrieval_info, overlap_info)
        """
        # Retrieve top-k frames
        frames_data = self.retrieve_top_k_frames(question, k=k)
        
        # Sort frames temporally
        sorted_frames_data = self.sort_frames_temporally(frames_data)
        
        # Extract frame indices from frame_dir paths
        frame_indices = []
        for frame_data in sorted_frames_data:
            frame_dir = frame_data['frame_dir']
            filename = os.path.basename(frame_dir)
            frame_idx = int(os.path.splitext(filename)[0])
            frame_indices.append(frame_idx)
        
        # Extract frames from video
        print(f"Extracting {len(frame_indices)} retrieved frames from video...")
        frames = self.extract_frames_by_indices(frame_indices)
        
        # Prepare retrieval info (full lists for later visualization and evaluation)
        retrieval_info = {
            'method': 'vectorized_retrieval',
            'k': k,
            'actual_frames_retrieved': len(frames),
            'video_duration': self.config['duration'],
            'original_fps': self.config['fps'],
            'frame_indices': frame_indices,
            'similarity_scores': [float(f['__metrics__']) for f in sorted_frames_data],
            'frame_indices_sample': frame_indices[:min(100, len(frame_indices))],
            'similarity_scores_sample': [float(f['__metrics__']) for f in sorted_frames_data[:min(100, len(sorted_frames_data))]]
        }
        
        # Calculate ground truth overlap
        overlap_info = self.calculate_ground_truth_overlap(frame_indices, time_reference)
        
        return frames, retrieval_info, overlap_info
    
    def generate_answer(self, question: str, time_reference: str,
                       k: int = 256) -> Dict:
        """
        Generate answer for a single question
        
        Args:
            question: Question text
            time_reference: Ground truth time reference
            k: Number of frames to retrieve (default 256)
        
        Returns:
            Dict with answer and metadata
        """
        # Prepare data
        frames, retrieval_info, overlap_info = self.prepare_question_data(
            question, time_reference, k
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
            'retrieval_info': retrieval_info,
            'ground_truth_overlap': overlap_info
        }
