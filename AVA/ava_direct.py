"""
AVA Direct - Simplified inference using cached search results
"""
import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any
from .ava import AVA
from .prompt import PROMPTS
from .utils import logger


class AVADirect(AVA):
    """
    Simplified AVA that uses cached sorted_SA_score_result.json for direct VLM inference.
    Bypasses the tree search and uses pre-computed results.
    """
    
    def __post_init__(self):
        """Initialize AVADirect with event data loading"""
        super().__post_init__()
        
        # Load events data once for this video
        logger.info("Loading events data for AVADirect")
        self.events_data = self._load_events_data()
        logger.info(f"Loaded {len(self.events_data)} events")
    
    def _load_events_data(self) -> List[Dict[str, Any]]:
        """Load events from vdb_events.json"""
        vdb_events_path = os.path.join(self.working_dir, "vdb_events.json")
        
        if not os.path.exists(vdb_events_path):
            logger.error(f"Events file not found: {vdb_events_path}")
            return []
        
        try:
            with open(vdb_events_path, 'r') as f:
                vdb_data = json.load(f)
            return vdb_data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to load events data: {e}")
            return []
    
    def _get_questions_folder(self, retrieval_mode: str = "tri_view") -> str:
        """Get the appropriate questions folder name based on retrieval mode"""
        folder_mapping = {
            "tri_view": "questions",
            "events_only": "questions_events_only", 
            "entities_only": "questions_entities_only",
            "features_only": "questions_features_only"
        }
        return folder_mapping.get(retrieval_mode, "questions")
    
    def load_sorted_sa_cache(self, question_id: int, retrieval_mode: str = "tri_view") -> List[Dict[str, Any]]:
        """
        Load the sorted SA results from cache
        
        Args:
            question_id: ID of the question
            retrieval_mode: Retrieval mode (tri_view, events_only, etc.)
            
        Returns:
            List of sorted SA result entries
        """
        questions_folder = self._get_questions_folder(retrieval_mode)
        cache_path = os.path.join(
            self.video.work_dir,
            questions_folder,
            str(question_id),
            "sorted_SA_score_result.json"
        )
        
        print(f"  DEBUG: Looking for cache at: {cache_path}")
        print(f"  DEBUG: File exists: {os.path.exists(cache_path)}")
        
        if not os.path.exists(cache_path):
            logger.error(f"Cache file not found: {cache_path}")
            print(f"  ERROR: Cache file not found: {cache_path}")
            return []
        
        try:
            with open(cache_path, 'r') as f:
                sorted_results = json.load(f)
            logger.info(f"Loaded {len(sorted_results)} SA paths from cache for question {question_id}")
            return sorted_results
        except Exception as e:
            logger.error(f"Failed to load cache file {cache_path}: {e}")
            return []
    
    def _merge_overlapping_segments(self, segments: List[List[int]]) -> List[List[int]]:
        """
        Merge overlapping frame segments
        
        Args:
            segments: List of [start_frame, end_frame] pairs
            
        Returns:
            List of merged non-overlapping segments
        """
        if not segments:
            return []
        
        # Sort by start frame
        sorted_segs = sorted(segments, key=lambda x: x[0])
        
        merged = [sorted_segs[0]]
        for current in sorted_segs[1:]:
            last = merged[-1]
            # If overlapping or adjacent, merge
            if current[0] <= last[1]:
                merged[-1] = [last[0], max(last[1], current[1])]
            else:
                merged.append(current)
        
        logger.info(f"Merged {len(segments)} segments into {len(merged)} non-overlapping segments")
        return merged
    
    def extract_and_sample_frames(
        self, 
        sorted_results: List[Dict[str, Any]], 
        max_frames: int = 256
    ) -> Tuple[List[Any], List[int]]:
        """
        Extract all frame segments from all paths, merge overlapping segments,
        and uniformly sample frames
        
        Args:
            sorted_results: List of SA result entries from cache
            max_frames: Maximum number of frames to sample
            
        Returns:
            Tuple of (sampled_frames, frame_indices)
        """
        print(f"  DEBUG: extract_and_sample_frames called with {len(sorted_results)} sorted_results")
        
        # Collect all frame_durations from all paths
        all_segments = []
        for entry in sorted_results:
            frame_durations = entry.get("frame_durations", [])
            all_segments.extend(frame_durations)
        
        print(f"  DEBUG: Collected {len(all_segments)} total segments")
        
        if not all_segments:
            logger.warning("No frame segments found in sorted results")
            print(f"  WARNING: No frame segments found in sorted results")
            return [], []
        
        # Merge overlapping segments
        merged_segments = self._merge_overlapping_segments(all_segments)
        
        # Extract frames at 1 FPS for each merged segment
        all_frames = []
        all_frame_indices = []
        
        for segment in merged_segments:
            start_frame, end_frame = segment
            # Use get_frames_by_fps with 1 FPS
            frames, _, frame_indices = self.video.get_frames_by_fps(
                fps=1, 
                duration=(start_frame / self.video.config["fps"], end_frame / self.video.config["fps"])
            )
            all_frames.extend(frames)
            all_frame_indices.extend(frame_indices)
        
        logger.info(f"Extracted {len(all_frames)} frames from {len(merged_segments)} segments")
        
        # Uniform downsampling if needed
        if len(all_frames) > max_frames:
            downsampled_indices = np.linspace(0, len(all_frames)-1, max_frames, dtype=int)
            sampled_frames = [all_frames[i] for i in downsampled_indices]
            sampled_frame_indices = [all_frame_indices[i] for i in downsampled_indices]
            logger.info(f"Downsampled from {len(all_frames)} to {len(sampled_frames)} frames")
        else:
            sampled_frames = all_frames
            sampled_frame_indices = all_frame_indices
        
        return sampled_frames, sampled_frame_indices
    
    def get_overlapping_events(self, frame_segments: List[List[int]]) -> List[str]:
        """
        Get event descriptions that overlap with the given frame segments
        
        Args:
            frame_segments: List of [start_frame, end_frame] pairs
            
        Returns:
            List of unique event descriptions
        """
        overlapping_events = []
        seen_event_ids = set()
        
        for event in self.events_data:
            event_id = event.get("id")
            event_start, event_end = event.get("duration", [0, 0])
            
            # Check if event overlaps with any segment
            for seg_start, seg_end in frame_segments:
                # Check for overlap: not (event ends before segment starts OR event starts after segment ends)
                if not (event_end < seg_start or event_start > seg_end):
                    # Overlap detected - add event if not already added
                    if event_id not in seen_event_ids:
                        overlapping_events.append(event.get("description", ""))
                        seen_event_ids.add(event_id)
                    break  # Don't check other segments for this event
        
        logger.info(f"Found {len(overlapping_events)} overlapping events from {len(frame_segments)} segments")
        return overlapping_events
    
    def create_prompt(self, question: str, event_descriptions: List[str]) -> str:
        """
        Create the prompt with event context
        
        Args:
            question: The user's question
            event_descriptions: List of event descriptions
            
        Returns:
            Formatted prompt string
        """
        # Format event descriptions as numbered list
        if event_descriptions:
            event_text = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(event_descriptions))
        else:
            event_text = "(No event descriptions available)"
        
        prompt = PROMPTS["checkframe_and_answer_with_events"].format(
            user_query=question,
            event_descriptions=event_text
        )
        
        return prompt
    
    def prepare_question_data(
        self, 
        question_id: int, 
        retrieval_mode: str = "tri_view",
        max_frames: int = 256
    ) -> Tuple[List[Any], List[str]]:
        """
        Prepare all data needed for a single question
        
        Args:
            question_id: ID of the question
            retrieval_mode: Retrieval mode
            max_frames: Maximum number of frames
            
        Returns:
            Tuple of (sampled_frames, event_descriptions)
        """
        print(f"  DEBUG: prepare_question_data for Q{question_id}, retrieval_mode={retrieval_mode}")
        
        # Load cached sorted SA results
        sorted_results = self.load_sorted_sa_cache(question_id, retrieval_mode)
        print(f"  DEBUG: load_sorted_sa_cache returned {len(sorted_results) if sorted_results else 0} results")
        
        if not sorted_results:
            logger.error(f"No cached results found for question {question_id}")
            print(f"  ERROR: No cached results found for question {question_id}")
            return [], []
        
        # Collect all segments from all paths
        all_segments = []
        for entry in sorted_results:
            frame_durations = entry.get("frame_durations", [])
            all_segments.extend(frame_durations)
        
        # Extract and sample frames
        sampled_frames, _ = self.extract_and_sample_frames(sorted_results, max_frames)
        
        # Get overlapping events
        event_descriptions = self.get_overlapping_events(all_segments)
        
        return sampled_frames, event_descriptions
    
    def generate_direct_answer(
        self, 
        question: str, 
        question_id: int, 
        retrieval_mode: str = "tri_view",
        max_frames: int = 256
    ) -> str:
        """
        Generate answer using cached results and direct VLM inference
        
        Args:
            question: The question to answer
            question_id: ID of the question
            retrieval_mode: Retrieval mode
            max_frames: Maximum number of frames
            
        Returns:
            The generated answer
        """
        logger.info(f"Generating direct answer for question {question_id}")
        
        # Prepare data
        sampled_frames, event_descriptions = self.prepare_question_data(
            question_id, retrieval_mode, max_frames
        )
        
        if not sampled_frames:
            logger.error(f"No frames extracted for question {question_id}")
            return None
        
        # Create prompt
        prompt = self.create_prompt(question, event_descriptions)
        
        # Prepare VLM input (single question, no self-consistency)
        vlm_input = {
            "text": prompt,
            "video": sampled_frames
        }
        
        # Call VLM
        logger.info(f"Calling VLM with {len(sampled_frames)} frames and {len(event_descriptions)} events")
        try:
            response = self.llm_model.batch_generate_response([vlm_input])
            answer = response[0] if response else None
            logger.info(f"Generated answer for question {question_id}")
            return answer
        except Exception as e:
            logger.error(f"VLM generation failed for question {question_id}: {e}")
            return None
    
    @staticmethod
    def validate_cache_exists(
        dataset_name: str,
        video_id: int, 
        question_id: int, 
        retrieval_mode: str = "tri_view"
    ) -> Tuple[bool, str]:
        """
        Validate that required cache files exist
        
        Args:
            dataset_name: Name of the dataset
            video_id: Video ID
            question_id: Question ID
            retrieval_mode: Retrieval mode
            
        Returns:
            Tuple of (exists: bool, error_message: str)
        """
        # Construct paths
        questions_folder_mapping = {
            "tri_view": "questions",
            "events_only": "questions_events_only", 
            "entities_only": "questions_entities_only",
            "features_only": "questions_features_only"
        }
        questions_folder = questions_folder_mapping.get(retrieval_mode, "questions")
        
        cache_base = f"AVA_cache/{dataset_name}/{video_id}"
        sorted_sa_path = f"{cache_base}/{questions_folder}/{question_id}/sorted_SA_score_result.json"
        vdb_events_path = f"{cache_base}/kg/vdb_events.json"
        
        # Check files
        if not os.path.exists(sorted_sa_path):
            return False, f"Missing sorted_SA_score_result.json: {sorted_sa_path}"
        
        if not os.path.exists(vdb_events_path):
            return False, f"Missing vdb_events.json: {vdb_events_path}"
        
        return True, ""

