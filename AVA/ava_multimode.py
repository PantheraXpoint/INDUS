"""
AVA MultiMode - 7 configuration modes for video QA inference
Supports different combinations of frame sampling and event retrieval
"""
import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .ava import AVA
from .prompt import PROMPTS
from .utils import logger
from .storage import ImageNanoVectorDBStorage
from embeddings.JinaCLIP import JinaCLIP


class AVAMultiMode(AVA):
    """
    AVA with 7 configuration modes:
    1. Uniform sampling → Frames only
    2. Top-K retrieval → Frames only
    3. Uniform sampling → Events only
    4. Top-K retrieval → Events only
    5. All segments → Events only
    6. Uniform sampling → Frames + Events
    7. Top-K retrieval → Frames + Events
    """
    
    def __post_init__(self):
        """Initialize AVAMultiMode with event and feature data loading"""
        super().__post_init__()
        
        # Load events data once for this video
        logger.info("Loading events data for AVAMultiMode")
        self.events_data = self._load_events_data()
        logger.info(f"Loaded {len(self.events_data)} events")
        
        # Load video config
        self.video_config = self._load_video_config()
        logger.info(f"Video FPS: {self.video_config['fps']}, Duration: {self.video_config['duration']}s")
        
        # Features VDB will be loaded lazily for top-k modes
        # Parent class AVA may set self.features_vdb in __post_init__, 
        # which will be handled by the setter and stored in _features_vdb
        if not hasattr(self, '_features_vdb'):
            self._features_vdb = None
    
    def _load_events_data(self) -> List[Dict[str, Any]]:
        """Load events from vdb_events.json"""
        vdb_events_path = os.path.join(self.working_dir, "vdb_events.json")
        
        if not os.path.exists(vdb_events_path):
            logger.error(f"Events file not found: {vdb_events_path}")
            return []
        
        try:
            with open(vdb_events_path, 'r') as f:
                vdb_data = json.load(f)
            # Extract events from the data field
            events = vdb_data.get("data", [])
            logger.info(f"Loaded {len(events)} events from vdb_events.json")
            return events
        except Exception as e:
            logger.error(f"Failed to load events data: {e}")
            return []
    
    def _load_video_config(self) -> Dict[str, Any]:
        """Load video configuration from config.json"""
        config_path = os.path.join(self.video.work_dir, "config.json")
        
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return {"fps": 30.0, "duration": 0.0}  # Default fallback
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load video config: {e}")
            return {"fps": 30.0, "duration": 0.0}
    
    @property
    def features_vdb(self) -> ImageNanoVectorDBStorage:
        """Lazy load features vector database"""
        if self._features_vdb is None:
            logger.info("Loading features VDB for top-k retrieval")
            kg_dir = os.path.join(self.video.work_dir, "kg")
            vdb_path = os.path.join(kg_dir, "vdb_features.json")
            
            if not os.path.exists(vdb_path):
                raise FileNotFoundError(f"Features VDB not found: {vdb_path}")
            
            # Initialize embedding model
            embedding_model = JinaCLIP("jinaai/jina-clip-v1")
            embedding_dim = embedding_model.embedding_dim
            
            # Create global_config
            global_config = {
                "video": self.video,
                "working_dir": kg_dir,
                "embedding_batch_num": 100,
            }
            
            self._features_vdb = ImageNanoVectorDBStorage(
                namespace="features",
                global_config=global_config,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                meta_fields={"id", "frame_dir", "event"},
            )
            
            logger.info("Features VDB loaded successfully")
        
        return self._features_vdb
    
    @features_vdb.setter
    def features_vdb(self, value):
        """Setter for features_vdb (allows parent class to set it)"""
        self._features_vdb = value
    
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
        
        if not os.path.exists(cache_path):
            logger.error(f"Cache file not found: {cache_path}")
            return []
        
        try:
            with open(cache_path, 'r') as f:
                sorted_results = json.load(f)
            logger.info(f"Loaded {len(sorted_results)} SA paths from cache for question {question_id}")
            return sorted_results
        except Exception as e:
            logger.error(f"Failed to load cache file {cache_path}: {e}")
            return []
    
    def _merge_time_segments(self, segments: List[List[float]]) -> List[List[float]]:
        """
        Merge overlapping time segments (in seconds)
        
        Args:
            segments: List of [start_sec, end_sec] pairs
            
        Returns:
            List of merged non-overlapping segments
        """
        if not segments:
            return []
        
        # Sort by start time
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
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _events_to_segments(self, events: List[Dict[str, Any]]) -> List[List[float]]:
        """Extract time segments [start_sec, end_sec] from event list for logging."""
        segments = []
        for ev in events:
            dur = ev.get("duration", [0, 0])
            if len(dur) >= 2:
                segments.append([float(dur[0]), float(dur[1])])
        return segments
    
    def _frame_index_to_seconds(self, frame_idx: int) -> float:
        """Convert frame index to seconds"""
        return frame_idx / self.video_config['fps']
    
    def _seconds_to_frame_index(self, seconds: float) -> int:
        """Convert seconds to frame index"""
        return int(seconds * self.video_config['fps'])
    
    def extract_time_segments_from_cache(self, sorted_results: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Extract and merge all time segments from sorted SA cache
        
        Args:
            sorted_results: List of SA result entries from cache
            
        Returns:
            List of merged time segments in seconds
        """
        # Collect all frame_durations (in seconds) from all paths
        all_segments = []
        for entry in sorted_results:
            frame_durations = entry.get("frame_durations", [])
            all_segments.extend(frame_durations)
        
        logger.info(f"Collected {len(all_segments)} total time segments")
        
        # Merge overlapping segments
        merged_segments = self._merge_time_segments(all_segments)
        
        return merged_segments
    
    def uniform_sample_frames_from_segments(
        self, 
        time_segments: List[List[float]], 
        max_frames: int = 256
    ) -> Tuple[List[Any], List[int]]:
        """
        Uniformly sample frames from time segments
        
        Args:
            time_segments: List of [start_sec, end_sec] pairs
            max_frames: Maximum number of frames to sample
            
        Returns:
            Tuple of (sampled_frames, frame_indices)
        """
        # Collect all frame indices within time segments
        all_frame_indices = []
        
        for start_sec, end_sec in time_segments:
            start_frame = self._seconds_to_frame_index(start_sec)
            end_frame = self._seconds_to_frame_index(end_sec)
            
            # Add all frames in this segment
            for frame_idx in range(start_frame, end_frame + 1):
                all_frame_indices.append(frame_idx)
        
        # Remove duplicates and sort
        all_frame_indices = sorted(list(set(all_frame_indices)))
        
        logger.info(f"Collected {len(all_frame_indices)} frames from {len(time_segments)} segments")
        
        # Uniform downsampling if needed
        if len(all_frame_indices) > max_frames:
            # Use linspace to select evenly spaced indices
            selected_positions = np.linspace(0, len(all_frame_indices) - 1, max_frames, dtype=int)
            sampled_frame_indices = [all_frame_indices[i] for i in selected_positions]
            logger.info(f"Uniformly sampled from {len(all_frame_indices)} to {len(sampled_frame_indices)} frames")
        else:
            sampled_frame_indices = all_frame_indices
        
        # Extract frames from video
        sampled_frames = self.video.get_frames_by_indices(sampled_frame_indices)
        
        return sampled_frames, sampled_frame_indices
    
    def topk_sample_frames_from_segments(
        self,
        question: str,
        time_segments: List[List[float]],
        max_frames: int = 256
    ) -> Tuple[List[Any], List[int], List[float]]:
        """
        Sample top-k frames from time segments based on semantic similarity
        
        Args:
            question: The question text for retrieval
            time_segments: List of [start_sec, end_sec] pairs
            max_frames: Maximum number of frames to retrieve
            
        Returns:
            Tuple of (sampled_frames, frame_indices, similarity_scores)
        """
        # First, downsample to 1 FPS within segments
        sampled_1fps_indices = []
        
        for start_sec, end_sec in time_segments:
            # Sample at 1 FPS
            current_sec = start_sec
            while current_sec <= end_sec:
                frame_idx = self._seconds_to_frame_index(current_sec)
                sampled_1fps_indices.append(frame_idx)
                current_sec += 1.0  # Increment by 1 second
        
        # Remove duplicates and sort
        sampled_1fps_indices = sorted(list(set(sampled_1fps_indices)))
        
        logger.info(f"1 FPS sampling resulted in {len(sampled_1fps_indices)} frames from {len(time_segments)} segments")
        
        # Rewrite query for visual retrieval
        rewrite_prompt = PROMPTS["query_rewrite_for_visual_retrieval"].format(
            input_text=question
        )
        rewritten_query = self.llm_model.generate_response({"text": rewrite_prompt})
        logger.info(f"Rewritten query for visual retrieval: {rewritten_query[:100]}...")
        
        # Get query embedding
        query_embedding = self.features_vdb.embedding_model.get_text_features([rewritten_query])[0]
        
        # Get all frames data from VDB
        vdb_data = self.features_vdb.client_storage["data"]
        vdb_matrix = self.features_vdb.client_storage["matrix"]
        
        # Filter to only frames within our 1fps sampled indices
        filtered_indices = []
        filtered_vectors = []
        filtered_frame_indices = []
        
        for i, data_entry in enumerate(vdb_data):
            # Extract frame index from frame_dir
            frame_dir = data_entry.get("frame_dir", "")
            if frame_dir:
                filename = os.path.basename(frame_dir)
                frame_idx = int(os.path.splitext(filename)[0])
                
                if frame_idx in sampled_1fps_indices:
                    filtered_indices.append(i)
                    filtered_vectors.append(vdb_matrix[i])
                    filtered_frame_indices.append(frame_idx)
        
        logger.info(f"Found {len(filtered_indices)} frames in VDB matching 1fps sampled frames")
        
        if not filtered_vectors:
            logger.warning("No frames found in VDB for the given segments")
            return [], [], []
        
        # Compute cosine similarity
        filtered_vectors = np.array(filtered_vectors)
        # Normalize vectors
        filtered_vectors = filtered_vectors / (np.linalg.norm(filtered_vectors, axis=1, keepdims=True) + 1e-8)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(filtered_vectors, query_embedding)
        
        # Get top-k indices
        if len(similarities) > max_frames:
            top_k_positions = np.argsort(similarities)[-max_frames:][::-1]
        else:
            top_k_positions = np.argsort(similarities)[::-1]
        
        # Get corresponding frame indices and scores (then sort by time for output)
        topk_frame_indices = [filtered_frame_indices[i] for i in top_k_positions]
        topk_scores = [float(similarities[i]) for i in top_k_positions]
        # Sort by temporal order and reorder scores to match
        sorted_pairs = sorted(zip(topk_frame_indices, topk_scores))
        topk_frame_indices = [p[0] for p in sorted_pairs]
        topk_scores = [p[1] for p in sorted_pairs]
        
        logger.info(f"Selected top {len(topk_frame_indices)} frames based on similarity")
        
        # Extract frames from video
        topk_frames = self.video.get_frames_by_indices(topk_frame_indices)
        
        return topk_frames, topk_frame_indices, topk_scores
    
    def project_frames_to_events(self, frame_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Project frame indices to their corresponding events
        
        Args:
            frame_indices: List of frame indices
            
        Returns:
            List of event dictionaries in temporal order
        """
        # Convert frame indices to seconds
        frame_times = [self._frame_index_to_seconds(idx) for idx in frame_indices]
        
        # Find corresponding events
        selected_events = []
        seen_event_ids = set()
        
        for frame_time in frame_times:
            # Find event containing this frame time
            for event in self.events_data:
                event_id = event.get("__id__")
                event_start, event_end = event.get("duration", [0, 0])
                
                # Check if frame falls within this event
                if event_start <= frame_time <= event_end:
                    if event_id not in seen_event_ids:
                        selected_events.append(event)
                        seen_event_ids.add(event_id)
                    break  # Move to next frame
        
        # Sort events by start time
        selected_events.sort(key=lambda e: e.get("duration", [0, 0])[0])
        
        logger.info(f"Projected {len(frame_indices)} frames to {len(selected_events)} events")
        
        return selected_events
    
    def project_segments_to_events(self, time_segments: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Project time segments to their corresponding events
        
        Args:
            time_segments: List of [start_sec, end_sec] pairs
            
        Returns:
            List of event dictionaries in temporal order
        """
        selected_events = []
        seen_event_ids = set()
        
        for seg_start, seg_end in time_segments:
            # Find events that overlap with this segment
            for event in self.events_data:
                event_id = event.get("__id__")
                event_start, event_end = event.get("duration", [0, 0])
                
                # Check for overlap: not (event ends before segment OR event starts after segment)
                if not (event_end < seg_start or event_start > seg_end):
                    if event_id not in seen_event_ids:
                        selected_events.append(event)
                        seen_event_ids.add(event_id)
        
        # Sort events by start time
        selected_events.sort(key=lambda e: e.get("duration", [0, 0])[0])
        
        logger.info(f"Projected {len(time_segments)} segments to {len(selected_events)} events")
        
        return selected_events
    
    def format_frames_for_prompt(self, frame_indices: List[int]) -> str:
        """
        Format frame indices as timestamp list for prompt
        
        Args:
            frame_indices: List of frame indices
            
        Returns:
            Formatted string like "Image 1 @ 00:05:30\nImage 2 @ 00:06:15\n..."
        """
        formatted_lines = []
        for i, frame_idx in enumerate(frame_indices, start=1):
            timestamp = self._seconds_to_timestamp(self._frame_index_to_seconds(frame_idx))
            formatted_lines.append(f"Image {i} @ {timestamp}")
        
        return "\n".join(formatted_lines)
    
    def format_events_for_prompt(self, events: List[Dict[str, Any]]) -> str:
        """
        Format events as description list for prompt
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Formatted string like "1. [00:05:30 - 00:07:45] Description...\n2. ..."
        """
        formatted_lines = []
        for i, event in enumerate(events, start=1):
            event_start, event_end = event.get("duration", [0, 0])
            start_ts = self._seconds_to_timestamp(event_start)
            end_ts = self._seconds_to_timestamp(event_end)
            description = event.get("description", "")
            formatted_lines.append(f"{i}. [{start_ts} - {end_ts}] {description}")
        
        return "\n".join(formatted_lines)
    
    def format_frames_and_events_for_prompt(
        self, 
        frame_indices: List[int], 
        events: List[Dict[str, Any]]
    ) -> str:
        """
        Format frames and events together showing visual evidence for each event
        
        Args:
            frame_indices: List of frame indices
            events: List of event dictionaries
            
        Returns:
            Formatted string with events and their visual evidence
        """
        # Map frames to events
        frame_times = [self._frame_index_to_seconds(idx) for idx in frame_indices]
        
        formatted_lines = []
        for i, event in enumerate(events, start=1):
            event_start, event_end = event.get("duration", [0, 0])
            start_ts = self._seconds_to_timestamp(event_start)
            end_ts = self._seconds_to_timestamp(event_end)
            description = event.get("description", "")
            
            # Find frames that fall within this event
            matching_frames = []
            for img_num, (frame_idx, frame_time) in enumerate(zip(frame_indices, frame_times), start=1):
                if event_start <= frame_time <= event_end:
                    timestamp = self._seconds_to_timestamp(frame_time)
                    matching_frames.append(f"Image {img_num} @ {timestamp}")
            
            # Format event with visual evidence
            event_line = f"{i}. [{start_ts} - {end_ts}] {description}"
            formatted_lines.append(event_line)
            
            if matching_frames:
                visual_evidence = ", ".join([f"[{frame}]" for frame in matching_frames])
                formatted_lines.append(f"   > Visual Evidence: {visual_evidence}")
            else:
                formatted_lines.append(f"   > Visual Evidence: None in this range")
        
        return "\n".join(formatted_lines)
    
    def generate_answer_mode_1(
        self,
        question: str,
        question_id: int,
        retrieval_mode: str = "tri_view",
        max_frames: int = 256
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Config 1: Uniform sampling → Frames only
        
        Returns:
            Tuple of (answer, metadata)
        """
        logger.info(f"Mode 1: Uniform sampling → Frames only (Q{question_id})")
        
        # Load sorted SA cache
        sorted_results = self.load_sorted_sa_cache(question_id, retrieval_mode)
        if not sorted_results:
            return None, {"error": "No cache found"}
        
        # Extract and merge time segments
        time_segments = self.extract_time_segments_from_cache(sorted_results)
        
        # Uniform sample frames
        frames, frame_indices = self.uniform_sample_frames_from_segments(time_segments, max_frames)
        
        if not frames or len(frames) == 0:
            return None, {"error": "No frames extracted"}
        
        # Format frames for prompt
        frame_timestamps = self.format_frames_for_prompt(frame_indices)
        
        # Create prompt
        prompt = PROMPTS["frames_only"].format(
            frame_timestamps=frame_timestamps,
            user_query=question
        )
        
        # Log the final prompt
        logger.info(f"=== MODE 1 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Full prompt:\n{prompt}")
        logger.info(f"=== END PROMPT ===")
        
        # Call VLM
        logger.info(f"Calling VLM with {len(frames)} frames (Mode 1)")
        vlm_input = {"text": prompt, "video": frames}
        try:
            response = self.llm_model.batch_generate_response([vlm_input])
            response = response[0] if response else None
        except Exception as e:
            logger.error(f"VLM generation failed (Mode 1): {e}")
            return None, {"error": f"VLM generation failed: {str(e)}", "mode": 1}
        
        metadata = {
            "mode": 1,
            "num_frames": len(frames),
            "num_segments": len(time_segments),
            "frame_indices": frame_indices,
            "video_duration": self.video_config["duration"],
            "fps": self.video_config["fps"],
        }
        
        return response, metadata
    
    def generate_answer_mode_2(
        self,
        question: str,
        question_id: int,
        retrieval_mode: str = "tri_view",
        max_frames: int = 256
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Config 2: Top-K retrieval → Frames only
        
        Returns:
            Tuple of (answer, metadata)
        """
        logger.info(f"Mode 2: Top-K retrieval → Frames only (Q{question_id})")
        
        # Load sorted SA cache
        sorted_results = self.load_sorted_sa_cache(question_id, retrieval_mode)
        if not sorted_results:
            return None, {"error": "No cache found"}
        
        # Extract and merge time segments
        time_segments = self.extract_time_segments_from_cache(sorted_results)
        
        # Top-K sample frames
        frames, frame_indices, similarity_scores = self.topk_sample_frames_from_segments(question, time_segments, max_frames)
        
        if not frames or len(frames) == 0:
            return None, {"error": "No frames extracted"}
        
        # Format frames for prompt
        frame_timestamps = self.format_frames_for_prompt(frame_indices)
        
        # Create prompt
        prompt = PROMPTS["frames_only"].format(
            frame_timestamps=frame_timestamps,
            user_query=question
        )
        
        # Log the final prompt
        logger.info(f"=== MODE 2 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Full prompt:\n{prompt}")
        logger.info(f"=== END PROMPT ===")
        
        # Call VLM
        logger.info(f"Calling VLM with {len(frames)} frames (Mode 2)")
        vlm_input = {"text": prompt, "video": frames}
        try:
            response = self.llm_model.batch_generate_response([vlm_input])
            response = response[0] if response else None
        except Exception as e:
            logger.error(f"VLM generation failed (Mode 2): {e}")
            return None, {"error": f"VLM generation failed: {str(e)}", "mode": 2}
        
        metadata = {
            "mode": 2,
            "num_frames": len(frames),
            "num_segments": len(time_segments),
            "frame_indices": frame_indices,
            "similarity_scores": similarity_scores,
            "video_duration": self.video_config["duration"],
            "fps": self.video_config["fps"],
        }
        
        return response, metadata
    
    def generate_answer_mode_3(
        self,
        question: str,
        question_id: int,
        retrieval_mode: str = "tri_view",
        max_frames: int = 256
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Config 3: Uniform sampling → Events only
        
        Returns:
            Tuple of (answer, metadata)
        """
        logger.info(f"Mode 3: Uniform sampling → Events only (Q{question_id})")
        
        # Load sorted SA cache
        sorted_results = self.load_sorted_sa_cache(question_id, retrieval_mode)
        if not sorted_results:
            return None, {"error": "No cache found"}
        
        # Extract and merge time segments
        time_segments = self.extract_time_segments_from_cache(sorted_results)
        
        # Uniform sample frames (we need frame indices to project to events)
        _, frame_indices = self.uniform_sample_frames_from_segments(time_segments, max_frames)
        
        if not frame_indices:
            return None, {"error": "No frames extracted"}
        
        # Project frames to events
        events = self.project_frames_to_events(frame_indices)
        
        if not events:
            return None, {"error": "No events found"}
        
        # Format events for prompt
        event_descriptions = self.format_events_for_prompt(events)
        
        # Create prompt
        prompt = PROMPTS["events_only"].format(
            event_descriptions=event_descriptions,
            user_query=question
        )
        
        # Log the final prompt
        logger.info(f"=== MODE 3 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Number of events: {len(events)}")
        logger.info(f"Full prompt:\n{prompt}")
        logger.info(f"=== END PROMPT ===")
        
        # Call VLM (no frames needed)
        vlm_input = {"text": prompt}
        try:
            response = self.llm_model.batch_generate_response([vlm_input])
            response = response[0] if response else None
        except Exception as e:
            logger.error(f"VLM generation failed (Mode 3): {e}")
            return None, {"error": f"VLM generation failed: {str(e)}", "mode": 3}
        
        metadata = {
            "mode": 3,
            "num_events": len(events),
            "num_frames_sampled": len(frame_indices),
            "event_segments": self._events_to_segments(events),
            "video_duration": self.video_config["duration"],
            "fps": self.video_config["fps"],
        }
        
        return response, metadata
    
    def generate_answer_mode_4(
        self,
        question: str,
        question_id: int,
        retrieval_mode: str = "tri_view",
        max_frames: int = 256
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Config 4: Top-K retrieval → Events only
        
        Returns:
            Tuple of (answer, metadata)
        """
        logger.info(f"Mode 4: Top-K retrieval → Events only (Q{question_id})")
        
        # Load sorted SA cache
        sorted_results = self.load_sorted_sa_cache(question_id, retrieval_mode)
        if not sorted_results:
            return None, {"error": "No cache found"}
        
        # Extract and merge time segments
        time_segments = self.extract_time_segments_from_cache(sorted_results)
        
        # Top-K sample frames
        _, frame_indices, _ = self.topk_sample_frames_from_segments(question, time_segments, max_frames)
        
        if not frame_indices:
            return None, {"error": "No frames extracted"}
        
        # Project frames to events
        events = self.project_frames_to_events(frame_indices)
        
        if not events:
            return None, {"error": "No events found"}
        
        # Format events for prompt
        event_descriptions = self.format_events_for_prompt(events)
        
        # Create prompt
        prompt = PROMPTS["events_only"].format(
            event_descriptions=event_descriptions,
            user_query=question
        )
        
        # Log the final prompt
        logger.info(f"=== MODE 4 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Number of events: {len(events)}")
        logger.info(f"Full prompt:\n{prompt}")
        logger.info(f"=== END PROMPT ===")
        
        # Call VLM (no frames needed)
        vlm_input = {"text": prompt}
        try:
            response = self.llm_model.batch_generate_response([vlm_input])
            response = response[0] if response else None
        except Exception as e:
            logger.error(f"VLM generation failed (Mode 4): {e}")
            return None, {"error": f"VLM generation failed: {str(e)}", "mode": 4}
        
        metadata = {
            "mode": 4,
            "num_events": len(events),
            "num_frames_sampled": len(frame_indices),
            "event_segments": self._events_to_segments(events),
            "video_duration": self.video_config["duration"],
            "fps": self.video_config["fps"],
        }
        
        return response, metadata
    
    def generate_answer_mode_5(
        self,
        question: str,
        question_id: int,
        retrieval_mode: str = "tri_view"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Config 5: All segments → Events only
        
        Returns:
            Tuple of (answer, metadata)
        """
        logger.info(f"Mode 5: All segments → Events only (Q{question_id})")
        
        # Load sorted SA cache
        sorted_results = self.load_sorted_sa_cache(question_id, retrieval_mode)
        if not sorted_results:
            return None, {"error": "No cache found"}
        
        # Extract and merge time segments
        time_segments = self.extract_time_segments_from_cache(sorted_results)
        
        # Project segments directly to events
        events = self.project_segments_to_events(time_segments)
        
        if not events:
            return None, {"error": "No events found"}
        
        # Format events for prompt
        event_descriptions = self.format_events_for_prompt(events)
        
        # Create prompt
        prompt = PROMPTS["events_only"].format(
            event_descriptions=event_descriptions,
            user_query=question
        )
        
        # Log the final prompt
        logger.info(f"=== MODE 5 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Number of events: {len(events)}")
        logger.info(f"Full prompt:\n{prompt}")
        logger.info(f"=== END PROMPT ===")
        
        # Call VLM (no frames needed)
        vlm_input = {"text": prompt}
        try:
            response = self.llm_model.batch_generate_response([vlm_input])
            response = response[0] if response else None
        except Exception as e:
            logger.error(f"VLM generation failed (Mode 5): {e}")
            return None, {"error": f"VLM generation failed: {str(e)}", "mode": 5}
        
        metadata = {
            "mode": 5,
            "num_events": len(events),
            "num_segments": len(time_segments),
            "event_segments": self._events_to_segments(events),
            "video_duration": self.video_config["duration"],
            "fps": self.video_config["fps"],
        }
        
        return response, metadata
    
    def generate_answer_mode_6(
        self,
        question: str,
        question_id: int,
        retrieval_mode: str = "tri_view",
        max_frames: int = 256
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Config 6: Uniform sampling → Frames + Events
        
        Returns:
            Tuple of (answer, metadata)
        """
        logger.info(f"Mode 6: Uniform sampling → Frames + Events (Q{question_id})")
        
        # Load sorted SA cache
        sorted_results = self.load_sorted_sa_cache(question_id, retrieval_mode)
        if not sorted_results:
            return None, {"error": "No cache found"}
        
        # Extract and merge time segments
        time_segments = self.extract_time_segments_from_cache(sorted_results)
        
        # Uniform sample frames
        frames, frame_indices = self.uniform_sample_frames_from_segments(time_segments, max_frames)
        
        if not frames or len(frames) == 0:
            return None, {"error": "No frames extracted"}
        
        # Project frames to events
        events = self.project_frames_to_events(frame_indices)
        
        if not events:
            return None, {"error": "No events found"}
        
        # Format frames and events together
        event_descriptions_with_frames = self.format_frames_and_events_for_prompt(frame_indices, events)
        
        # Create prompt
        prompt = PROMPTS["frames_and_events_aligned"].format(
            event_descriptions_with_frames=event_descriptions_with_frames,
            user_query=question
        )
        
        # Log the final prompt
        logger.info(f"=== MODE 6 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Number of frames: {len(frames)}, Number of events: {len(events)}")
        logger.info(f"Full prompt:\n{prompt}")
        logger.info(f"=== END PROMPT ===")
        
        # Call VLM
        logger.info(f"Calling VLM with {len(frames)} frames and {len(events)} events (Mode 6)")
        vlm_input = {"text": prompt, "video": frames}
        try:
            response = self.llm_model.batch_generate_response([vlm_input])
            response = response[0] if response else None
        except Exception as e:
            logger.error(f"VLM generation failed (Mode 6): {e}")
            return None, {"error": f"VLM generation failed: {str(e)}", "mode": 6}
        
        metadata = {
            "mode": 6,
            "num_frames": len(frames),
            "num_events": len(events),
            "frame_indices": frame_indices,
            "event_segments": self._events_to_segments(events),
            "video_duration": self.video_config["duration"],
            "fps": self.video_config["fps"],
        }
        
        return response, metadata
    
    def generate_answer_mode_7(
        self,
        question: str,
        question_id: int,
        retrieval_mode: str = "tri_view",
        max_frames: int = 256
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Config 7: Top-K retrieval → Frames + Events
        
        Returns:
            Tuple of (answer, metadata)
        """
        logger.info(f"Mode 7: Top-K retrieval → Frames + Events (Q{question_id})")
        
        # Load sorted SA cache
        sorted_results = self.load_sorted_sa_cache(question_id, retrieval_mode)
        if not sorted_results:
            return None, {"error": "No cache found"}
        
        # Extract and merge time segments
        time_segments = self.extract_time_segments_from_cache(sorted_results)
        
        # Top-K sample frames
        frames, frame_indices, similarity_scores = self.topk_sample_frames_from_segments(question, time_segments, max_frames)
        
        if not frames or len(frames) == 0:
            return None, {"error": "No frames extracted"}
        
        # Project frames to events
        events = self.project_frames_to_events(frame_indices)
        
        if not events:
            return None, {"error": "No events found"}
        
        # Format frames and events together
        event_descriptions_with_frames = self.format_frames_and_events_for_prompt(frame_indices, events)
        
        # Create prompt
        prompt = PROMPTS["frames_and_events_aligned"].format(
            event_descriptions_with_frames=event_descriptions_with_frames,
            user_query=question
        )
        
        # Log the final prompt
        logger.info(f"=== MODE 7 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Number of frames: {len(frames)}, Number of events: {len(events)}")
        logger.info(f"Full prompt:\n{prompt}")
        logger.info(f"=== END PROMPT ===")
        
        # Call VLM
        logger.info(f"Calling VLM with {len(frames)} frames and {len(events)} events (Mode 7)")
        vlm_input = {"text": prompt, "video": frames}
        try:
            response = self.llm_model.batch_generate_response([vlm_input])
            response = response[0] if response else None
        except Exception as e:
            logger.error(f"VLM generation failed (Mode 7): {e}")
            return None, {"error": f"VLM generation failed: {str(e)}", "mode": 7}
        
        metadata = {
            "mode": 7,
            "num_frames": len(frames),
            "num_events": len(events),
            "frame_indices": frame_indices,
            "similarity_scores": similarity_scores,
            "event_segments": self._events_to_segments(events),
            "video_duration": self.video_config["duration"],
            "fps": self.video_config["fps"],
        }
        
        return response, metadata
    
    def generate_answer(
        self,
        question: str,
        question_id: int,
        config_mode: int,
        retrieval_mode: str = "tri_view",
        max_frames: int = 256
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate answer using specified configuration mode
        
        Args:
            question: The question text
            question_id: Question ID
            config_mode: Configuration mode (1-7)
            retrieval_mode: Retrieval mode for cache folder
            max_frames: Maximum frames (used by modes 1-4, 6-7)
            
        Returns:
            Tuple of (answer, metadata)
        """
        mode_functions = {
            1: self.generate_answer_mode_1,
            2: self.generate_answer_mode_2,
            3: self.generate_answer_mode_3,
            4: self.generate_answer_mode_4,
            5: self.generate_answer_mode_5,
            6: self.generate_answer_mode_6,
            7: self.generate_answer_mode_7,
        }
        
        if config_mode not in mode_functions:
            raise ValueError(f"Invalid config_mode: {config_mode}. Must be 1-7.")
        
        mode_func = mode_functions[config_mode]
        
        # Mode 5 doesn't use max_frames
        if config_mode == 5:
            return mode_func(question, question_id, retrieval_mode)
        else:
            return mode_func(question, question_id, retrieval_mode, max_frames)

