"""
IndusMultiMode - Multimode VLM reasoning using INDUS seed events.

This class is conceptually similar to `AVA.ava_multimode.AVAMultiMode`, but:
    - It does NOT inherit from AVAMultiMode (only from AVA).
    - It only supports configuration modes 3–7 (events-only and frames+events).
    - It uses INDUS seed events JSON to obtain per-question temporal segments
      and event IDs, while still loading rich event/entity/feature content
      from the existing VDB files under the video working directory.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import numpy as np

from AVA.ava import AVA
from AVA.prompt import PROMPTS
from AVA.utils import logger
from AVA.storage import ImageNanoVectorDBStorage
from embeddings.JinaCLIP import JinaCLIP


@dataclass
class IndusMultiMode(AVA):
    """
    Multimode VLM reasoning driven by INDUS seed events.

    Differences from AVAMultiMode:
        - Uses INDUS seed_events_* JSON for per-question event IDs and durations.
        - Only supports configuration modes 3–7.
        - Keeps relying on the existing VDBs (events/entities/features) for
          full event descriptions and frame features.

    Attributes added on top of AVA:
        seed_events_by_question: Mapping from question_id -> list of INDUS
            seed event dicts (each with id / __id__, duration in seconds, etc.).
    """

    # Mapping: question_id -> list[seed_event_dict]
    seed_events_by_question: Dict[int, List[Dict[str, Any]]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        """Initialize IndusMultiMode with event/VDB loading."""
        # Initialize base AVA (sets working_dir, VDB storages, etc.)
        super().__post_init__()

        # Load all events for this video from VDB (vdb_events.json)
        logger.info("Loading events data for IndusMultiMode")
        self.events_data = self._load_events_data()
        logger.info(f"Loaded {len(self.events_data)} events from VDB")

        # Load video config (fps, duration) if available
        self.video_config = self._load_video_config()
        logger.info(
            f"Video FPS: {self.video_config['fps']}, "
            f"Duration: {self.video_config['duration']}s"
        )

        # Features VDB may be created by AVA; keep lazy handle consistent
        if not hasattr(self, "_features_vdb"):
            self._features_vdb = None

    # -------------------------------------------------------------------------
    # Data loading helpers
    # -------------------------------------------------------------------------

    def _load_events_data(self) -> List[Dict[str, Any]]:
        """
        Load events from vdb_events.json under the video's kg directory.

        This provides rich event descriptions and durations that we align
        with INDUS seed event IDs.
        """
        vdb_events_path = os.path.join(self.working_dir, "vdb_events.json")

        if not os.path.exists(vdb_events_path):
            logger.error(f"Events file not found: {vdb_events_path}")
            return []

        try:
            with open(vdb_events_path, "r") as f:
                vdb_data = json.load(f)
            # Expect structure {"data": [event_dict, ...]}
            events = vdb_data.get("data", [])
            logger.info(f"Loaded {len(events)} events from vdb_events.json")
            return events
        except Exception as e:
            logger.error(f"Failed to load events data: {e}")
            return []

    def _load_video_config(self) -> Dict[str, Any]:
        """
        Load video configuration (fps, duration) from config.json in the
        video's working directory.
        """
        config_path = os.path.join(self.video.work_dir, "config.json")

        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            # Reasonable default fallback
            return {"fps": 30.0, "duration": 0.0}

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            # Ensure required keys exist
            if "fps" not in config:
                config["fps"] = 30.0
            if "duration" not in config:
                config["duration"] = 0.0
            return config
        except Exception as e:
            logger.error(f"Failed to load video config: {e}")
            return {"fps": 30.0, "duration": 0.0}

    # -------------------------------------------------------------------------
    # Features VDB lazy loader (copied from AVAMultiMode pattern)
    # -------------------------------------------------------------------------

    @property
    def features_vdb(self) -> ImageNanoVectorDBStorage:
        """Lazy load features vector database if needed."""
        if self._features_vdb is None:
            logger.info("Loading features VDB for top-k retrieval (IndusMultiMode)")
            kg_dir = os.path.join(self.video.work_dir, "kg")
            vdb_path = os.path.join(kg_dir, "vdb_features.json")

            if not os.path.exists(vdb_path):
                raise FileNotFoundError(f"Features VDB not found: {vdb_path}")

            # Initialize embedding model
            embedding_model = JinaCLIP("jinaai/jina-clip-v1")
            embedding_dim = embedding_model.embedding_dim

            # Create global_config for features storage
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

            logger.info("Features VDB loaded successfully (IndusMultiMode)")

        return self._features_vdb

    @features_vdb.setter
    def features_vdb(self, value):
        """Setter for features_vdb (allows parent AVA to set it)."""
        self._features_vdb = value

    # -------------------------------------------------------------------------
    # INDUS seed events utilities
    # -------------------------------------------------------------------------

    def _get_seed_events(self, question_id: int) -> List[Dict[str, Any]]:
        """Return INDUS seed events for a given question_id."""
        return self.seed_events_by_question.get(int(question_id), [])

    def _get_vdb_events_for_seed(
        self, question_id: int
    ) -> List[Dict[str, Any]]:
        """
        Map INDUS seed event IDs to corresponding VDB events for this video.

        Preserves the order of the seed events (e.g., by Borda score).
        Falls back to the seed event dict itself if a matching VDB entry
        cannot be found.
        """
        seeds = self._get_seed_events(question_id)
        if not seeds:
            return []

        # Build lookup map from VDB events by ID
        id_to_event: Dict[str, Dict[str, Any]] = {}
        for ev in self.events_data:
            ev_id = ev.get("id") or ev.get("__id__")
            if ev_id:
                id_to_event[ev_id] = ev

        aligned_events: List[Dict[str, Any]] = []
        for seed in seeds:
            sid = seed.get("id") or seed.get("__id__")
            if sid and sid in id_to_event:
                ev = dict(id_to_event[sid])  # shallow copy
                # Optionally attach seed-specific metadata (e.g., Borda score)
                for k in ["borda_score"]:
                    if k in seed:
                        ev[k] = seed[k]
                aligned_events.append(ev)
            else:
                # Fallback to seed event itself if VDB entry is missing
                aligned_events.append(seed)

        return aligned_events

    def _get_time_segments_for_question(
        self, question_id: int
    ) -> List[List[float]]:
        """
        Build merged time segments (in seconds) from INDUS seed events for a
        given question.

        Note: durations in seed_events are already in seconds.
        """
        seeds = self._get_seed_events(question_id)
        if not seeds:
            return []

        segments: List[List[float]] = []
        for ev in seeds:
            dur = ev.get("duration", [0, 0])
            if isinstance(dur, (list, tuple)) and len(dur) >= 2:
                try:
                    start = float(dur[0])
                    end = float(dur[1])
                    segments.append([start, end])
                except (TypeError, ValueError):
                    continue

        logger.info(
            f"Collected {len(segments)} raw segments from INDUS seed events "
            f"for question {question_id}"
        )
        return self._merge_time_segments(segments)

    # -------------------------------------------------------------------------
    # Generic temporal / formatting helpers
    # -------------------------------------------------------------------------

    def _merge_time_segments(
        self, segments: List[List[float]]
    ) -> List[List[float]]:
        """
        Merge overlapping time segments (in seconds).

        Args:
            segments: List of [start_sec, end_sec] pairs.
        """
        if not segments:
            return []

        sorted_segs = sorted(segments, key=lambda x: x[0])
        merged: List[List[float]] = [sorted_segs[0]]
        for current in sorted_segs[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = [last[0], max(last[1], current[1])]
            else:
                merged.append(current)

        logger.info(
            f"Merged {len(segments)} segments into {len(merged)} "
            f"non-overlapping segments"
        )
        return merged

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS string."""
        seconds = max(0.0, float(seconds))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _events_to_segments(
        self, events: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """Extract time segments [start_sec, end_sec] from event list."""
        segments: List[List[float]] = []
        for ev in events:
            dur = ev.get("duration", [0, 0])
            if isinstance(dur, (list, tuple)) and len(dur) >= 2:
                try:
                    segments.append([float(dur[0]), float(dur[1])])
                except (TypeError, ValueError):
                    continue
        return segments

    def _frame_index_to_seconds(self, frame_idx: int) -> float:
        """Convert frame index to seconds using video FPS."""
        return float(frame_idx) / float(self.video_config["fps"])

    def _seconds_to_frame_index(self, seconds: float) -> int:
        """Convert seconds to frame index using video FPS."""
        return int(float(seconds) * float(self.video_config["fps"]))

    # -------------------------------------------------------------------------
    # Frame sampling over time segments
    # -------------------------------------------------------------------------

    def uniform_sample_frames_from_segments(
        self, time_segments: List[List[float]], max_frames: int = 256
    ) -> Tuple[List[Any], List[int]]:
        """
        Uniformly sample frames from time segments.

        Args:
            time_segments: List of [start_sec, end_sec] pairs (seconds).
            max_frames: Maximum number of frames to sample.

        Returns:
            (sampled_frames, frame_indices)
        """
        all_frame_indices: List[int] = []

        for start_sec, end_sec in time_segments:
            start_frame = self._seconds_to_frame_index(start_sec)
            end_frame = self._seconds_to_frame_index(end_sec)
            if end_frame < start_frame:
                continue
            for frame_idx in range(start_frame, end_frame + 1):
                all_frame_indices.append(frame_idx)

        # Remove duplicates and sort
        all_frame_indices = sorted(set(all_frame_indices))
        logger.info(
            f"Collected {len(all_frame_indices)} frames from "
            f"{len(time_segments)} segments (uniform)"
        )

        if len(all_frame_indices) > max_frames and max_frames > 0:
            positions = np.linspace(
                0, len(all_frame_indices) - 1, max_frames, dtype=int
            )
            sampled_frame_indices = [all_frame_indices[i] for i in positions]
            logger.info(
                f"Downsampled frames from {len(all_frame_indices)} "
                f"to {len(sampled_frame_indices)} (uniform)"
            )
        else:
            sampled_frame_indices = all_frame_indices

        sampled_frames = self.video.get_frames_by_indices(sampled_frame_indices)
        return sampled_frames, sampled_frame_indices

    def topk_sample_frames_from_segments(
        self,
        question: str,
        time_segments: List[List[float]],
        max_frames: int = 256,
    ) -> Tuple[List[Any], List[int], List[float]]:
        """
        Sample top-K frames from time segments based on semantic similarity.

        Args:
            question: The question text.
            time_segments: List of [start_sec, end_sec] pairs (seconds).
            max_frames: Maximum number of frames to retrieve.

        Returns:
            (frames, frame_indices, similarity_scores)
        """
        # 1 FPS sampling within segments
        sampled_1fps_indices: List[int] = []
        for start_sec, end_sec in time_segments:
            current_sec = float(start_sec)
            end_sec = float(end_sec)
            while current_sec <= end_sec:
                frame_idx = self._seconds_to_frame_index(current_sec)
                sampled_1fps_indices.append(frame_idx)
                current_sec += 1.0

        sampled_1fps_indices = sorted(set(sampled_1fps_indices))
        logger.info(
            f"1 FPS sampling resulted in {len(sampled_1fps_indices)} frames "
            f"from {len(time_segments)} segments"
        )

        if not sampled_1fps_indices:
            return [], [], []

        # Rewrite query for visual retrieval
        rewrite_prompt = PROMPTS["query_rewrite_for_visual_retrieval"].format(
            input_text=question
        )
        rewritten_query = self.llm_model.generate_response({"text": rewrite_prompt})
        logger.info(
            f"Rewritten query for visual retrieval "
            f"(first 100 chars): {rewritten_query[:100]}..."
        )

        # Query embedding
        query_embedding = self.features_vdb.embedding_model.get_text_features(
            [rewritten_query]
        )[0]

        vdb_data = self.features_vdb.client_storage["data"]
        vdb_matrix = self.features_vdb.client_storage["matrix"]

        filtered_vectors: List[np.ndarray] = []
        filtered_frame_indices: List[int] = []

        for i, data_entry in enumerate(vdb_data):
            frame_dir = data_entry.get("frame_dir", "")
            if not frame_dir:
                continue
            filename = os.path.basename(frame_dir)
            try:
                frame_idx = int(os.path.splitext(filename)[0])
            except ValueError:
                continue
            if frame_idx in sampled_1fps_indices:
                filtered_vectors.append(vdb_matrix[i])
                filtered_frame_indices.append(frame_idx)

        logger.info(
            f"Found {len(filtered_frame_indices)} frames in VDB matching "
            f"1 FPS sampled frames"
        )

        if not filtered_vectors:
            logger.warning("No frames found in VDB for the given segments")
            return [], [], []

        filtered_vectors_np = np.array(filtered_vectors, dtype=np.float32)
        # Normalize vectors
        filtered_vectors_np /= (
            np.linalg.norm(filtered_vectors_np, axis=1, keepdims=True) + 1e-8
        )
        query_embedding = query_embedding.astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding) + 1e-8

        similarities = np.dot(filtered_vectors_np, query_embedding)

        if len(similarities) > max_frames and max_frames > 0:
            top_k_positions = np.argsort(similarities)[-max_frames:][::-1]
        else:
            top_k_positions = np.argsort(similarities)[::-1]

        topk_frame_indices = [int(filtered_frame_indices[i]) for i in top_k_positions]
        topk_scores = [float(similarities[i]) for i in top_k_positions]

        # Sort by temporal order for downstream alignment
        sorted_pairs = sorted(zip(topk_frame_indices, topk_scores))
        topk_frame_indices = [p[0] for p in sorted_pairs]
        topk_scores = [p[1] for p in sorted_pairs]

        logger.info(
            f"Selected top {len(topk_frame_indices)} frames based on similarity"
        )

        topk_frames = self.video.get_frames_by_indices(topk_frame_indices)
        return topk_frames, topk_frame_indices, topk_scores

    # -------------------------------------------------------------------------
    # Event / frame alignment and formatting
    # -------------------------------------------------------------------------

    def project_frames_to_events(
        self, frame_indices: List[int], events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Project frame indices to their corresponding events.

        Only considers the provided `events` list (typically INDUS-aligned
        VDB events for the given question).
        """
        if not frame_indices or not events:
            return []

        frame_times = [self._frame_index_to_seconds(idx) for idx in frame_indices]
        selected_events: List[Dict[str, Any]] = []
        seen_event_ids = set()

        for frame_time in frame_times:
            for event in events:
                event_id = event.get("__id__") or event.get("id")
                duration = event.get("duration", [0, 0])
                if not isinstance(duration, (list, tuple)) or len(duration) < 2:
                    continue
                start, end = float(duration[0]), float(duration[1])
                if start <= frame_time <= end:
                    if event_id not in seen_event_ids:
                        selected_events.append(event)
                        seen_event_ids.add(event_id)
                    break

        selected_events.sort(key=lambda e: e.get("duration", [0, 0])[0])
        logger.info(
            f"Projected {len(frame_indices)} frames to {len(selected_events)} events"
        )
        return selected_events

    def project_segments_to_events(
        self, time_segments: List[List[float]], events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Project time segments to their corresponding events.

        Only considers the provided `events` list.
        """
        if not time_segments or not events:
            return []

        selected_events: List[Dict[str, Any]] = []
        seen_event_ids = set()

        for seg_start, seg_end in time_segments:
            seg_start = float(seg_start)
            seg_end = float(seg_end)
            for event in events:
                event_id = event.get("__id__") or event.get("id")
                duration = event.get("duration", [0, 0])
                if not isinstance(duration, (list, tuple)) or len(duration) < 2:
                    continue
                ev_start, ev_end = float(duration[0]), float(duration[1])
                # Overlap if not (event ends before seg OR event starts after seg)
                if not (ev_end < seg_start or ev_start > seg_end):
                    if event_id not in seen_event_ids:
                        selected_events.append(event)
                        seen_event_ids.add(event_id)

        selected_events.sort(key=lambda e: e.get("duration", [0, 0])[0])
        logger.info(
            f"Projected {len(time_segments)} segments to {len(selected_events)} events"
        )
        return selected_events

    def format_events_for_prompt(self, events: List[Dict[str, Any]]) -> str:
        """
        Format events as description list for prompt:
            "1. [HH:MM:SS - HH:MM:SS] Description"
        """
        lines: List[str] = []
        for i, event in enumerate(events, start=1):
            duration = event.get("duration", [0, 0])
            start_ts = self._seconds_to_timestamp(duration[0])
            end_ts = self._seconds_to_timestamp(duration[1])
            description = event.get("description", "")
            lines.append(f"{i}. [{start_ts} - {end_ts}] {description}")
        return "\n".join(lines)

    def format_frames_and_events_for_prompt(
        self, frame_indices: List[int], events: List[Dict[str, Any]]
    ) -> str:
        """
        Format frames and events together, showing visual evidence for each event.
        """
        if not events:
            return ""

        frame_times = [self._frame_index_to_seconds(idx) for idx in frame_indices]
        lines: List[str] = []

        for i, event in enumerate(events, start=1):
            duration = event.get("duration", [0, 0])
            start_sec, end_sec = float(duration[0]), float(duration[1])
            start_ts = self._seconds_to_timestamp(start_sec)
            end_ts = self._seconds_to_timestamp(end_sec)
            description = event.get("description", "")

            # Find frames that fall within this event
            matching_frames: List[str] = []
            for img_num, (frame_idx, frame_time) in enumerate(
                zip(frame_indices, frame_times), start=1
            ):
                if start_sec <= frame_time <= end_sec:
                    timestamp = self._seconds_to_timestamp(frame_time)
                    matching_frames.append(f"Image {img_num} @ {timestamp}")

            event_line = f"{i}. [{start_ts} - {end_ts}] {description}"
            lines.append(event_line)
            if matching_frames:
                visual_evidence = ", ".join(f"[{m}]" for m in matching_frames)
                lines.append(f"   > Visual Evidence: {visual_evidence}")
            else:
                lines.append("   > Visual Evidence: None in this range")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Modes 3–7 implementations
    # -------------------------------------------------------------------------

    def generate_answer_mode_3(
        self,
        question: str,
        question_id: int,
        retrieval_mode: str = "tri_view",
        max_frames: int = 256,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Mode 3: Uniform sampling → Events only.

        Uses INDUS seed events to build segments, samples frames for coverage
        (not passed to the VLM), and then projects frames to events to build
        a textual event-only prompt.
        """
        logger.info(
            f"IndusMultiMode - Mode 3 (Uniform → Events only) | Q{question_id}"
        )

        time_segments = self._get_time_segments_for_question(question_id)
        if not time_segments:
            return None, {"error": "No INDUS segments found", "mode": 3}

        # Sample frames only to choose representative events
        _, frame_indices = self.uniform_sample_frames_from_segments(
            time_segments, max_frames
        )
        if not frame_indices:
            return None, {"error": "No frames extracted", "mode": 3}

        # Restrict events to INDUS seed-aligned ones
        question_events = self._get_vdb_events_for_seed(question_id)
        if not question_events:
            return None, {"error": "No events found for INDUS seeds", "mode": 3}

        events = self.project_frames_to_events(frame_indices, question_events)
        if not events:
            # Fall back to using all question_events directly
            events = question_events

        if not events:
            return None, {"error": "No events found", "mode": 3}

        event_descriptions = self.format_events_for_prompt(events)
        prompt = PROMPTS["events_only"].format(
            event_descriptions=event_descriptions, user_query=question
        )

        logger.info("=== INDUS MODE 3 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Number of events: {len(events)}")
        logger.info(prompt)
        logger.info("=== END PROMPT ===")

        # VLM call is text-only for events-only mode
        vlm_input = {"text": prompt}
        try:
            response_list = self.llm_model.batch_generate_response([vlm_input])
            response = response_list[0] if response_list else None
        except Exception as e:
            logger.error(f"VLM generation failed (Indus Mode 3): {e}")
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
        max_frames: int = 256,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Mode 4: Top-K retrieval → Events only.

        Uses INDUS seed segments to perform top-K frame retrieval, then
        projects those frames to question-specific events and builds an
        events-only textual prompt.
        """
        logger.info(
            f"IndusMultiMode - Mode 4 (Top-K → Events only) | Q{question_id}"
        )

        time_segments = self._get_time_segments_for_question(question_id)
        if not time_segments:
            return None, {"error": "No INDUS segments found", "mode": 4}

        frames, frame_indices, similarity_scores = self.topk_sample_frames_from_segments(
            question, time_segments, max_frames
        )
        if not frame_indices:
            return None, {"error": "No frames extracted", "mode": 4}

        question_events = self._get_vdb_events_for_seed(question_id)
        if not question_events:
            return None, {"error": "No events found for INDUS seeds", "mode": 4}

        events = self.project_frames_to_events(frame_indices, question_events)
        if not events:
            events = question_events

        if not events:
            return None, {"error": "No events found", "mode": 4}

        event_descriptions = self.format_events_for_prompt(events)
        prompt = PROMPTS["events_only"].format(
            event_descriptions=event_descriptions, user_query=question
        )

        logger.info("=== INDUS MODE 4 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Number of events: {len(events)}")
        logger.info(prompt)
        logger.info("=== END PROMPT ===")

        vlm_input = {"text": prompt}
        try:
            response_list = self.llm_model.batch_generate_response([vlm_input])
            response = response_list[0] if response_list else None
        except Exception as e:
            logger.error(f"VLM generation failed (Indus Mode 4): {e}")
            return None, {"error": f"VLM generation failed: {str(e)}", "mode": 4}

        metadata = {
            "mode": 4,
            "num_events": len(events),
            "num_frames_sampled": len(frame_indices),
            "similarity_scores": similarity_scores,
            "event_segments": self._events_to_segments(events),
            "video_duration": self.video_config["duration"],
            "fps": self.video_config["fps"],
        }
        return response, metadata

    def generate_answer_mode_5(
        self,
        question: str,
        question_id: int,
        retrieval_mode: str = "tri_view",
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Mode 5: All segments → Events only.

        Uses all merged INDUS segments and all corresponding question-specific
        events; no frame sampling is involved.
        """
        logger.info(
            f"IndusMultiMode - Mode 5 (All segments → Events only) | Q{question_id}"
        )

        time_segments = self._get_time_segments_for_question(question_id)
        if not time_segments:
            return None, {"error": "No INDUS segments found", "mode": 5}

        question_events = self._get_vdb_events_for_seed(question_id)
        if not question_events:
            return None, {"error": "No events found for INDUS seeds", "mode": 5}

        events = self.project_segments_to_events(time_segments, question_events)
        if not events:
            events = question_events

        if not events:
            return None, {"error": "No events found", "mode": 5}

        event_descriptions = self.format_events_for_prompt(events)
        prompt = PROMPTS["events_only"].format(
            event_descriptions=event_descriptions, user_query=question
        )

        logger.info("=== INDUS MODE 5 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Number of events: {len(events)}")
        logger.info(prompt)
        logger.info("=== END PROMPT ===")

        vlm_input = {"text": prompt}
        try:
            response_list = self.llm_model.batch_generate_response([vlm_input])
            response = response_list[0] if response_list else None
        except Exception as e:
            logger.error(f"VLM generation failed (Indus Mode 5): {e}")
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
        max_frames: int = 256,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Mode 6: Uniform sampling → Frames + Events.

        Uses INDUS segments for uniform frame sampling, then aligns sampled
        frames with question-specific events and sends both visual and textual
        evidence to the VLM.
        """
        logger.info(
            f"IndusMultiMode - Mode 6 (Uniform → Frames + Events) | Q{question_id}"
        )

        time_segments = self._get_time_segments_for_question(question_id)
        if not time_segments:
            return None, {"error": "No INDUS segments found", "mode": 6}

        frames, frame_indices = self.uniform_sample_frames_from_segments(
            time_segments, max_frames
        )
        if not frames:
            return None, {"error": "No frames extracted", "mode": 6}

        question_events = self._get_vdb_events_for_seed(question_id)
        if not question_events:
            return None, {"error": "No events found for INDUS seeds", "mode": 6}

        events = self.project_frames_to_events(frame_indices, question_events)
        if not events:
            events = question_events

        if not events:
            return None, {"error": "No events found", "mode": 6}

        event_descriptions_with_frames = self.format_frames_and_events_for_prompt(
            frame_indices, events
        )

        prompt = PROMPTS["frames_and_events_aligned"].format(
            event_descriptions_with_frames=event_descriptions_with_frames,
            user_query=question,
        )

        logger.info("=== INDUS MODE 6 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(
            f"Number of frames: {len(frames)}, Number of events: {len(events)}"
        )
        logger.info(prompt)
        logger.info("=== END PROMPT ===")

        vlm_input = {"text": prompt, "video": frames}
        try:
            response_list = self.llm_model.batch_generate_response([vlm_input])
            response = response_list[0] if response_list else None
        except Exception as e:
            logger.error(f"VLM generation failed (Indus Mode 6): {e}")
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
        max_frames: int = 256,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Mode 7: Top-K retrieval → Frames + Events.

        Uses INDUS segments for top-K frame retrieval from features VDB, then
        aligns those frames with question-specific events and passes both to
        the VLM.
        """
        logger.info(
            f"IndusMultiMode - Mode 7 (Top-K → Frames + Events) | Q{question_id}"
        )

        time_segments = self._get_time_segments_for_question(question_id)
        if not time_segments:
            return None, {"error": "No INDUS segments found", "mode": 7}

        frames, frame_indices, similarity_scores = self.topk_sample_frames_from_segments(
            question, time_segments, max_frames
        )
        if not frames:
            return None, {"error": "No frames extracted", "mode": 7}

        question_events = self._get_vdb_events_for_seed(question_id)
        if not question_events:
            return None, {"error": "No events found for INDUS seeds", "mode": 7}

        events = self.project_frames_to_events(frame_indices, question_events)
        if not events:
            events = question_events

        if not events:
            return None, {"error": "No events found", "mode": 7}

        event_descriptions_with_frames = self.format_frames_and_events_for_prompt(
            frame_indices, events
        )

        prompt = PROMPTS["frames_and_events_aligned"].format(
            event_descriptions_with_frames=event_descriptions_with_frames,
            user_query=question,
        )

        logger.info("=== INDUS MODE 7 PROMPT ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(
            f"Number of frames: {len(frames)}, Number of events: {len(events)}"
        )
        logger.info(prompt)
        logger.info("=== END PROMPT ===")

        vlm_input = {"text": prompt, "video": frames}
        try:
            response_list = self.llm_model.batch_generate_response([vlm_input])
            response = response_list[0] if response_list else None
        except Exception as e:
            logger.error(f"VLM generation failed (Indus Mode 7): {e}")
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

    # -------------------------------------------------------------------------
    # Public entrypoint
    # -------------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        question_id: int,
        config_mode: int,
        retrieval_mode: str = "tri_view",
        max_frames: int = 256,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Generate answer using specified configuration mode (3–7).
        """
        mode_functions = {
            3: self.generate_answer_mode_3,
            4: self.generate_answer_mode_4,
            5: self.generate_answer_mode_5,
            6: self.generate_answer_mode_6,
            7: self.generate_answer_mode_7,
        }

        if config_mode not in mode_functions:
            raise ValueError(
                f"Invalid config_mode for IndusMultiMode: {config_mode}. "
                "Must be 3, 4, 5, 6, or 7."
            )

        mode_func = mode_functions[config_mode]

        if config_mode == 5:
            # Mode 5 ignores max_frames
            return mode_func(question, question_id, retrieval_mode)
        else:
            return mode_func(question, question_id, retrieval_mode, max_frames)


