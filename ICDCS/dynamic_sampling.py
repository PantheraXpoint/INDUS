"""
Dynamic Video Frame Sampling with Scene Segmentation
Implements FastVID DySeg algorithm for adaptive frame sampling
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import sys
import os

# Add embeddings directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embeddings'))
from JinaCLIP import JinaCLIP


@dataclass
class Segment:
    """Represents a video segment"""
    start_frame: int
    end_frame: int
    duration: float  # in seconds
    frame_count: int
    embeddings: np.ndarray  # CLIP embeddings for frames in this segment
    density: float = 0.0
    sampling_rate: float = 1.0  # seconds per frame
    allocated_frames: int = 0
    selected_frames: List[int] = None  # frame indices selected for this segment
    
    def __post_init__(self):
        if self.selected_frames is None:
            self.selected_frames = []


@dataclass
class Batch:
    """Represents a batch of frames"""
    frames: List[Tuple[int, np.ndarray]]  # List of (frame_index, frame_data)
    segment_ids: List[int]  # Which segments these frames belong to


class DynamicSampler:
    """
    Dynamic video frame sampler using scene segmentation and adaptive sampling.
    
    Implements 5-stage pipeline:
    1. Scene Segmentation (FastVID DySeg)
    2. Density Heuristic
    3. Adaptive Frame Allocation
    4. Uniform Sampling Within Segment
    5. Batch Assembly
    """
    
    def __init__(
        self,
        embedding_model: Optional[JinaCLIP] = None,
        buffer_size: int = 150,
        similarity_threshold: float = 0.75,
        min_segments: int = 8,
        base_rate: float = 1.0,  # base sampling rate in seconds/frame
        min_rate: float = 1.0,
        max_rate: float = 30.0,
        min_frames_per_segment: int = 2,
        batch_capacity: int = 10,
        density_method: str = "hybrid",  # "duration", "variance", or "hybrid"
        alpha: float = 0.5,  # weight for duration in hybrid method
        beta: float = 0.5,  # weight for variance in hybrid method
        verbose: bool = False,  # Enable verbose debugging output
    ):
        """
        Initialize the dynamic sampler.
        
        Args:
            embedding_model: JinaCLIP model for generating embeddings. If None, will initialize.
            buffer_size: Number of frames to buffer for processing
            similarity_threshold: Threshold for detecting scene boundaries (cosine similarity)
            min_segments: Minimum number of segments to create
            base_rate: Base sampling rate in seconds per frame
            min_rate: Minimum sampling rate (seconds per frame)
            max_rate: Maximum sampling rate (seconds per frame)
            min_frames_per_segment: Minimum frames to sample per segment
            batch_capacity: Maximum frames per batch
            density_method: Method for computing density ("duration", "variance", or "hybrid")
            alpha: Weight for duration component in hybrid method
            beta: Weight for variance component in hybrid method
            verbose: Enable verbose debugging output
        """
        self.embedding_model = embedding_model
        if self.embedding_model is None:
            print("Initializing JinaCLIP model...")
            self.embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        
        self.buffer_size = buffer_size
        self.similarity_threshold = similarity_threshold
        self.min_segments = min_segments
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.min_frames_per_segment = min_frames_per_segment
        self.batch_capacity = batch_capacity
        self.density_method = density_method
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        
    def extract_clip_embeddings(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract CLIP embeddings for a list of frames.
        
        Args:
            frames: List of frames (BGR format from cv2)
            
        Returns:
            Embeddings array of shape (num_frames, embedding_dim)
        """
        # Convert BGR to RGB and to PIL Images
        pil_images = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_images.append(pil_image)
        
        # Get embeddings
        embeddings = self.embedding_model.get_image_features(pil_images)
        return embeddings
    
    def compute_cosine_similarity(self, e1: np.ndarray, e2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    
    def stage1_scene_segmentation(
        self, 
        embeddings: np.ndarray, 
        fps: float,
        frame_indices: List[int]
    ) -> List[Segment]:
        """
        Stage 1: Scene Segmentation (FastVID DySeg)
        
        Args:
            embeddings: CLIP embeddings for frames, shape (num_frames, embedding_dim)
            fps: Frames per second of the video
            frame_indices: Original frame indices in the video
            
        Returns:
            List of Segment objects
        """
        num_frames = len(embeddings)
        
        # Compute inter-frame similarities
        similarities = []
        for i in range(num_frames - 1):
            sim = self.compute_cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Detect boundaries
        # S_1: Minimum similarity points (for min_segments)
        # S_2: Similarity below threshold
        boundaries = set()
        
        # S_1: Get top (min_segments - 1) lowest similarity points
        if len(similarities) >= self.min_segments - 1:
            min_indices = np.argsort(similarities)[:self.min_segments - 1]
            boundaries.update(min_indices.tolist())
        
        # S_2: Points below threshold
        below_threshold = np.where(similarities < self.similarity_threshold)[0]
        boundaries.update(below_threshold.tolist())
        
        # Convert to sorted list and add start/end
        boundaries = sorted(boundaries)
        boundaries = [0] + [b + 1 for b in boundaries] + [num_frames]
        
        # Remove duplicates and ensure sorted
        boundaries = sorted(list(set(boundaries)))
        # Create segments
        segments = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            segment_embeddings = embeddings[start_idx:end_idx]
            segment_frame_indices = frame_indices[start_idx:end_idx]
            
            duration = (end_idx - start_idx) / fps
            frame_count = end_idx - start_idx
            
            segment = Segment(
                start_frame=segment_frame_indices[0],
                end_frame=segment_frame_indices[-1],
                duration=duration,
                frame_count=frame_count,
                embeddings=segment_embeddings
            )
            segments.append(segment)
        
        # Verbose output
        if self.verbose:
            print("\n" + "─" * 65)
            print("Step 1: Segment (DySeg)")
            print("─" * 65)
            print("Output:")
            for i, seg in enumerate(segments, 1):
                print(f"  Seg_{i}: {seg.frame_count} frames, {seg.duration:.1f} sec")
        
        return segments
    
    def stage2_density_heuristic(self, segments: List[Segment]) -> List[Segment]:
        """
        Stage 2: Compute density for each segment.
        
        Args:
            segments: List of segments
            
        Returns:
            Segments with density values computed
        """
        # Store raw variance values for verbose output
        raw_variances = []
        raw_densities = []
        
        for segment in segments:
            if self.density_method == "duration":
                # Option A: Duration-based
                if segment.duration > 0:
                    segment.density = 1.0 / segment.duration
                else:
                    segment.density = 1.0
                raw_variances.append(None)
                raw_densities.append(segment.density)
                    
            elif self.density_method == "variance":
                # Option B: Variance-based
                if len(segment.embeddings) > 1:
                    raw_var = np.var(segment.embeddings, axis=0).mean()
                    raw_variances.append(raw_var)
                    segment.density = raw_var
                    raw_densities.append(raw_var)
                else:
                    raw_variances.append(0.0)
                    segment.density = 0.0
                    raw_densities.append(0.0)
                    
            elif self.density_method == "hybrid":
                # Option C: Hybrid (recommended)
                duration_component = 1.0 / segment.duration if segment.duration > 0 else 1.0
                variance_component = np.var(segment.embeddings, axis=0).mean() if len(segment.embeddings) > 1 else 0.0
                raw_variances.append(variance_component)
                
                # Normalize components (simple min-max normalization)
                # For duration: typically ranges from ~0.03 to 1.0+ (for 1-30 seconds)
                # For variance: typically ranges from 0.0 to ~0.1
                # We'll use raw values and let alpha/beta handle weighting
                segment.density = self.alpha * duration_component + self.beta * variance_component
                raw_densities.append(segment.density)
            else:
                raise ValueError(f"Unknown density method: {self.density_method}")
        
        # Normalize densities to [0, 1] for verbose output
        if self.verbose and raw_densities:
            min_density = min(d for d in raw_densities if d is not None)
            max_density = max(d for d in raw_densities if d is not None)
            density_range = max_density - min_density if max_density > min_density else 1.0
            
            normalized_densities = []
            for i, seg in enumerate(segments):
                if density_range > 0:
                    norm_density = (raw_densities[i] - min_density) / density_range
                else:
                    norm_density = 0.5
                normalized_densities.append(norm_density)
            
            # Verbose output
            print("\n" + "─" * 65)
            print("Step 2: Variance-Based Density" if self.density_method == "variance" else f"Step 2: {self.density_method.capitalize()}-Based Density")
            print("─" * 65)
            for i, seg in enumerate(segments):
                if self.density_method == "variance":
                    var_val = raw_variances[i] if raw_variances[i] is not None else 0.0
                    # Simple interpretation based on variance value
                    if var_val < 0.05:
                        interpretation = "static/empty"
                    elif var_val < 0.15:
                        interpretation = "slow motion"
                    elif var_val < 0.25:
                        interpretation = "moderate motion"
                    else:
                        interpretation = "fast motion/transitions"
                    print(f"Seg_{i+1}: var(embeddings) = {var_val:.2f} ({interpretation})")
                else:
                    print(f"Seg_{i+1}: density = {raw_densities[i]:.4f}")
                print(f"Normalize to [0,1]: ρ_{i+1}={normalized_densities[i]:.2f}")
        
        return segments
    
    def stage3_adaptive_frame_allocation(self, segments: List[Segment]) -> List[Segment]:
        """
        Stage 3: Adaptive Frame Allocation.
        
        Args:
            segments: List of segments with density computed
            
        Returns:
            Segments with sampling_rate and allocated_frames computed
        """
        # Store intermediate values for verbose output
        allocation_info = []
        
        for segment in segments:
            # Compute segment sampling rate
            if segment.density > 0:
                rate_i_raw = self.base_rate / segment.density
            else:
                rate_i_raw = self.max_rate
            
            # Apply constraints
            rate_i = np.clip(rate_i_raw, self.min_rate, self.max_rate)
            segment.sampling_rate = rate_i
            
            # Compute frames needed
            n_i_raw = segment.duration / rate_i
            n_i = int(np.ceil(n_i_raw))
            
            # Apply min/max per segment
            n_i_before_constraint = n_i
            n_i = np.clip(n_i, self.min_frames_per_segment, segment.frame_count)
            segment.allocated_frames = n_i
            
            allocation_info.append({
                'rate_raw': rate_i_raw,
                'rate_final': rate_i,
                'n_raw': n_i_raw,
                'n_before_constraint': n_i_before_constraint,
                'n_final': n_i,
                'min_applied': n_i_before_constraint < self.min_frames_per_segment
            })
        
        # Verbose output
        if self.verbose:
            print("\n" + "─" * 65)
            print("Step 3: Allocation")
            print("─" * 65)
            for i, (segment, info) in enumerate(zip(segments, allocation_info)):
                print(f"rate_{i+1} = {self.base_rate} / {segment.density:.2f} = {info['rate_raw']:.1f} sec/frame", end="")
                if info['rate_final'] != info['rate_raw']:
                    print(f" → clipped to {info['rate_final']:.1f} sec/frame", end="")
                print()
                print(f"  → n_{i+1} = {segment.duration:.0f}/{info['rate_final']:.1f} = {info['n_raw']:.1f} frames", end="")
                if info['min_applied']:
                    print(f" (min={self.min_frames_per_segment})", end="")
                if info['n_final'] != info['n_before_constraint']:
                    print(f" → {info['n_final']} frames", end="")
                print()
            total_frames = sum(seg.allocated_frames for seg in segments)
            print(f"After min constraint: {', '.join(f'n_{i+1}={seg.allocated_frames}' for i, seg in enumerate(segments))} (total: {total_frames} frames)")
        
        return segments
    
    def stage4_uniform_sampling(self, segments: List[Segment], all_frames: List[np.ndarray]) -> List[Segment]:
        """
        Stage 4: Uniform Sampling Within Segment.
        
        Args:
            segments: List of segments with allocated_frames computed
            all_frames: All frames from the video buffer
            
        Returns:
            Segments with selected_frames populated
        """
        for segment in segments:
            n_i = segment.allocated_frames
            k_i = segment.frame_count
            
            if n_i >= k_i:
                # Sample all frames
                segment.selected_frames = list(range(segment.start_frame, segment.end_frame + 1))
            else:
                # Sample uniformly: every (k_i / n_i)-th frame
                step = k_i / n_i
                indices = []
                for i in range(n_i):
                    idx = int(segment.start_frame + i * step)
                    # Ensure we don't exceed segment bounds
                    idx = min(idx, segment.end_frame)
                    indices.append(idx)
                
                # Remove duplicates and sort
                segment.selected_frames = sorted(list(set(indices)))
        
        # Verbose output
        if self.verbose:
            print("\n" + "─" * 65)
            print("Step 4: Uniform Sampling")
            print("─" * 65)
            for i, segment in enumerate(segments, 1):
                frames_str = str(segment.selected_frames)
                if len(frames_str) > 50:
                    frames_str = frames_str[:47] + "..."
                print(f"Seg_{i}: Pick frames {frames_str} (evenly spaced in {segment.frame_count} frames)")
        
        return segments
    
    def stage5_batch_assembly(
        self, 
        segments: List[Segment], 
        all_frames: List[np.ndarray]
    ) -> List[Batch]:
        """
        Stage 5: Greedy Bin-Packing into Batches.
        
        Args:
            segments: List of segments with selected_frames populated
            all_frames: All frames from the video buffer
            
        Returns:
            List of Batch objects
        """
        batches = []
        current_batch = []
        current_batch_segments = []
        
        # Create segment queue with frames
        segment_queue = []
        for seg in segments:
            for frame_idx in seg.selected_frames:
                segment_queue.append((frame_idx, seg))
        
        # Greedy packing
        for frame_idx, seg in segment_queue:
            # Check if adding this frame would exceed capacity
            if len(current_batch) + 1 <= self.batch_capacity:
                # Add frame to current batch
                if frame_idx < len(all_frames):
                    current_batch.append((frame_idx, all_frames[frame_idx]))
                    current_batch_segments.append(id(seg))
            else:
                # Current batch is full, start new batch
                if current_batch:
                    batches.append(Batch(
                        frames=current_batch.copy(),
                        segment_ids=list(set(current_batch_segments))
                    ))
                
                # Start new batch with current frame
                current_batch = [(frame_idx, all_frames[frame_idx])] if frame_idx < len(all_frames) else []
                current_batch_segments = [id(seg)]
        
        # Add remaining batch
        if current_batch:
            batches.append(Batch(
                frames=current_batch,
                segment_ids=list(set(current_batch_segments))
            ))
        
        # Verbose output
        if self.verbose:
            print("\n" + "─" * 65)
            print("Step 5: Batch Assembly")
            print("─" * 65)
            # Create mapping from segment id to segment index
            seg_id_to_idx = {id(seg): i+1 for i, seg in enumerate(segments)}
            
            for i, batch in enumerate(batches, 1):
                # Group frames by segment
                seg_frames = {}
                for frame_idx, _ in batch.frames:
                    # Find which segment this frame belongs to
                    for seg in segments:
                        if frame_idx in seg.selected_frames:
                            seg_idx = seg_id_to_idx[id(seg)]
                            if seg_idx not in seg_frames:
                                seg_frames[seg_idx] = []
                            seg_frames[seg_idx].append(frame_idx)
                            break
                
                # Format output
                batch_parts = []
                for seg_idx in sorted(seg_frames.keys()):
                    frames = sorted(seg_frames[seg_idx])
                    frames_str = ",".join(map(str, frames))
                    batch_parts.append(f"Seg{seg_idx}_frames: {frames_str}")
                
                batch_str = ", ".join(batch_parts)
                print(f"Batch_{i}: [{batch_str}]")
        
        return batches
    
    def process_video_buffer(
        self, 
        frames: List[np.ndarray], 
        fps: float,
        frame_indices: Optional[List[int]] = None
    ) -> Tuple[List[Segment], List[Batch]]:
        """
        Process a buffer of video frames through all 5 stages.
        
        Args:
            frames: List of video frames (BGR format)
            fps: Frames per second of the video
            frame_indices: Optional list of original frame indices. If None, uses 0..len(frames)-1
            
        Returns:
            Tuple of (segments, batches)
        """
        if frame_indices is None:
            frame_indices = list(range(len(frames)))
        
        # Verbose: Show input parameters
        if self.verbose:
            print("\n" + "─" * 65)
            print("Input: Video buffer")
            print("─" * 65)
            print(f"  - Frames: {len(frames)}")
            print(f"  - batch_size: {self.batch_capacity}")
            print(f"  - base_rate: {self.base_rate}")
            print()
        
        if not self.verbose:
            print(f"Stage 1: Scene Segmentation - Processing {len(frames)} frames...")
        # Stage 1: Extract embeddings
        embeddings = self.extract_clip_embeddings(frames)
        
        # Stage 1: Scene segmentation
        segments = self.stage1_scene_segmentation(embeddings, fps, frame_indices)
        if not self.verbose:
            print(f"  → Detected {len(segments)} segments")
        
        # Stage 2: Density heuristic
        if not self.verbose:
            print("Stage 2: Density Heuristic...")
        segments = self.stage2_density_heuristic(segments)
        
        # Stage 3: Adaptive frame allocation
        if not self.verbose:
            print("Stage 3: Adaptive Frame Allocation...")
        segments = self.stage3_adaptive_frame_allocation(segments)
        if not self.verbose:
            total_allocated = sum(seg.allocated_frames for seg in segments)
            print(f"  → Allocated {total_allocated} frames across {len(segments)} segments")
        
        # Stage 4: Uniform sampling
        if not self.verbose:
            print("Stage 4: Uniform Sampling Within Segments...")
        segments = self.stage4_uniform_sampling(segments, frames)
        
        # Stage 5: Batch assembly
        if not self.verbose:
            print("Stage 5: Batch Assembly...")
        batches = self.stage5_batch_assembly(segments, frames)
        if not self.verbose:
            print(f"  → Created {len(batches)} batches")
        
        return segments, batches
    
    def process_video(
        self, 
        video_path: str, 
        max_frames: Optional[int] = None
    ) -> Tuple[List[Segment], List[Batch], Dict]:
        """
        Process a video file and return segments and batches.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            Tuple of (segments, batches, video_info)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': total_frames / fps if fps > 0 else 0
        }
        
        print(f"Video info: {total_frames} frames, {fps:.2f} fps, {video_info['duration']:.2f}s")
        
        # Read frames into buffer
        frames = []
        frame_indices = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            frames.append(frame)
            frame_indices.append(frame_count)
            frame_count += 1
            
            # Process in buffer_size chunks
            if len(frames) >= self.buffer_size:
                # Process this buffer
                buffer_segments, buffer_batches = self.process_video_buffer(
                    frames, fps, frame_indices
                )
                
                # For now, we'll process the first buffer and return
                # In a full implementation, you'd accumulate segments/batches
                cap.release()
                return buffer_segments, buffer_batches, video_info
        
        # Process remaining frames
        if frames:
            segments, batches = self.process_video_buffer(frames, fps, frame_indices)
            cap.release()
            return segments, batches, video_info
        
        cap.release()
        return [], [], video_info

if __name__ == "__main__":
    sampler = DynamicSampler(verbose=True, min_frames_per_segment=3, buffer_size=900, base_rate=0.1)
    segments, batches, video_info = sampler.process_video("datas/front.mp4")