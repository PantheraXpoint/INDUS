#!/usr/bin/env python3
"""
Test script for DynamicSampler class
Reads a video using cv2 and tests the dynamic sampling pipeline
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add AVA directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'AVA'))
from dynamic_sampling import DynamicSampler, Segment, Batch


def test_dynamic_sampling(video_path: str, max_frames: int = 150, verbose: bool = False):
    """
    Test the dynamic sampling pipeline on a video.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process
    """
    print("=" * 70)
    print("Testing Dynamic Video Frame Sampling")
    print("=" * 70)
    print(f"Video path: {video_path}")
    print(f"Max frames: {max_frames}")
    print()
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return False
    
    try:
        # Initialize sampler
        print("Initializing DynamicSampler...")
        sampler = DynamicSampler(
            embedding_model=None,  # Will auto-initialize
            buffer_size=150,
            similarity_threshold=0.75,
            min_segments=8,
            base_rate=15.0,  # Match example
            min_rate=1.0,
            max_rate=30.0,
            min_frames_per_segment=2,
            batch_capacity=10,
            density_method="variance",  # Match example (can be "duration", "variance", or "hybrid")
            alpha=0.5,
            beta=0.5,
            verbose=verbose  # Enable verbose output
        )
        print("✓ DynamicSampler initialized")
        print()
        
        # Process video
        print("Processing video...")
        segments, batches, video_info = sampler.process_video(video_path, max_frames=max_frames)
        print()
        
        # Print results
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Video Info:")
        print(f"  - FPS: {video_info['fps']:.2f}")
        print(f"  - Total frames: {video_info['total_frames']}")
        print(f"  - Duration: {video_info['duration']:.2f}s")
        print(f"  - Resolution: {video_info['width']}x{video_info['height']}")
        print()
        
        print(f"Segments: {len(segments)}")
        print("-" * 70)
        for i, seg in enumerate(segments, 1):
            print(f"Segment {i}:")
            print(f"  - Frames: {seg.start_frame} to {seg.end_frame} ({seg.frame_count} frames)")
            print(f"  - Duration: {seg.duration:.2f}s")
            print(f"  - Density: {seg.density:.4f}")
            print(f"  - Sampling rate: {seg.sampling_rate:.2f} s/frame")
            print(f"  - Allocated frames: {seg.allocated_frames}")
            print(f"  - Selected frames: {len(seg.selected_frames)} frames")
            if seg.selected_frames:
                print(f"    → Frame indices: {seg.selected_frames[:10]}{'...' if len(seg.selected_frames) > 10 else ''}")
        print()
        
        print(f"Batches: {len(batches)}")
        print("-" * 70)
        total_batch_frames = 0
        for i, batch in enumerate(batches, 1):
            print(f"Batch {i}:")
            print(f"  - Frames: {len(batch.frames)}")
            print(f"  - Segments: {len(batch.segment_ids)}")
            if batch.frames:
                frame_indices = [f[0] for f in batch.frames]
                print(f"  - Frame indices: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")
            total_batch_frames += len(batch.frames)
        print()
        
        print(f"Summary:")
        print(f"  - Total segments: {len(segments)}")
        print(f"  - Total batches: {len(batches)}")
        print(f"  - Total frames in batches: {total_batch_frames}")
        print(f"  - Average frames per batch: {total_batch_frames / len(batches) if batches else 0:.2f}")
        print()
        
        # Visualize: save sample frames from first batch
        if batches and batches[0].frames:
            print("Saving sample frames from first batch...")
            output_dir = Path("test_output")
            output_dir.mkdir(exist_ok=True)
            
            for i, (frame_idx, frame) in enumerate(batches[0].frames[:5]):  # Save first 5 frames
                output_path = output_dir / f"sample_frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(output_path), frame)
                print(f"  → Saved {output_path}")
            print()
        
        print("=" * 70)
        print("Test completed successfully!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Dynamic Video Frame Sampling")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Maximum number of frames to process (default: 150)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debugging output"
    )
    
    args = parser.parse_args()
    
    success = test_dynamic_sampling(args.video, args.max_frames, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

