import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
from utils.grounding import GroundingDetector
from llms.QwenLM import QwenLM
from embeddings.JinaCLIP import JinaCLIP
from AVA.prompt import PROMPTS


def parse_time_reference(time_ref: str, fps: float = 30.0) -> Tuple[int, int]:
    """
    Parse time_reference string to frame numbers.
    Handles formats like:
    - "00:1:20" (hours:minutes:seconds)
    - "03:05:40" (hours:minutes:seconds)
    - "48:40-48:48" (minutes:seconds range)
    - "36:00-36:40" (minutes:seconds range)
    - "1:27:35" (hours:minutes:seconds)
    - "00:01:42" (hours:minutes:seconds)
    - "N/A"
    
    Returns:
        Tuple of (start_frame, end_frame). If single time, end_frame = start_frame + 1
    """
    if time_ref == "N/A" or not time_ref:
        return (0, 0)
    
    # Handle range format (e.g., "48:40-48:48")
    if '-' in time_ref:
        parts = time_ref.split('-')
        start_time = parts[0].strip()
        end_time = parts[1].strip()
        start_frame = time_to_frames(start_time, fps)
        end_frame = time_to_frames(end_time, fps)
        return (start_frame, end_frame)
    
    # Single time format
    frame = time_to_frames(time_ref, fps)
    return (frame, frame + 1)


def time_to_frames(time_str: str, fps: float) -> int:
    """
    Convert time string to frame number.
    Handles formats: "HH:MM:SS", "MM:SS", "M:SS"
    """
    parts = time_str.split(':')
    
    if len(parts) == 3:
        # Format: HH:MM:SS
        hours, minutes, seconds = map(float, parts)
        total_seconds = hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        # Format: MM:SS
        minutes, seconds = map(float, parts)
        total_seconds = minutes * 60 + seconds
    else:
        raise ValueError(f"Unknown time format: {time_str}")
    
    return int(total_seconds * fps)


def extract_frames_from_events(
    video_path: str,
    events: List[Dict],
    target_fps: float = 2.0
) -> List[Tuple[int, Image.Image]]:
    """
    Extract frames from video at target_fps rate within event durations.
    
    Args:
        video_path: Path to video file
        events: List of event dictionaries with metadata.duration
        target_fps: Target frames per second to sample
        
    Returns:
        List of tuples (frame_number, PIL.Image)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / target_fps))  # Frames to skip between samples
    
    extracted_frames = []
    
    for event in events:
        if 'metadata' not in event or 'duration' not in event['metadata']:
            continue
        
        duration = event['metadata']['duration']  # [start_time_sec, end_time_sec]
        start_sec = duration[0]
        end_sec = duration[1]

        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)
        
        # Sample frames at target_fps within this event duration
        current_frame = start_frame
        while current_frame <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                extracted_frames.append((current_frame, pil_image))
            
            current_frame += frame_interval
    
    cap.release()
    return extracted_frames


def extract_query_objects(question: str, answer_statements: List[str]) -> List[str]:
    """
    Extract object names from question and answer statements.
    Looks for shop names, landmarks, and other objects mentioned.
    """
    objects = []
    
    # Extract capitalized words (likely proper nouns/object names)
    # From question
    words = re.findall(r'\b[A-Z][a-z]+\b', question)
    objects.extend(words)
    
    # From answer statements
    for statement in answer_statements:
        words = re.findall(r'\b[A-Z][a-z]+\b', statement)
        objects.extend(words)
    
    # Also look for common patterns like "X shop", "X called Y", etc.
    patterns = [
        r'([A-Z][a-z]+)\s+(?:shop|store|café|cafe|restaurant|theater|theatre)',
        r'(?:called|named)\s+([A-Z][a-z]+)',
        r'([A-Z][a-z]+)\s+(?:before|after|when)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        objects.extend(matches)
        for statement in answer_statements:
            matches = re.findall(pattern, statement, re.IGNORECASE)
            objects.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_objects = []
    for obj in objects:
        obj_lower = obj.lower()
        # Filter out common words that aren't objects
        if obj_lower not in seen and obj_lower not in ['the', 'a', 'an', 'camera', 'wearer']:
            seen.add(obj_lower)
            unique_objects.append(obj)
    
    return unique_objects


def analyze_and_filter_events(
    events: List[Dict],
    frame_numbers: List[int],
    frame_scores: List[float],
    video_fps: float,
    time_reference: str,
    gt_start_frame: int,
    gt_end_frame: int
) -> Dict[str, Any]:
    """
    Analyze events: FIRST filter out events with no positive scores,
    THEN calculate time lengths and check if they hit time_reference.
    
    Args:
        events: List of event dictionaries with metadata.duration
        frame_numbers: List of frame numbers that were extracted
        frame_scores: List of detection scores for each frame
        video_fps: Video frames per second
        time_reference: Ground truth time reference string
        gt_start_frame: Ground truth start frame number
        gt_end_frame: Ground truth end frame number
        
    Returns:
        Dictionary containing:
        - 'event_analysis': List of event analysis dicts (only for filtered events)
        - 'filtered_events': Events that have at least one frame with score > 0
        - 'total_time_length': Total time length of all original events
        - 'filtered_time_length': Total time length of filtered events
        - 'filtered_events_hitting_gt': Filtered events that also hit GT
    """
    # Create a mapping from frame_number to score for quick lookup
    frame_to_score = dict(zip(frame_numbers, frame_scores))

    # STEP 0: Check if any event hits the GT time reference
    events_hitting_gt = []
    for event_idx, event in enumerate(events):
        if 'metadata' not in event or 'duration' not in event['metadata']:
            continue
        
        duration = event['metadata']['duration']  # [start_time_sec, end_time_sec]
        start_sec = duration[0]
        end_sec = duration[1]
                
        # Convert to frame numbers for comparison
        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)
        
        # Check if event overlaps with ground truth time reference
        hits_gt = False
        if gt_start_frame > 0:
            # Check if event overlaps with GT time range
            # Event overlaps if: event_start <= gt_end AND event_end >= gt_start
            hits_gt = (start_frame <= gt_end_frame and end_frame >= gt_start_frame)
        
        if hits_gt:
            events_hitting_gt.append(event)
    
    # STEP 1: Filter events - keep only those with at least one positive score
    filtered_events = []
    total_time_length = 0.0
    
    threshold = np.percentile([event["score_embedding"] for event in events], 90)
    frame_threshold = np.percentile([score for score in frame_scores], 50)
    for event_idx, event in enumerate(events):
        if 'metadata' not in event or 'duration' not in event['metadata']:
            continue
        
        duration = event['metadata']['duration']  # [start_time_sec, end_time_sec]
        start_sec = duration[0]
        end_sec = duration[1]
        
        # Calculate time length for statistics (before filtering)
        time_length = end_sec - start_sec
        total_time_length += time_length
        
        # Convert to frame numbers for comparison
        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)
        
        # Check if any extracted frames from this event have score > 0
        has_positive_score = False
        
        # Find all extracted frames that fall within this event's duration
        for frame_num, score in frame_to_score.items():
            if start_frame <= frame_num <= end_frame:
                if score > frame_threshold:
                    has_positive_score = True
                    break  # Found at least one positive score, no need to continue
        if event["score_embedding"] >= threshold:
            has_positive_score = True
        
        # Filter: keep events with at least one positive score
        if has_positive_score:
            filtered_events.append(event)
    
    # STEP 2: Calculate time lengths and check GT hits for FILTERED events only
    event_analysis = []
    filtered_events_hitting_gt = []
    filtered_time_length = 0.0
    
    for event_idx, filtered_event in enumerate(filtered_events):
        if 'metadata' not in filtered_event or 'duration' not in filtered_event['metadata']:
            continue
        
        duration = filtered_event['metadata']['duration']  # [start_time_sec, end_time_sec]
        start_sec = duration[0]
        end_sec = duration[1]
        
        # Calculate time length of this filtered event
        time_length = end_sec - start_sec
        filtered_time_length += time_length
        
        # Convert to frame numbers for comparison
        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)
        
        # Check if filtered event overlaps with ground truth time reference
        hits_gt = False
        if gt_start_frame > 0:
            # Check if event overlaps with GT time range
            # Event overlaps if: event_start <= gt_end AND event_end >= gt_start
            hits_gt = (start_frame <= gt_end_frame and end_frame >= gt_start_frame)
        
        # Get all frame scores for this event
        event_frame_scores = []
        for frame_num, score in frame_to_score.items():
            if start_frame <= frame_num <= end_frame:
                event_frame_scores.append(score)
        
        # Store analysis for this filtered event
        event_info = {
            'event_id': filtered_event.get('id', f'event_{event_idx}'),
            'start_sec': start_sec,
            'end_sec': end_sec,
            'time_length': time_length,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'hits_gt': hits_gt,
            'num_extracted_frames': len(event_frame_scores),
            'max_score_in_event': max(event_frame_scores) if event_frame_scores else 0.0,
            'avg_score_in_event': sum(event_frame_scores) / len(event_frame_scores) if event_frame_scores else 0.0
        }
        event_analysis.append(event_info)
        
        # Track filtered events that hit GT
        if hits_gt:
            filtered_events_hitting_gt.append(filtered_event)
    
    return {
        'event_analysis': event_analysis,  # Only for filtered events
        'filtered_events': filtered_events,
        'total_time_length': total_time_length,
        'filtered_time_length': filtered_time_length,
        'filtered_events_hitting_gt': filtered_events_hitting_gt,
        'events_hitting_gt': events_hitting_gt,
        'num_original_events': len(events),
        'num_filtered_events': len(filtered_events),
        'num_filtered_events_hitting_gt': len(filtered_events_hitting_gt)
    }


def process_question_folder(
    video_path: str,
    question_folder: str,
    detector: GroundingDetector,
    llm: QwenLM = None,
    embedding_model: JinaCLIP = None,
    output_dir: str = "iterative_pruning_output"
):
    """
    Process a single question folder:
    1. Read pruned_events.json to get event durations
    2. Read metadata.json to get query and GT time_reference
    3. Sample frames at 2 fps from event durations
    4. Run grounding detection
    5. Visualize results
    """
    question_folder = Path(question_folder)
    
    # Read metadata
    metadata_path = question_folder / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    question = metadata.get('question', '')
    answer_statements = metadata.get('answer_statements', [])
    time_reference = metadata.get('time_reference', 'N/A')
    if time_reference == "N/A":
        return
    question_id = metadata.get('question_id', 0)
    
    print(f"\nProcessing Question {question_id}: {question}")
    print(f"Time Reference: {time_reference}")
    
    # Read pruned events
    events_path = question_folder / "pruned_events.json"
    with open(events_path, 'r') as f:
        events = json.load(f)
    
    print(f"Found {len(events)} events")
    
    # Get video FPS
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Parse time reference to frames
    gt_start_frame, gt_end_frame = parse_time_reference(time_reference, video_fps)
    
    output_path = Path(output_dir) / f"{video_path.split('/')[-1].split('.')[0]}_q{question_id}"
    extracted_frames = []
    json_file = output_path / f"summary_q{question_id}.json"
    sucess_read = False
    try:
        with open(json_file, 'r') as f:
            summary = json.load(f)
        sucess_read = True
    except Exception as e:
        sucess_read = False 
    if output_path.exists():
        return
    if not output_path.exists() or not sucess_read:
        # Extract frames at 2 fps from event durations
        print("Extracting frames at 2 fps from event durations...")
        extracted_frames = extract_frames_from_events(video_path, events, target_fps=0.1)
        print(f"Extracted {len(extracted_frames)} frames")
        
        if not extracted_frames:
            print("No frames extracted. Skipping...")
            return
        
        # Extract query objects using LLM
        if llm is not None:
            print("Extracting objects using QwenLM...")
            # Create prompt to extract objects/entities from question and statements
            prompt = f"""Given the following question and answer statements, extract all relevant objects, entities, places, or things that can be searched by a open-world object detection model in a video.

    Question: {question}

    Answer Statements:
    {chr(10).join([f"- {stmt}" for stmt in answer_statements])}

    Please list all objects, entities, places, shops, landmarks, or things mentioned that can be searched by a open-world object detection model in a video. Return only a comma-separated list of objects, one per line. Be specific and do not include any explanatory text.

    Objects to search for:"""
            
            try:
                llm_response = llm.generate_response({"text": prompt}, max_new_tokens=256, temperature=0.3)
                # Parse the response to extract objects
                # Split by newlines and commas, clean up
                query_objects = []
                for line in llm_response.strip().split('\n'):
                    # Remove numbering, bullets, etc.
                    line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                    line = re.sub(r'^[-•]\s*', '', line)
                    # Split by comma
                    objects_in_line = [obj.strip() for obj in line.split(',')]
                    query_objects.extend([obj for obj in objects_in_line if obj and len(obj) > 1])
                
                # Remove duplicates while preserving order
                seen = set()
                unique_objects = []
                for obj in query_objects:
                    obj_lower = obj.lower()
                    if obj_lower not in seen and obj_lower not in ['the', 'a', 'an', 'camera', 'wearer']:
                        seen.add(obj_lower)
                        unique_objects.append(obj)
                
                query_objects = unique_objects
                print(f"LLM extracted objects: {query_objects}")
            except Exception as e:
                print(f"Error using LLM for object extraction: {e}")
                print("Falling back to simple extraction...")
                query_objects = extract_query_objects(question, answer_statements)
                print(f"Fallback extracted objects: {query_objects}")
        else:
            # Fallback to simple extraction if LLM not available
            query_objects = extract_query_objects(question, answer_statements)
            print(f"Query objects (simple extraction): {query_objects}")
        
        
        text_labels = query_objects
        
        # Run grounding detection on all frames
        print("Running grounding detection...")
        frame_numbers = [frame_num for frame_num, _ in extracted_frames]
        images = [img for _, img in extracted_frames]
        
        # Process in batches to avoid memory issues
        batch_size = 8
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_results = detector.detect(
                batch_images,
                text_labels,
                threshold=0.1,
                verbose=False
            )
            all_results.extend(batch_results)
        
        # Calculate scores for each frame
        # Use maximum score from all detections in the frame
        frame_scores = []
        for result in all_results:
            if result['scores']:
                max_score = max(result['scores']) # we can use the sum, avg score instead of the max score
            else:
                max_score = 0.0
            frame_scores.append(max_score)
    else:
        frame_scores = []
        frame_numbers = []
        frame_scores = summary['frame_scores']
        frame_numbers = summary['frame_numbers']
    events_scores = []
    template = PROMPTS["keyword_extraction"]
    prompt = template.format(input_text=question+" "+chr(10).join([f"- {stmt}" for stmt in answer_statements]))
    llm_response = llm.generate_response({"text": prompt}, max_new_tokens=256, temperature=0.3)

    query_embedding = embedding_model.get_text_features([llm_response.strip()])
    event_embeddings = embedding_model.get_text_features([event["content"] for event in events])
    events_scores = np.dot(event_embeddings, query_embedding.T)
    for event_idx, event in enumerate(events):
        event["score_embedding"] = float(events_scores[event_idx][0])
    # Analyze and filter events
    print("\nAnalyzing events...")
    event_analysis_result = analyze_and_filter_events(
        events=events,
        frame_numbers=frame_numbers,
        frame_scores=frame_scores,
        video_fps=video_fps,
        time_reference=time_reference,
        gt_start_frame=gt_start_frame,
        gt_end_frame=gt_end_frame
    )
    
    print(f"Total events: {event_analysis_result['num_original_events']}")
    print(f"Events with positive scores (filtered): {event_analysis_result['num_filtered_events']}")
    print(f"Filtered events hitting GT time: {event_analysis_result['num_filtered_events_hitting_gt']}")
    print(f"Total time length of all events: {event_analysis_result['total_time_length']:.2f} seconds")
    print(f"Total time length of filtered events: {event_analysis_result['filtered_time_length']:.2f} seconds")
    
    # Create visualization
    # if output_path.exists():
    #     return
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Scores over time
    frame_times = [frame_num / video_fps for frame_num in frame_numbers]
    
    plt.figure(figsize=(15, 6))
    # plt.plot(frame_times, frame_scores, 'b-', linewidth=2, label='Detection Score')
    plt.scatter(frame_times, frame_scores, c=frame_scores, cmap='viridis', s=50, alpha=0.6)
    
    # Mark GT time reference
    if gt_start_frame > 0:
        gt_time = gt_start_frame / video_fps
        plt.axvline(x=gt_time, color='r', linestyle='--', linewidth=2, label=f'GT Time Reference: {time_reference}')
        plt.scatter([gt_time], [max(frame_scores) if frame_scores else 0], 
                   color='red', s=200, marker='*', zorder=5, label='GT Time')
    
    # Highlight highest score region
    if frame_scores:
        max_score_idx = np.argmax(frame_scores)
        max_score_time = frame_times[max_score_idx]
        max_score_value = frame_scores[max_score_idx]
        
        plt.axvline(x=max_score_time, color='g', linestyle='--', linewidth=2, 
                   label=f'Highest Score: {max_score_value:.3f} at {max_score_time:.2f}s')
        plt.scatter([max_score_time], [max_score_value], 
                   color='green', s=200, marker='*', zorder=5)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Detection Score', fontsize=12)
    plt.title(f'Question {question_id}: {question[:60]}...', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = output_path / f"score_timeline_q{question_id}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved score timeline to {plot_path}")
    plt.close()
    
    # Plot 2: Frame-by-frame visualization with highest score frames (with detections)
    if frame_scores and len(extracted_frames) > 0:
        # Get top N frames with highest scores
        top_n = min(10, len(extracted_frames))
        top_indices = np.argsort(frame_scores)[-top_n:][::-1]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, top_idx in enumerate(top_indices):
            if idx >= len(axes):
                break
            
            frame_num, img = extracted_frames[top_idx]
            score = frame_scores[top_idx]
            time_sec = frame_num / video_fps
            result = all_results[top_idx]
            
            # Visualize detections on the image
            annotated_img = detector.visualize([img], [result])[0]
            
            axes[idx].imshow(annotated_img)
            axes[idx].set_title(f'Frame {frame_num}\nTime: {time_sec:.2f}s\nScore: {score:.3f}', 
                              fontsize=10)
            axes[idx].axis('off')
            
            # Mark if this is near GT time
            if gt_start_frame > 0 and abs(frame_num - gt_start_frame) < video_fps * 5:
                axes[idx].add_patch(plt.Rectangle((0, 0), img.width, img.height, 
                                                 fill=False, edgecolor='red', linewidth=3))
        
        plt.suptitle(f'Top {top_n} Frames by Detection Score - Question {question_id}', fontsize=14)
        plt.tight_layout()
        
        top_frames_path = output_path / f"top_frames_q{question_id}.png"
        plt.savefig(top_frames_path, dpi=150, bbox_inches='tight')
        print(f"Saved top frames to {top_frames_path}")
        plt.close()
        
        # Also create a visualization showing frames around GT time reference
        if gt_start_frame > 0:
            # Find frames closest to GT time
            gt_time_sec = gt_start_frame / video_fps
            time_diffs = [abs(t - gt_time_sec) for t in frame_times]
            closest_indices = np.argsort(time_diffs)[:min(5, len(extracted_frames))]
            
            fig, axes = plt.subplots(1, len(closest_indices), figsize=(4*len(closest_indices), 4))
            if len(closest_indices) == 1:
                axes = [axes]
            
            for idx, frame_idx in enumerate(closest_indices):
                frame_num, img = extracted_frames[frame_idx]
                time_sec = frame_times[frame_idx]
                score = frame_scores[frame_idx]
                result = all_results[frame_idx]
                
                annotated_img = detector.visualize([img], [result])[0]
                axes[idx].imshow(annotated_img)
                axes[idx].set_title(f'Frame {frame_num}\nTime: {time_sec:.2f}s\nScore: {score:.3f}\n(GT: {gt_time_sec:.2f}s)', 
                                  fontsize=10)
                axes[idx].axis('off')
            
            plt.suptitle(f'Frames Near GT Time Reference ({time_reference}) - Question {question_id}', fontsize=14)
            plt.tight_layout()
            
            gt_frames_path = output_path / f"gt_time_frames_q{question_id}.png"
            plt.savefig(gt_frames_path, dpi=150, bbox_inches='tight')
            print(f"Saved GT time frames to {gt_frames_path}")
            plt.close()
    
    # Save detection results summary
    summary = {
        'question_id': question_id,
        'question': question,
        'time_reference': time_reference,
        'gt_start_frame': int(gt_start_frame),
        'gt_end_frame': int(gt_end_frame),
        'num_events': len(events),
        # 'num_frames_extracted': len(extracted_frames),
        # 'query_objects': query_objects,
        'max_score': float(max(frame_scores)) if frame_scores else 0.0,
        'max_score_frame': int(frame_numbers[np.argmax(frame_scores)]) if frame_scores else 0,
        'max_score_time': float(frame_times[np.argmax(frame_scores)]) if frame_scores else 0.0,
        'frame_scores': [float(s) for s in frame_scores],
        'frame_numbers': frame_numbers,
        'frame_times': [float(t) for t in frame_times],
        'event_analysis': {
            'num_original_events': event_analysis_result['num_original_events'],
            'num_filtered_events': event_analysis_result['num_filtered_events'],
            'num_filtered_events_hitting_gt': event_analysis_result['num_filtered_events_hitting_gt'],
            'total_time_length_seconds': float(event_analysis_result['total_time_length']),
            'filtered_time_length_seconds': float(event_analysis_result['filtered_time_length']),
            'event_details': event_analysis_result['event_analysis'],  # Only filtered events
            'events_hitting_gt': event_analysis_result['events_hitting_gt']
        }
    }
    
    summary_path = output_path / f"summary_q{question_id}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


def main(args):
    """Main function to process all question folders."""
    output_dir = "iterative_pruning_output_lvbench"
    
    # Initialize grounding detector
    print("Initializing GroundingDetector...")
    detector = GroundingDetector()
    # detector = None
    
    # Initialize QwenLM for object extraction
    print("Initializing QwenLM...")
    try:
        llm = QwenLM()
        print("QwenLM initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize QwenLM: {e}")
        print("Will use simple extraction method instead")
        llm = None
    embedding_model = JinaCLIP()
    print("JinaCLIP initialized successfully")
    
    base_path = Path("../original/Project-Ava/SAE-k2-lvbench_buffer_matrix")
    base_folders = [base_path / folder for folder in os.listdir(base_path)]
    for base_folder in base_folders:        
        video_path = f"{args[0]}/videos/{base_folder.name}.mp4"
        # Find all question folders
        base_path = base_folder
        question_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('q')])
        
        print(f"Found {len(question_folders)} question folders")
        
        # Process each question folderx
        for q_folder in question_folders:
            try:
                process_question_folder(video_path, str(q_folder), detector, llm, embedding_model, output_dir)
            except Exception as e:
                print(f"Error processing {q_folder}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(args)
