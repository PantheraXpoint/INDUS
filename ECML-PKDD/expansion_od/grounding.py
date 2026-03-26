import re
import requests
from PIL import Image
import torch
from PIL import ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from typing import List, Union, Dict, Any, Set, Tuple


class GroundingDetector:
    """A class for grounded object detection using OwlViT model with batch support."""
    
    def __init__(self, model_name: str = "google/owlvit-base-patch32"):
        """
        Initialize the grounding detector.
        
        Args:
            model_name: HuggingFace model identifier for OwlViT
        """
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        self.model = self.model.cuda()
    
    def detect(
        self,
        images: Union[Image.Image, List[Image.Image]],
        text_labels: Union[List[str], List[List[str]]],
        threshold: float = 0.1,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in images based on text queries with batch support.
        
        Args:
            images: Single PIL Image or list of PIL Images
            text_labels: List of text queries (applied to all images) or 
                        list of lists (one per image)
            threshold: Confidence threshold for detections
            verbose: Whether to print detection results
            
        Returns:
            List of dictionaries, one per image, each containing:
            - 'boxes': List of bounding boxes [xmin, ymin, xmax, ymax]
            - 'scores': List of confidence scores
            - 'text_labels': List of detected text labels
        """
        # Normalize inputs to batch format
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(text_labels[0], str):
            # Same text labels for all images
            text_labels = [text_labels] * len(images)
        
        # Ensure text_labels matches number of images
        if len(text_labels) != len(images):
            raise ValueError(
                f"Number of text_label lists ({len(text_labels)}) must match "
                f"number of images ({len(images)})"
            )
        
        # Truncate text labels to avoid tokenization issues
        # OwlViT has a max length limit, so we truncate each text label
        max_length = 77  # Typical max length for CLIP-based models
        truncated_text_labels = []
        for label_list in text_labels:
            truncated_list = []
            for label in label_list:
                # Simple word-based truncation (rough approximation)
                words = label.split()
                if len(words) > max_length:
                    truncated_label = ' '.join(words[:max_length])
                else:
                    truncated_label = label
                truncated_list.append(truncated_label)
            truncated_text_labels.append(truncated_list)
        
        # Process inputs
        inputs = self.processor(
            text=truncated_text_labels, 
            images=images, 
            return_tensors="pt"
        )
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = self.model(**inputs)
        
        # Get target sizes for each image
        target_sizes = torch.tensor([(img.height, img.width) for img in images])
        
        # Post-process results (use truncated labels for consistency)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            text_labels=truncated_text_labels
        )
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["text_labels"]
            
            # Convert to lists
            boxes_list = [[round(coord, 2) for coord in box.tolist()] for box in boxes]
            scores_list = [round(score.item(), 3) for score in scores]
            labels_list = labels
            
            formatted_result = {
                "boxes": boxes_list,
                "scores": scores_list,
                "text_labels": labels_list
            }
            formatted_results.append(formatted_result)
            
            if verbose:
                print(f"\nImage {i + 1}:")
                for box, score, label in zip(boxes_list, scores_list, labels_list):
                    print(f"  Detected {label} with confidence {score} at location {box}")
        
        return formatted_results
    
    def visualize(
        self,
        images: Union[Image.Image, List[Image.Image]],
        results: List[Dict[str, Any]],
        output_paths: Union[str, List[str]] = None,
        outline_color: str = "red",
        outline_width: int = 2
    ) -> List[Image.Image]:
        """
        Draw bounding boxes on images based on detection results.
        
        Args:
            images: Single PIL Image or list of PIL Images
            results: Detection results from detect() method
            output_paths: Single output path or list of paths to save images.
                         If None, images won't be saved.
            outline_color: Color for bounding box outlines
            outline_width: Width of bounding box outlines
            
        Returns:
            List of annotated PIL Images
        """
        # Normalize inputs
        if isinstance(images, Image.Image):
            images = [images]
        
        if len(images) != len(results):
            raise ValueError(
                f"Number of images ({len(images)}) must match "
                f"number of results ({len(results)})"
            )
        
        if output_paths is not None:
            if isinstance(output_paths, str):
                output_paths = [output_paths]
            if len(output_paths) != len(images):
                raise ValueError(
                    f"Number of output paths ({len(output_paths)}) must match "
                    f"number of images ({len(images)})"
                )
        
        annotated_images = []
        for i, (image, result) in enumerate(zip(images, results)):
            # Create a copy to avoid modifying original
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["text_labels"]
            
            for box, score, label in zip(boxes, scores, labels):
                draw.rectangle(box, outline=outline_color, width=outline_width)
            
            annotated_images.append(annotated_image)
            
            # Save if output path provided
            if output_paths is not None:
                annotated_image.save(output_paths[i])
        
        return annotated_images


def extract_query_objects(question: str, answer_statements: List[str], llm) -> List[str]:
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
            llm_response = llm.batch_generate_response([{"text": prompt}], max_new_tokens=256, temperature=0.3)[0]
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
            return query_objects
        except Exception as e:
            print(f"Error using LLM for object extraction: {e}")
            return []
    return []
        


def _event_duration(event: Dict) -> Tuple[float, float]:
    """Get (start_sec, end_sec) from event (VDB has 'duration' at top level)."""
    d = event.get("duration")
    if isinstance(d, (list, tuple)) and len(d) >= 2:
        return float(d[0]), float(d[1])
    meta = event.get("metadata") or {}
    d = meta.get("duration")
    if isinstance(d, (list, tuple)) and len(d) >= 2:
        return float(d[0]), float(d[1])
    return 0.0, 0.0


def filter_events_by_grounding(
    video_path: str,
    events: List[Dict],
    query_objects: List[str],
    detector: "GroundingDetector",
    threshold: float = 0.1,
    target_fps: float = 0.5,
    batch_size: int = 8,
) -> Set[str]:
    """
    Keep only event IDs for events where at least one frame shows a query object (score >= threshold).
    events: list of dicts with 'id' (or '__id__') and 'duration' [start_sec, end_sec].
    """
    import cv2

    if not query_objects or not events or not video_path:
        return set()
    cap = cv2.VideoCapture("../" + video_path)
    if not cap.isOpened():
        return set()
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(video_fps / target_fps))
    to_detect: List[Tuple[str, int, Image.Image]] = []
    for ev in events:
        eid = ev.get("id") or ev.get("__id__")
        if not eid:
            continue
        start_sec, end_sec = _event_duration(ev)
        if end_sec <= start_sec:
            continue
        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)
        current = start_frame
        while current <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                to_detect.append((eid, current, Image.fromarray(frame_rgb)))
            current += frame_interval
    cap.release()
    if not to_detect:
        return set()
    images = [img for _, _, img in to_detect]
    all_scores = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        results = detector.detect(batch, query_objects, threshold=threshold, verbose=False)
        for r in results:
            all_scores.append(max(r["scores"]) if r["scores"] else 0.0)
    keep: Set[str] = set()
    for (eid, _, _), score in zip(to_detect, all_scores):
        if score >= threshold:
            keep.add(eid)
    return keep


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = GroundingDetector()

    # Single image example
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    text_labels = ["a photo of a cat", "a photo of a dog"]
    
    results = detector.detect(image, text_labels, threshold=0.1, verbose=True)
    annotated_images = detector.visualize(image, results, output_paths="result.png")
    
    # Batch example
    print("\n" + "="*50)
    print("Batch Processing Example:")
    print("="*50)
    
    # Multiple images with same queries
    length_batch = 64
    images_batch = [image] * length_batch  # Using same image twice for demo
    results_batch = detector.detect(
        images_batch,
        text_labels,
        threshold=0.1,
        verbose=True
    )
    # annotated_batch = detector.visualize(
    #     images_batch,
    #     results_batch,
    #     output_paths=[f"result_batch_{i}.png" for i in range(length_batch)]
    # )
    # )
