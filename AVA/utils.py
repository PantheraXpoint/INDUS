import asyncio
import html
import io
import csv
import json
import logging
import os
import re
import time
import coloredlogs
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List, Dict
import xml.etree.ElementTree as ET
from .prompt import PROMPTS
from llms.BaseModel import BaseLanguageModel
from embeddings.object_search import SearchSystem
from typing import Optional
from PIL import Image
import numpy as np
import tiktoken
import cv2

ENCODER = None

logger = logging.getLogger("videorag")
coloredlogs.install(level='INFO', logger=logger)


def set_logger(log_file: str):
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

def clean_json(result:str):
    result = result.replace("```", "").replace("json", "").strip()
    # pattern = r"\{.*?\}"
    # pattern = r'\{\s*"Entities":\s*\[.*?\],\s*"Relations":\s*\[.*?\]\s*\}'
    # matches = re.findall(pattern, result, re.S)
    # return matches[0]
    return result

# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input
    
    result = html.unescape(input.strip().replace("/", " "))
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

def calculate_cosine_similarity(mat1, mat2):
    """
    Calculate cosine similarity between two matrices using numpy.
    
    Args:
        mat1 (np.ndarray): First matrix, shape (n1, feature_dim)
        mat2 (np.ndarray): Second matrix, shape (n2, feature_dim)
    
    Returns:
        np.ndarray: Cosine similarity matrix, shape (n1, n2)
    """
    # Normalize the matrices
    mat1_norms = np.linalg.norm(mat1, axis=1, keepdims=True)  # Shape (n1, 1)
    mat2_norms = np.linalg.norm(mat2, axis=1, keepdims=True)  # Shape (n2, 1)
    
    # Avoid division by zero
    mat1 = mat1 / (mat1_norms + 1e-8)
    mat2 = mat2 / (mat2_norms + 1e-8)
    
    # Compute cosine similarity
    similarity_matrix = np.dot(mat1, mat2.T)  # Shape (n1, n2)
    return similarity_matrix

def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def tri_view_retrieval(
    query: str,
    event_search_system: SearchSystem,
    object_search_system: SearchSystem,
    llm: BaseLanguageModel,
    retrieval_mode: str = "both"  # "events_only", "entities_only", or "both"
):
    # Validate retrieval_mode parameter
    valid_modes = ["events_only", "entities_only", "both"]
    if retrieval_mode not in valid_modes:
        raise ValueError(f"retrieval_mode must be one of {valid_modes}, got: {retrieval_mode}")
    
    top_k_for_events = 50
    top_k_for_entities = 5
    S = 1/2
    
    # Initialize results
    events_result = []
    entities_result = []
    batch_inputs = []
    
    time_extraction_prompt = PROMPTS["time_extraction"].format(input_text=query)
    batch_inputs.append({"text": time_extraction_prompt})

    # Prepare prompts based on retrieval mode
    if retrieval_mode in ["events_only", "both"]:
        keywords_prompt = PROMPTS["keyword_extraction"].format(input_text=query)
        batch_inputs.append({"text": keywords_prompt})
    
    if retrieval_mode in ["entities_only", "both"]:
        rewrite_entity_prompt = PROMPTS["query_rewrite_for_entity_retrieval"].format(input_text=query)
        batch_inputs.append({"text": rewrite_entity_prompt})
    
    # Generate responses
    batch_outputs = llm.batch_generate_response(batch_inputs)
    try:
        duration_filter = json.loads(batch_outputs[0]) if batch_outputs[0] != "None" else None
    except:
        duration_filter = None

    # Process results based on mode
    output_idx = 1
    if retrieval_mode in ["events_only", "both"]:
        keywords_response = batch_outputs[output_idx]
        print("Rewrite event response: ", keywords_response)
        filter_expr = build_filter_expression_for_time(duration_filter)
        events_result = event_search_system.search_by_description(keywords_response, top_k_for_events, filter_expr)
        output_idx += 1
    
    if retrieval_mode in ["entities_only", "both"]:
        rewrite_entity_response = batch_outputs[output_idx]
        print("Rewrite entity response: ", rewrite_entity_response)
        filter_expr = build_filter_expression_for_objects(duration_filter)
        entities_result = object_search_system.search_by_description(rewrite_entity_response, top_k_for_entities, filter_expr)
    
    # Initialize scoring dictionaries
    events_from_events = {}
    events_from_entities = {}
    
    # Process events if needed
    if retrieval_mode in ["events_only", "both"] and events_result:
        for event in events_result:
            event["id"] = int(event["id"].split("_")[0])
            score = event["similarity_score"]
            events_from_events[event["id"]] = max(score, events_from_events.get(event["id"], 0))

    # Process entities and their relationship to events
    if retrieval_mode in ["entities_only", "both"] and entities_result:
        # Calculate entity-based event scores
        for entity in entities_result:
            for event_id in entity.get("event_id", [entity["id"]]):
                event_id = int(event_id)
                if event_id not in events_from_entities:
                    events_from_entities[event_id] = entity["similarity_score"]
                else:
                    events_from_entities[event_id] = max(events_from_entities[event_id], entity["similarity_score"])
    
    # Sort results
    events_from_events = sorted(events_from_events.items(), key=lambda x: x[1], reverse=True)
    events_from_entities = sorted(events_from_entities.items(), key=lambda x: x[1], reverse=True)
    print("Events from events: ", events_from_events)
    print("Events from entities: ", events_from_entities)
    
    # Calculate event scores using normalized Borda Count
    event_scores = {}
    
    # Add event-based scores
    if events_from_events:
        events_from_events_scores_sum = sum([score for _, score in events_from_events])
        for event_id, score in events_from_events:
            event_scores[event_id] = score / events_from_events_scores_sum * S
    
    # Add entity-based scores
    if events_from_entities:
        events_from_entities_scores_sum = sum([score for _, score in events_from_entities])
        for event_id, score in events_from_entities:
            if event_id not in event_scores:
                event_scores[event_id] = score / events_from_entities_scores_sum * S
            else:
                event_scores[event_id] += score / events_from_entities_scores_sum * S
    
    # Handle case where no events are found
    if not event_scores:
        return []
    
    # sort event_scores by score
    event_scores = sorted(event_scores.items(), key=lambda x: x[1], reverse=True)
    # This must be considered to use or not.
    event_scores = event_scores[:top_k_for_events] if len(event_scores) > top_k_for_events else event_scores
    # Build final results
    final_results = []
    for event_id, score in event_scores:
        # Find event description
        event_description = ""
        # Find related entities
        related_entities = []       
        if events_result:
            matching_events = [event for event in events_result if event["id"] == event_id]
            event_duration = None
            if len(matching_events) > 0:
                matching_events = matching_events[0]
                event_duration = [int(matching_events['faiss_metadata']['start_time']), int(matching_events['faiss_metadata']['end_time'])]
                event_description = matching_events['faiss_metadata']['description']
                filter_objects = None
                if matching_events['faiss_metadata']['objects'] != "":
                    filter_objects = {"track_id": matching_events['faiss_metadata']['objects'].split(",")}
                filter_expr = build_filter_expression(filter_objects)
                event_objects = object_search_system.search_by_description(event_description,
                                                                        top_k_for_entities,
                                                                        filter_expr)
                for object in event_objects:
                    filtered = [(f, b) for f, b in zip(object['frame_numbers'], object['bbox_history']) if event_duration[1] >= f >= event_duration[0]]
                    if filtered:
                        frame_numbers, bbox_history = zip(*filtered)
                        object['frame_numbers'] = list(frame_numbers)
                        object['bbox_history'] = list(bbox_history)
                    related_entities.append({
                        "id": int(object['id']),
                        "class_name": object['class_name'],
                        "bbox_history": object['bbox_history'],
                        "frame_numbers": object['frame_numbers']
                    })

        final_results.append({
            "event_id": [event_id],
            "event_description": event_description,
            "event_duration": event_duration,
            "query": [query],
            "score": score,
            "entities": related_entities
        })
    
    return final_results

def filter_answer_generation(results: list, llm: BaseLanguageModel, video_path: str, question_id: int):
    cap = cv2.VideoCapture(video_path)
    batch_inputs = []
    for result in results:
        frames = []
        # TODO: dirty code, should be cleaned up.
        try:
            step = result["entities"][0]['frame_numbers'][1] - result["entities"][0]['frame_numbers'][0]
        except:
            step = 2
        num_frames = 10
        try:
            new_step = (result["event_duration"][1] - result["event_duration"][0]) // num_frames
        except:
            new_step = 1
            result["event_duration"] = [0, 0]
        i = 1
        while step * i <= new_step:
            i += 1
        step = step * (i - 1) if i > 1 else step
        for frame_number in range(result["event_duration"][0], result["event_duration"][1]+1, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                continue
            for entity in result["entities"]:
                if frame_number in entity["frame_numbers"]:
                    bbox_history = entity["bbox_history"][entity["frame_numbers"].index(frame_number)]
                    cv2.rectangle(frame, bbox_history[:2], bbox_history[2:], (0, 255, 0), 1)
                    cv2.putText(frame,"Track ID: " + str(entity["id"]) + ", " + entity["class_name"],
                                (bbox_history[0], bbox_history[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            # cv2.imwrite(f"debug/frame_{frame_number}.jpg", frame)
            # frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        tracks_json = []
        for entity in result["entities"]:
            bbox_history = entity["bbox_history"]
            h, w = frame.shape[:2]
            bbox_history = [(int(x1/w*1000), int(y1/h*1000), int(x2/w*1000), int(y2/h*1000)) for x1, y1, x2, y2 in bbox_history]
            frame_numbers_normalized = [int(frame - result["event_duration"][0]) for frame in entity["frame_numbers"]]
            tracks_json.append({
                "track_id": entity["id"],
                "class": entity["class_name"],
                "frame_numbers": frame_numbers_normalized,
                "boxes": bbox_history
            })
        tracks_json = json.dumps(tracks_json)
        batch_inputs.append({
            "video": frames,
            "text": PROMPTS["summary_and_answer_augmented"].format(query=result["query"], description=result["event_description"], tracks_json=tracks_json)
            # "text": PROMPTS["visual_filter_description"].format(query=result["query"], description=result["event_description"])
        })
    batch_outputs = llm.batch_generate_response(batch_inputs)
    answers = []
    for output, result in zip(batch_outputs, results):
        try:
            answer = parse_json_response(output)
            answer['score'] = result['score']
            answers.append(answer)
        except:
            answers.append({
                "Analysis": "No answer found",
                "Answer": "[]",
                "score": 0.0
            })
    # Save the answers to a file
    result_path = f'database/{os.path.basename(video_path)[:-4]}/{question_id}/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = os.path.join(result_path, "answers.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    return answers

def parse_json_response(response: str):
    clean = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    data = json.loads(clean)
    if isinstance(data.get("track_ids"), str):
        try:
            data["track_ids"] = json.loads(data["track_ids"])
        except json.JSONDecodeError:
            pass
    return data


def chunk_text(text: str):
    """
    Split long text into smaller chunks (approx. max_words each)
    """
    sentences = text.split(".")
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def build_filter_expression(filter: Optional[Dict[str, Any]] = None) -> str:
    """Build a filter expression for Milvus."""
    if filter is None:
        return None
    filter_expr = []
    for key, value in filter.items():
        if isinstance(value, list):
            # Use 'in' operator for lists instead of many OR conditions
            # This avoids Milvus "unsupported expr proto node" error with long OR chains
            if len(value) == 0:
                continue
            elif len(value) == 1:
                filter_expr.append(f"{key} == {value[0]}")
            else:
                # Convert items to appropriate format
                # Try to convert to int if possible (for numeric fields like track_id)
                formatted_values = []
                for item in value:
                    item_str = str(item).strip()
                    try:
                        # Try converting to int for numeric fields
                        formatted_values.append(str(int(item_str)))
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        formatted_values.append(f'"{item_str}"')
                value_str = ", ".join(formatted_values)
                filter_expr.append(f"{key} in [{value_str}]")
        else:
            filter_expr.append(f"{key} == {value}")
    if not filter_expr:
        return None
    return " and ".join(filter_expr) if len(filter_expr) > 1 else filter_expr[0]

def build_filter_expression_for_time(filter: Optional[Dict[str, Any]] = None) -> str:
    """Build a filter expression for Milvus for time."""
    if filter is None:
        return None
    filter_expr = []
    filter_expr.append(f"start_time <= {filter[1]}")
    filter_expr.append(f"end_time >= {filter[0]}")
    return " and ".join(filter_expr)

def build_filter_expression_for_objects(filter: Optional[Dict[str, Any]] = None) -> str:
    """Build a filter expression for Milvus for time."""
    if filter is None:
        return None
    filter_expr = []
    filter_expr.append(f"frame_number >= {filter[0]}")
    filter_expr.append(f"frame_number <= {filter[1]}")
    return " and ".join(filter_expr)