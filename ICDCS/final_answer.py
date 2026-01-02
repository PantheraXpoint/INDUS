"""
Script to generate final answers from queries using knowledge graph information.

This script loads a knowledge graph JSON file, formats the information,
and uses prompt templates to generate answers to user queries.
"""

import json
import os
import sys
from typing import Optional, Dict, Any, List
from PIL import Image
import cv2
import glob
import time
import re
import ast

# Add parent directory to path for imports (must be before importing time_ref)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from time_ref import overlap_reference_helper

try:
    from AVA.prompt import PROMPTS
except ImportError:
    # Fallback for direct execution
    import importlib.util
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.py')
    spec = importlib.util.spec_from_file_location("prompt", prompt_path)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    PROMPTS = prompt_module.PROMPTS

def extract_questions_from_response(response):
    questions = []

    # Find all vqa("...") entries (single string argument)
    single_q_matches = re.findall(r'vqa\("([^"]+)"\)', response)
    questions.extend(single_q_matches)

    # Find vqa([...]) blocks (list of questions)
    list_q_matches = re.findall(r'vqa\(\s*(\[[^\]]*\])\s*\)', response)
    for list_q_str in list_q_matches:
        try:
            # Safely evaluate the string to a list
            list_q = ast.literal_eval(list_q_str)
            if isinstance(list_q, list):
                questions.extend(list_q)
        except Exception:
            pass  # ignore malformed lists

    return questions

def extract_frames_for_event_pair(
    video_path: str,
    event1_start: int,
    event1_end: int,
    event2_start: int,
    event2_end: int,
    num_frames: int = 10
) -> List[Image.Image]:
    """
    Extract frames from video for a pair of events.
    Similar to the frame extraction logic in utils.py:301-368.
    
    Args:
        video_path: Path to the video file
        event1_start: Start frame number for event 1
        event1_end: End frame number for event 1
        event2_start: Start frame number for event 2
        event2_end: End frame number for event 2
        num_frames: Number of frames to extract per event
        
    Returns:
        List of PIL Image objects
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        return frames
    
    # Extract frames for event 1
    event1_duration = event1_end - event1_start
    step1 = max(1, event1_duration // num_frames) if event1_duration > 0 else 1
    
    for frame_number in range(event1_start, event1_end + 1, step1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (540, 360))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    # Extract frames for event 2
    event2_duration = event2_end - event2_start
    step2 = max(1, event2_duration // num_frames) if event2_duration > 0 else 1
    
    for frame_number in range(event2_start, event2_end + 1, step2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (540, 360))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames


def find_connected_event_pairs(
    graph_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Find event pairs that are directly connected via event→event connections only.    
    Args:
        graph_data: Dictionary containing graph data with nodes and edges
        query: User query to help filter relevant connections
        
    Returns:
        List of dictionaries containing event pair information
    """
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    # Create mappings
    events = {node['id']: node for node in nodes if node.get('type') == 'event'}
    
    # Filter out event→event connections
    event_to_event_edges = [
        edge for edge in edges
        if edge.get('connection', '') == 'event→event'
    ]
    # Find event pairs directly connected via event→event edges
    event_pairs = []
    seen_pairs = set()
    
    for edge in event_to_event_edges:
        source_id = edge.get('source_id', '')
        target_id = edge.get('target_id', '')
        
        # Skip if not both are events
        if source_id not in events or target_id not in events:
            continue
        
        # Skip self-connections
        if source_id == target_id:
            continue
        
        # Create a unique pair identifier
        pair_key = tuple(sorted([source_id, target_id]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        
        # Get event information
        event1 = events.get(source_id)
        event2 = events.get(target_id)
        
        if event1 and event2:
            event1_metadata = event1.get('metadata', {})
            event2_metadata = event2.get('metadata', {})
            
            event1_start = event1_metadata.get('start_time', 0)
            event1_end = event1_metadata.get('end_time', 0)
            event2_start = event2_metadata.get('start_time', 0)
            event2_end = event2_metadata.get('end_time', 0)
            
            # Get descriptions
            event1_desc = (
                event1_metadata.get('full_description', '') or
                event1_metadata.get('description', '') or
                event1.get('content', '')
            )
            event2_desc = (
                event2_metadata.get('full_description', '') or
                event2_metadata.get('description', '') or
                event2.get('content', '')
            )
            
            # Get edge information
            edge_score = edge.get('score', 0.0)
            edge_type = edge.get('type', '')
            
            event_pairs.append({
                'event1_id': source_id,
                'event1_description': event1_desc,
                'event1_start': event1_start,
                'event1_end': event1_end,
                'event2_id': target_id,
                'event2_description': event2_desc,
                'event2_start': event2_start,
                'event2_end': event2_end,
                'edge_score': edge_score,
                'edge_type': edge_type,
                'connection_type': 'event_to_event'
            })

    event_pairs = sorted(event_pairs, key=lambda x: x['edge_score'], reverse=True)
    
    return event_pairs


def find_graph_json_path(graph_folder_path: str, graph_filename: Optional[str] = None) -> str:
    """
    Find the graph JSON file path with fallback logic.
    
    Args:
        graph_folder_path: Path to the folder containing graph JSON files
        graph_filename: Optional custom filename (e.g., "best_subgraph_stage2_budget20.json")
                       If None or not found, falls back to "best_subgraph.json"
        
    Returns:
        Path to the graph JSON file to use
    """
    # If custom filename is provided, try to use it
    if graph_filename:
        custom_path = os.path.join(graph_folder_path, graph_filename)
        if os.path.exists(custom_path):
            return custom_path
    
    # Fall back to default
    default_path = os.path.join(graph_folder_path, "best_subgraph.json")
    return default_path

def load_graph_data_with_metadata(graph_folder_path: str, graph_filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Load graph data and ensure query_metadata exists.
    If the custom graph file doesn't have query_metadata, load it from best_subgraph.json.
    
    Args:
        graph_folder_path: Path to the folder containing graph JSON files
        graph_filename: Optional custom filename
        
    Returns:
        Dictionary containing the graph data with query_metadata
    """
    graph_json_path = find_graph_json_path(graph_folder_path, graph_filename)
    graph_data = load_graph_data(graph_json_path)
    
    # If query_metadata is missing, try to load it from best_subgraph.json
    if 'query_metadata' not in graph_data or not graph_data.get('query_metadata'):
        default_path = os.path.join(graph_folder_path, "best_subgraph.json")
        if os.path.exists(default_path) and default_path != graph_json_path:
            try:
                default_data = load_graph_data(default_path)
                if 'query_metadata' in default_data:
                    graph_data['query_metadata'] = default_data['query_metadata']
                    print(f"Loaded query_metadata from {default_path}")
            except Exception as e:
                print(f"Warning: Could not load query_metadata from {default_path}: {e}")
    
    return graph_data

def load_graph_data(json_path: str) -> Dict[str, Any]:
    """
    Load the knowledge graph data from JSON file.
    
    Args:
        json_path: Path to the JSON file containing graph data
        
    Returns:
        Dictionary containing the graph data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def check_overlap(nodes: List[Dict[str, Any]], time_reference: str):
    """
    Check if the overlap of the nodes and edges is greater than 0.5.
    """
    time_list = []
    for node in nodes:
        if node["type"] == "event":
            start_time = node["metadata"]["duration"][0]
            end_time = node["metadata"]["duration"][1]
            if end_time > start_time:
                time_list.append((start_time, end_time))
        if node["type"] == "object":
            for duration in node["metadata"]["durations"]:
                start_time = duration[0]
                end_time = duration[1]
                if end_time > start_time:
                    time_list.append((start_time, end_time))
    overlap = overlap_reference_helper(time_reference, time_list)
    return overlap


def format_nodes_as_segments(graph_data: Dict[str, Any], limited_ratio: float, time_reference: str, event_nodes: List[str] = None) -> str:
    """
    Format graph nodes (events and objects) into a readable segment format.
    Groups objects under their associated events.
    
    Args:
        graph_data: Dictionary containing graph data with nodes
        
    Returns:
        Formatted string representing video segments in the format:
        Event A:
        description: ...
        Objects in Event A:
        Object A1:
        description:....
        Object A2:
        description:....
    """
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    if limited_ratio < 1.0:
        nodes = nodes[:int(len(nodes) * limited_ratio)]
        edges = edges[:int(len(edges) * limited_ratio)]
    overlap = check_overlap(nodes, time_reference)
    print(f"Overlap: {overlap}")
    print(f"Limited nodes: {len(nodes)}")
    print(f"Limited edges: {len(edges)}")
    # Create mappings for quick lookup
    node_map = {node['id']: node for node in nodes}
    events = [node for node in nodes if node.get('type') == 'event' and node['id'] in event_nodes]
    # events = [node for node in nodes if node.get('type') == 'event']
    objects = {node['id']: node for node in nodes if node.get('type') == 'object'}
    
    # Build mapping: event_id -> list of object_ids connected to it
    event_to_objects: Dict[str, List[str]] = {}
    graph_statistics = {
        'event_count': len(events),
        'object_count': len(objects),
    }
    for edge in edges:
        edge_type = edge.get('type', '')
        source_id = edge.get('source_id', '')
        target_id = edge.get('target_id', '')
        
        # Handle event_to_object edges
        if (edge_type == 'event_object' or edge_type == 'event_to_object') and source_id in node_map and target_id in objects:
            if source_id not in event_to_objects:
                event_to_objects[source_id] = []
            if target_id not in event_to_objects[source_id]:
                event_to_objects[source_id].append(target_id)
        
        # Handle object_to_event edges (reverse direction)
        elif (edge_type == 'object_event' or edge_type == 'object_to_event') and target_id in node_map and source_id in objects:
            if target_id not in event_to_objects:
                event_to_objects[target_id] = []
            if source_id not in event_to_objects[target_id]:
                event_to_objects[target_id].append(source_id)
    
    # Format output: Event with its objects grouped
    segments = []
    seen_event_ids = set()
    
    for event in events:
        event_id = event['id']
        
        # Skip duplicate events
        if event_id in seen_event_ids:
            continue
        seen_event_ids.add(event_id)
        
        # Get event description
        metadata = event.get('metadata', {})
        description = metadata.get('full_description', '') or metadata.get('description', '') or event.get('content', '')
        
        # Format event header
        event_label = f"Event {event_id}:"
        segments.append(event_label)
        segments.append("")
        segments.append(f"description: {description}")
        segments.append("")
        
        # Get objects associated with this event
        associated_objects = event_to_objects.get(event_id, [])
        # if len(associated_objects) > 10:
        #     associated_objects = associated_objects[:10]
        
        if associated_objects:
            segments.append(f"Objects in Event {event_id}:")
            segments.append("")
            
            for obj_id in associated_objects:
                if obj_id not in objects:
                    continue
                
                obj = objects[obj_id]
                obj_metadata = obj.get('metadata', {})
                obj_description = obj_metadata.get('description', '') or obj.get('content', '')
                obj_class = obj_metadata.get('class_name', '')
                
                # Format object
                obj_label = f"Object {obj_id}"
                if obj_class:
                    obj_label += f" ({obj_class})"
                obj_label += ":"
                segments.append(obj_label)
                segments.append("")
                segments.append(f"description: {obj_description}")
                segments.append("")
        else:
            segments.append(f"Objects in Event {event_id}:")
            segments.append("(No objects associated)")
            segments.append("")
    
    return "\n".join(segments), overlap, graph_statistics


def format_graph_summary(graph_data: Dict[str, Any]) -> str:
    """
    Create a summary of the graph structure.
    
    Args:
        graph_data: Dictionary containing graph data
        
    Returns:
        Formatted summary string
    """
    stats = graph_data.get('statistics', {})
    summary_parts = [
        f"Knowledge Graph Summary:",
        f"- Total Nodes: {stats.get('total_nodes', 0)}",
        f"- Total Edges: {stats.get('total_edges', 0)}",
        f"- Events: {stats.get('event_count', 0)}",
        f"- Objects: {stats.get('object_count', 0)}"
    ]
    
    edge_summary = graph_data.get('edge_type_summary', {})
    if edge_summary:
        summary_parts.append("\nEdge Types:")
        for edge_type, count in edge_summary.items():
            summary_parts.append(f"  - {edge_type}: {count}")
    
    return "\n".join(summary_parts)

def generate_question_list(query: str, prompt_template: str, llm_model: Optional[Any] = None) -> List[str]:
    """
    Generate a list of questions from a query using a prompt template.
    """
    prompt_template_str = PROMPTS[prompt_template]
    formatted_prompt = prompt_template_str.format(
        question_and_options=query
    )
    response = llm_model.batch_generate_response([{"text": formatted_prompt}], max_new_tokens=512, temperature=0.5)[0]

    question_list = extract_questions_from_response(
        response
    )
    return question_list

def generate_reasoning_answer(
    graph_folder_path: str,
    prompt_template: str = "Reasoning",
    llm_model: Optional[Any] = None,
    event_to_event_descriptions: List[str] = None,
    event_nodes: List[str] = None,
    graph_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a reasoning answer to a query using knowledge graph information.
    """
        # Load graph data
    graph_json_path = find_graph_json_path(graph_folder_path, graph_filename)
    if not os.path.exists(graph_json_path):
        return {
            'error': f"Graph JSON file not found: {graph_json_path}"
        }
    graph_data = load_graph_data_with_metadata(graph_folder_path, graph_filename)

    # Find the summary json file in the graph folder
    query = graph_data.get('query_metadata', {}).get('query', "")

    if query is None or query == "":
        raise ValueError("Query is not found in the summary file")
    
    question_list = generate_question_list(query, prompt_template, llm_model)

    if len(question_list) <= 1:
        return None
    batch_inputs = []
    for question in question_list:
        batch_inputs.append({"text": question})
    batch_outputs = llm_model.batch_generate_response(batch_inputs, max_new_tokens=512, temperature=0.5)
    question_list = [output for output in batch_outputs]
    results = []
    for question in question_list:
        # Format the graph information as video segments
        time_reference = graph_data["query_metadata"]["time_reference"]
        video_segments, overlap, graph_statistics = format_nodes_as_segments(graph_data, limited_ratio=1.0, time_reference=time_reference, event_nodes=event_nodes)
        graph_statistics['edges'] = len(event_to_event_descriptions)
        # Get the prompt template
        if prompt_template not in PROMPTS:
            raise ValueError(f"Prompt template '{prompt_template}' not found. Available templates: {list(PROMPTS.keys())}")
        
        prompt_template_str = PROMPTS["generate_reasoning_answer"]
        
        # Format the prompt with the query and video segments
        formatted_prompt = prompt_template_str.format(
            user_query=question,
            video_segments=video_segments,
            event_to_event_descriptions="\n".join(event_to_event_descriptions)
        )
        
        results.append({
            'query': question,
            'prompt_template': prompt_template,
            'formatted_prompt': formatted_prompt,
            'graph_statistics': graph_statistics,
            'overlap': overlap,
        })
        
    # Generate answer using LLM if provided
    if llm_model is not None:
        try:
            # Prepare input for LLM
            llm_input = []
            for result in results:
                llm_input.append({"text": result['formatted_prompt']})
            
            # Generate response
            if hasattr(llm_model, 'batch_generate_response'):
                responses = llm_model.batch_generate_response(llm_input, max_new_tokens=512, temperature=0.5)
            else:
                raise ValueError("LLM model must have 'generate_response' or 'batch_generate_response' method")
            
            for result, response in zip(results, responses):
                result['llm_response'] = response
            
            # Try to parse JSON response if it's in JSON format
            try:
                import re
                # Extract JSON from response if it's wrapped in markdown code blocks
                breakpoint()
                for result in results:
                    analysis_match = re.search(r'"Analysis"\s*:\s*"([^"]*)"', result['llm_response'])
                    result['analysis'] = analysis_match.group(1) if analysis_match else result['llm_response']
            except Exception as e:
                for result in results:
                    result['analysis'] = None
        except Exception as e:
            for result in results:
                result['error'] = str(e)
                result['analysis'] = None
    return results
    

def generate_e2e_answer(
    graph_folder_path: str,
    prompt_template: str = "summary_and_answer",
    llm_model: Optional[Any] = None,
    video_path: Optional[str] = None,
    graph_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate an end-to-end answer to a query using knowledge graph information.
    
    Args:
        query: User's query/question
        graph_json_path: Path to the knowledge graph JSON file
        prompt_template: Name of the prompt template to use (default: "summary_and_answer")
        llm_model: Optional LLM model instance to generate the answer
        use_summary: Whether to include graph summary in the prompt
        graph_filename: Optional custom graph JSON filename (falls back to "best_subgraph.json" if not found)
        
    Returns:
        Dictionary containing the answer, analysis, and formatted prompt
    """
    # Load graph data
    graph_json_path = find_graph_json_path(graph_folder_path, graph_filename)
    if not os.path.exists(graph_json_path):
        return {
            'error': f"Graph JSON file not found: {graph_json_path}"
        }
    graph_data = load_graph_data_with_metadata(graph_folder_path, graph_filename)

    # Find the summary json file in the graph folder
    query = graph_data.get('query_metadata', {}).get('query', "")
    query = query + " " + ", ".join(graph_data.get('query_metadata', {}).get('options', []))

    if query is None or query == "":
        raise ValueError("Query is not found in the summary file")
    # Find connected event pairs (keeping only event→event connections)
    event_pairs = find_connected_event_pairs(graph_data)
    
    # Format event connections description
    connection_descriptions = []
    event_nodes = []
    for pair in event_pairs[:50]:  # Limit to first 10 pairs to avoid overwhelming
        event1_id = pair['event1_id']
        event2_id = pair['event2_id']
        if event1_id not in event_nodes:
            event_nodes.append(event1_id)
        if event2_id not in event_nodes:
            event_nodes.append(event2_id)
        if len(event_nodes) > 20:
            break
        event1_desc = pair['event1_description']
        event2_desc = pair['event2_description']
        edge_score = pair.get('edge_score', 0.0)
        
        # Build connection description
        conn_desc = f"Event {event1_id} → Event {event2_id} (connection score: {edge_score:.4f}):\n"
        conn_desc += f"Event {event1_id}:\n"
        conn_desc += f"  description: {event1_desc}\n"
        conn_desc += f"  frames: {pair['event1_start']} to {pair['event1_end']}\n\n"
        
        conn_desc += f"Event {event2_id}:\n"
        conn_desc += f"  description: {event2_desc}\n"
        conn_desc += f"  frames: {pair['event2_start']} to {pair['event2_end']}\n\n"
        
        connection_descriptions.append(conn_desc)
    
    # Extract frames for event pairs if video path is provided and LLM supports video
    frames_list = []
    if video_path and os.path.exists(video_path) and llm_model is not None:
        for pair in event_pairs[:50]:  # Limit to 5 pairs for frame extraction
            frames = extract_frames_for_event_pair(
                video_path=video_path,
                event1_start=int(pair['event1_start']),
                event1_end=int(pair['event1_end']),
                event2_start=int(pair['event2_start']),
                event2_end=int(pair['event2_end']),
                num_frames=1
            )
            if frames:
                frames_list.append(frames)
    
    # Format the graph information as video segments (original format)
    video_segments = []
    
    # Add event connection descriptions
    if connection_descriptions:
        for connection_description in connection_descriptions:
            video_segments.append("Event Connections (event→event): " + connection_description)
    
    # Get the prompt template
    if prompt_template not in PROMPTS:
        raise ValueError(f"Prompt template '{prompt_template}' not found. Available templates: {list(PROMPTS.keys())}")
    
    prompt_template_str = PROMPTS[prompt_template]    
    
    # Format the prompt with the query and video segments
    results = []
    for video_segment in video_segments:
        formatted_prompt = prompt_template_str.format(
            user_query=query,
            event_pair_info=video_segment
        )    
        results.append({"formatted_prompt": formatted_prompt})
    # Generate answer using LLM if provided
    if llm_model is not None:
        # Prepare input for LLM
        batch_inputs = []
        for idx, result in enumerate(results):
            if idx < len(frames_list):
                batch_inputs.append({"text": result["formatted_prompt"], "video": frames_list[idx]})
            else:
                batch_inputs.append({"text": result["formatted_prompt"]})
        start_time = time.time()
        try:
            batch_outputs = llm_model.batch_generate_response(batch_inputs, max_new_tokens=512, temperature=0.5)
        except Exception as e:
            batch_outputs = [""] * len(batch_inputs)
        end_time = time.time()
        print(f"Time taken for VLM generation: {end_time - start_time} seconds")
        for result, output in zip(results, batch_outputs):
            result['llm_response'] = output
    return results, event_nodes


def generate_final_answer(
    graph_folder_path: str,
    prompt_template: str = "summary_and_answer",
    llm_model: Optional[Any] = None,
    event_to_event_descriptions: List[str] = None,
    event_nodes: List[str] = None,
    graph_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate an answer to a query using knowledge graph information.
    
    Args:
        query: User's query/question
        graph_json_path: Path to the knowledge graph JSON file
        prompt_template: Name of the prompt template to use (default: "summary_and_answer")
        llm_model: Optional LLM model instance to generate the answer
        use_summary: Whether to include graph summary in the prompt
        graph_filename: Optional custom graph JSON filename (falls back to "best_subgraph.json" if not found)
        
    Returns:
        Dictionary containing the answer, analysis, and formatted prompt
    """
    # Load graph data
    graph_json_path = find_graph_json_path(graph_folder_path, graph_filename)
    graph_data = load_graph_data_with_metadata(graph_folder_path, graph_filename)
    query = graph_data.get('query_metadata', {}).get('query', "")
    query = query + " " + ", ".join(graph_data.get('query_metadata', {}).get('options', []))

    if query is None or query == "":
        raise ValueError("Query is not found in the summary file")
    
    # Format the graph information as video segments
    time_reference = graph_data["query_metadata"]["time_reference"]
    video_segments, overlap, graph_statistics = format_nodes_as_segments(graph_data, limited_ratio=1.0, time_reference=time_reference, event_nodes=event_nodes)
    graph_statistics['edges'] = len(event_to_event_descriptions)
    # Get the prompt template
    if prompt_template not in PROMPTS:
        raise ValueError(f"Prompt template '{prompt_template}' not found. Available templates: {list(PROMPTS.keys())}")
    
    prompt_template_str = PROMPTS[prompt_template]
    
    # Format the prompt with the query and video segments
    formatted_prompt = prompt_template_str.format(
        user_query=query,
        video_segments=video_segments,
        event_to_event_descriptions="\n".join(event_to_event_descriptions)
    )
    
    result = {
        'query': query,
        'prompt_template': prompt_template,
        'formatted_prompt': formatted_prompt,
        'graph_statistics': graph_statistics,
        'overlap': overlap,
    }
    
    # Generate answer using LLM if provided
    if llm_model is not None:
        try:
            # Prepare input for LLM
            llm_input = {"text": formatted_prompt}
            
            
            # Generate response
            if hasattr(llm_model, 'batch_generate_response'):
                response = llm_model.batch_generate_response([llm_input], max_new_tokens=512, temperature=0.5)[0]
            else:
                raise ValueError("LLM model must have 'generate_response' or 'batch_generate_response' method")
            
            result['llm_response'] = response
            
            # Try to parse JSON response if it's in JSON format
            try:
                import re
                # Extract JSON from response if it's wrapped in markdown code blocks
                answer_match = re.search(r'"Answer"\s*:\s*"([^"]*)"', response)
                analysis_match = re.search(r'"Analysis"\s*:\s*"([^"]*)"', response)
                result['answer'] = answer_match.group(1) if answer_match else response
                result['analysis'] = analysis_match.group(1) if analysis_match else response
            except Exception as e:
                result['answer'] = None
                result['analysis'] = None
        except Exception as e:
            result['error'] = str(e)
            result['answer'] = None
    else:
        result['answer'] = None
        result['note'] = 'No LLM model provided. Only formatted prompt is returned.'
    
    return result


def run_ava_100_benchmark(graph_folder: str, video_path: str, vlm_port: int = 8002, llm_port: int = 8000, llm_only: bool = False, postfix: str = '', graph_filename: Optional[str] = None):
    # Initialize LLM model if specified
    vlm_model = None
    llm_model = None    
    try:
        from llms.init_model import init_model
        vlm_model = init_model("qwenvl_vllm", num_gpus=1, model_type="Qwen/Qwen2.5-VL-7B-Instruct-AWQ", port=vlm_port)
        llm_model = init_model("qwenvl_vllm", num_gpus=1, model_type="Qwen/Qwen2.5-14B-Instruct-AWQ", port=llm_port)
        print(f"Initialized VLM model: qwenvl and LLM model: qwenvl_vllm")
    except Exception as e:
        print(f"Warning: Could not initialize LLM model: {e}")
        print("Continuing without LLM model...")
    questions_folder = glob.glob(os.path.join(graph_folder, "q*"))
    replace_str = "_llm_only" if llm_only else "_full"
    replace_str = replace_str + "_" + postfix
    for question_folder in sorted(questions_folder):
        # Save to file if specified
        output = f"{question_folder}/final_answer{replace_str}.json"
        # Check if file exists and is complete (has more than just time_taken)
        should_skip = False
        if os.path.exists(output):
            try:
                with open(output, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    # If file has more than just time_taken, skip it
                    if len(existing_data) > 1 or 'query' in existing_data or 'formatted_prompt' in existing_data:
                        should_skip = True
            except:
                # If file is corrupted, regenerate it
                pass
        if should_skip:
            print(f"Skipping {output} (already exists and is complete)")
            continue
        start_time = time.time()
        # Generate answer
        if not llm_only:
            e2e_results, event_nodes = generate_e2e_answer(
                graph_folder_path=question_folder,
                prompt_template="generate_event_to_event_description",
                llm_model=vlm_model,
                video_path=video_path,
                graph_filename=graph_filename
            )

            result = generate_final_answer(
                graph_folder_path=question_folder,
                prompt_template="generate_final_answer_with_e2e",
                llm_model=llm_model,
                event_to_event_descriptions=[e2e_result['llm_response'] for e2e_result in e2e_results],
                event_nodes=event_nodes,
                graph_filename=graph_filename
            )
        else:
            result = generate_final_answer(
                graph_folder_path=question_folder,
                prompt_template="generate_final_answer_with_e2e",
                llm_model=llm_model,
                event_to_event_descriptions=[],
                graph_filename=graph_filename
            )
        end_time = time.time()
        
        # Ensure result is a dictionary
        if result is None:
            result = {'error': 'generate_final_answer returned None'}
        elif not isinstance(result, dict):
            result = {'error': f'generate_final_answer returned unexpected type: {type(result)}'}
        
        result['time_taken'] = end_time - start_time
        print(f"Time taken: {result['time_taken']} seconds")
        
        # Debug: Print what keys are in result
        print(f"Result keys: {list(result.keys())}")
        
        # Print results
        print("\n" + "="*80)
        print("QUERY:")
        print("="*80)
        print(result.get('query',''))
        
        print("\n" + "="*80)
        print("GRAPH STATISTICS:")
        print("="*80)
        for key, value in result.get('graph_statistics',{}).items():
            print(f"  {key}: {value}")
        
        if result.get('answer'):
            print("\n" + "="*80)
            print("ANSWER:")
            print("="*80)
            print(result.get('answer',''))
            
            if result.get('analysis'):
                print("\n" + "="*80)
                print("ANALYSIS:")
                print("="*80)
                print(result.get('analysis',''))
        
        if result.get('error'):
            print("\n" + "="*80)
            print("ERROR:")
            print("="*80)
            print(result.get('error',''))
        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"\nResults saved to: {output}")
        
        # Optionally print formatted prompt (can be very long)
        prompt_file = output.replace('.json', '_prompt.txt')
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(result.get('formatted_prompt','') + "\n")
        print(f"Formatted prompt saved to: {prompt_file}")


def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate answers from queries using knowledge graph')
    parser.add_argument('--graph_folder', type=str, 
                       default='ava100_results/ego1/q0',
                       help='Path to knowledge graph folder')
    parser.add_argument('--prompt_template', type=str, default='summary_and_answer',
                       help='Name of prompt template to use')
    parser.add_argument('--output', type=str, help='Output file path (optional)')
    parser.add_argument('--video_path', type=str, help='Path to video file (optional, for frame extraction)')
    parser.add_argument('--no_summary', action='store_true', 
                       help='Exclude graph summary from prompt')
    parser.add_argument('--model', type=str, help='LLM model name (optional)')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs for model')
    
    args = parser.parse_args()
    
    # Initialize LLM model if specified
    llm_model = None
    if args.model:
        try:
            from llms.init_model import init_model
            llm_model = init_model(args.model, args.gpus)
            print(f"Initialized LLM model: {args.model}")
        except Exception as e:
            print(f"Warning: Could not initialize LLM model: {e}")
            print("Continuing without LLM model...")
    
    # Generate answer
    e2e_results = generate_e2e_answer(
        graph_folder_path=args.graph_folder,
        prompt_template="generate_event_to_event_description",
        llm_model=llm_model,
        video_path=args.video_path
    )

    result = generate_final_answer(
        graph_folder_path=args.graph_folder,
        prompt_template="generate_final_answer_with_e2e",
        llm_model=llm_model,
        event_to_event_descriptions=[e2e_result['llm_response'] for e2e_result in e2e_results]
    )
    
    # Print results
    print("\n" + "="*80)
    print("QUERY:")
    print("="*80)
    print(result['query'])
    
    print("\n" + "="*80)
    print("GRAPH STATISTICS:")
    print("="*80)
    for key, value in result['graph_statistics'].items():
        print(f"  {key}: {value}")
    
    if result.get('answer'):
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result['answer'])
        
        if result.get('analysis'):
            print("\n" + "="*80)
            print("ANALYSIS:")
            print("="*80)
            print(result['analysis'])
    
    if result.get('error'):
        print("\n" + "="*80)
        print("ERROR:")
        print("="*80)
        print(result['error'])
    
    # Save to file if specified
    if args.output:
        output_data = {
            'query': result['query'],
            'answer': result.get('answer'),
            'analysis': result.get('analysis'),
            'graph_statistics': result['graph_statistics'],
            'prompt_template': result['prompt_template']
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")
    
    # Optionally print formatted prompt (can be very long)
    if args.output and 'formatted_prompt' in result:
        prompt_file = args.output.replace('.json', '_prompt.txt')
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(result['formatted_prompt'])
        print(f"Formatted prompt saved to: {prompt_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate answers from queries using knowledge graph')
    parser.add_argument('--video_range', type=str, help='Video range')
    parser.add_argument('--vlm_port', type=int, default=8002, help='VLM port')
    parser.add_argument('--llm_port', type=int, default=8000, help='LLM port')
    parser.add_argument('--llm_only', action='store_true', help='Only run LLM')
    parser.add_argument('--postfix', type=str, default='', help='Postfix for output file')
    parser.add_argument('--graph_filename', type=str, default=None, help='Custom graph JSON filename (e.g., "best_subgraph_stage2_budget20.json"). Falls back to "best_subgraph.json" if not found.')
    # main()
    args = parser.parse_args()
    video_range = args.video_range.split("-")
    video_range = [int(video_range[0]), int(video_range[1])]
    dataset_names = ["citytour1", "ego1", "traffic1", "wildlife1", "citytour2", "ego2", "traffic2", "wildlife2"]
    for idx, dataset_name in enumerate(dataset_names):
        if idx < video_range[0] or idx > video_range[1]:
            continue
        run_ava_100_benchmark(graph_folder=f"50_nodes_limit_seeds_10/ava100_results/{dataset_name}", video_path=f"datas/AVA100/videos/{dataset_name}.mp4", vlm_port=args.vlm_port, llm_port=args.llm_port, llm_only=args.llm_only, postfix=args.postfix, graph_filename=args.graph_filename)

