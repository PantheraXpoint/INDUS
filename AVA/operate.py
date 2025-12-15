import time
import os
import json
import re
import numpy as np
from json_repair import repair_json
from typing import Union
from embeddings.BaseEmbeddingModel import BaseEmbeddingModel
from llms.BaseModel import BaseVideoModel, BaseLanguageModel
from .base import (
    BaseGraphStorage,
    BaseVectorStorage,
)
from .prompt import PROMPTS
from .events import extract_events
from .entities import extract_entities_and_relations
from .tree_search import TreeSearch
from video_utils import VideoRepresentation
from bert_score import BERTScorer
import time


def extract_knowledge_graph(
        llm: BaseVideoModel,
        embedding_model: BaseEmbeddingModel,
        knowledge_graph_inst: BaseGraphStorage,
        events_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage,
        relations_vdb: BaseVectorStorage,
        features_vdb: BaseVectorStorage,
        global_config: dict,
        if_check: bool = True
)->Union[BaseGraphStorage]:
    profiling: dict = {}
    if events_vdb.is_empty() or if_check:
        profiling["extract_events"] = {}
        profiling["extract_events"]["total"] = time.time()
        events, profiling_extract_events = extract_events(
            llm=llm,
            video=global_config["video"],
            global_config=global_config,
        )
        profiling["extract_events"]["total"] = time.time() - profiling["extract_events"]["total"]
        profiling["extract_events"].update(profiling_extract_events)
        print(profiling["extract_events"])
    else:
        events = events_vdb.get_datas()
    
    if entities_vdb.is_empty() or relations_vdb.is_empty() or if_check:
        profiling["extract_entities_and_relations"] = {}
        profiling["extract_entities_and_relations"]["total"] = time.time()
        entities, relations, profiling_extract_entities_and_relations = extract_entities_and_relations(
            llm=llm,
            embedding_model=embedding_model,
            events=events,
            video=global_config["video"],
            global_config=global_config,
        )
        profiling["extract_entities_and_relations"]["total"] = time.time() - profiling["extract_entities_and_relations"]["total"]
        profiling["extract_entities_and_relations"].update(profiling_extract_entities_and_relations)
        print(profiling["extract_entities_and_relations"])
    else:
        entities = entities_vdb.get_datas()
        relations = relations_vdb.get_datas()
    
    # add to event_vdb
    datas_for_vdb = {
        event["id"]:{
            "id": event["id"],
            "description": event["description"],
            "duration": event["duration"],
            "content": [event["description"]]
        } for event in events
    }
    
    events_vdb.upsert(datas_for_vdb)
    
    # add to entities_vdb
    datas_for_vdb = {
        entity["id"]:{
            "id": entity["id"],
            "descriptions": entity["descriptions"],
            "timestamps": entity["timestamps"],
            "frame_indices": entity["frame_indices"],
            "durations": entity["durations"],
            "events": entity["events"],
            "content": entity["descriptions"]
        } for entity in entities
    }
    
    entities_vdb.upsert(datas_for_vdb)
    
    # add to relations_vdb
    datas_for_vdb = {
        relation["id"]:{
            "id": relation["id"],
            "entity1": relation["entity1"],
            "entity2": relation["entity2"],
            "description": relation["description"],
            "content": [relation["description"]]
        } for relation in relations
    }
    
    relations_vdb.upsert(datas_for_vdb)
    
    # add to features vdb
    if features_vdb.is_empty():
        profiling["extract_features"] = time.time()
        video = global_config["video"]
        datas_for_vdb = {}
        frame_path = video.frames_dir
        for event in events:
            duration = event["duration"]
            for i in range(int(duration[0]), int(duration[1])):
                _, _, index = video.get_frames_by_fps(fps=1, duration=(i,i))
                if not index:
                    continue
                frame_dir = os.path.join(frame_path, f"{index[0]}.jpg")
                datas_for_vdb[f"{i}"] = {
                    "content": [frame_dir],
                    "event": event["id"],
                    "frame_dir": frame_dir
                }
        
        features_vdb.upsert(datas_for_vdb)
        profiling["extract_features"] = time.time() - profiling["extract_features"]
    
    for event in events:
        if not knowledge_graph_inst.has_node(event["id"]):
            knowledge_graph_inst.upsert_node(
                node_id=event["id"],
                node_data={
                    "type": "event",
                    "id": event["id"],
                    "description": event["description"],
                }
            )
    
    for entity in entities:
        if not knowledge_graph_inst.has_node(entity["id"]):
            knowledge_graph_inst.upsert_node(
                node_id=entity["id"],
                node_data={
                    "type": "entity",
                    "id": entity["id"],
                    "description": f" ".join(entity["descriptions"]),
                }
            )
        for event_id in entity["events"]:
            if not knowledge_graph_inst.has_edge(entity["id"], event_id):
                knowledge_graph_inst.upsert_edge(
                    entity["id"],
                    event_id,
                    edge_data={
                        "type": "belong_to",
                        "id1": entity["id"],
                        "id2": event_id,
                    }
                )
    
    for relation in relations:
        if not knowledge_graph_inst.has_edge(relation["entity1"], relation["entity2"]):
            knowledge_graph_inst.upsert_edge(
                relation["entity1"],
                relation["entity2"],
                edge_data={
                    "type": "relation",
                    "id1": relation["entity1"],
                    "id2": relation["entity2"],
                    "description": relation["description"]
                }
            )
    
    # add event time relation
    total_events = len(events)
    for i in range(total_events-1):
        if not knowledge_graph_inst.has_edge(events[i]["id"], events[i+1]["id"]):
            knowledge_graph_inst.upsert_edge(
                events[i]["id"],
                events[i+1]["id"],
                edge_data={
                    "type": "event_time_relation",
                    "id1": events[i]["id"],
                    "id2": events[i+1]["id"],
                }
            )
    
    return knowledge_graph_inst, profiling

def tree_search(
    query: str,
    llm: BaseLanguageModel,
    video: VideoRepresentation,
    events_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,    
    features_vdb: BaseVectorStorage,
    retrieval_mode: str = "tri_view",
):
    from .utils import logger
    logger.info(f"STEP - Tree Search Start")
    
    # Tree search initialization
    tree_init_start = time.time()
    tree_search = TreeSearch(query, llm, video, events_vdb, entities_vdb, features_vdb, retrieval_mode=retrieval_mode)
    tree_init_end = time.time()
    logger.info(f"TIMING - Tree Search Initialization ({retrieval_mode}): {tree_init_end - tree_init_start:.4f} seconds")
    
    # Tree search execution
    tree_execution_start = time.time()
    tree_search.search()
    tree_execution_end = time.time()
    logger.info(f"TIMING - Tree Search Traversal: {tree_execution_end - tree_execution_start:.4f} seconds")
    
    # Collect tree information
    collect_info_start = time.time()
    tree_information = tree_search.collect_tree_information()
    collect_info_end = time.time()
    logger.info(f"TIMING - Collect Tree Information: {collect_info_end - collect_info_start:.4f} seconds")
    
    return tree_information

def generate_sa_self_consistency_result(
    tree_information: dict,
    llm: BaseLanguageModel,
    self_consistency_num: int = 1,
):
    # Start timing for self-consistency generation
    self_consistency_start = time.time()
    from .utils import logger
    logger.info(f"STEP - Self-Consistency Generation Start")
    
    # Prepare input prompts
    prompt_prep_start = time.time()
    all_input_prompts = []
    
    for sa_node in tree_information:
        input_prompt = sa_node["input_prompt"]
        for _ in range(self_consistency_num):
            all_input_prompts.append(
                {
                    "text": input_prompt,
                    # "text": input_prompt  + f"""
                    # This is the timestamps and descriptions of the events:
                    # {str(tree_information[0]['structed_information'][0]['event_data'])}
                    # """,
                }
            )
    prompt_prep_end = time.time()
    logger.info(f"TIMING - Prepare Input Prompts: {prompt_prep_end - prompt_prep_start:.4f} seconds")
    
    # Batched generate
    batch_generate_start = time.time()
    all_responses = llm.batch_generate_response(all_input_prompts)
    batch_generate_end = time.time()
    logger.info(f"TIMING - Batch Generate Responses: {batch_generate_end - batch_generate_start:.4f} seconds")
    
    # Process responses
    process_responses_start = time.time()
    for i in range(0, len(all_responses), self_consistency_num):
        node_responses = all_responses[i:i+self_consistency_num]
        tree_information[int(i/self_consistency_num)]["responses"] = node_responses
    process_responses_end = time.time()
    logger.info(f"TIMING - Process Responses: {process_responses_end - process_responses_start:.4f} seconds")
    
    return tree_information

def calculate_sa_score(
    node_results: list,
):
    # Start timing for SA score calculation
    score_calc_start = time.time()
    from .utils import logger
    logger.info(f"STEP - SA Score Calculation Start")
    
    # Initialize BERT scorer
    scorer_init_start = time.time()
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")
    scorer_init_end = time.time()
    logger.info(f"TIMING - BERT Scorer Initialization: {scorer_init_end - scorer_init_start:.4f} seconds")
    
    # calculate the score of each node
    node_processing_start = time.time()
    for i, node_result in enumerate(node_results):
        node_start = time.time()
        responses = node_result["responses"]
        answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        thoughts_counts = {"A": [], "B": [], "C": [], "D": []}
        node_score = {"A": [0.0, 0.0], "B": [0.0, 0.0], "C": [0.0, 0.0], "D": [0.0, 0.0]}
        
        def extract_json(llm_output):
            json_pattern = r'{.*}'
            match = re.search(json_pattern, llm_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    return parsed_json
                except json.JSONDecodeError as e:
                    return f"JSON parse error: {e}"
            else:
                return None
        
        # Parse responses
        response_parsing_start = time.time()
        for response in responses:
            json_result = extract_json(response)
            if isinstance(json_result, dict) and "Answer" in json_result.keys() and "Analysis" in json_result.keys():
                answer = json_result["Answer"]
                thoughts = json_result["Analysis"]
                if answer in answer_counts.keys():
                    answer_counts[answer] += 1
                    thoughts_counts[answer].append(thoughts)
        response_parsing_end = time.time()
        
        self_consistency_num = len(responses)
        
        # Calculate answer scores
        answer_score_start = time.time()
        for choice in answer_counts.keys():
            node_score[choice][0] = 1.0 * answer_counts[choice] / self_consistency_num
        answer_score_end = time.time()
            
        # Calculate thoughts scores
        thoughts_score_start = time.time()
        for choice in answer_counts.keys():
            if len(thoughts_counts[choice]) > 1:
                text1 = []
                text2 = []
                f1_scores = []
                for i in range(len(thoughts_counts[choice])):
                    for j in range(len(thoughts_counts[choice])):
                        if i != j:
                            text1.append(thoughts_counts[choice][i])
                            text2.append(thoughts_counts[choice][j])
                    f1,_, _ = scorer.score(text1, text2)
                    f1 = f1.mean().item()
                    f1_scores.append(f1)
                node_score[choice][1] = 1.0 * sum(f1_scores) / len(f1_scores)
        thoughts_score_end = time.time()
        
        node_result["scores"] = node_score
        node_end = time.time()
        logger.info(f"TIMING - Node {i} Processing: {node_end - node_start:.4f} seconds (parsing: {response_parsing_end - response_parsing_start:.4f}, answer: {answer_score_end - answer_score_start:.4f}, thoughts: {thoughts_score_end - thoughts_score_start:.4f})")
    
    node_processing_end = time.time()
    logger.info(f"TIMING - All Nodes Processing: {node_processing_end - node_processing_start:.4f} seconds")
    
    return node_results

def choose_best_sa_answer(
    score_results: list,
    alpha: float = 0.3,
):
    # Start timing for best answer selection
    choose_best_start = time.time()
    from .utils import logger
    logger.info(f"STEP - Choose Best Answer Start")
    
    # Calculate final scores
    final_score_calc_start = time.time()
    for score_result in score_results:
        scores = {
            choice: score_result["scores"][choice][0] * alpha + score_result["scores"][choice][1] * (1 - alpha)
            for choice in score_result["scores"]
        }
        best_choice = max(scores, key=scores.get)
        best_score = scores[best_choice]
        score_result["final_score"] = {best_choice: best_score}
    final_score_calc_end = time.time()
    logger.info(f"TIMING - Calculate Final Scores: {final_score_calc_end - final_score_calc_start:.4f} seconds")

    # Sort results
    sort_start = time.time()
    sorted_score_results = sorted(
        score_results,
        key=lambda x: list(x["final_score"].values())[0],
        reverse=True
    )
    sort_end = time.time()
    logger.info(f"TIMING - Sort Results: {sort_end - sort_start:.4f} seconds")

    return sorted_score_results

def generate_ca_self_consistency_result(
    query: str,
    sorted_sa_nodes: list,
    llm: BaseVideoModel,
    video: VideoRepresentation,
    self_consistency_num: int = 4,
    max_frames: int = 256,
    max_retries: int = 3,
):
    # Start timing for CA self-consistency generation
    ca_self_consistency_start = time.time()
    from .utils import logger
    logger.info(f"STEP - CA Self-Consistency Generation Start")
    
    # Select candidate nodes
    candidate_selection_start = time.time()
    candidate_nodes = []
    for sa_node in sorted_sa_nodes:
        if len(candidate_nodes) == 2:
            break
        
        if not candidate_nodes:
            candidate_nodes.append(sa_node)
        else:
            if list(sa_node["final_score"].keys())[0] != list(candidate_nodes[0]["final_score"].keys())[0]:
                candidate_nodes.append(sa_node)
    candidate_selection_end = time.time()
    logger.info(f"TIMING - Select Candidate Nodes: {candidate_selection_end - candidate_selection_start:.4f} seconds")
    
    # Process each candidate node
    for node_idx, candidate_node in enumerate(candidate_nodes):
        node_processing_start = time.time()
        logger.info(f"STEP - Processing CA Node {node_idx + 1}")
        
        # Extract frame durations
        frame_extraction_start = time.time()
        frame_durations = candidate_node["frame_durations"]
        frame_extraction_end = time.time()
        logger.info(f"TIMING - Extract Frame Durations: {frame_extraction_end - frame_extraction_start:.4f} seconds")
        
        # Sample frames
        frame_sampling_start = time.time()
        frames = []
        for frame_duration in frame_durations:
            sampled_frames, _, _ = video.get_frames_by_fps(fps=1, duration=frame_duration)
            frames.extend(sampled_frames)
        frame_sampling_end = time.time()
        logger.info(f"TIMING - Sample Frames: {frame_sampling_end - frame_sampling_start:.4f} seconds")
        
        # Downsample if needed
        downsample_start = time.time()
        if len(frames) > max_frames:
            downsampled_indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
            frames = [frames[i] for i in downsampled_indices]
        downsample_end = time.time()
        logger.info(f"TIMING - Downsample Frames: {downsample_end - downsample_start:.4f} seconds")
        
        # Create prompt
        prompt_creation_start = time.time()
        text_prompt = PROMPTS["checkframe_and_answer_COT"].format(
            user_query=query
        )
        prompt_creation_end = time.time()
        logger.info(f"TIMING - Create CA Prompt: {prompt_creation_end - prompt_creation_start:.4f} seconds")
        
        # Prepare batch inputs
        batch_prep_start = time.time()
        batch_input_prompts = []
        for _ in range(self_consistency_num):
            batch_input_prompts.append(
                {
                    "text": text_prompt,
                    "video": frames
                }
            )
        batch_prep_end = time.time()
        logger.info(f"TIMING - Prepare Batch Inputs: {batch_prep_end - batch_prep_start:.4f} seconds")
        
        # Generate responses with retries
        generation_start = time.time()
        for retry in range(max_retries):
            try:
                all_responses = llm.batch_generate_response(batch_input_prompts)
                break
            except Exception as e:
                logger.info(f"CA Generation retry {retry + 1} failed: {e}")
                continue
        generation_end = time.time()
        logger.info(f"TIMING - Generate CA Responses: {generation_end - generation_start:.4f} seconds")
        
        candidate_node["responses"] = all_responses
        node_processing_end = time.time()
        logger.info(f"TIMING - CA Node {node_idx + 1} Total Processing: {node_processing_end - node_processing_start:.4f} seconds")
    
    # Clean candidate nodes
    cleanup_start = time.time()
    cleaned_candidate_nodes = [
        {
            "action": "CA",
            "depth": candidate_node["depth"],
            "path": candidate_node["path"],
            "responses": candidate_node["responses"]
        } for candidate_node in candidate_nodes
    ]
    cleanup_end = time.time()
    logger.info(f"TIMING - Clean Candidate Nodes: {cleanup_end - cleanup_start:.4f} seconds")
    
    ca_self_consistency_end = time.time()
    logger.info(f"TIMING - CA Self-Consistency Generation Total: {ca_self_consistency_end - ca_self_consistency_start:.4f} seconds")
    
    return cleaned_candidate_nodes

def calculate_ca_score(
    node_results: list,
):
    # Start timing for CA score calculation
    ca_score_calc_start = time.time()
    from .utils import logger
    logger.info(f"STEP - CA Score Calculation Start")
    
    # Initialize BERT scorer
    scorer_init_start = time.time()
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")
    scorer_init_end = time.time()
    logger.info(f"TIMING - BERT Scorer Initialization: {scorer_init_end - scorer_init_start:.4f} seconds")
    
    # Process each CA node
    node_processing_start = time.time()
    for node_idx, node_result in enumerate(node_results):
        node_start = time.time()
        logger.info(f"STEP - Processing CA Node {node_idx + 1}")
        
        responses = node_result["responses"]
        answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        thoughts_counts = {"A": [], "B": [], "C": [], "D": []}
        node_score = {"A": [0.0, 0.0], "B": [0.0, 0.0], "C": [0.0, 0.0], "D": [0.0, 0.0]}
        
        def extract_json(llm_output):
            json_pattern = r'{.*}'
            match = re.search(json_pattern, llm_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    return parsed_json
                except json.JSONDecodeError as e:
                    return f"JSON parse error: {e}"
            else:
                return None
        
        # Parse responses
        response_parsing_start = time.time()
        for response in responses:
            json_result = extract_json(response)
            if isinstance(json_result, dict) and "Answer" in json_result.keys() and "Analysis" in json_result.keys():
                answer = json_result["Answer"]
                thoughts = json_result["Analysis"]
                if answer in answer_counts.keys():
                    answer_counts[answer] += 1
                    thoughts_counts[answer].append(thoughts)
        response_parsing_end = time.time()
        logger.info(f"TIMING - Parse CA Responses: {response_parsing_end - response_parsing_start:.4f} seconds")
        
        self_consistency_num = len(responses)
        
        # Calculate answer scores
        answer_score_start = time.time()
        for choice in answer_counts.keys():
            node_score[choice][0] = 1.0 * answer_counts[choice] / self_consistency_num
        answer_score_end = time.time()
        logger.info(f"TIMING - Calculate CA Answer Scores: {answer_score_end - answer_score_start:.4f} seconds")
        
        # Calculate thoughts scores
        thoughts_score_start = time.time()
        for choice in answer_counts.keys():
            if len(thoughts_counts[choice]) > 1:
                text1 = []
                text2 = []
                f1_scores = []
                for i in range(len(thoughts_counts[choice])):
                    for j in range(len(thoughts_counts[choice])):
                        if i != j:
                            text1.append(thoughts_counts[choice][i])
                            text2.append(thoughts_counts[choice][j])
                    f1,_, _ = scorer.score(text1, text2)
                    f1 = f1.mean().item()
                    f1_scores.append(f1)
                node_score[choice][1] = 1.0 * sum(f1_scores) / len(f1_scores)
        thoughts_score_end = time.time()
        logger.info(f"TIMING - Calculate CA Thoughts Scores: {thoughts_score_end - thoughts_score_start:.4f} seconds")
        
        node_result["scores"] = node_score
        node_end = time.time()
        logger.info(f"TIMING - CA Node {node_idx + 1} Total Processing: {node_end - node_start:.4f} seconds")
    
    node_processing_end = time.time()
    logger.info(f"TIMING - All CA Nodes Processing: {node_processing_end - node_processing_start:.4f} seconds")
    
    ca_score_calc_end = time.time()
    logger.info(f"TIMING - CA Score Calculation Total: {ca_score_calc_end - ca_score_calc_start:.4f} seconds")
    
    return node_results

def choose_best_ca_answer(
    score_results: list,
    alpha: float = 0.3,
):
    # Start timing for CA best answer selection
    ca_choose_best_start = time.time()
    from .utils import logger
    logger.info(f"STEP - Choose Best CA Answer Start")
    
    # Calculate final scores
    final_score_calc_start = time.time()
    for score_result in score_results:
        scores = {
            choice: score_result["scores"][choice][0] * alpha + score_result["scores"][choice][1] * (1 - alpha)
            for choice in score_result["scores"]
        }
        best_choice = max(scores, key=scores.get)
        best_score = scores[best_choice]
        score_result["final_score"] = {best_choice: best_score}
    final_score_calc_end = time.time()
    logger.info(f"TIMING - Calculate CA Final Scores: {final_score_calc_end - final_score_calc_start:.4f} seconds")
    
    # Sort results
    sort_start = time.time()
    sorted_score_results = sorted(
        score_results,
        key=lambda x: list(x["final_score"].values())[0],
        reverse=True
    )
    sort_end = time.time()
    logger.info(f"TIMING - Sort CA Results: {sort_end - sort_start:.4f} seconds")
    
    ca_choose_best_end = time.time()
    logger.info(f"TIMING - Choose Best CA Answer Total: {ca_choose_best_end - ca_choose_best_start:.4f} seconds")
    
    return sorted_score_results
                    
            