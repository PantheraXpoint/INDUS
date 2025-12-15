import json
import random
import copy
import time

import numpy as np
from llms.BaseModel import BaseLanguageModel, BaseVideoModel
from typing import Union, List, Dict
from .storage import BaseVectorStorage
from .prompt import PROMPTS
from json_repair import repair_json
from video_utils import VideoRepresentation

def tri_view_retrieval(
    query: str,
    llm: Union[BaseLanguageModel, BaseVideoModel],
    events_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    features_vdb: BaseVectorStorage,
    generation: int = 0
):
    # Start timing for tri-view retrieval
    tri_view_start = time.time()
    from .utils import logger
    logger.info(f"STEP - Tri-View Retrieval Start (generation {generation})")
    
    top_k_for_events = 5
    top_k_for_entities = 5
    top_k_for_features = 32
    S = 1
    
    # Prepare prompts
    prompt_prep_start = time.time()
    keywords_prompt = PROMPTS["keyword_extraction"].format(
        input_text=query
    )
    
    rewrite_entity_prompt = PROMPTS["query_rewrite_for_entity_retrieval"].format(
        input_text=query
    )
    
    rewrite_feature_prompt = PROMPTS["query_rewrite_for_visual_retrieval"].format(
        input_text=query
    )
    
    batch_inputs = [
        {
            "text": keywords_prompt
        },
        {
            "text": rewrite_entity_prompt
        },
        {
            "text": rewrite_feature_prompt
        }
    ]
    prompt_prep_end = time.time()
    logger.info(f"TIMING - Prepare Prompts: {prompt_prep_end - prompt_prep_start:.4f} seconds")
    
    # Batch LLM generation
    batch_generate_start = time.time()
    batch_outputs = llm.batch_generate_response(batch_inputs)
    batch_generate_end = time.time()
    logger.info(f"TIMING - Batch LLM Generation: {batch_generate_end - batch_generate_start:.4f} seconds")
    
    keywords_response = batch_outputs[0]
    rewrite_entity_response = batch_outputs[1]
    rewrite_feature_response = batch_outputs[2]
    
    # Vector database queries
    vector_queries_start = time.time()
    events_result = events_vdb.query(keywords_response, top_k=top_k_for_events)
    entities_result = entities_vdb.query(rewrite_entity_response, top_k=top_k_for_entities)
    features_result = features_vdb.query(rewrite_feature_response, top_k=top_k_for_features)
    vector_queries_end = time.time()
    logger.info(f"TIMING - Vector Database Queries: {vector_queries_end - vector_queries_start:.4f} seconds")

    # Event similarity scores calculation
    similarity_calc_start = time.time()
    events_from_events = {event["id"]: event["__metrics__"] for event in events_result}
    events_from_entities = {}
    for entity in entities_result:
        for event_id in entity["events"]:
            if event_id not in events_from_entities:
                events_from_entities[event_id] = entity["__metrics__"]
            else:
                events_from_entities[event_id] += entity["__metrics__"]
    events_from_features = {}
    for feature in features_result:
        if feature["event"] not in events_from_features:
            events_from_features[feature["event"]] = feature["__metrics__"]
        else:
            events_from_features[feature["event"]] += feature["__metrics__"]
    similarity_calc_end = time.time()
    logger.info(f"TIMING - Event Similarity Calculation: {similarity_calc_end - similarity_calc_start:.4f} seconds")
    
    # Sort events by scores
    sorting_start = time.time()
    events_from_events = sorted(events_from_events.items(), key=lambda x: x[1], reverse=True)
    events_from_entities = sorted(events_from_entities.items(), key=lambda x: x[1], reverse=True)
    events_from_features = sorted(events_from_features.items(), key=lambda x: x[1], reverse=True)
    sorting_end = time.time()
    logger.info(f"TIMING - Sort Events by Scores: {sorting_end - sorting_start:.4f} seconds")
    
    # Normalized Borda Count
    borda_count_start = time.time()
    event_scores = {}
    events_from_events_scores_sum = sum([score for _, score in events_from_events])
    events_from_entities_scores_sum = sum([score for _, score in events_from_entities])
    events_from_features_scores_sum = sum([score for _, score in events_from_features])
    
    for event_id, score in events_from_events:
        event_scores[event_id] = score / events_from_events_scores_sum * S
    for event_id, score in events_from_entities:
        if event_id not in event_scores:
            event_scores[event_id] = score / events_from_entities_scores_sum * S
        else:
            event_scores[event_id] += score / events_from_entities_scores_sum * S
    for event_id, score in events_from_features:
        if event_id not in event_scores:
            event_scores[event_id] = score / events_from_features_scores_sum * S
        else:
            event_scores[event_id] += score / events_from_features_scores_sum * S
    borda_count_end = time.time()
    logger.info(f"TIMING - Normalized Borda Count: {borda_count_end - borda_count_start:.4f} seconds")
            
    # Prepare results
    result_prep_start = time.time()
    results = [{"event_id": [event_id], "query": [query], "score": event_scores[event_id], "generation": generation+1} for event_id in event_scores]
    
    for result in results:
        event_id = result["event_id"][0]
        event_data = events_vdb.get_data(event_id)
        result["event_data"] = [event_data]

    results = sorted(results, key=lambda x: x["event_data"][0]["duration"][0])
    result_prep_end = time.time()
    logger.info(f"TIMING - Prepare Results: {result_prep_end - result_prep_start:.4f} seconds")
    
    end_time = time.time()
    
    return results

def events_only_retrieval(
    query: str,
    llm: Union[BaseLanguageModel, BaseVideoModel],
    events_vdb: BaseVectorStorage,
    generation: int = 0
):
    """Retrieval using only events vector database"""
    # Start timing for events-only retrieval
    events_only_start = time.time()
    from .utils import logger
    logger.info(f"STEP - Events-Only Retrieval Start (generation {generation})")
    
    top_k_for_events = 5
    S = 1
    
    # Prepare prompt
    prompt_prep_start = time.time()
    keywords_prompt = PROMPTS["keyword_extraction"].format(
        input_text=query
    )
    prompt_prep_end = time.time()
    logger.info(f"TIMING - Prepare Keywords Prompt: {prompt_prep_end - prompt_prep_start:.4f} seconds")
    
    # LLM generation
    llm_generation_start = time.time()
    keywords_response = llm.generate_response({"text": keywords_prompt})
    llm_generation_end = time.time()
    logger.info(f"TIMING - LLM Generation for Keywords: {llm_generation_end - llm_generation_start:.4f} seconds")
    
    # Vector database query
    vector_query_start = time.time()
    events_result = events_vdb.query(keywords_response, top_k=top_k_for_events)
    vector_query_end = time.time()
    logger.info(f"TIMING - Events Vector Database Query: {vector_query_end - vector_query_start:.4f} seconds")

    # Event similarity scores calculation
    similarity_calc_start = time.time()
    events_from_events = {event["id"]: event["__metrics__"] for event in events_result}
    similarity_calc_end = time.time()
    logger.info(f"TIMING - Event Similarity Calculation: {similarity_calc_end - similarity_calc_start:.4f} seconds")
    
    # Sort events by scores
    sorting_start = time.time()
    events_from_events = sorted(events_from_events.items(), key=lambda x: x[1], reverse=True)
    sorting_end = time.time()
    logger.info(f"TIMING - Sort Events by Scores: {sorting_end - sorting_start:.4f} seconds")
    
    # Normalized scoring
    scoring_start = time.time()
    event_scores = {}
    events_from_events_scores_sum = sum([score for _, score in events_from_events])
    
    for event_id, score in events_from_events:
        event_scores[event_id] = score / events_from_events_scores_sum * S
    scoring_end = time.time()
    logger.info(f"TIMING - Normalized Scoring: {scoring_end - scoring_start:.4f} seconds")
            
    # Prepare results
    result_prep_start = time.time()
    results = [{"event_id": [event_id], "query": [query], "score": event_scores[event_id], "generation": generation+1} for event_id in event_scores]
    
    for result in results:
        event_id = result["event_id"][0]
        event_data = events_vdb.get_data(event_id)
        result["event_data"] = [event_data]

    results = sorted(results, key=lambda x: x["event_data"][0]["duration"][0])
    result_prep_end = time.time()
    logger.info(f"TIMING - Prepare Results: {result_prep_end - result_prep_start:.4f} seconds")
    
    end_time = time.time()
    logger.info(f"TIMING - Events-Only Retrieval Total: {end_time - events_only_start:.4f} seconds")
    
    return results

def entities_only_retrieval(
    query: str,
    llm: Union[BaseLanguageModel, BaseVideoModel],
    entities_vdb: BaseVectorStorage,
    events_vdb: BaseVectorStorage,
    generation: int = 0
):
    """Retrieval using only entities vector database"""
    # Start timing for entities-only retrieval
    entities_only_start = time.time()
    from .utils import logger
    logger.info(f"STEP - Entities-Only Retrieval Start (generation {generation})")
    
    top_k_for_entities = 5
    S = 1
    
    # Prepare prompt
    prompt_prep_start = time.time()
    rewrite_entity_prompt = PROMPTS["query_rewrite_for_entity_retrieval"].format(
        input_text=query
    )
    prompt_prep_end = time.time()
    logger.info(f"TIMING - Prepare Entity Rewrite Prompt: {prompt_prep_end - prompt_prep_start:.4f} seconds")
    
    # LLM generation
    llm_generation_start = time.time()
    rewrite_entity_response = llm.generate_response({"text": rewrite_entity_prompt})
    llm_generation_end = time.time()
    logger.info(f"TIMING - LLM Generation for Entity Rewrite: {llm_generation_end - llm_generation_start:.4f} seconds")
    
    # Vector database query
    vector_query_start = time.time()
    entities_result = entities_vdb.query(rewrite_entity_response, top_k=top_k_for_entities)
    vector_query_end = time.time()
    logger.info(f"TIMING - Entities Vector Database Query: {vector_query_end - vector_query_start:.4f} seconds")

    # Event similarity scores calculation from entities
    similarity_calc_start = time.time()
    events_from_entities = {}
    for entity in entities_result:
        for event_id in entity["events"]:
            if event_id not in events_from_entities:
                events_from_entities[event_id] = entity["__metrics__"]
            else:
                events_from_entities[event_id] += entity["__metrics__"]
    similarity_calc_end = time.time()
    logger.info(f"TIMING - Event Similarity Calculation from Entities: {similarity_calc_end - similarity_calc_start:.4f} seconds")
    
    # Sort events by scores
    sorting_start = time.time()
    events_from_entities = sorted(events_from_entities.items(), key=lambda x: x[1], reverse=True)
    sorting_end = time.time()
    logger.info(f"TIMING - Sort Events by Scores: {sorting_end - sorting_start:.4f} seconds")
    
    # Normalized scoring
    scoring_start = time.time()
    event_scores = {}
    events_from_entities_scores_sum = sum([score for _, score in events_from_entities])
    
    for event_id, score in events_from_entities:
        event_scores[event_id] = score / events_from_entities_scores_sum * S
    scoring_end = time.time()
    logger.info(f"TIMING - Normalized Scoring: {scoring_end - scoring_start:.4f} seconds")
            
    # Prepare results
    result_prep_start = time.time()
    results = [{"event_id": [event_id], "query": [query], "score": event_scores[event_id], "generation": generation+1} for event_id in event_scores]
    
    for result in results:
        event_id = result["event_id"][0]
        event_data = events_vdb.get_data(event_id)
        result["event_data"] = [event_data]

    results = sorted(results, key=lambda x: x["event_data"][0]["duration"][0])
    result_prep_end = time.time()
    logger.info(f"TIMING - Prepare Results: {result_prep_end - result_prep_start:.4f} seconds")
    
    end_time = time.time()
    logger.info(f"TIMING - Entities-Only Retrieval Total: {end_time - entities_only_start:.4f} seconds")
    
    return results

def features_only_retrieval(
    query: str,
    llm: Union[BaseLanguageModel, BaseVideoModel],
    features_vdb: BaseVectorStorage,
    events_vdb: BaseVectorStorage,
    generation: int = 0
):
    """Retrieval using only features vector database"""
    # Start timing for features-only retrieval
    features_only_start = time.time()
    from .utils import logger
    logger.info(f"STEP - Features-Only Retrieval Start (generation {generation})")
    
    top_k_for_features = 32
    S = 1
    
    # Prepare prompt
    prompt_prep_start = time.time()
    rewrite_feature_prompt = PROMPTS["query_rewrite_for_visual_retrieval"].format(
        input_text=query
    )
    prompt_prep_end = time.time()
    logger.info(f"TIMING - Prepare Feature Rewrite Prompt: {prompt_prep_end - prompt_prep_start:.4f} seconds")
    
    # LLM generation
    llm_generation_start = time.time()
    rewrite_feature_response = llm.generate_response({"text": rewrite_feature_prompt})
    llm_generation_end = time.time()
    logger.info(f"TIMING - LLM Generation for Feature Rewrite: {llm_generation_end - llm_generation_start:.4f} seconds")
    
    # Vector database query
    vector_query_start = time.time()
    features_result = features_vdb.query(rewrite_feature_response, top_k=top_k_for_features)
    vector_query_end = time.time()
    logger.info(f"TIMING - Features Vector Database Query: {vector_query_end - vector_query_start:.4f} seconds")

    # Event similarity scores calculation from features
    similarity_calc_start = time.time()
    events_from_features = {}
    for feature in features_result:
        if feature["event"] not in events_from_features:
            events_from_features[feature["event"]] = feature["__metrics__"]
        else:
            events_from_features[feature["event"]] += feature["__metrics__"]
    similarity_calc_end = time.time()
    logger.info(f"TIMING - Event Similarity Calculation from Features: {similarity_calc_end - similarity_calc_start:.4f} seconds")
    
    # Sort events by scores
    sorting_start = time.time()
    events_from_features = sorted(events_from_features.items(), key=lambda x: x[1], reverse=True)
    sorting_end = time.time()
    logger.info(f"TIMING - Sort Events by Scores: {sorting_end - sorting_start:.4f} seconds")
    
    # Normalized scoring
    scoring_start = time.time()
    event_scores = {}
    events_from_features_scores_sum = sum([score for _, score in events_from_features])
    
    for event_id, score in events_from_features:
        event_scores[event_id] = score / events_from_features_scores_sum * S
    scoring_end = time.time()
    logger.info(f"TIMING - Normalized Scoring: {scoring_end - scoring_start:.4f} seconds")
            
    # Prepare results
    result_prep_start = time.time()
    results = [{"event_id": [event_id], "query": [query], "score": event_scores[event_id], "generation": generation+1} for event_id in event_scores]
    
    for result in results:
        event_id = result["event_id"][0]
        event_data = events_vdb.get_data(event_id)
        result["event_data"] = [event_data]

    results = sorted(results, key=lambda x: x["event_data"][0]["duration"][0])
    result_prep_end = time.time()
    logger.info(f"TIMING - Prepare Results: {result_prep_end - result_prep_start:.4f} seconds")
    
    end_time = time.time()
    logger.info(f"TIMING - Features-Only Retrieval Total: {end_time - features_only_start:.4f} seconds")
    
    return results

class EventList:
    def __init__(self, datas:List[Dict], Limited_length:int=16):
        self.datas = []
        self.Limited_length = Limited_length
        self.max_generation = max([event_chunk["generation"] for event_chunk in self.datas]) if self.datas else 0
        self.event_positions = self.get_event_positions()
        self.insert(datas)
        
    def insert(self, new_datas: List[Dict]):
        for new_data in new_datas:
            event_id = new_data["event_id"][0]
            if event_id not in self.event_positions:
                self.datas.append(new_data)
        
        # self.drop()
        self.merge_adjacent_events()
        self.drop()
        self.event_positions = self.get_event_positions()
    
    def get_event_positions(self):
        positions = {}
        for i, event_chunk in enumerate(self.datas):
            for j, event_id in enumerate(event_chunk["event_id"]):
                positions[event_id] = [i, j]
        
        return positions
        
    def merge_adjacent_events(self):
        self.datas.sort(key=lambda x: x["event_data"][0]["duration"][0])
        merged_datas = []
        
        for event_chunk in self.datas:
            if not merged_datas:
                merged_datas.append(event_chunk)
                continue
            
            last_chunk = merged_datas[-1]
            last_end = last_chunk["event_data"][-1]["duration"][1]
            current_start = event_chunk["event_data"][0]["duration"][0]
            current_end = event_chunk["event_data"][-1]["duration"][1]
            
            if last_end >= current_start:
                if last_end >= current_end:
                    continue
                else:
                    if last_end > current_start:
                        overlap_start_index = next(i for i, event in enumerate(event_chunk["event_data"]) if event["duration"][0] >= last_end)
                        # overlap_start_index = next((i for i, event in enumerate(event_chunk["event_data"]) if event["duration"][1] > last_end))
                        event_chunk["event_data"] = event_chunk["event_data"][overlap_start_index:]

                    last_chunk["event_data"].extend(event_chunk["event_data"])
                    # last_chunk["score"] = max(last_chunk["score"], event_chunk["score"])
                    last_chunk["score"] = last_chunk["score"] + event_chunk["score"]
                    last_chunk["generation"] = max(last_chunk["generation"], event_chunk["generation"])
                    last_chunk["event_id"].extend(event_chunk["event_id"])
                    last_chunk["query"].extend(event_chunk["query"])
            else:
                merged_datas.append(event_chunk)
        
        self.datas = merged_datas
        self.event_positions = self.get_event_positions()
    
    def format_information(self, limited_ratio=None):
        if limited_ratio is not None:
            candidate_datas = sorted(self.datas, key=lambda x: x["score"], reverse=True)
            candidate_datas = candidate_datas[:int(len(candidate_datas) * limited_ratio)]
        else:
            candidate_datas = self.datas
        
        formatted_information = ""
        sorted_datas = sorted(candidate_datas, key=lambda x: x["event_data"][0]["duration"][0])
        sorted_durations = []
        
        for event_chunk in sorted_datas:
            description = ""
            if len(event_chunk["event_data"]) == 1:
                description = event_chunk["event_data"][0]["description"]
            else:
                description_list = [event["description"] for event in event_chunk["event_data"]]
                description = "After, ".join(description_list)
                
            formatted_information += "{}s - {}s, {}\n".format(
                event_chunk["event_data"][0]["duration"][0],
                event_chunk["event_data"][-1]["duration"][1],
                description
            )
            
            sorted_durations.append([event_chunk["event_data"][0]["duration"][0], event_chunk["event_data"][-1]["duration"][1]])
        return formatted_information, sorted_durations

    def structed_information(self):
        return self.datas
    
    def format_durations(self, limited_frames:int=128, limited_ratio=None):
        if limited_ratio is not None:
            candidate_datas = sorted(self.datas, key=lambda x: x["score"], reverse=True)
            candidate_datas = candidate_datas[:int(len(candidate_datas) * limited_ratio)]
        else:
            candidate_datas = self.datas
        
        sorted_datas = sorted(candidate_datas, key=lambda x: x["event_data"][0]["duration"][0])
        durations = []
        for event_chunk in sorted_datas:
            durations.append([event_chunk["event_data"][0]["duration"][0], event_chunk["event_data"][-1]["duration"][1]])
        return durations
    
    def drop(self, drop_ratio=None):
        if drop_ratio is not None:
            cur_length = len(self.datas)
            drop_length = int(cur_length * drop_ratio)
            self.datas.sort(key=lambda x: x["score"], reverse=True)
            self.datas = self.datas[:cur_length - drop_length]
            self.datas = sorted(self.datas, key=lambda x: x["event_data"][0]["duration"][0])
        else:
            cur_length = len(self.datas)
            if cur_length > self.Limited_length:
                self.datas.sort(key=lambda x: x["score"], reverse=True)
                self.datas = self.datas[:self.Limited_length]
                
                self.datas = sorted(self.datas, key=lambda x: x["event_data"][0]["duration"][0])
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        return self.datas[index]
    
    def __iter__(self):
        return iter(self.datas)
    
    def __contains__(self, item):
        return item in self.datas


class Node:
    _node_counter = 0
    
    def __init__(self, state, query, action=None, parent=None, initial_event_list=None, events_vdb=None, entities_vdb=None, features_vdb=None, llm=None, video=None):
        # Start timing for node initialization
        node_init_start = time.time()
        
        # Assign unique node ID
        Node._node_counter += 1
        self.node_id = Node._node_counter
        
        self.action_function = {
            "RQ": self._re_query,
            "F": self._forward,
            "B": self._backward,
            "SA": self._prepare_summary_and_answer_input,
            # "CA": self._prepare_checkframe_and_answer_input
        }
        self.queries = [query]
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.event_list = copy.deepcopy(initial_event_list)
        self.events_vdb = events_vdb
        self.entities_vdb = entities_vdb
        self.features_vdb = features_vdb
        self.llm = llm
        self.video = video
        
        self.input_prompt = None
        self.responses = None
        
        self.max_retry_times = 5
        
        self.path = []
        if parent:
            self.path = copy.deepcopy(parent.path)
        if action:
            self.path.append(action)
            
        self.inference_input = None
        
        # Log node initialization timing
        node_init_end = time.time()
        from .utils import logger
        logger.info(f"TIMING - Node {self.node_id} ({action}) Initialization: {node_init_end - node_init_start:.4f} seconds")

    def apply_action(self):
        if self.action in self.action_function:
            self.action_function[self.action]()

    def _re_query(self):
        from .utils import logger
        logger.info(f"Node {self.node_id} - Re-query")
        re_query_start = time.time()
        
        # Format information step
        format_info_start = time.time()
        retrieved_information, _ = self.event_list.format_information()
        query_prompt = PROMPTS["re-query"].format(
            video_segments=retrieved_information,
            user_query=self.queries[0]
        )
        format_info_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Format Information: {format_info_end - format_info_start:.4f} seconds")
        
        # LLM generation step
        llm_generation_start = time.time()
        sub_query = self.queries[0]
        for i in range(self.max_retry_times):
            try:
                response = self.llm.generate_response({"text": query_prompt})
                response = repair_json(response)
                response = json.loads(response)
                sub_query = response["sub_query"]
                if sub_query:
                    break
            except Exception as e:
                logger.info(f"Retry {i+1} failed, error: {e}")
        llm_generation_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} LLM Generation for Re-query: {llm_generation_end - llm_generation_start:.4f} seconds")
        logger.info(f"Node {self.node_id} sub_query: {sub_query}")

        # Tri-view retrieval step
        tri_view_start = time.time()
        events_result = tri_view_retrieval(sub_query, self.llm, self.events_vdb, self.entities_vdb, self.features_vdb, generation=self.event_list.max_generation)
        # events_result = events_only_retrieval(sub_query, self.llm, self.events_vdb, generation=self.event_list.max_generation)
        # events_result = entities_only_retrieval(sub_query, self.llm, self.entities_vdb, self.events_vdb, generation=self.event_list.max_generation)
        # events_result = features_only_retrieval(sub_query, self.llm, self.features_vdb, self.events_vdb, generation=self.event_list.max_generation)
        tri_view_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Tri-view Retrieval in Re-query: {tri_view_end - tri_view_start:.4f} seconds")
        
        # Event list insertion step
        insert_start = time.time()
        self.event_list.insert(events_result)
        self.queries.append(sub_query)
        insert_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Event List Insertion: {insert_end - insert_start:.4f} seconds")
        
        re_query_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Re-query Action Total: {re_query_end - re_query_start:.4f} seconds")
        
    def _forward(self):
        from .utils import logger
        logger.info(f"Node {self.node_id} - Forward")
        forward_start = time.time()
        
        # Drop events step
        drop_start = time.time()
        self.event_list.drop(drop_ratio=0.2)
        drop_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Drop Events: {drop_end - drop_start:.4f} seconds")
        
        # Get previous events step
        get_previous_start = time.time()
        for event_chunk in self.event_list:
            event_id = event_chunk["event_id"][0]
            previous_event = self.events_vdb.get_previous_data(event_id)
            if previous_event:
                event_chunk["event_id"].insert(0, previous_event["__id__"])
                event_chunk["event_data"].insert(0, previous_event)
                event_chunk["query"].insert(0, event_chunk["query"][0])
        get_previous_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Get Previous Events: {get_previous_end - get_previous_start:.4f} seconds")
        
        # Merge adjacent events step
        merge_start = time.time()
        self.event_list.merge_adjacent_events()
        self.event_list.event_positions = self.event_list.get_event_positions()
        merge_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Merge Adjacent Events: {merge_end - merge_start:.4f} seconds")
        
        forward_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Forward Action Total: {forward_end - forward_start:.4f} seconds")

    def _backward(self):
        from .utils import logger
        logger.info(f"Node {self.node_id} - Backward")
        backward_start = time.time()
        
        # Drop events step
        drop_start = time.time()
        self.event_list.drop(drop_ratio=0.2)
        drop_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Drop Events: {drop_end - drop_start:.4f} seconds")
        
        # Get next events step
        get_next_start = time.time()
        for event_chunk in self.event_list:
            event_id = event_chunk["event_id"][-1]
            next_event = self.events_vdb.get_next_data(event_id)
            if next_event:
                event_chunk["event_id"].append(next_event["__id__"])
                event_chunk["event_data"].append(next_event)
                event_chunk["query"].append(event_chunk["query"][-1])
        get_next_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Get Next Events: {get_next_end - get_next_start:.4f} seconds")
        
        # Merge adjacent events step
        merge_start = time.time()
        self.event_list.merge_adjacent_events()
        self.event_list.event_positions = self.event_list.get_event_positions()
        merge_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Merge Adjacent Events: {merge_end - merge_start:.4f} seconds")
        
        backward_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Backward Action Total: {backward_end - backward_start:.4f} seconds")

    def _prepare_summary_and_answer_input(self):
        from .utils import logger
        logger.info(f"Node {self.node_id} - Prepare summary and answer input")
        prepare_start = time.time()
        
        # Format information step
        format_info_start = time.time()
        retrieved_information, retrieved_durations = self.event_list.format_information(limited_ratio=0.75)
        format_info_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Format Information for Summary: {format_info_end - format_info_start:.4f} seconds")
        
        # Create prompt step
        prompt_creation_start = time.time()
        summary_prompt = PROMPTS["summary_and_answer_COT"].format(
            video_segments=retrieved_information,
            user_query=self.queries[0]
        )
        prompt_creation_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Create Summary Prompt: {prompt_creation_end - prompt_creation_start:.4f} seconds")
        
        # Set node properties step
        set_properties_start = time.time()
        self.input_prompt = summary_prompt
        inference_input = {"text": summary_prompt}
        self.inference_input = inference_input
        self.retrieved_durations = retrieved_durations
        set_properties_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Set Node Properties: {set_properties_end - set_properties_start:.4f} seconds")
        
        prepare_end = time.time()
        logger.info(f"TIMING - Node {self.node_id} Prepare Summary and Answer Input Total: {prepare_end - prepare_start:.4f} seconds")
        return inference_input

class TreeSearch:
    ACTIONS = ["SA", "RQ", "F", "B"]
    
    def __init__(self, 
                 query: str, 
                 llm: Union[BaseLanguageModel, BaseVideoModel],
                 video: VideoRepresentation,
                 events_vdb: BaseVectorStorage,
                 entities_vdb: BaseVectorStorage,
                 features_vdb: BaseVectorStorage,
                 max_depth: int = 3,
                 retrieval_mode: str = "tri_view"):
        self.query = query
        self.llm = llm
        self.video = video
        self.events_vdb = events_vdb
        self.entities_vdb = entities_vdb
        self.features_vdb = features_vdb
        self.max_depth = max_depth
        self.retrieval_mode = retrieval_mode
        self.event_list = self.init_event_list(query, llm, events_vdb, entities_vdb, features_vdb, retrieval_mode)
        self.root = Node(state={}, action="Root", query=query, initial_event_list=self.event_list, events_vdb=events_vdb, entities_vdb=entities_vdb, features_vdb=features_vdb, llm=llm, video=video)
        
        self.actions = self.ACTIONS

    def init_event_list(self, query, llm, events_vdb, entities_vdb, features_vdb, retrieval_mode):
        init_start = time.time()
        
        if retrieval_mode == "tri_view":
            events_result = tri_view_retrieval(query, llm, events_vdb, entities_vdb, features_vdb)
        elif retrieval_mode == "events_only":
            events_result = events_only_retrieval(query, llm, events_vdb)
        elif retrieval_mode == "entities_only":
            events_result = entities_only_retrieval(query, llm, entities_vdb, events_vdb)
        elif retrieval_mode == "features_only":
            events_result = features_only_retrieval(query, llm, features_vdb, events_vdb)
        else:
            # Default to tri_view
            events_result = tri_view_retrieval(query, llm, events_vdb, entities_vdb, features_vdb)
            
        event_list = EventList(events_result)
        init_end = time.time()
        from .utils import logger
        logger.info(f"TIMING - Initialize Event List ({retrieval_mode}): {init_end - init_start:.4f} seconds")
        
        return event_list

    def merge_event_lists(self, event_lists, select_ratio=None):
        event_ids = set()
        if select_ratio is not None:
            for event_list in event_lists:
                event_list.drop(drop_ratio=(1-select_ratio))
            
        for event_list in event_lists:
            for event_chunk in event_list:
                for event_id in event_chunk["event_id"]:
                    event_ids.add(event_id)
        
        event_results = [{"event_id": [event_id], "query": [""], "score": 0.0, "generation": 0} for event_id in event_ids]
        for event_result in event_results:
            event_id = event_result["event_id"][0]
            event_data = self.events_vdb.get_data(event_id)
            event_result["event_data"] = [event_data]
            
        return EventList(event_results, Limited_length=1000)
                    
    def search(self):
        search_start = time.time()
        self._search(self.root, 0)
        search_end = time.time()
        from .utils import logger
        logger.info(f"TIMING - Tree Search Algorithm: {search_end - search_start:.4f} seconds")
        # self._delay_inference()

    def _search(self, node, current_depth):
        if current_depth >= self.max_depth or node.action == "SA" or node.action == "CA":
            return
        
        if current_depth == self.max_depth - 1:
            from .utils import logger
            logger.info(f"STEP - Creating SA Node at depth {current_depth}")
            sa_node_start = time.time()
            child_node = Node(state="node", query=node.queries[0], action="SA", parent=node, initial_event_list=node.event_list, events_vdb=self.events_vdb, entities_vdb=self.entities_vdb, features_vdb=self.features_vdb, llm=self.llm, video=self.video)
            child_node.apply_action()
            node.children.append(child_node)
            sa_node_end = time.time()
            logger.info(f"TIMING - SA Node {child_node.node_id} Creation and Action: {sa_node_end - sa_node_start:.4f} seconds")
            return

        for action in self.actions:
            from .utils import logger
            logger.info(f"STEP - Creating {action} Node at depth {current_depth}")
            node_creation_start = time.time()
            child_node = Node(state="node", query=node.queries[0], action=action, parent=node, initial_event_list=node.event_list, events_vdb=self.events_vdb, entities_vdb=self.entities_vdb, features_vdb=self.features_vdb, llm=self.llm, video=self.video)
            child_node.apply_action()
            node.children.append(child_node)
            node_creation_end = time.time()
            logger.info(f"TIMING - {action} Node {child_node.node_id} Creation and Action: {node_creation_end - node_creation_start:.4f} seconds")
            self._search(child_node, current_depth + 1)
            
    
    def collect_tree_information(self):
        collect_start = time.time()
        tree_information = []

        def collect_answers(node, depth, path):
            if node.action in ["SA"]:
                tree_information.append({
                    "action": node.action,
                    "depth": depth,
                    "path": path,
                    "input_prompt": node.input_prompt,
                    "frame_durations": node.retrieved_durations,
                    "structed_information": node.event_list.datas,
                })
            for child in node.children:
                collect_answers(child, depth + 1, path + [child.action])

        collect_answers(self.root, 0, [self.root.action])
        collect_end = time.time()
        from .utils import logger
        logger.info(f"TIMING - Collect Tree Information: {collect_end - collect_start:.4f} seconds")

        return tree_information
        