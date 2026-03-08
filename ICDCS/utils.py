import os
import json
import time
from llms.BaseModel import BaseVideoModel
from bert_score import score, BERTScorer
import json
import ast
from typing import Optional

def semantic_chunking(
    llm: BaseVideoModel,
    descriptions: list,
    chunk_durations: list,
    file_path: str,
    global_config: dict,
    threshold: float = 0.65,
    reprocess: bool = False,
    window_size: int = 8,
    max_retries: int = 5,
    batch_size: int = 16,
):
    profiling: dict = {}
    
    # start from first-unmerged description
    merged_timestamp = events[-1]["duration"][-1] if events else 0
    chunk_duration = global_config["video_chunk_duration"]
    num_merged_descriptions = merged_timestamp // chunk_duration + (1 if merged_timestamp % chunk_duration > 0 else 0)
    
    unmerged_descriptions = [descriptions[i] for i in range(num_merged_descriptions, len(descriptions))]
    
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)
    
    # cal bert score within window_size
    profiling["cal_bert_score"] = time.time()
    description_list1 = []
    description_list2 = []
    for i in range(len(unmerged_descriptions)):
        for j in range(i+1, min(i + 1 + window_size, len(unmerged_descriptions))):
            description_list1.append(unmerged_descriptions[i])
            description_list2.append(unmerged_descriptions[j])
    _, recall, _ = scorer.score(description_list1, description_list2, batch_size=batch_size)
    recall = recall.tolist()
    profiling["cal_bert_score"] = time.time() - profiling["cal_bert_score"]
    scores_metric = [[0.0] * len(unmerged_descriptions) for _ in range(len(unmerged_descriptions))]
    unsed_count = 0
    for i in range(len(unmerged_descriptions)):
        for j in range(i+1, min(i + 1 + window_size, len(unmerged_descriptions))):
            scores_metric[i][j] = recall[unsed_count]
            unsed_count += 1
    
    
    def cal_chunk_score(i, j, scores_metric):
        chunk_score = 0
        chunk_count = 0
        for k in range(i, j+1):
            for l in range(k+1, j+1):
                chunk_score += scores_metric[k][l]
                chunk_count += 1
        return 1.0 * chunk_score / chunk_count
    
    partitions = []
    start_index = 0
    while start_index < len(unmerged_descriptions):
        end_index = start_index + 1
        while end_index < len(unmerged_descriptions) and end_index - start_index < window_size and cal_chunk_score(start_index, end_index, scores_metric) > threshold and scores_metric[end_index-1][end_index] > threshold:
            end_index += 1
        partitions.append((start_index, end_index-1))
        start_index = end_index
    
    profiling["summarize_descriptions"] = time.time()
    batch_inputs = []
    summary_indices = []
    for i in range(len(partitions)):
        partition = partitions[i]
        # if partition[0] == partition[1]:
        #     continue
        # else:
        summary_indices.append(i)
        prompt = PROMPTS["summarize_descriptions"].format(
            inputs=[descriptions[j] for j in range(partition[0], partition[1])]
        )
        batch_inputs.append({"text": prompt})
    
    for i in range(max_retries):
        batch_summaries = llm.batch_generate_response(batch_inputs, max_new_tokens=1024, temperature=0.5, max_batch_size=batch_size)
        if batch_summaries:
            break
        else:
            print(f"Attempt {i} failed. Retrying...")
    profiling["summarize_descriptions"] = time.time() - profiling["summarize_descriptions"]
    for i in range(len(partitions)):
        partition = partitions[i]
        if partition[0] == partition[1]:
            start_timestamp = chunk_durations[partition[0]+num_merged_descriptions][0]
            end_timestamp = chunk_durations[partition[1]+num_merged_descriptions][1]
            events.append({
                "duration": [start_timestamp, end_timestamp],
                "description": unmerged_descriptions[partition[0]],
            })
        else:
            start_timestamp = chunk_durations[partition[0]+num_merged_descriptions][0]
            end_timestamp = chunk_durations[partition[1]+num_merged_descriptions][1]
            
            summary_index = summary_indices.index(i)
            summary = batch_summaries[summary_index]
            
            events.append({
                "duration": [start_timestamp, end_timestamp],
                "description": summary,
            })
        
        with open(source_file, "w") as f:
            json.dump(events, f, indent=4)
    
    with open(scores_file, "w") as f:
        json.dump(scores_metric, f, indent=4)
    
    events = format_events(events)
    return events, profiling

def extract_analysis_value(raw: str) -> str:
    """
    Extracts obj["Analysis"] from inputs like:
      '{\\n  "Analysis": "The user\\'s ..."}'
    or:
      '{ "Analysis": "..." }'
    """
    s = raw.strip()

    # 1) Try normal JSON first
    try:
        obj = json.loads(s)
        # If it's a JSON string containing JSON, decode again
        if isinstance(obj, str):
            obj = json.loads(obj)
        if isinstance(obj, dict) and "Analysis" in obj:
            return obj["Analysis"]
    except json.JSONDecodeError:
        pass

    # 2) If it's wrapped in Python quotes (like your example), decode as Python string literal
    #    This converts \\n -> newline and \\' -> '
    try:
        decoded = ast.literal_eval(s)  # important: DO NOT strip outer quotes first
        if isinstance(decoded, dict) and "Analysis" in decoded:
            return decoded["Analysis"]

        if isinstance(decoded, str):
            # decoded is now the inner JSON text
            obj = json.loads(decoded)
            return obj["Analysis"]
    except Exception:
        pass

    # 3) Last resort: fix the common invalid JSON escape \'
    #    (Turn it into a normal apostrophe) and try again.
    try:
        fixed = s.replace("\\'", "'")
        obj = json.loads(fixed)
        return obj["Analysis"]
    except Exception as e:
        raise ValueError(f'Could not extract "Analysis": {e}')

if __name__ == "__main__":
    text = '{\n  "Analysis": "To answer the user\'s query about the parking cost options near Shoppers Drug Mart, I reviewed the provided knowledge graph information and event-to-event connections. However, the video frames and knowledge graph information do not contain explicit details about parking costs at Shoppers Drug Mart. The events and objects described in the knowledge graph and video frames provide a rich description of the urban environment, including scenes with a \'Shoppers Drug Mart\' sign and various parking lots. However, none of these scenes directly mention parking costs or provide a clear indication of the parking rates at the specific location near Shoppers Drug Mart.\n\nThe event-to-event connections provided some context about the urban environment, including the presence of parking lots and the presence of \'Shoppers Drug Mart\' in different scenes. However, none of these connections directly address the parking costs. \n\nOne event, Event-47edb4b460b4bae657d948df31fbdb18, does mention a flat-rate parking cost of $10 in a snowy urban environment. While this event does not directly reference Shoppers Drug Mart, it provides a possible parking cost scenario that could be relevant to the user\'s query. The user is asking for specific parking cost information, and while the video frames and knowledge graph provide a detailed urban environment, they do not contain the exact information requested.\n\nTherefore, based on the available information, I cannot provide a definitive answer to the user\'s query about parking cost options near Shoppers Drug Mart. The user may need to provide more specific information about the location or seek additional sources for the parking cost details."\n}'
    print(extract_analysis_value(text))