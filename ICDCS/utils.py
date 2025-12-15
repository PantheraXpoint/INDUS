import os
import json
import time
from llms.BaseModel import BaseVideoModel
from bert_score import score, BERTScorer

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