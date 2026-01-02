from typing import Dict, Any
import json
import glob
import os

def time_to_seconds(time_str: str):
    """
    Convert a time string to seconds.
    """
    if len(time_str.split(":")) == 2:
        minutes, seconds = time_str.split(":")
        return int(minutes) * 60 + int(seconds)
    elif len(time_str.split(":")) == 3:
        hours, minutes, seconds = time_str.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    else:
        raise ValueError(f"Invalid time string: {time_str}")

from typing import List, Tuple

def percentage_overlap(time_list: List[Tuple[int, int]], time_ref: Tuple[int, int]) -> float:
    ref_start, ref_end = time_ref

    # Guard against inverted intervals
    if ref_end < ref_start:
        return 0.0
        raise ValueError("time_ref must have ref_end >= ref_start")

    # Special case: time_ref is a single point
    if ref_start == ref_end:
        point = ref_start
        for start, end in time_list:
            if end <= start:
                continue  # skip invalid/empty chunks
            # check if point lies inside [start, end) (or any convention you use)
            if start <= point < end:
                return 1.0
        return 0.0

    # General case: time_ref is an interval
    ref_length = ref_end - ref_start
    
    # Merge overlapping intervals within the reference range to avoid counting overlaps multiple times
    relevant_intervals = []
    for start, end in time_list:
        if end <= start:
            continue  # skip invalid/empty chunks
        # Clip to reference range
        s = max(start, ref_start)
        e = min(end, ref_end)
        if e > s:
            relevant_intervals.append((s, e))
    
    if not relevant_intervals:
        return 0.0
    
    # Sort intervals and merge overlapping ones
    relevant_intervals.sort()
    merged = [relevant_intervals[0]]
    for current_start, current_end in relevant_intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            # Overlapping or adjacent, merge
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # Non-overlapping, add as new interval
            merged.append((current_start, current_end))
    
    # Calculate total covered length
    total_covered = sum(end - start for start, end in merged)
    
    # Cap at 1.0 (100% coverage maximum)
    return min(1.0, total_covered / ref_length) if ref_length > 0 else 0.0

def overlap_reference_helper(time_ref: str, time_list: List[Tuple[int, int]]):
    time_ref = time_ref.strip()
    if time_ref == "N/A" or time_ref == "" or time_ref == "None" or time_ref == "None-None":
        return None
    if "-" in time_ref:
        start_time, end_time = time_ref.split("-")
        if start_time in ["", "None"]:
            start_time = end_time
        if end_time in ["", "None"]:
            end_time = start_time
        start_time_seconds = time_to_seconds(start_time)
        end_time_seconds = time_to_seconds(end_time)
    elif "," in time_ref:
        points_overlap = []
        points = time_ref.split(",")
        for point in points:
            point = point.strip()
            points_overlap.append(percentage_overlap(time_list, (time_to_seconds(point), time_to_seconds(point))))
        return sum(points_overlap) / len(points_overlap)
    else:
        start_time_seconds = time_to_seconds(time_ref)
        end_time_seconds = start_time_seconds
    return percentage_overlap(time_list, (start_time_seconds, end_time_seconds))



def overlap_reference(graph_data: Dict[str, Any], time_reference: Dict[str, Any]):
    """
    Check if the time reference of the graph data overlaps with the time reference of the time reference.
    """
    time_list = []
    for node in graph_data["nodes"]:
        if node["type"] == "event":
            start_time = node["metadata"]["start_time"] // 30 # frames to seconds
            end_time = node["metadata"]["end_time"] // 30 # frames to seconds
            time_list.append((start_time, end_time))
    time_ref = time_reference["time_reference"].strip()
    return overlap_reference_helper(time_ref, time_list)

def load_data(data_path: str):
    with open(data_path, 'r') as f:
        return json.load(f)

def run_ava100_benchmark():
    datasets = ["citytour", "ego", "traffic", "wildlife"]
    for dataset in datasets:
        data_path = f"datas/AVA100/{dataset}.json"
        with open(data_path, 'r') as f:
            data = json.load(f)
        for item in data:
            video_key = item["video_key"]
            for question in item["qa"]:
                graph_datas = glob.glob(f"ava100_results/{video_key}/q{question['question_id']}/subgraph_*.json")
                overlap_list = []
                for graph_data in graph_datas:
                    graph_data = load_data(graph_data)
                    time_reference = load_data(f"datas/AVA100/{dataset}.json")
                    time_reference = time_reference[int(video_key[-1])-1]["qa"][int(question["question_id"])]
                    overlap = overlap_reference(graph_data, time_reference)
                    if overlap is not None:
                        overlap_list.append(overlap)
                if len(overlap_list) > 0:
                    print(f"Video: {video_key}, Question: {question['question_id']}, Overlap: {sum(overlap_list)}")
                else:
                    print(f"Video: {video_key}, Question: {question['question_id']}, No overlap")

def run_lvbench_benchmark():
    video_folders = glob.glob("AVA_cache/LVBench/*")
    lvbench_data = load_data("datas/LVBench/LVBench.json")
    overlap_dataset = []
    for video_folder in video_folders:
        if not os.path.exists(os.path.join(video_folder, "config.json")):
            print(f"No config.json in {video_folder}")
            continue
        video_key = load_data(os.path.join(video_folder, "config.json"))["source_path"].split("/")[-1].split(".")[0]
        res = []
        for idx, item in enumerate(lvbench_data):
            if item["key"] == video_key:
                for q_id, question in enumerate(item["qa"]):
                    if not os.path.exists(f"{video_folder}/questions/{q_id}/sorted_SA_score_result.json"):
                        print(f"No sorted_SA_score_result.json in {video_folder}/questions/{q_id}")
                        continue
                    graph_datas = load_data(f"{video_folder}/questions/{q_id}/sorted_SA_score_result.json")
                    overlap_list = []
                    for graph_data in graph_datas:
                        if graph_data["depth"] != 3: # only consider the first depth
                            continue
                        graph_data = graph_data["frame_durations"]
                        graph_data = [(start, end) for start, end in graph_data]
                        time_reference = question["time_reference"].strip()
                        # print(time_reference)
                        overlap = overlap_reference_helper(time_reference, graph_data)
                        if overlap is not None:
                            overlap_list.append(overlap)
                    if len(overlap_list) > 0:
                        print(f"Video: {video_key}, Question: {q_id}, Overlap: {sum(overlap_list) / len(overlap_list)}")
                        overlap_dataset.append(sum(overlap_list) / len(overlap_list))
                        res.append({"video_id": idx, "question_id": q_id, "overlap": sum(overlap_list) / len(overlap_list)})
                    
    print(f"Average Overlap: {sum(overlap_dataset) / len(overlap_dataset)}")
    json.dump(res, open("lvbench_overlap.json", "w"), indent=4)


def run_ava100_benchmark_ava():
    video_folders = glob.glob("AVA_cache/AVA100/*")
    overlap_dataset = []
    for video_folder in video_folders:
        video_key = load_data(os.path.join(video_folder, "config.json"))["source_path"].split("/")[-1].split(".")[0]
        data = load_data(f"datas/AVA100/{video_key[:-1]}.json")
        for item in data:
            if item["video_key"] == video_key:
                for q_id, question in enumerate(item["qa"]):
                    graph_datas = load_data(f"{video_folder}/questions/{q_id}/sorted_SA_score_result.json")
                    overlap_list = []
                    for graph_data in graph_datas:
                        if graph_data["depth"] != 3: # only consider the first depth
                            continue
                        graph_data = graph_data["frame_durations"]
                        graph_data = [(start, end) for start, end in graph_data]
                        time_reference = question["time_reference"].strip()
                        # print(time_reference)
                        overlap = overlap_reference_helper(time_reference, graph_data)
                        if overlap is not None:
                            overlap_list.append(overlap)
                    if len(overlap_list) > 0:
                        print(f"Video: {video_key}, Question: {q_id}, Overlap: {sum(overlap_list) / len(overlap_list)}")
                        overlap_dataset.append(sum(overlap_list) / len(overlap_list))
                    
    print(f"Average Overlap: {sum(overlap_dataset) / len(overlap_dataset)}")


if __name__ == "__main__":
    # run_ava100_benchmark_ava()
    run_lvbench_benchmark()