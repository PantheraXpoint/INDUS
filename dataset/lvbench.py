import json
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from video_utils import VideoRepresentation
from typing import Union

class LVBench(Dataset):
    def __init__(self, json_file="datas/LVBench/LVBench.json", videos_path="datas/LVBench/videos", work_path="AVA_cache/LVBench/"):
        """
        Args:
            json_file (string): Path to the JSON file with video data.
            videos_path (string): Directory with all the videos.
        Self:
            video_info: 
                video_path -> source video path
                others -> other video dataset information
            work_path: directory to save the processed video frames
        """
        with open(json_file, 'r') as f:
            self.video_infos = json.load(f)
        
        self.videos_path = videos_path
        self.work_path = work_path
        self.init_event_list_all = json.load(open("ECML-PKDD/lvbench_retrieval/seed_events_LVBench.json", "r"))
        for video_info in self.video_infos:
            video_info["video_path"] = os.path.join(videos_path, f'{video_info["key"]}.mp4')

    def __len__(self):
        return len(self.video_list)

    def get_video_info(self, video_id:int):
        video_info = self.video_infos[video_id-1]

        return video_info
    
    def get_video(self, video_id):        
        source_path = self.video_infos[video_id-1]["video_path"]
        work_path = os.path.join(self.work_path, f"{video_id}")
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        
        return VideoRepresentation(source_path, work_path)
    
    def get_init_event_list(self, video_id: int, question_id):
        source_name = self.video_infos[video_id-1]["video_path"].split("/")[-1].split(".")[0]
        all_question_ids = []
        for event in self.init_event_list_all:
            if event["video_key"] == source_name:
                all_question_ids.append(event)
        # get top 20
        first_20_events = []
        all_borda_events = []
        for event in all_question_ids[int(question_id)]["seed_events"]:
            if event["borda_score"] is not None:
                all_borda_events.append(event)
        borda_events_sorted = sorted(all_borda_events, key=lambda x: x["borda_score"], reverse=True)
        first_20_events.extend([event["id"] for event in borda_events_sorted[:20-len(first_20_events)]])
        return first_20_events
    
    def get_video_for_vlm(self, video_id: Union[int, str]):
        """Get VideoRepresentation for VLM Direct - separate method to avoid breaking existing code"""
        video_info = self.get_video_info(video_id)
        source_path = video_info["video_path"]
        work_path = os.path.join(self.work_path, f"{video_id}")
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        
        return VideoRepresentation(source_path, work_path)