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
        
        for video_info in self.video_infos:
            video_info["video_path"] = os.path.join(videos_path, f'{video_info["key"]}.mp4')

    def __len__(self):
        return len(self.video_infos)

    def get_video_info(self, video_id:int):
        video_info = self.video_infos[video_id-1]

        return video_info
    
    def get_video(self, video_id):        
        source_path = self.video_infos[video_id-1]["video_path"]
        base_path = os.path.join("database", os.path.basename(source_path)[:-4])
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        object_faiss_db_path = os.path.join(base_path, "object_embeddings.db")
        event_faiss_db_path = os.path.join(base_path, "event_embeddings.db")
        object_sqlite_db_path = os.path.join(base_path, "tracked_objects.db")
        # if not os.path.exists(object_faiss_db_path):
        #     return None
        # if not os.path.exists(event_faiss_db_path):
        #     return None
        # if not os.path.exists(object_sqlite_db_path):
        #     return None
        return source_path, object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path