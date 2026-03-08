# import json
# import os
# from torch.utils.data import Dataset
# from PIL import Image
# import numpy as np
# from video_utils import VideoRepresentation
# from typing import Union

# class AVA100(Dataset):
#     def __init__(self, json_file="datas/AVA100", videos_path="datas/AVA100/videos", work_path="AVA_cache/AVA100/"):
#         """
#         Args:
#             json_file (string): Path to the JSON file with video data.
#             videos_path (string): Directory with all the videos.
#         """
#         json_file_list = [
#             os.path.join(json_file, "ego.json"),
#             os.path.join(json_file, "citytour.json"),
#             os.path.join(json_file, "wildlife.json"),
#             os.path.join(json_file, "traffic.json"),
#         ]
#         self.video_infos_list = []
#         for json_file in json_file_list:
#             if os.path.exists(json_file):
#                 with open(json_file, 'r') as f:
#                     self.video_infos_list.append(json.load(f))
#             else:
#                 self.video_infos_list.append([])
        
#         self.videos_path = videos_path
#         self.work_path = work_path
    
#     def __len__(self):
#         pass
    
#     def get_video_info(self, video_id: Union[int, str]):
#         video_id = int(video_id)
#         video_list_idx = (video_id-1) // 2
#         video_idx = (video_id-1) % 2
#         video_info = self.video_infos_list[video_list_idx][video_idx]
#         video_info["video_path"] = os.path.join(self.videos_path, f'{video_info["video_key"]}.mp4')
        
#         qas = video_info["qa"]
#         for qa in qas:
#             question = qa["query"]
#             options = qa["options"]
#             concat_question = f"{question}\n{options[0]}\n{options[1]}\n{options[2]}\n{options[3]}"
#             qa["question"] = concat_question
        
#         video_info["qa"] = qas
            
#         return video_info
    
#     def get_video(self, video_id: Union[int, str]):
#         video_info = self.get_video_info(video_id)
#         video_path = video_info["video_path"]
#         base_path = os.path.join("database", os.path.basename(video_path)[:-4])
#         if not os.path.exists(base_path):
#             os.makedirs(base_path)
#         object_faiss_db_path = os.path.join(base_path, "object_embeddings.db")
#         event_faiss_db_path = os.path.join(base_path, "event_embeddings.db")
#         object_sqlite_db_path = os.path.join(base_path, "tracked_objects.db")
#         return video_path, object_faiss_db_path, event_faiss_db_path, object_sqlite_db_path


import json
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from video_utils import VideoRepresentation
from typing import Union

class AVA100(Dataset):
    def __init__(self, json_file="datas/AVA100", videos_path="datas/AVA100/videos", work_path="AVA_cache/AVA100/"):
        """
        Args:
            json_file (string): Path to the JSON file with video data.
            videos_path (string): Directory with all the videos.
        """
        json_file_list = [
            os.path.join(json_file, "ego.json"),
            os.path.join(json_file, "citytour.json"),
            os.path.join(json_file, "wildlife.json"),
            os.path.join(json_file, "traffic.json"),
        ]
        self.video_infos_list = []
        for json_file in json_file_list:
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    self.video_infos_list.append(json.load(f))
            else:
                self.video_infos_list.append([])
        
        self.videos_path = videos_path
        self.work_path = work_path
        self.init_event_list_all = json.load(open("ECML-PKDD/ava100_retrieval/seed_events_AVA100.json", "r"))
    
    def __len__(self):
        pass
    
    def get_init_event_list(self, video_id: int, question_id: int):
        source_name = self.get_video_info(video_id)["video_path"].split("/")[-1].split(".")[0]
        seed_events = None
        for event in self.init_event_list_all:
            if event["video_key"] == source_name and event["question_id"] == question_id:
                seed_events = event["seed_events"]
        # get top 20
        first_20_events = []
        all_borda_events = []
        for event in seed_events:
            if event["borda_score"] is not None:
                all_borda_events.append(event)
        borda_events_sorted = sorted(all_borda_events, key=lambda x: x["borda_score"], reverse=True)
        first_20_events.extend([event["id"] for event in borda_events_sorted[:20-len(first_20_events)]])
        return first_20_events
    
    def get_video_info(self, video_id: Union[int, str]):
        video_id = int(video_id)
        video_list_idx = (video_id-1) // 2
        video_idx = (video_id-1) % 2
        video_info = self.video_infos_list[video_list_idx][video_idx]
        video_info["video_path"] = os.path.join(self.videos_path, f'{video_info["video_key"]}.mp4')
        
        qas = video_info["qa"]
        for qa in qas:
            question = qa["query"]
            options = qa["options"]
            concat_question = f"{question}\n{options[0]}\n{options[1]}\n{options[2]}\n{options[3]}"
            qa["question"] = concat_question
        
        video_info["qa"] = qas
            
        return video_info
    
    def get_video(self, video_id: Union[int, str]):
        video_info = self.get_video_info(video_id)
        video_path = video_info["video_path"]
        work_path = os.path.join(self.work_path, f"{video_id}")
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        
        return VideoRepresentation(video_path, work_path)