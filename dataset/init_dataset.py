from dataset.lvbench import LVBench
from dataset.videomme import VideoMME
from dataset.ava100 import AVA100
from dataset.single_video import SingleVideo

dataset_zoo = {
    "lvbench": LVBench,
    "videomme": VideoMME,
    "ava100": AVA100,
    "single_video": SingleVideo
}

video_idx_zoo = {
    "lvbench": [1, 103],
    "videomme": [601, 900],
    "ava100": [1, 8]
}

def get_video_idx(dataset_name):
    if dataset_name not in video_idx_zoo:
        supported_datasets = ", ".join(video_idx_zoo.keys())
        raise ValueError(f"Dataset {dataset_name} not found in video_idx_zoo. Supported datasets: {supported_datasets}")
    return video_idx_zoo[dataset_name]

def init_dataset(dataset_name, video_path=None):
    if dataset_name not in dataset_zoo:
        supported_datasets = ", ".join(dataset_zoo.keys())
        raise ValueError(f"Dataset {dataset_name} not found in dataset_zoo. Supported datasets: {supported_datasets}")
    
    # Handle different dataset initialization parameters
    if dataset_name == "single_video":
        return dataset_zoo[dataset_name](video_path=video_path)
    else:
        # For other datasets (lvbench, videomme, ava100), use default parameters
        return dataset_zoo[dataset_name]()