"""Paths and constants for temporal clustering experiment."""
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ECML = SCRIPT_DIR.parent
PROJECT_ROOT = ECML.parent

SEED_PATHS = [
    ECML / "ava100_analysis" / "seed_events_AVA100.json",
    # ECML / "lvbench_retrieval" / "seed_events_LVBench.json",
]
TOP_K = [20]
THRESHOLD_T = [0, 20, 40, 60, 80, 100, 120]
# GT_BASE = PROJECT_ROOT / "datas" / "LVBench"
GT_BASE = PROJECT_ROOT / "datas" / "AVA100"
# GT_FILES = ["LVBench.json"]
GT_FILES = ["citytour.json", "ego.json", "traffic.json", "wildlife.json"]

# (k, T) used for cluster-boundary expansion (forward_backward / evt_obj_evt)
EXPAND_K = 20
EXPAND_T = 0
DATASET = "AVA100"
# DATASET = "LVBench"
