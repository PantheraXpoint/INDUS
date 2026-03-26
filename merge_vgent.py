from typing import Any


import json
from pathlib import Path
seed_events_paths = [Path("/home/panthera/avas/Project-Ava/ECML-PKDD/ava100_retrieval/indus_explore/seed_events_vgent_3.json"), Path("top_k_events_retrieval/old/seed_events_AVA100_k60.json")]
merged_seed_events = []

data = []
for entry in seed_events_paths:
    data_entries = json.loads(entry.read_text())
    if isinstance(data_entries, dict) and data_entries.get("results"):
        data_entries = data_entries["results"]
    for idx, data_entry in enumerate(data_entries):
        if (
            len(data) > idx
            and data[idx].get("video_key") == data_entry.get("video_key")
            and data[idx].get("question_id") == data_entry.get("question_id")
        ):
            data[idx]["seed_events"].extend(data_entry.get("seed_events", []))
        else:
            data.append(data_entry)
json.dump(data, open("merged_seed_events.json", "w"), indent=2)