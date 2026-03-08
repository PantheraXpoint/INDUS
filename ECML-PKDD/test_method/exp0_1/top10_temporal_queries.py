#!/usr/bin/env python3
"""One-off: top-10 queries by count of seed events with borda_score null (descending)."""
import json
from pathlib import Path

ECML = Path(__file__).resolve().parent.parent.parent  # ECML-PKDD
seed_path = ECML / "ava100_retrieval" / "seed_events_AVA100.json"
if not seed_path.exists():
    seed_path = ECML / "lvbench_retrieval" / "seed_events_LVBench.json"

data = json.loads(seed_path.read_text())
rows = []
for entry in data:
    vk = entry.get("video_key")
    qid = entry.get("question_id")
    events = entry.get("seed_events") or []
    n_null = sum(1 for e in events if e.get("borda_score") is None)
    rows.append((vk, qid, n_null))

rows.sort(key=lambda x: -x[2])
print("Top-10 queries by number of seed events with borda_score null (descending):")
print("-" * 60)
for i, (vk, qid, n) in enumerate(rows[:10], 1):
    print(f"  {i:2}. video_key={vk!r}  question_id={qid}  count={n}")
