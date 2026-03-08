#!/usr/bin/env python3
"""
Expand INDUS seed events using two expansion methods:
1. forward_backward: Temporal expansion (previous/next events)
2. evt_obj_evt: Event → Object → Event (2-hop via objects)

Input: seed_events_{dataset}.json
Output: 
  - seed_events_{dataset}_expanded_{mode}_{hops}.json (deduplicated events)
  - seed_events_{dataset}_expansion_trace_{mode}_{hops}.json (full expansion paths)

Expansion uses only VDB (events, entities); the graph is not loaded or used.
Event→entity mapping is built by inverting entity["events"] in entities_vdb.

Note (LVBench): If expansion results equal the original seeds, check:
  1) KG not found: AVA_cache/LVBench must exist; each folder's config.json should have
     source_path (or video_path) whose filename stem matches seed file "video_key".
  2) No new events: seed_event_ids must come from the same KG build (same event IDs as
     in that video's vdb_events.json); otherwise prev/next and graph lookups find nothing.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from AVA.storage import TextNanoVectorDBStorage
from embeddings.JinaCLIP import JinaCLIP


def initialize_vdbs(kg_dir: str, embedding_model, embedding_dim: int = 768):
    """Initialize VDBs and graph storage directly (no KnowledgeGraphInterface)."""
    global_config = {
        "working_dir": kg_dir,
        "embedding_batch_num": 64,
        "cosine_better_than_threshold": 0.1,
    }
    
    events_vdb = TextNanoVectorDBStorage(
        namespace="events",
        global_config=global_config,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        meta_fields={"id", "name", "description", "duration"},
    )
    
    entities_vdb = TextNanoVectorDBStorage(
        namespace="entities",
        global_config=global_config,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        meta_fields={"id", "descriptions", "timestamps", "frame_indices", "durations", "events"},
    )
    
    print(f"  [OK] VDBs loaded from {kg_dir} (graph not used)")
    return events_vdb, entities_vdb, None


def _build_event_to_entities(entities_vdb) -> Dict[str, List[str]]:
    """Build event_id -> [entity_ids] by inverting entity['events'] in entities_vdb (VDB-only, no graph)."""
    event_to_entities = defaultdict(list)
    try:
        for data in entities_vdb.get_datas():
            entity_id = data.get("id") or data.get("__id__")
            if not entity_id:
                continue
            for eid in data.get("events") or []:
                if eid:
                    event_to_entities[eid].append(entity_id)
    except Exception as e:
        print("  [Warning] Failed to build event-to-entities index: ", e)
    return dict(event_to_entities)


def resolve_kg_dir(project_root: Path, video_key: str, dataset_entry: str, verbose: bool = True) -> Optional[str]:
    """Resolve KG directory for a video_key and dataset. Returns path to kg dir or None."""
    kg_dir = None
    if dataset_entry == "AVA100":
        base_db = project_root / "AVA_cache" / "AVA100"
        for video_index in range(1, 9):
            config_path = base_db / str(video_index) / "config.json"
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text())
                    source_path = config.get("source_path", "")
                    if source_path and Path(source_path).stem == video_key:
                        kg_dir = str(base_db / str(video_index) / "kg")
                        break
                except Exception:
                    pass
    elif dataset_entry == "LVBench":
        base_db = project_root / "AVA_cache" / "LVBench"
        if not base_db.exists():
            if verbose:
                print(f"  [Warning] LVBench cache not found at {base_db}")
        else:
            for folder in base_db.iterdir():
                if not folder.is_dir():
                    continue
                config_path = folder / "config.json"
                if not config_path.exists():
                    continue
                try:
                    config = json.loads(config_path.read_text())
                    source_path = config.get("source_path") or config.get("video_path") or ""
                    stem = Path(source_path).stem if source_path else ""
                    if stem == video_key or folder.name == video_key:
                        kg_dir = str(folder / "kg")
                        break
                except Exception:
                    pass
            if not kg_dir and verbose:
                try:
                    stems = []
                    for f in base_db.iterdir():
                        if f.is_dir():
                            cp = f / "config.json"
                            if cp.exists():
                                c = json.loads(cp.read_text())
                                sp = c.get("source_path") or c.get("video_path") or ""
                                stems.append((f.name, Path(sp).stem if sp else "(no path)"))
                    print(f"  [Debug] LVBench folders seen (name, source_path stem): {stems[:10]}{'...' if len(stems) > 10 else ''}")
                except Exception:
                    pass
    if kg_dir and not Path(kg_dir).exists():
        kg_dir = None
    return kg_dir


def expand_forward_backward(seed_event_ids: Set[str], events_vdb) -> Set[str]:
    """
    Expand events forward and backward (temporal).
    For each seed event, get previous and next events.
    Uses a set so each event ID is added at most once (no duplication from prev/next).
    """
    expanded = set()  # set ensures no duplicate event IDs
    for event_id in seed_event_ids:
        try:
            prev_event = events_vdb.get_previous_data(event_id)
            if prev_event:
                prev_id = prev_event.get("__id__") or prev_event.get("id")
                if prev_id:
                    expanded.add(prev_id)
                    # print(f"    [Forward/Backward] {event_id} -> previous: {prev_id}")
        except Exception as e:
            print(f"    [Warning] Failed to get previous for {event_id}: {e}")
        try:
            next_event = events_vdb.get_next_data(event_id)
            if next_event:
                next_id = next_event.get("__id__") or next_event.get("id")
                if next_id:
                    expanded.add(next_id)
                    # print(f"    [Forward/Backward] {event_id} -> next: {next_id}")
        except Exception as e:
            print(f"    [Warning] Failed to get next for {event_id}: {e}")
    return expanded


def expand_evt_obj_evt(seed_event_ids: Set[str], entities_vdb, event_to_entities: Dict[str, List[str]]) -> Set[str]:
    """
    Expand events via objects (Event -> Object -> Event) using VDB only.
    For each seed event, get objects from event_to_entities (inverted from entity['events']),
    then get events containing those objects from entities_vdb.
    Uses a set for expanded so each event ID is added at most once (no duplication across objects).
    """
    expanded = set()  # set ensures no duplicate event IDs even if same event reached via multiple objects
    for event_id in seed_event_ids:
        try:
            object_ids = event_to_entities.get(event_id) or []
            if not object_ids:
                continue
            print(f"    [Event->Object->Event] {event_id} -> {len(object_ids)} objects (VDB)")
            for obj_id in object_ids:
                try:
                    obj_data = entities_vdb.get_data(obj_id)
                    if obj_data and "events" in obj_data:
                        events_list = obj_data["events"]
                        if isinstance(events_list, list):
                            for evt_id in events_list:
                                if evt_id and evt_id != event_id:
                                    expanded.add(evt_id)
                except Exception as e:
                    print(f"      [Warning] Failed to get events for object {obj_id}: {e}")
        except Exception as e:
            print(f"    [Warning] Failed to expand event {event_id}: {e}")
    return expanded


def expand_events_multi_hop(
    seed_event_ids: List[str],
    expansion_func,
    num_hops: int,
    events_vdb=None,
    entities_vdb=None,
    event_to_entities: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Set[str], Dict[str, Any]]:
    """
    Apply expansion function for multiple hops.
    Returns: (all_expanded_event_ids, trace_structure)
    
    Trace structure:
    {
      "seed_event_id": {
        "hop_1": {
          "expanded_event_id_1": {},
          "expanded_event_id_2": {}
        },
        "hop_2": {
          "expanded_event_id_1": {
            "next_event_1": {},
            "next_event_2": {}
          }
        }
      }
    }
    """
    trace = {}
    current_pool = set(seed_event_ids)
    all_expanded = set()
    
    # Track which events belong to which seed (for trace building)
    event_to_seed = {seed_id: seed_id for seed_id in seed_event_ids}
    
    # Initialize trace for each seed event
    for seed_id in seed_event_ids:
        trace[seed_id] = {}
    
    for hop in range(1, num_hops + 1):
        print(f"  Hop {hop}: Expanding from {len(current_pool)} events...")
        hop_expanded = set()
        hop_trace = {}  # event_id -> list of expanded event_ids
        
        # Expand from current pool
        for event_id in current_pool:
            # Prepare arguments based on expansion function
            if expansion_func == expand_forward_backward:
                new_events = expansion_func({event_id}, events_vdb)
            elif expansion_func == expand_evt_obj_evt:
                new_events = expansion_func({event_id}, entities_vdb, event_to_entities or {})
            else:
                new_events = set()
            
            # Remove events already in pool (deduplication for this hop)
            new_events = new_events - current_pool
            
            if new_events:
                hop_expanded.update(new_events)
                hop_trace[event_id] = list(new_events)
                print(f"    {event_id} → {len(new_events)} new events")
                
                # Track which seed these new events belong to
                seed_id = event_to_seed.get(event_id)
                if seed_id:
                    for new_evt_id in new_events:
                        event_to_seed[new_evt_id] = seed_id
            else:
                hop_trace[event_id] = []
        
        # Update trace structure (nested)
        for seed_id in seed_event_ids:
            hop_key = f"hop_{hop}"
            if hop == 1:
                # First hop: seed events expand directly
                if seed_id in hop_trace:
                    trace[seed_id][hop_key] = {
                        evt_id: {} for evt_id in hop_trace[seed_id]
                    }
            else:
                # Subsequent hops: expand from events in previous hop
                # Collect all events from previous hop for this seed
                prev_hop_key = f"hop_{hop-1}"
                if prev_hop_key in trace[seed_id]:
                    # Recursively collect all event IDs from previous hop
                    def collect_events(d):
                        events = set()
                        for k, v in d.items():
                            events.add(k)
                            if isinstance(v, dict) and v:  # Non-empty dict
                                events.update(collect_events(v))
                        return events
                    
                    prev_events = collect_events(trace[seed_id][prev_hop_key])
                    
                    # Find expansions from these previous events
                    hop_expansions = {}
                    for prev_evt_id in prev_events:
                        if prev_evt_id in hop_trace and hop_trace[prev_evt_id]:
                            hop_expansions[prev_evt_id] = {
                                evt_id: {} for evt_id in hop_trace[prev_evt_id]
                            }
                    
                    if hop_expansions:
                        trace[seed_id][hop_key] = hop_expansions
        
        # Update pool for next hop
        current_pool.update(hop_expanded)
        all_expanded.update(hop_expanded)
        
        if not hop_expanded:
            print(f"  Hop {hop}: No new events found. Stopping expansion.")
            break
    
    return all_expanded, trace


def _event_ids_in_vdb(events_vdb) -> Set[str]:
    """Return set of event IDs that exist in the events VDB (avoids get_data index error for missing IDs)."""
    try:
        return {
            (d.get("id") or d.get("__id__"))
            for d in events_vdb.get_datas()
            if d.get("id") or d.get("__id__")
        }
    except Exception:
        return set()


def fetch_event_data(event_ids: Set[str], events_vdb) -> List[Dict[str, Any]]:
    """Fetch full event data from VDB for given event IDs. Only IDs present in the VDB are fetched."""
    events = []
    for event_id in event_ids:
        try:
            event_data = events_vdb.get_data(event_id)
            if event_data:
                # Create serializable dict (remove vector, clear borda_score)
                payload = {}
                for k, v in event_data.items():
                    if k in ("__vector__", "__metrics__", "content"):
                        continue
                    if hasattr(v, "tolist"):
                        payload[k] = v.tolist()
                    else:
                        payload[k] = v
                
                # Clear borda_score for expanded events
                if "borda_score" in payload:
                    payload["borda_score"] = None
                
                events.append(payload)
        except Exception as e:
            print(f"    [Warning] Failed to fetch event {event_id}: {e}")
    
    return events
