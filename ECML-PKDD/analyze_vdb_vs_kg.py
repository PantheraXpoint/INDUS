#!/usr/bin/env python3
"""
Compare event–object connectivity when derived from VDB vs from the knowledge graph.

Purpose: Explain why evt_obj_evt expansion with VDB edges yields much higher total event
duration than with the graph. The script:
  1) Connectivity report: builds event→entities from VDB vs graph, simulates one-hop EOE size.
  2) Expansion comparison (optional, with --seed-file): runs actual expansion from seed events
     in two ways — (a) VDB: event→entities from entity['events'], entity→events from entities_vdb;
     (b) KG: event→entities and entity→events from the graph (event/entity node neighbors).
     Compares total event count and total event duration per query to see why they differ.

Usage:
  python analyze_vdb_vs_kg.py --cache AVA_cache --dataset AVA100 [--out-dir ...]
  python analyze_vdb_vs_kg.py --cache AVA_cache --dataset AVA100 --seed-file ECML-PKDD/.../seed_events_AVA100.json [--out-dir ...]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from AVA.storage import TextNanoVectorDBStorage, NetworkXStorage
from embeddings.JinaCLIP import JinaCLIP


def load_vdbs(kg_dir: str, embedding_model, embedding_dim: int = 768) -> Tuple[Any, Any]:
    """Load events and entities VDBs from kg_dir."""
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
    return events_vdb, entities_vdb


def load_graph(kg_dir: str) -> Optional[Any]:
    """Load NetworkX graph from kg_dir (event_knowledge_graph). Returns None if no graph file."""
    global_config = {"working_dir": kg_dir}
    try:
        storage = NetworkXStorage(
            namespace="event_knowledge_graph",
            global_config=global_config,
        )
        if storage._graph.number_of_nodes() == 0:
            return None
        return storage
    except Exception:
        return None


def build_event_to_entities_vdb(entities_vdb) -> Dict[str, List[str]]:
    """Event -> [entity_ids] from entities_vdb (invert entity['events'])."""
    out = defaultdict(list)
    for data in entities_vdb.get_datas():
        entity_id = data.get("id") or data.get("__id__")
        if not entity_id:
            continue
        for eid in data.get("events") or []:
            if eid:
                out[eid].append(entity_id)
    return dict(out)


def build_event_to_entities_kg(graph_storage) -> Dict[str, List[str]]:
    """Event -> [entity_ids] from graph: neighbors of event node with type 'entity'."""
    out = defaultdict(list)
    if graph_storage is None:
        return {}
    for event_id in list(graph_storage._graph.nodes):
        if not event_id.startswith("Event-"):
            continue
        edges = graph_storage.get_node_edges(event_id)
        if not edges:
            continue
        for u, v in edges:
            other = v if u == event_id else u
            node_data = graph_storage.get_node(other)
            if node_data and node_data.get("type") == "entity":
                out[event_id].append(other)
    return dict(out)


def build_entity_to_events_kg(graph_storage) -> Dict[str, List[str]]:
    """Entity -> [event_ids] from graph: neighbors of entity node that are Event-* (for KG-only EOE expansion)."""
    out = defaultdict(list)
    if graph_storage is None:
        return {}
    for node_id in list(graph_storage._graph.nodes):
        if not graph_storage.has_node(node_id):
            continue
        node_data = graph_storage.get_node(node_id)
        if not node_data or node_data.get("type") != "entity":
            continue
        edges = graph_storage.get_node_edges(node_id)
        if not edges:
            continue
        for u, v in edges:
            other = v if u == node_id else u
            if other.startswith("Event-"):
                out[node_id].append(other)
    return dict(out)


def events_in_vdb(events_vdb) -> Set[str]:
    """Set of event IDs present in events VDB."""
    try:
        return {
            (d.get("id") or d.get("__id__"))
            for d in events_vdb.get_datas()
            if d.get("id") or d.get("__id__")
        }
    except Exception:
        return set()


def count_expanded_events_one_hop(
    event_id: str,
    entity_ids: List[str],
    entities_vdb,
    events_vdb_valid: Set[str],
) -> int:
    """Given event_id and its object (entity) IDs, count how many distinct other events we'd add (evt->obj->evt), only in VDB."""
    expanded = set()
    for obj_id in entity_ids:
        try:
            obj_data = entities_vdb.get_data(obj_id)
            if not obj_data or "events" not in obj_data:
                continue
            for eid in obj_data.get("events") or []:
                if eid and eid != event_id and eid in events_vdb_valid:
                    expanded.add(eid)
        except Exception:
            continue
    return len(expanded)


# ---- Expansion from seed events (VDB vs KG) ----

def expand_evt_obj_evt_vdb(
    seed_event_ids: Set[str],
    entities_vdb,
    event_to_entities_vdb: Dict[str, List[str]],
    valid_event_ids: Set[str],
) -> Set[str]:
    """One-hop EOE expansion using VDB: event→entities from event_to_entities_vdb, entity→events from entities_vdb. Only IDs in valid_event_ids."""
    expanded = set()
    for event_id in seed_event_ids:
        object_ids = event_to_entities_vdb.get(event_id) or []
        for obj_id in object_ids:
            try:
                obj_data = entities_vdb.get_data(obj_id)
                if not obj_data or "events" not in obj_data:
                    continue
                for eid in obj_data.get("events") or []:
                    if eid and eid != event_id and eid in valid_event_ids:
                        expanded.add(eid)
            except Exception:
                continue
    return expanded


def expand_evt_obj_evt_kg(
    seed_event_ids: Set[str],
    event_to_entities_kg: Dict[str, List[str]],
    entity_to_events_kg: Dict[str, List[str]],
    valid_event_ids: Set[str],
) -> Set[str]:
    """One-hop EOE expansion using graph only: event→entities and entity→events from KG. Only IDs in valid_event_ids."""
    expanded = set()
    for event_id in seed_event_ids:
        object_ids = event_to_entities_kg.get(event_id) or []
        for obj_id in object_ids:
            for eid in entity_to_events_kg.get(obj_id) or []:
                if eid and eid != event_id and eid in valid_event_ids:
                    expanded.add(eid)
    return expanded


def event_duration_frames(event_data: Dict[str, Any]) -> float:
    """Return event duration as a scalar (frames or seconds). duration can be [start, end] or a number."""
    d = event_data.get("duration")
    if d is None:
        return 0.0
    if isinstance(d, (list, tuple)) and len(d) >= 2:
        return float(d[1] - d[0])
    try:
        return float(d)
    except (TypeError, ValueError):
        return 0.0


def total_duration_and_count(event_ids: Set[str], events_vdb) -> Tuple[float, int]:
    """Sum of event durations (frames) and count of events successfully fetched. Only IDs present in VDB are used."""
    total = 0.0
    count = 0
    for eid in event_ids:
        try:
            data = events_vdb.get_data(eid)
            if data:
                total += event_duration_frames(data)
                count += 1
        except Exception:
            continue
    return total, count


def resolve_kg_dir(video_key: str, dataset: str, cache_root: Path) -> Optional[str]:
    """Resolve video_key to kg directory path (same logic as expand_indus_seed_events). Returns None if not found."""
    base_db = cache_root / dataset
    if not base_db.exists():
        return None
    if dataset == "AVA100":
        for video_index in range(1, 9):
            config_path = base_db / str(video_index) / "config.json"
            if not config_path.exists():
                continue
            try:
                config = json.loads(config_path.read_text())
                source_path = config.get("source_path", "")
                if source_path and Path(source_path).stem == video_key:
                    kg_dir = str(base_db / str(video_index) / "kg")
                    if Path(kg_dir).exists():
                        return kg_dir
                    return None
            except Exception:
                pass
        return None
    # LVBench
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
                if Path(kg_dir).exists():
                    return kg_dir
                return None
        except Exception:
            pass
    return None


def run_expansion_comparison_for_entry(
    entry: Dict[str, Any],
    events_vdb,
    entities_vdb,
    graph_storage: Optional[Any],
    event_to_entities_vdb: Dict[str, List[str]],
    event_to_entities_kg: Dict[str, List[str]],
    entity_to_events_kg: Dict[str, List[str]],
    valid_event_ids: Set[str],
) -> Dict[str, Any]:
    """Run one-hop EOE expansion from seed_event_ids using VDB and KG; return per-query comparison (counts and total duration)."""
    seed_ids = set(entry.get("seed_event_ids") or [])
    if not seed_ids:
        return {
            "video_key": entry.get("video_key"),
            "question_id": entry.get("question_id"),
            "skip": True,
            "reason": "no_seed_event_ids",
        }
    seed_ids = seed_ids & valid_event_ids

    expanded_vdb = expand_evt_obj_evt_vdb(seed_ids, entities_vdb, event_to_entities_vdb, valid_event_ids)
    expanded_kg = set()
    if graph_storage is not None:
        expanded_kg = expand_evt_obj_evt_kg(seed_ids, event_to_entities_kg, entity_to_events_kg, valid_event_ids)

    all_ids_vdb = seed_ids | expanded_vdb
    all_ids_kg = seed_ids | expanded_kg

    duration_vdb, count_vdb = total_duration_and_count(all_ids_vdb, events_vdb)
    duration_kg, count_kg = total_duration_and_count(all_ids_kg, events_vdb)

    return {
        "video_key": entry.get("video_key"),
        "question_id": entry.get("question_id"),
        "skip": False,
        "n_seed": len(seed_ids),
        "n_expanded_vdb": len(expanded_vdb),
        "n_expanded_kg": len(expanded_kg),
        "n_total_vdb": len(all_ids_vdb),
        "n_total_kg": len(all_ids_kg),
        "total_duration_vdb": round(duration_vdb, 2),
        "total_duration_kg": round(duration_kg, 2),
        "duration_diff_vdb_minus_kg": round(duration_vdb - duration_kg, 2),
        "count_vdb": count_vdb,
        "count_kg": count_kg,
    }


def collect_kg_dirs(cache_root: Path, dataset: str) -> List[Tuple[str, Path]]:
    """Return list of (video_id_or_key, kg_path) for the dataset."""
    base = cache_root / dataset
    if not base.exists():
        return []
    result = []
    if dataset == "AVA100":
        for i in range(1, 9):
            kg = base / str(i) / "kg"
            if kg.exists() and (kg / "vdb_events.json").exists():
                result.append((str(i), kg))
    else:
        # LVBench: any subdir with kg/
        for d in base.iterdir():
            if not d.is_dir():
                continue
            kg = d / "kg"
            if kg.exists() and (kg / "vdb_events.json").exists():
                result.append((d.name, kg))
    return result


def run_analysis_for_kg(
    video_id: str,
    kg_path: Path,
    dataset: str,
    events_vdb,
    entities_vdb,
    graph_storage: Optional[Any],
) -> Dict[str, Any]:
    """Compare VDB vs KG event–entity links and one-hop expansion size for one video."""
    event_to_entities_vdb = build_event_to_entities_vdb(entities_vdb)
    event_to_entities_kg = build_event_to_entities_kg(graph_storage)
    valid_event_ids = events_in_vdb(events_vdb)

    # Restrict to events that exist in events VDB (apples to apples)
    events_common = valid_event_ids

    # Per-event comparison
    n_events = len(events_common)
    total_links_vdb = 0
    total_links_kg = 0
    events_with_more_vdb = 0
    events_with_more_kg = 0
    diff_counts = []  # (n_vdb - n_kg) per event
    total_expanded_vdb = 0
    total_expanded_kg = 0

    for event_id in events_common:
        ent_vdb = set(event_to_entities_vdb.get(event_id) or [])
        ent_kg = set(event_to_entities_kg.get(event_id) or [])
        n_vdb = len(ent_vdb)
        n_kg = len(ent_kg)
        total_links_vdb += n_vdb
        total_links_kg += n_kg
        if n_vdb > n_kg:
            events_with_more_vdb += 1
        elif n_kg > n_vdb:
            events_with_more_kg += 1
        diff_counts.append(n_vdb - n_kg)

        # Simulate one-hop expansion size (how many events we'd add)
        expanded_vdb = count_expanded_events_one_hop(
            event_id, list(ent_vdb), entities_vdb, valid_event_ids
        )
        expanded_kg = count_expanded_events_one_hop(
            event_id, list(ent_kg), entities_vdb, valid_event_ids
        )
        total_expanded_vdb += expanded_vdb
        total_expanded_kg += expanded_kg

    avg_links_vdb = total_links_vdb / n_events if n_events else 0
    avg_links_kg = total_links_kg / n_events if n_events else 0
    avg_diff = sum(diff_counts) / len(diff_counts) if diff_counts else 0
    avg_expanded_vdb = total_expanded_vdb / n_events if n_events else 0
    avg_expanded_kg = total_expanded_kg / n_events if n_events else 0

    return {
        "video_id": video_id,
        "dataset": dataset,
        "kg_path": str(kg_path),
        "n_events_in_vdb": n_events,
        "has_graph": graph_storage is not None,
        "total_event_entity_links_vdb": total_links_vdb,
        "total_event_entity_links_kg": total_links_kg,
        "avg_entities_per_event_vdb": round(avg_links_vdb, 2),
        "avg_entities_per_event_kg": round(avg_links_kg, 2),
        "events_with_more_links_in_vdb": events_with_more_vdb,
        "events_with_more_links_in_kg": events_with_more_kg,
        "avg_diff_entities_per_event_vdb_minus_kg": round(avg_diff, 2),
        "total_one_hop_expanded_events_vdb": total_expanded_vdb,
        "total_one_hop_expanded_events_kg": total_expanded_kg,
        "avg_one_hop_expanded_per_event_vdb": round(avg_expanded_vdb, 2),
        "avg_one_hop_expanded_per_event_kg": round(avg_expanded_kg, 2),
        "ratio_avg_expanded_vdb_over_kg": round(avg_expanded_vdb / avg_expanded_kg, 2) if avg_expanded_kg else None,
    }


def print_and_save_report(
    all_results: List[Dict],
    out_dir: Path,
    expansion_results: Optional[List[Dict]] = None,
) -> None:
    """Print summary and per-video stats; save JSON and text report. If expansion_results, append expansion comparison section."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("VDB vs KG: EVENT–OBJECT CONNECTIVITY AND EOE EXPANSION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("This report compares event→entity links from (1) VDB = invert entity['events']")
    lines.append("vs (2) KG = event node neighbors with type 'entity', and simulates one-hop expansion size.")
    lines.append("When the graph is built from the same pipeline (AVA/operate.py), VDB and KG are identical,")
    lines.append("so ratio 1.0 is expected. Use this to verify alignment or to compare different cache builds.")
    lines.append("")

    for r in all_results:
        vid = r["video_id"]
        ds = r["dataset"]
        lines.append("-" * 80)
        lines.append(f"  {ds} / video {vid}")
        lines.append("-" * 80)
        lines.append(f"  Events in VDB: {r['n_events_in_vdb']}")
        lines.append(f"  Has graph: {r['has_graph']}")
        lines.append(f"  Total event–entity links (VDB): {r['total_event_entity_links_vdb']}")
        lines.append(f"  Total event–entity links (KG):  {r['total_event_entity_links_kg']}")
        lines.append(f"  Avg entities per event (VDB): {r['avg_entities_per_event_vdb']}")
        lines.append(f"  Avg entities per event (KG):  {r['avg_entities_per_event_kg']}")
        lines.append(f"  Events with more links in VDB than KG: {r['events_with_more_links_in_vdb']}")
        lines.append(f"  Events with more links in KG than VDB: {r['events_with_more_links_in_kg']}")
        lines.append(f"  Avg diff (VDB − KG) entities per event: {r['avg_diff_entities_per_event_vdb_minus_kg']}")
        lines.append("  One-hop evt→obj→evt expansion (simulated):")
        lines.append(f"    Total expanded events (VDB): {r['total_one_hop_expanded_events_vdb']}")
        lines.append(f"    Total expanded events (KG):  {r['total_one_hop_expanded_events_kg']}")
        lines.append(f"    Avg expanded per seed event (VDB): {r['avg_one_hop_expanded_per_event_vdb']}")
        lines.append(f"    Avg expanded per seed event (KG):  {r['avg_one_hop_expanded_per_event_kg']}")
        if r.get("ratio_avg_expanded_vdb_over_kg") is not None:
            lines.append(f"    Ratio (VDB/KG) avg expanded per event: {r['ratio_avg_expanded_vdb_over_kg']}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("SUMMARY (aggregate over all videos)")
    lines.append("=" * 80)
    if all_results:
        n_videos = len(all_results)
        total_links_vdb = sum(r["total_event_entity_links_vdb"] for r in all_results)
        total_links_kg = sum(r["total_event_entity_links_kg"] for r in all_results)
        total_exp_vdb = sum(r["total_one_hop_expanded_events_vdb"] for r in all_results)
        total_exp_kg = sum(r["total_one_hop_expanded_events_kg"] for r in all_results)
        avg_per_event_vdb = sum(r["avg_one_hop_expanded_per_event_vdb"] for r in all_results) / n_videos
        avg_per_event_kg = sum(r["avg_one_hop_expanded_per_event_kg"] for r in all_results) / n_videos
        lines.append(f"  Videos: {n_videos}")
        lines.append(f"  Total event–entity links (VDB): {total_links_vdb}")
        lines.append(f"  Total event–entity links (KG):  {total_links_kg}")
        lines.append(f"  Total one-hop expanded events (VDB): {total_exp_vdb}")
        lines.append(f"  Total one-hop expanded events (KG):  {total_exp_kg}")
        lines.append(f"  Avg one-hop expanded per seed event (VDB): {avg_per_event_vdb:.2f}")
        lines.append(f"  Avg one-hop expanded per seed event (KG):  {avg_per_event_kg:.2f}")
        if avg_per_event_kg > 0:
            ratio = avg_per_event_vdb / avg_per_event_kg
            lines.append(f"  Ratio VDB/KG (avg expanded per event): {ratio:.2f}")
        # Interpretation when ratio is 1.0
        all_identical = all(
            r.get("total_event_entity_links_vdb") == r.get("total_event_entity_links_kg")
            for r in all_results
        )
        if all_results and all_identical:
            lines.append("")
            lines.append("INTERPRETATION (ratio ≈ 1.0):")
            lines.append("  The graph is built from the same entity['events'] data as the VDB (see AVA/operate.py).")
            lines.append("  So event→entity links are identical by construction; ratio 1.0 is expected.")
            lines.append("  If you observed higher total event duration with VDB-based EOE in real runs, possible")
            lines.append("  causes: (1) different cache/build when the graph was used, (2) different expansion")
            lines.append("  options (e.g. --max-entities-per-event caps), or (3) different seed sets.")
    lines.append("")

    # Expansion-from-seed comparison (when --seed-file was used)
    if expansion_results:
        lines.append("")
        lines.append("=" * 80)
        lines.append("EXPANSION FROM SEED EVENTS: VDB vs KG (one-hop evt→obj→evt)")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Each query: seed events expanded with (1) VDB: event→entity from entity['events'], entity→events from entities_vdb;")
        lines.append("(2) KG: event→entity and entity→events from graph neighbors. Total duration = sum of event durations (frames) over unique events.")
        lines.append("")
        skipped = [r for r in expansion_results if r.get("skip")]
        reported = [r for r in expansion_results if not r.get("skip")]
        for r in reported:
            lines.append(f"  {r.get('video_key')} Q{r.get('question_id')}: n_seed={r.get('n_seed')} | VDB: expanded={r.get('n_expanded_vdb')} total_events={r.get('n_total_vdb')} duration={r.get('total_duration_vdb')} | KG: expanded={r.get('n_expanded_kg')} total_events={r.get('n_total_kg')} duration={r.get('total_duration_kg')} | duration_diff(VDB−KG)={r.get('duration_diff_vdb_minus_kg')}")
        if skipped:
            lines.append("")
            lines.append(f"  Skipped {len(skipped)} entries: {[r.get('reason', '?') for r in skipped[:5]]}{'...' if len(skipped) > 5 else ''}")
        if reported:
            lines.append("")
            lines.append("  Summary (expansion comparison):")
            n_q = len(reported)
            avg_dur_vdb = sum(r["total_duration_vdb"] for r in reported) / n_q
            avg_dur_kg = sum(r["total_duration_kg"] for r in reported) / n_q
            avg_diff = sum(r["duration_diff_vdb_minus_kg"] for r in reported) / n_q
            lines.append(f"    Queries: {n_q} | Avg total duration (VDB): {avg_dur_vdb:.2f} | Avg total duration (KG): {avg_dur_kg:.2f} | Avg duration diff (VDB−KG): {avg_diff:.2f}")
        lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    report_file = out_dir / "vdb_vs_kg_report.txt"
    report_file.write_text(report_text, encoding="utf-8")
    print(f"Report saved: {report_file}")

    json_file = out_dir / "vdb_vs_kg_results.json"
    payload = {"per_video": all_results}
    if expansion_results is not None:
        payload["expansion_per_query"] = expansion_results
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON saved: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare event–object connectivity from VDB vs graph to explain EOE expansion duration difference."
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="AVA_cache",
        help="Cache root (default: AVA_cache)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="AVA100",
        choices=["AVA100", "LVBench"],
        help="Dataset (default: AVA100)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="ECML-PKDD/vdb_vs_kg_output",
        help="Output directory for report and JSON (default: ECML-PKDD/vdb_vs_kg_output)",
    )
    parser.add_argument(
        "--seed-file",
        type=str,
        default=None,
        help="Optional seed events JSON (same format as expand_indus_seed_events). Run expansion from seeds with VDB vs KG and compare total duration per query.",
    )
    args = parser.parse_args()

    cache_root = project_root / args.cache
    kg_dirs = collect_kg_dirs(cache_root, args.dataset)
    if not kg_dirs:
        print(f"No KG directories found under {cache_root / args.dataset}")
        return

    print("Loading embedding model...")
    embedding_model = JinaCLIP("jinaai/jina-clip-v1")
    all_results = []

    for video_id, kg_path in kg_dirs:
        print(f"Processing {args.dataset} / {video_id} ...")
        try:
            events_vdb, entities_vdb = load_vdbs(str(kg_path), embedding_model)
            graph_storage = load_graph(str(kg_path))
            r = run_analysis_for_kg(
                video_id,
                kg_path,
                args.dataset,
                events_vdb,
                entities_vdb,
                graph_storage,
            )
            all_results.append(r)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    expansion_results = None
    if args.seed_file:
        seed_path = Path(args.seed_file)
        if not seed_path.exists():
            print(f"Seed file not found: {seed_path}")
        else:
            print(f"Loading seed file: {seed_path}")
            with open(seed_path, "r", encoding="utf-8") as f:
                seed_data = json.load(f)
            if not isinstance(seed_data, list):
                print("Seed file must be a list of entries.")
            else:
                expansion_results = []
                current_video_key = None
                events_vdb = entities_vdb = graph_storage = None
                event_to_entities_vdb = event_to_entities_kg = entity_to_events_kg = {}
                valid_event_ids = set()

                for idx, entry in enumerate(seed_data):
                    video_key = entry.get("video_key")
                    if video_key != current_video_key:
                        current_video_key = video_key
                        kg_dir = resolve_kg_dir(video_key, args.dataset, cache_root)
                        if not kg_dir or not Path(kg_dir).exists():
                            events_vdb = entities_vdb = graph_storage = None
                            event_to_entities_vdb = event_to_entities_kg = entity_to_events_kg = {}
                            valid_event_ids = set()
                            expansion_results.append({
                                "video_key": video_key,
                                "question_id": entry.get("question_id"),
                                "skip": True,
                                "reason": "no_kg_dir",
                            })
                            continue
                        try:
                            events_vdb, entities_vdb = load_vdbs(kg_dir, embedding_model)
                            graph_storage = load_graph(kg_dir)
                            valid_event_ids = events_in_vdb(events_vdb)
                            event_to_entities_vdb = build_event_to_entities_vdb(entities_vdb)
                            event_to_entities_kg = build_event_to_entities_kg(graph_storage)
                            entity_to_events_kg = build_entity_to_events_kg(graph_storage)
                        except Exception as e:
                            events_vdb = entities_vdb = graph_storage = None
                            event_to_entities_vdb = event_to_entities_kg = entity_to_events_kg = {}
                            valid_event_ids = set()
                            expansion_results.append({
                                "video_key": video_key,
                                "question_id": entry.get("question_id"),
                                "skip": True,
                                "reason": f"load_error: {e}",
                            })
                            continue

                    if events_vdb is None:
                        expansion_results.append({
                            "video_key": video_key,
                            "question_id": entry.get("question_id"),
                            "skip": True,
                            "reason": "no_kg_dir",
                        })
                        continue

                    comp = run_expansion_comparison_for_entry(
                        entry,
                        events_vdb,
                        entities_vdb,
                        graph_storage,
                        event_to_entities_vdb,
                        event_to_entities_kg,
                        entity_to_events_kg,
                        valid_event_ids,
                    )
                    expansion_results.append(comp)

                print(f"Expansion comparison: {len([r for r in expansion_results if not r.get('skip')])} queries processed.")

    if not all_results and not expansion_results:
        print("No results to report.")
    else:
        print_and_save_report(all_results, Path(args.out_dir), expansion_results=expansion_results)
    print("Done.")


if __name__ == "__main__":
    main()
