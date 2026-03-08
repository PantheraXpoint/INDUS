#!/usr/bin/env python3
"""
Check alignment between VDB (vdb_events.json, vdb_entities.json) and the knowledge graph
(graph_event_knowledge_graph.graphml or .graphml.xml).

Reports per video:
  - events/entities in VDB but not in the graph
  - events/entities in the graph but not in VDB

Supports AVA100 and LVBench under AVA_cache/.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# GraphML namespace (ElementTree uses full URI in braces)
GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"

# Graph file may be named .graphml or .graphml.xml (e.g. LVBench)
GRAPH_FILENAMES = ["graph_event_knowledge_graph.graphml", "graph_event_knowledge_graph.graphml.xml"]


def get_graph_path(kg_path: Path) -> Optional[Path]:
    """Return path to the graph file if it exists (tries both .graphml and .graphml.xml)."""
    for name in GRAPH_FILENAMES:
        p = kg_path / name
        if p.exists():
            return p
    return None


def load_json_file(file_path: Path) -> dict:
    """Load JSON (supports single-line large files)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_graphml_ids(graph_path: Path) -> Tuple[Set[str], Set[str]]:
    """
    Parse graph_event_knowledge_graph.graphml and return event_ids and entity_ids (node IDs only).
    """
    event_ids = set()
    entity_ids = set()

    tree = ET.parse(graph_path)
    root = tree.getroot()
    ns = GRAPHML_NS

    def find_all(elem, local_tag: str) -> List:
        return elem.findall(f".//{{{ns}}}{local_tag}")

    def find_data(elem, key: str) -> Optional[str]:
        for d in elem.findall(f"{{{ns}}}data"):
            if d.get("key") == key and d.text:
                return d.text
        return None

    for node in find_all(root, "node"):
        nid = node.get("id") or find_data(node, "d1")
        if not nid:
            continue
        if nid.startswith("Event-"):
            event_ids.add(nid)
        elif nid.startswith("Entity-"):
            entity_ids.add(nid)

    # Include IDs that appear only as edge endpoints
    for edge in find_all(root, "edge"):
        src = edge.get("source")
        tgt = edge.get("target")
        if src and src.startswith("Event-"):
            event_ids.add(src)
        elif src and src.startswith("Entity-"):
            entity_ids.add(src)
        if tgt and tgt.startswith("Event-"):
            event_ids.add(tgt)
        elif tgt and tgt.startswith("Entity-"):
            entity_ids.add(tgt)

    return event_ids, entity_ids


def get_vdb_ids(kg_path: Path) -> Tuple[Set[str], Set[str]]:
    """Return (event_ids from vdb_events.json, entity_ids from vdb_entities.json)."""
    event_ids = set()
    entity_ids = set()
    events_file = kg_path / "vdb_events.json"
    entities_file = kg_path / "vdb_entities.json"
    if events_file.exists():
        data = load_json_file(events_file)
        for item in data.get("data", []):
            eid = item.get("id") or item.get("__id__")
            if eid:
                event_ids.add(eid)
    if entities_file.exists():
        data = load_json_file(entities_file)
        for item in data.get("data", []):
            eid = item.get("id") or item.get("__id__")
            if eid:
                entity_ids.add(eid)
    return event_ids, entity_ids


def check_alignment(
    video_id: str,
    kg_path: Path,
    dataset_name: str,
) -> Optional[Dict]:
    """
    For one video: load graph (if present) + VDB, compare IDs and return alignment counts/samples.
    If no graph file exists, returns a result with has_graph=False and zero alignment counts.
    """
    vdb_events, vdb_entities = get_vdb_ids(kg_path)
    graph_path = get_graph_path(kg_path)

    if graph_path is None:
        return {
            "video_id": video_id,
            "dataset": dataset_name,
            "has_graph": False,
            "num_events_vdb": len(vdb_events),
            "num_entities_vdb": len(vdb_entities),
            "num_events_graph": 0,
            "num_entities_graph": 0,
            "alignment": {
                "events_only_in_vdb": 0,
                "events_only_in_graph": 0,
                "entities_only_in_vdb": 0,
                "entities_only_in_graph": 0,
                "event_ids_only_vdb": [],
                "event_ids_only_graph": [],
                "entity_ids_only_vdb": [],
                "entity_ids_only_graph": [],
            },
        }

    graph_events, graph_entities = parse_graphml_ids(graph_path)
    events_only_vdb = vdb_events - graph_events
    events_only_graph = graph_events - vdb_events
    entities_only_vdb = vdb_entities - graph_entities
    entities_only_graph = graph_entities - vdb_entities

    return {
        "video_id": video_id,
        "dataset": dataset_name,
        "has_graph": True,
        "num_events_vdb": len(vdb_events),
        "num_entities_vdb": len(vdb_entities),
        "num_events_graph": len(graph_events),
        "num_entities_graph": len(graph_entities),
        "alignment": {
            "events_only_in_vdb": len(events_only_vdb),
            "events_only_in_graph": len(events_only_graph),
            "entities_only_in_vdb": len(entities_only_vdb),
            "entities_only_in_graph": len(entities_only_graph),
            "event_ids_only_vdb": list(events_only_vdb)[:20],
            "event_ids_only_graph": list(events_only_graph)[:20],
            "entity_ids_only_vdb": list(entities_only_vdb)[:20],
            "entity_ids_only_graph": list(entities_only_graph)[:20],
        },
    }


def collect_video_dirs(base_path: Path) -> List[str]:
    """Return sorted list of numeric video directory names that have kg/ with graph or VDB data."""
    if not base_path.exists():
        return []
    ids = []
    for d in base_path.iterdir():
        if not d.is_dir() or not d.name.isdigit():
            continue
        kg = d / "kg"
        if not kg.exists():
            continue
        if get_graph_path(kg) is not None:
            ids.append(d.name)
        elif (kg / "vdb_events.json").exists() or (kg / "vdb_entities.json").exists():
            ids.append(d.name)
    return sorted(ids, key=int)


def run_analysis(cache_root: Path, datasets: List[str]) -> Dict:
    """Run alignment check for given datasets under cache_root."""
    cache_root = Path(cache_root)
    all_results = []
    for dataset_name in datasets:
        base = cache_root / dataset_name
        for vid in collect_video_dirs(base):
            kg_path = base / vid / "kg"
            r = check_alignment(vid, kg_path, dataset_name)
            if r:
                all_results.append(r)
    return {"per_video": all_results, "datasets": datasets}


def print_report(results: Dict, out_dir: Path) -> None:
    """Print and save alignment report (text + JSON)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("VDB–GRAPH ALIGNMENT REPORT")
    lines.append("=" * 80)

    for r in results["per_video"]:
        vid = r["video_id"]
        ds = r["dataset"]
        lines.append(f"\n--- {ds} / video {vid} ---")
        lines.append(f"  VDB: events={r['num_events_vdb']}, entities={r['num_entities_vdb']}")
        if r.get("has_graph", True):
            lines.append(f"  Graph: events={r['num_events_graph']}, entities={r['num_entities_graph']}")
            al = r["alignment"]
            lines.append(
                f"  Alignment: events only_vdb={al['events_only_in_vdb']}, only_graph={al['events_only_in_graph']}; "
                f"entities only_vdb={al['entities_only_in_vdb']}, only_graph={al['entities_only_in_graph']}"
            )
            if (
                al["events_only_in_vdb"]
                or al["events_only_in_graph"]
                or al["entities_only_in_vdb"]
                or al["entities_only_in_graph"]
            ):
                if al["event_ids_only_vdb"]:
                    lines.append(f"    Sample events only in VDB: {al['event_ids_only_vdb'][:5]}")
                if al["event_ids_only_graph"]:
                    lines.append(f"    Sample events only in graph: {al['event_ids_only_graph'][:5]}")
                if al["entity_ids_only_vdb"]:
                    lines.append(f"    Sample entities only in VDB: {al['entity_ids_only_vdb'][:5]}")
                if al["entity_ids_only_graph"]:
                    lines.append(f"    Sample entities only in graph: {al['entity_ids_only_graph'][:5]}")
        else:
            lines.append("  Graph: no graph file found (skipped alignment)")

    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    total_ev_vdb = sum(r["alignment"]["events_only_in_vdb"] for r in results["per_video"])
    total_ev_gr = sum(r["alignment"]["events_only_in_graph"] for r in results["per_video"])
    total_ent_vdb = sum(r["alignment"]["entities_only_in_vdb"] for r in results["per_video"])
    total_ent_gr = sum(r["alignment"]["entities_only_in_graph"] for r in results["per_video"])
    lines.append(f"Total events only in VDB (across all videos): {total_ev_vdb}")
    lines.append(f"Total events only in graph: {total_ev_gr}")
    lines.append(f"Total entities only in VDB: {total_ent_vdb}")
    lines.append(f"Total entities only in graph: {total_ent_gr}")

    report_text = "\n".join(lines)
    print(report_text)
    report_file = out_dir / "alignment_report.txt"
    report_file.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved: {report_file}")

    # JSON summary (no sample ID lists)
    export = []
    for r in results["per_video"]:
        ex = {k: v for k, v in r.items() if k != "alignment"}
        al = r["alignment"].copy()
        for k in ("event_ids_only_vdb", "event_ids_only_graph", "entity_ids_only_vdb", "entity_ids_only_graph"):
            al.pop(k, None)
        ex["alignment"] = al
        export.append(ex)
    json_path = out_dir / "alignment_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"per_video": export, "datasets": results["datasets"]}, f, indent=2)
    print(f"JSON summary saved: {json_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check alignment between VDB and knowledge graph (GraphML)")
    parser.add_argument("--cache", type=str, default="AVA_cache", help="Cache root (default: AVA_cache)")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["AVA100", "LVBench"],
        help="Datasets to analyze (default: AVA100 LVBench)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="ECML-PKDD/kg_alignment_output",
        help="Output directory for report (default: ECML-PKDD/kg_alignment_output)",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache)
    results = run_analysis(cache_root, args.datasets)
    if not results["per_video"]:
        print("No videos with kg data found.")
        return

    print_report(results, Path(args.out))
    print("\nDone.")


if __name__ == "__main__":
    main()
