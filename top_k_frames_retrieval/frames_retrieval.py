#!/usr/bin/env python3
"""
Top-k frame retrieval from the features VDB (query rewrite → embed → VDB).
Output: seed_events-style JSON; events that contain the top-k frames (id, description, duration only);
per-query frame_scores JSON and actual frame images in a dedicated directory.
Records average_latency_seconds_per_query.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from AVA.prompt import PROMPTS
from AVA.storage import ImageNanoVectorDBStorage
from embeddings.JinaCLIP import JinaCLIP
from llms.init_model import init_model


def parse_lvbench_question_options(full_question: str):
    full_question = (full_question or "").strip()
    if not full_question:
        return "", ""
    lines = full_question.split("\n")
    question_parts = []
    options_parts = []
    in_options = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("(A)") or stripped.startswith("(a)"):
            in_options = True
        if in_options:
            options_parts.append(stripped)
        else:
            question_parts.append(stripped)
    return "\n".join(question_parts).strip(), "\n".join(options_parts).strip()


def load_ava100_queries(base_dir: Path):
    for name in ["citytour", "ego", "traffic", "wildlife"]:
        path = base_dir / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for video in data:
            video_key = video.get("video_key", "")
            for qa in video.get("qa", []):
                question = qa.get("query", "")
                opts = qa.get("options", [])
                options_str = "\n".join(opts) if opts else ""
                question_id = qa.get("question_id")
                yield video_key, question_id, question, options_str


def load_lvbench_queries(base_dir: Path):
    path = base_dir / "LVBench.json"
    data = json.loads(path.read_text())
    for video in data:
        video_key = video.get("key") or video.get("video_key", "")
        for qa in video.get("qa", []):
            full = qa.get("question") or qa.get("query", "")
            question, options_str = parse_lvbench_question_options(full)
            question_id = qa.get("uid") or qa.get("question_id", "")
            yield video_key, question_id, question, options_str


def _build_ava100_kg_mapping():
    base_db = project_root / "AVA_cache" / "AVA100"
    mapping = {}
    for video_index in range(1, 9):
        config_path = base_db / str(video_index) / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
            source_path = config.get("source_path", "")
            if source_path:
                video_key = Path(source_path).stem
                kg_dir = base_db / str(video_index) / "kg"
                if (kg_dir / "vdb_events.json").exists() and (kg_dir / "vdb_entities.json").exists() and (kg_dir / "vdb_features.json").exists():
                    mapping[video_key] = str(kg_dir)
        except Exception:
            pass
    return mapping


def _build_lvbench_kg_mapping():
    base_db = project_root / "AVA_cache" / "LVBench"
    mapping = {}
    if not base_db.exists():
        return mapping
    for folder in base_db.iterdir():
        if not folder.is_dir():
            continue
        config_path = folder / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
            source_path = config.get("source_path", "")
            if source_path:
                video_key = Path(source_path).stem
                kg_dir = folder / "kg"
                if (kg_dir / "vdb_events.json").exists() and (kg_dir / "vdb_entities.json").exists() and (kg_dir / "vdb_features.json").exists():
                    mapping[video_key] = str(kg_dir)
        except Exception:
            pass
    return mapping


def _load_events_data(kg_dir: str):
    """Load events from vdb_events.json; return list of event dicts (id, description, duration)."""
    path = Path(kg_dir) / "vdb_events.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data.get("data", [])
    except Exception:
        return []


def _load_features_vdb(kg_dir: str, embedding_model, embedding_dim: int = 768):
    global_config = {
        "working_dir": kg_dir,
        "embedding_batch_num": 100,
        "cosine_better_than_threshold": 0.1,
    }
    return ImageNanoVectorDBStorage(
        namespace="features",
        global_config=global_config,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        meta_fields={"id", "frame_dir", "event"},
    )


def _event_id_to_info(events_data: list) -> dict:
    """Build map event_id -> {id, description, duration} (no score)."""
    out = {}
    for ev in events_data:
        eid = ev.get("id") or ev.get("__id__")
        if not eid:
            continue
        out[eid] = {
            "__id__": eid,
            "id": eid,
            "description": ev.get("description", ""),
            "duration": ev.get("duration", []),
        }
    return out


def _resolve_frame_path(frame_dir: str) -> Path:
    """Resolve frame path: if relative, from project_root."""
    p = Path(frame_dir)
    if not p.is_absolute():
        p = project_root / p
    return p


def main():
    parser = argparse.ArgumentParser(description="Top-k frame retrieval (query rewrite → features VDB)")
    parser.add_argument("--dataset", choices=["AVA100", "LVBench"], required=True)
    parser.add_argument("--top-k", type=int, default=256, help="Number of frames to retrieve per query")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: top_k_frames_retrieval/out_<dataset>)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--no-copy-frames", action="store_true", help="Skip copying frame images to output")
    args = parser.parse_args()

    base_dir = project_root / "datas" / args.dataset
    if not base_dir.exists():
        raise FileNotFoundError(f"Query directory not found: {base_dir}")

    kg_mapping = _build_ava100_kg_mapping() if args.dataset == "AVA100" else _build_lvbench_kg_mapping()
    queries = list(load_ava100_queries(base_dir) if args.dataset == "AVA100" else load_lvbench_queries(base_dir))

    llm = init_model("qwenvl_vllm", num_gpus=1, model_type=args.model, port=args.port)
    embedding_model = JinaCLIP("jinaai/jina-clip-v1")
    embedding_dim = embedding_model.embedding_dim

    out_dir = Path(args.output_dir) if args.output_dir else project_root / "top_k_frames_retrieval" / f"out_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_outputs_dir = out_dir / "frame_outputs"
    frame_outputs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    latencies = []

    for idx, (video_key, question_id, question, options_str) in enumerate(queries):
        kg_dir = kg_mapping.get(video_key)
        question_with_options = f"{question}\n{options_str}".strip() if options_str else question

        if not kg_dir:
            results.append({
                "dataset": args.dataset,
                "video_key": video_key,
                "question_id": question_id,
                "question": question,
                "options": options_str,
                "localization_time_seconds": None,
                "skip_reason": "no_kg",
                "llm_outputs": {},
                "seed_event_ids": [],
                "seed_events": [],
            })
            continue

        seed_event_ids = []
        seed_events = []
        rewritten_query = ""
        frame_scores = []
        frame_results = []
        query_slug = f"{video_key}_{question_id}"

        try:
            # Time: load VDB + load events + LLM call + embedding + retrieve + result assembly (exclude file writes)
            t0 = time.perf_counter()
            events_data = _load_events_data(kg_dir)
            id_to_event = _event_id_to_info(events_data)
            features_vdb = _load_features_vdb(kg_dir, embedding_model, embedding_dim)
            rewrite_prompt = PROMPTS["query_rewrite_for_visual_retrieval"].format(input_text=question_with_options)
            rewritten_query = llm.batch_generate_response([{"text": rewrite_prompt}])[0]
            rewritten_query = (rewritten_query or "").strip()
            frame_results = features_vdb.query(rewritten_query, top_k=args.top_k)
            frame_scores = [{"rank": r + 1, "score": float(f["__metrics__"])} for r, f in enumerate(frame_results)]
            event_ids_containing_frames = []
            for f in frame_results:
                eid = f.get("event")
                if eid and eid not in event_ids_containing_frames:
                    event_ids_containing_frames.append(eid)
            seed_event_ids = event_ids_containing_frames
            for eid in seed_event_ids:
                if eid in id_to_event:
                    seed_events.append(id_to_event[eid])
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
        except Exception as ex:
            print(f"Error for {video_key} q{question_id}: {ex}")

        # Store results (not included in latency)
        if kg_dir and not args.no_copy_frames and frame_results:
            try:
                query_frames_dir = frame_outputs_dir / query_slug
                query_frames_dir.mkdir(parents=True, exist_ok=True)
                for rank, f in enumerate(frame_results, start=1):
                    frame_dir = f.get("frame_dir")
                    if not frame_dir:
                        continue
                    src = _resolve_frame_path(frame_dir)
                    if src.exists():
                        shutil.copy2(src, query_frames_dir / f"{rank}.jpg")
            except Exception:
                pass
        if kg_dir and frame_scores is not None:
            try:
                scores_path = frame_outputs_dir / query_slug / "scores.json"
                scores_path.parent.mkdir(parents=True, exist_ok=True)
                with open(scores_path, "w") as sf:
                    json.dump(frame_scores, sf, indent=2)
            except Exception:
                pass

        results.append({
            "dataset": args.dataset,
            "video_key": video_key,
            "question_id": question_id,
            "question": question,
            "options": options_str,
            "localization_time_seconds": None,
            "llm_outputs": {"query_rewrite_for_visual_retrieval": rewritten_query if kg_dir else ""},
            "seed_event_ids": seed_event_ids,
            "seed_events": seed_events,
        })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(queries)}")

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    output = {
        "average_latency_seconds_per_query": round(avg_latency, 4),
        "num_queries": len(queries),
        "results": results,
    }
    out_json = out_dir / f"seed_events_{args.dataset}.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_json} and frame_outputs under {out_dir} (avg latency {avg_latency:.4f}s per query)")


if __name__ == "__main__":
    main()
