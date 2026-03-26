#!/usr/bin/env python3
"""
Top-k event retrieval from the events VDB using LLM keyword extraction.
Input: same as ECML-PKDD/run_indus_prompts (datas/<dataset>, AVA_cache/<dataset>/.../kg).
Output: seed_events-style JSON with average_latency_seconds_per_query.
"""
import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from AVA.prompt import PROMPTS
from AVA.storage import TextNanoVectorDBStorage
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


def _load_events_vdb(kg_dir: str, embedding_model, embedding_dim: int = 768):
    global_config = {
        "working_dir": kg_dir,
        "embedding_batch_num": 64,
        "cosine_better_than_threshold": 0.1,
    }
    return TextNanoVectorDBStorage(
        namespace="events",
        global_config=global_config,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        meta_fields={"id", "name", "description", "duration"},
    )


def main():
    parser = argparse.ArgumentParser(description="Top-k event retrieval (keyword → events VDB)")
    parser.add_argument("--dataset", choices=["AVA100", "LVBench"], required=True)
    parser.add_argument("--top-k", type=int, default=80, help="Number of events to retrieve per query")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: top_k_events_retrieval/seed_events_{dataset}.json)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    base_dir = project_root / "datas" / args.dataset
    if not base_dir.exists():
        raise FileNotFoundError(f"Query directory not found: {base_dir}")

    kg_mapping = _build_ava100_kg_mapping() if args.dataset == "AVA100" else _build_lvbench_kg_mapping()
    queries = list(load_ava100_queries(base_dir) if args.dataset == "AVA100" else load_lvbench_queries(base_dir))

    llm = init_model("qwenvl_vllm", num_gpus=1, model_type=args.model, port=args.port)
    embedding_model = JinaCLIP("jinaai/jina-clip-v1")
    embedding_dim = embedding_model.embedding_dim

    out_path = Path(args.output) if args.output else project_root / "top_k_events_retrieval" / f"seed_events_{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    latencies = []

    for idx, (video_key, question_id, question, options_str) in enumerate(queries):
        kg_dir = kg_mapping.get(video_key)

        # For LVBench, the original data may contain both question and options
        # in a single string field. Split on the first '?' to separate them.
        question_for_query = (question or "").strip()
        options_for_output = options_str
        if args.dataset == "LVBench":
            combined = (question or "").strip()
            if "?" in combined:
                q_part, rest = combined.split("?", 1)
                question_for_query = (q_part + "?").strip()
                options_for_output = rest.strip()
            else:
                question_for_query = combined
                options_for_output = (options_str or "").strip()

        if not kg_dir:
            results.append({
                "dataset": args.dataset,
                "video_key": video_key,
                "question_id": question_id,
                "question": question_for_query,
                "options": options_for_output,
                "skip_reason": "no_kg",
                "seed_event_ids": [],
                "seed_events": [],
                "llm_outputs": {},
            })
            continue

        try:
            # Time: load VDB + embedding + retrieve + result assembly (exclude file writes)
            t0 = time.perf_counter()
            events_vdb = _load_events_vdb(kg_dir, embedding_model, embedding_dim)
            keyword_str = question_for_query
            event_results = events_vdb.query(keyword_str, top_k=args.top_k)
            seed_event_ids = [e["id"] for e in event_results if e.get("id")]
            seed_events = []
            for e in event_results:
                ev = {
                    "__id__": e.get("id", ""),
                    "id": e.get("id", ""),
                    "description": e.get("description", ""),
                    "duration": e.get("duration", []),
                }
                if e.get("__metrics__") is not None:
                    ev["retrieval_score"] = float(e["__metrics__"])
                seed_events.append(ev)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
        except Exception as ex:
            seed_event_ids = []
            seed_events = []
            keyword_str = ""
            print(f"Error for {video_key} q{question_id}: {ex}")

        results.append({
            "dataset": args.dataset,
            "video_key": video_key,
            "question_id": question_id,
            "question": question_for_query,
            "options": options_for_output,
            "localization_time_seconds": None,
            "llm_outputs": {"keyword_extraction": keyword_str if kg_dir else ""},
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
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path} (avg latency {avg_latency:.4f}s per query)")


if __name__ == "__main__":
    main()
