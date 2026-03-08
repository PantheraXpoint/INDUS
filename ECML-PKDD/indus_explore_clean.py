#!/usr/bin/env python3
"""
Budgeted INDUS exploration: Batch 1 only (base ∪ FB(base)).

Pipeline per query:
  Batch 1: base seeds (top-K Borda) ∪ FB(base). Send combined set to VLM → verified_batch1.
           exploration_history = base ∪ FB(base).

  Per-batch budget: Batch 1 may send at most total_budget events to the VLM.

Final output = verified_batch1.

Modes:
  - No VLM (default): no LLM calls; every candidate is treated as verified. Evaluation uses
    exactly the events sent (up to budget) for Batch 1.
  - VLM (--use-vlm): Prompt 1 (query_event_extraction) once per query; Prompt 2
    (event_relevance_verification) per candidate. Verified set = subset passing
    the chosen rule (any/majority/all yes). Evaluation uses this smaller verified set.

Reads:
  - ECML-PKDD/{dataset}_retrieval/seed_events_{dataset}.json

Outputs under ECML-PKDD/{dataset}_retrieval/indus_explore/:
  - seed_events_{dataset}_indus_explore_final.json
"""

import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional


# ── path setup ────────────────────────────────────────────────────────────────────

_ecml_dir = Path(__file__).resolve().parent          # ECML-PKDD
project_root = _ecml_dir.parent                     # repo root (Project-Ava)
sys.path.insert(0, str(_ecml_dir))
sys.path.insert(0, str(project_root))

from expand_indus_seed_events import (  # type: ignore[attr-defined]
    initialize_vdbs,
    _build_event_to_entities,
    resolve_kg_dir,
    expand_forward_backward,
    _event_ids_in_vdb,
    fetch_event_data,
)

try:
    import indus_prompts  # used only when --use-vlm
except ImportError:
    indus_prompts = None  # type: ignore[assignment]


# ── types ─────────────────────────────────────────────────────────────────────────

VideoKey = str
QuestionId = int
QueryKey = Tuple[VideoKey, QuestionId]


# ── helpers: seed selection and sorting ──────────────────────────────────────────

def _sort_key_event(e: Dict[str, Any]) -> Tuple[int, float]:
    """Sort seed events: by borda_score descending; None last."""
    score = e.get("borda_score")
    if score is None:
        return (1, 0.0)
    try:
        return (0, -float(score))
    except (TypeError, ValueError):
        return (1, 0.0)


def select_top_k_borda_seed_ids(entry: Dict[str, Any], top_k: int) -> List[str]:
    """
    Top-K Borda-only seeds (no temporal). Deduplicated by event id.
    Change: borda_score == None is treated as MAX score (kept + ranked highest).
    """
    seed_events = entry.get("seed_events") or []
    if not seed_events or top_k <= 0:
        return []

    def _borda_score_max_none(ev: Dict[str, Any]) -> float:
        s = ev.get("borda_score")
        if s is None:
            return float("inf")
        try:
            return float(s)
        except Exception:
            return float("-inf")

    sorted_scored = sorted(
        seed_events,
        key=lambda ev: (-_borda_score_max_none(ev), _sort_key_event(ev)),
    )

    seen_ids: Set[str] = set()
    out: List[str] = []
    for ev in sorted_scored:
        eid = ev.get("id") or ev.get("__id__")
        if not eid or eid in seen_ids:
            continue
        seen_ids.add(eid)
        out.append(eid)
        if len(out) >= top_k:
            break
    return out


# ── VLM verification (no-VLM stub vs real LLM) ────────────────────────────────────

def vlm_verify_events_stub(
    stage: str,
    video_key: str,
    question_id: int,
    question: str,
    candidate_event_ids: List[str],
) -> Set[str]:
    """No-VLM mode: return all candidates as verified (no LLM call)."""
    _ = (stage, video_key, question_id, question)
    return set(candidate_event_ids)


def _parse_query_events_from_llm(llm_output: str) -> List[str]:
    out = (llm_output or "").strip()
    if not out:
        return []
    out = re.sub(r"^```\w*\n?", "", out).strip()
    out = re.sub(r"\n?```$", "", out).strip()
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    events = data.get("query_events")
    if isinstance(events, list):
        return [str(e).strip() for e in events if e]
    return []


def _format_query_events_for_prompt(query_events: List[str]) -> str:
    return "\n".join(f"{i+1}. {e}" for i, e in enumerate(query_events))


def _get_event_description(events_vdb, event_id: str) -> str:
    try:
        data = events_vdb.get_data(event_id)
    except (IndexError, KeyError):
        return ""
    if not data:
        return ""
    desc = data.get("description") or data.get("name") or ""
    if isinstance(desc, list):
        desc = " ".join(str(d) for d in desc)
    dur = data.get("duration")
    if isinstance(dur, (list, tuple)) and len(dur) >= 2:
        try:
            prefix = f"[{float(dur[0])}-{float(dur[1])}] "
            return prefix + str(desc).strip()
        except (TypeError, ValueError):
            pass
    return str(desc).strip() or event_id


def _parse_relevance_response(llm_output: str, num_expected: int) -> List[str]:
    out = (llm_output or "").strip()
    if not out:
        return ["no"] * num_expected
    out = re.sub(r"^```\w*\n?", "", out).strip()
    out = re.sub(r"\n?```$", "", out).strip()
    try:
        arr = json.loads(out)
    except json.JSONDecodeError:
        return ["no"] * num_expected
    if not isinstance(arr, list):
        return ["no"] * num_expected
    result = []
    for v in arr[:num_expected]:
        s = (str(v).strip().lower() if v is not None else "no")
        result.append("yes" if s == "yes" else "no")
    while len(result) < num_expected:
        result.append("no")
    return result[:num_expected]


def _verify_batch_with_vlm(
    llm: Any,
    query_events: List[str],
    candidate_event_ids: List[str],
    events_vdb: Any,
    stage: str,
    verify_rule: str = "any",
    max_batch_size: int = 64,
    out_matrix: Optional[Dict[str, Any]] = None,
) -> Set[str]:
    if not query_events or not candidate_event_ids:
        if out_matrix is not None:
            out_matrix["candidate_ids"] = list(candidate_event_ids)
            out_matrix["matrix"] = []
            out_matrix["verified_ids"] = []
        return set(candidate_event_ids) if verify_rule == "any" else set()

    query_events_formatted = _format_query_events_for_prompt(query_events)
    M = len(query_events)
    verified: Set[str] = set()
    matrix_rows: List[List[str]] = []

    prompts: List[str] = []
    for eid in candidate_event_ids:
        desc = _get_event_description(events_vdb, eid)
        prompt = indus_prompts.INDUS_PROMPT["event_relevance_verification"].format(
            query_events_formatted=query_events_formatted,
            candidate_event_description=desc,
        )
        prompts.append(prompt)

    for start in range(0, len(prompts), max_batch_size):
        end = min(start + max_batch_size, len(prompts))
        batch_inputs = [{"text": p} for p in prompts[start:end]]
        batch_outputs = llm.batch_generate_response(batch_inputs)
        for i, raw in enumerate(batch_outputs):
            idx = start + i
            if idx >= len(candidate_event_ids):
                break
            eid = candidate_event_ids[idx]
            row = _parse_relevance_response(raw, M)
            matrix_rows.append(row)
            if verify_rule == "any":
                if any(r == "yes" for r in row):
                    verified.add(eid)
            elif verify_rule == "majority":
                if sum(1 for r in row if r == "yes") > M // 2:
                    verified.add(eid)
            else:  # "all"
                if all(r == "yes" for r in row):
                    verified.add(eid)

    if out_matrix is not None:
        out_matrix["candidate_ids"] = list(candidate_event_ids)
        out_matrix["matrix"] = matrix_rows
        out_matrix["verified_ids"] = sorted(verified)

    return verified


# ── core per-query pipeline ──────────────────────────────────────────────────────

def process_query(
    entry: Dict[str, Any],
    dataset: str,
    events_vdb,
    valid_event_ids: Set[str],
    top_k_seeds: int,
    total_budget: int,
    use_vlm: bool = False,
    llm: Any = None,
    verify_rule: str = "any",
) -> Dict[str, Any]:
    video_key = entry.get("video_key")
    question_id = entry.get("question_id")
    question = entry.get("question", "")
    options = entry.get("options", "")

    if video_key is None or question_id is None:
        raise ValueError("Entry must contain video_key and question_id")
    try:
        qid_int: int = int(question_id)
    except (TypeError, ValueError):
        qid_int = -1

    # Prompt 1 (optional, VLM mode)
    query_events: List[str] = []
    if use_vlm and llm is not None and indus_prompts is not None:
        prompt_p1 = indus_prompts.INDUS_PROMPT["query_event_extraction"].format(
            question=question,
            options=options or "",
        )
        out_p1 = llm.batch_generate_response([{"text": prompt_p1}])[0]
        query_events = _parse_query_events_from_llm(out_p1)
        if not query_events:
            query_events = ["Relevant to the question or options"]

    vlm_trace_batches: List[Dict[str, Any]] = []

    def _verify_batch(candidate_event_ids: List[str], stage: str) -> Set[str]:
        if use_vlm and llm is not None and query_events and candidate_event_ids:
            trace_entry: Dict[str, Any] = {"stage": stage}
            verified = _verify_batch_with_vlm(
                llm=llm,
                query_events=query_events,
                candidate_event_ids=candidate_event_ids,
                events_vdb=events_vdb,
                stage=stage,
                verify_rule=verify_rule,
                out_matrix=trace_entry,
            )
            vlm_trace_batches.append(trace_entry)
            return verified

        return vlm_verify_events_stub(
            stage=stage,
            video_key=str(video_key),
            question_id=qid_int,
            question=f"{question}\n{options}",
            candidate_event_ids=candidate_event_ids,
        )

    # Batch 1: seeds ∪ FB(seeds)
    seed_ids = select_top_k_borda_seed_ids(entry, top_k_seeds)
    verified_batch1: Set[str] = set()

    if seed_ids:
        fb_all = expand_forward_backward(set(seed_ids), events_vdb) & valid_event_ids
        fb_candidates = fb_all - set(seed_ids)
        batch1_ids = set(seed_ids) | fb_candidates
        batch1_to_verify = list(seed_ids) + sorted(batch1_ids - set(seed_ids))

        if len(batch1_to_verify) > total_budget:
            batch1_to_verify = batch1_to_verify[:total_budget]

        if batch1_to_verify:
            verified_batch1 = _verify_batch(batch1_to_verify, "batch1")

    final_ids: Set[str] = verified_batch1 & valid_event_ids
    final_events = fetch_event_data(final_ids, events_vdb)

    final_entry = {
        "dataset": dataset,
        "video_key": video_key,
        "question_id": question_id,
        "question": question,
        "options": options,
        "localization_time_seconds": entry.get("localization_time_seconds"),
        "seed_event_ids": sorted(final_ids),
        "seed_events": final_events,
        "indus_explore_meta": {
            "top_k_seeds": top_k_seeds,
            "total_budget": total_budget,
            "verified_batch1_count": len(verified_batch1),
            "final_event_count": len(final_ids),
        },
    }

    if use_vlm and vlm_trace_batches:
        final_entry["vlm_verification_trace"] = {
            "query_events": query_events,
            "verify_rule": verify_rule,
            "batches": vlm_trace_batches,
        }

    return final_entry


# ── main dataset-level driver ────────────────────────────────────────────────────

def process_dataset(
    dataset: str,
    top_k_seeds: int,
    total_budget: int,
    use_vlm: bool = False,
    llm_port: int = 8000,
    llm_model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ",
    verify_rule: str = "any",
) -> None:
    assert dataset in ("AVA100", "LVBench")

    llm = None
    if use_vlm:
        if indus_prompts is None:
            raise RuntimeError("indus_prompts not available; cannot use --use-vlm")
        from llms.init_model import init_model
        llm = init_model("qwenvl_vllm", num_gpus=1, model_type=llm_model, port=llm_port)
        print(f"VLM mode: LLM loaded (port={llm_port}, model={llm_model}), verify_rule={verify_rule}")
    else:
        print("No-VLM mode: all candidates treated as verified (evaluation = budget-sized sets)")

    retrieval_dir = _ecml_dir / f"{dataset.lower()}_retrieval"
    seed_path = retrieval_dir / f"seed_events_{dataset}.json"
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_path}")

    seed_data = json.loads(seed_path.read_text())
    if not isinstance(seed_data, list):
        raise ValueError(f"Expected list in {seed_path}, got {type(seed_data)}")

    out_dir = retrieval_dir / "indus_explore"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_entries: List[Dict[str, Any]] = []

    current_video_key: Optional[str] = None
    events_vdb = entities_vdb = None
    valid_event_ids: Set[str] = set()

    for idx, entry in enumerate(seed_data):
        video_key = entry.get("video_key")
        question_id = entry.get("question_id")
        dataset_entry = entry.get("dataset", dataset)
        print(f"\n[{idx + 1}/{len(seed_data)}] {dataset} video={video_key} Q{question_id}")

        if video_key is None:
            print("  [Warning] Missing video_key; skipping")
            continue

        if video_key != current_video_key:
            current_video_key = video_key
            kg_dir = resolve_kg_dir(project_root, str(video_key), dataset_entry)
            if not kg_dir:
                print(f"  [Warning] KG directory not found for video_key={video_key!r}; skipping.")
                continue
            print(f"  Loading VDBs from {kg_dir}")
            from embeddings.JinaCLIP import JinaCLIP

            embedding_model = JinaCLIP("jinaai/jina-clip-v1")
            events_vdb, entities_vdb, _ = initialize_vdbs(kg_dir, embedding_model)
            valid_event_ids = _event_ids_in_vdb(events_vdb)

        if not events_vdb:
            print("  [Warning] VDBs not initialized; skipping entry.")
            continue

        try:
            final_entry = process_query(
                entry=entry,
                dataset=dataset,
                events_vdb=events_vdb,
                valid_event_ids=valid_event_ids,
                top_k_seeds=top_k_seeds,
                total_budget=total_budget,
                use_vlm=use_vlm,
                llm=llm,
                verify_rule=verify_rule,
            )
        except Exception as e:
            print(f"  [Error] Failed to process query: {e}")
            import traceback
            traceback.print_exc()
            continue

        final_entries.append(final_entry)

    final_out = out_dir / f"seed_events_{dataset}_indus_explore_final.json"
    with final_out.open("w") as f:
        json.dump(final_entries, f, indent=2)
    print(f"\n✓ Wrote final outputs: {final_out} ({len(final_entries)} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Budgeted INDUS exploration (Batch1 only) writing final.json"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="AVA100",
        choices=["AVA100", "LVBench"],
        help="Dataset name (default: AVA100).",
    )
    parser.add_argument(
        "--top-k-seeds",
        type=int,
        default=20,
        help="Top-K Borda seeds per query (initial seeds, default: 20).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Per-batch VLM budget: Batch 1 may send at most this many events to the VLM (required).",
    )
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="Enable VLM verification: run query_event_extraction (Prompt 1) and event_relevance_verification (Prompt 2). Default: no VLM (all candidates treated as verified).",
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        default=8000,
        help="LLM server port when --use-vlm (default: 8000).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct-AWQ",
        help="LLM model name when --use-vlm (default: Qwen/Qwen2.5-14B-Instruct-AWQ).",
    )
    parser.add_argument(
        "--verify-rule",
        type=str,
        default="any",
        choices=["any", "majority", "all"],
        help="When --use-vlm: keep candidate if any / majority / all expected events get 'yes' (default: any).",
    )

    args = parser.parse_args()

    process_dataset(
        dataset=args.dataset,
        top_k_seeds=args.top_k_seeds,
        total_budget=args.budget,
        use_vlm=args.use_vlm,
        llm_port=args.llm_port,
        llm_model=args.llm_model,
        verify_rule=args.verify_rule,
    )


if __name__ == "__main__":
    main()