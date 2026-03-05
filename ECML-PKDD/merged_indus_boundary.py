#!/usr/bin/env python3
"""
MERGED SCRIPT: Batch1-only INDUS (seeds ∪ FB(seeds), optional VLM)  ∩
              Temporal clustering boundary → FB expansion (full/selective grounding).

Final output per query:
  final_ids = merge(methodA_ids, methodB_ids)
Default merge = intersection ("selective of both").

Method A (INDUS Batch1-only):
  - seed_ids = Top-K Borda (borda_score None treated as +inf)
  - fb_ids = forward_backward(seed_ids)
  - candidates = seed_ids ∪ fb_ids
  - budget cap: take first seeds then extras (deterministic)
  - if --use-vlm: verify candidates with Prompt1+Prompt2 -> verified_A
    else verified_A = candidates_sent

Method B (clustering boundary FB expansion):
  - top intervals -> merge_gap -> boundary_ids
  - expand_and_metrics (run_fb=True, run_evo=False)
  - kept_B = top_seed_ids ∪ fb_events_kept (from expand_and_metrics)

Final ids:
  - default: verified_A ∩ kept_B
  - optional: verified_A ∪ kept_B (use --merge-mode union)

Writes:
  ECML-PKDD/{dataset}_retrieval/indus_explore/seed_events_{dataset}_merged_final_{merge_mode}_{fb_mode}.json
"""

import sys
import json
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ── path setup ────────────────────────────────────────────────────────────────────
_ecml_dir = Path(__file__).resolve().parent          # ECML-PKDD
project_root = _ecml_dir.parent                     # repo root (Project-Ava)
sys.path.insert(0, str(_ecml_dir))
sys.path.insert(0, str(project_root))

# INDUS helpers (Method A)
from expand_indus_seed_events import (  # type: ignore[attr-defined]
    expand_forward_backward,
    fetch_event_data,
    _event_ids_in_vdb,
)

# Expansion-OD helpers (Method B)
from expansion_od.expansion import (  # type: ignore[attr-defined]
    initialize_vdbs as od_initialize_vdbs,
    resolve_kg_dir as od_resolve_kg_dir,
    get_video_path,
    expand_and_metrics,
)
from expansion_od.clustering import (
    events_to_id_intervals,
    top_k_id_intervals,
    merge_gap,
    cluster_boundary_event_ids,
)
from expansion_od.gt import load_gt, load_questions
from llms.QwenLM import QwenLM

try:
    import indus_prompts  # used only when --use-vlm
except ImportError:
    indus_prompts = None  # type: ignore[assignment]


# ── types ─────────────────────────────────────────────────────────────────────────
VideoKey = str
QuestionId = int
QueryKey = Tuple[VideoKey, QuestionId]


# ── JSON safety ───────────────────────────────────────────────────────────────────
def _jsonify(x):
    from pathlib import Path
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonify(v) for v in x]
    if isinstance(x, set):
        return sorted(_jsonify(v) for v in x)
    if isinstance(x, tuple):
        return [_jsonify(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    try:
        import numpy as np
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
    except Exception:
        pass
    return str(x)


# ── Method A: Top-K Borda (None treated as +inf) ─────────────────────────────────
def _sort_key_event(e: Dict[str, Any]) -> Tuple[int, float]:
    score = e.get("borda_score")
    if score is None:
        return (1, 0.0)
    try:
        return (0, -float(score))
    except (TypeError, ValueError):
        return (1, 0.0)


def select_top_k_borda_seed_ids(entry: Dict[str, Any], top_k: int) -> List[str]:
    """
    Dedup by event id.
    Change: borda_score == None is treated as MAX score (ranked highest).
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

    seen: Set[str] = set()
    out: List[str] = []
    for ev in sorted_scored:
        eid = ev.get("id") or ev.get("__id__")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        out.append(eid)
        if len(out) >= top_k:
            break
    return out


# ── VLM verification (Method A optional) ─────────────────────────────────────────
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
        except Exception:
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


def _verify_candidates_with_vlm(
    llm: Any,
    query_events: List[str],
    candidate_ids: List[str],
    events_vdb: Any,
    verify_rule: str = "any",
    max_batch_size: int = 64,
    out_matrix: Optional[Dict[str, Any]] = None,
) -> Set[str]:
    if not query_events or not candidate_ids:
        if out_matrix is not None:
            out_matrix["candidate_ids"] = list(candidate_ids)
            out_matrix["matrix"] = []
            out_matrix["verified_ids"] = []
        return set(candidate_ids) if verify_rule == "any" else set()

    query_events_formatted = _format_query_events_for_prompt(query_events)
    M = len(query_events)

    prompts: List[str] = []
    for eid in candidate_ids:
        desc = _get_event_description(events_vdb, eid)
        prompts.append(
            indus_prompts.INDUS_PROMPT["event_relevance_verification"].format(
                query_events_formatted=query_events_formatted,
                candidate_event_description=desc,
            )
        )

    verified: Set[str] = set()
    matrix_rows: List[List[str]] = []

    for start in range(0, len(prompts), max_batch_size):
        end = min(start + max_batch_size, len(prompts))
        batch_inputs = [{"text": p} for p in prompts[start:end]]
        batch_outputs = llm.batch_generate_response(batch_inputs)

        for i, raw in enumerate(batch_outputs):
            idx = start + i
            if idx >= len(candidate_ids):
                break
            eid = candidate_ids[idx]
            row = _parse_relevance_response(raw, M)
            matrix_rows.append(row)

            if verify_rule == "any":
                if any(r == "yes" for r in row):
                    verified.add(eid)
            elif verify_rule == "majority":
                if sum(1 for r in row if r == "yes") > M // 2:
                    verified.add(eid)
            else:  # all
                if all(r == "yes" for r in row):
                    verified.add(eid)

    if out_matrix is not None:
        out_matrix["candidate_ids"] = list(candidate_ids)
        out_matrix["matrix"] = matrix_rows
        out_matrix["verified_ids"] = sorted(verified)

    return verified


# ── Method A per-query ───────────────────────────────────────────────────────────
def method_a_indus_batch1(
    entry: Dict[str, Any],
    events_vdb: Any,
    valid_event_ids: Set[str],
    top_k_seeds: int,
    budget: int,
    use_vlm: bool,
    llm: Any,
    verify_rule: str,
) -> Tuple[Set[str], Dict[str, Any]]:
    """
    Returns:
      verified_A_ids: Set[str]
      trace: dict (optional diagnostics)
    """
    trace: Dict[str, Any] = {}
    seed_ids = select_top_k_borda_seed_ids(entry, top_k_seeds)
    trace["seed_ids"] = list(seed_ids)

    if not seed_ids:
        trace["batch1_sent"] = []
        trace["verified_A_ids"] = []
        return set(), trace

    fb_all = expand_forward_backward(set(seed_ids), events_vdb) & valid_event_ids
    fb_candidates = fb_all - set(seed_ids)

    batch1_ids = set(seed_ids) | fb_candidates
    # deterministic ordering: seeds first, then extras sorted
    batch1_to_send = list(seed_ids) + sorted(batch1_ids - set(seed_ids))

    if len(batch1_to_send) > budget:
        batch1_to_send = batch1_to_send[:budget]

    trace["batch1_sent"] = list(batch1_to_send)

    if not use_vlm:
        verified = set(batch1_to_send)
        trace["verified_A_ids"] = sorted(verified)
        return verified, trace

    if indus_prompts is None:
        raise RuntimeError("indus_prompts not available; cannot use --use-vlm")

    question = entry.get("question", "")
    options = entry.get("options", "")

    # Prompt 1 once per query
    prompt_p1 = indus_prompts.INDUS_PROMPT["query_event_extraction"].format(
        question=question,
        options=options or "",
    )
    out_p1 = llm.batch_generate_response([{"text": prompt_p1}])[0]
    query_events = _parse_query_events_from_llm(out_p1) or ["Relevant to the question or options"]
    trace["query_events"] = list(query_events)

    matrix_trace: Dict[str, Any] = {}
    verified = _verify_candidates_with_vlm(
        llm=llm,
        query_events=query_events,
        candidate_ids=batch1_to_send,
        events_vdb=events_vdb,
        verify_rule=verify_rule,
        out_matrix=matrix_trace,
    )
    trace["vlm_matrix"] = matrix_trace
    trace["verified_A_ids"] = sorted(verified)
    return verified, trace


# ── Method B per-query ───────────────────────────────────────────────────────────
def method_b_boundary_fb(
    entry: Dict[str, Any],
    vk: str,
    qid: Any,
    gt: Dict,
    events_vdb: Any,
    entities_vdb: Any,
    valid_event_ids: Set[str],
    fb_mode: str,
    questions: Optional[Dict],
    grounding_detector: Any,
    grounding_threshold: float,
    llm_for_grounding: Optional[QwenLM],
    expand_k: int,
    expand_t: int,
    dataset: str,
) -> Tuple[Set[str], Dict[str, Any]]:
    """
    Returns:
      kept_B_ids: Set[str]
      trace: dict
    """
    trace: Dict[str, Any] = {}
    events = entry.get("seed_events") or []
    id_intervals = events_to_id_intervals(events)
    if not id_intervals:
        trace["reason"] = "no_id_intervals"
        return set(), trace

    top = top_k_id_intervals(id_intervals, expand_k, events)
    merged = merge_gap([x[1] for x in top], expand_t)
    boundary_ids = cluster_boundary_event_ids(top, merged)

    top_ids = set([x[0] for x in top if x and x[0]])
    trace["top_ids"] = sorted(top_ids)
    trace["boundary_ids"] = list(boundary_ids) if boundary_ids else []

    kept_ids: Set[str] = set()
    kept_ids |= (top_ids & valid_event_ids)

    if not boundary_ids:
        trace["reason"] = "no_boundary_ids"
        return kept_ids, trace

    gt_seg = gt.get((vk, str(qid))) if gt else None

    fb_selective = (fb_mode == "selective")
    video_path = get_video_path(vk, dataset) if fb_selective else None

    query_objects = []
    if fb_selective:
        # Grounding-based filtering needs question parsing
        from expansion_od.grounding import extract_query_objects
        qinfo = (questions or {}).get((vk, str(qid))) or {}
        query_objects = extract_query_objects(
            qinfo.get("question", ""),
            qinfo.get("answer_statements", []),
            llm_for_grounding,
        ) or []
    trace["fb_selective"] = fb_selective
    trace["query_objects"] = query_objects if fb_selective else []

    try:
        _hit_fb, _time_fb, _hit_evo, _time_evo, fb_events_kept = expand_and_metrics(
            boundary_ids,
            gt_seg,
            events_vdb,
            entities_vdb,
            run_fb=True,
            run_evo=False,
            fb_selective=fb_selective,
            video_path=video_path,
            query_objects=query_objects or None,
            grounding_detector=grounding_detector,
            grounding_threshold=grounding_threshold,
            original_merged=merged,
        )
        if fb_events_kept:
            kept_ids |= (set(fb_events_kept) & valid_event_ids)
        trace["fb_events_kept_n"] = len(fb_events_kept or [])
    except Exception as e:
        trace["error"] = str(e)

    trace["kept_B_ids_n"] = len(kept_ids)
    return kept_ids, trace


# ── main driver ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Merge INDUS Batch1-only with boundary FB expansion (selective-of-both).")
    parser.add_argument("--dataset", choices=["AVA100", "LVBench"], default="AVA100")
    parser.add_argument("--top-k-seeds", type=int, default=20)
    parser.add_argument("--budget", type=int, required=True)

    # Method B config
    parser.add_argument("--fb-mode", choices=["full", "selective"], default="full")
    parser.add_argument("--expand-k", type=int, default=20)
    parser.add_argument("--expand-t", type=int, default=0)
    parser.add_argument("--grounding-threshold", type=float, default=0.1)

    # Merge behavior
    parser.add_argument("--merge-mode", choices=["intersection", "union"], default="union")

    # Method A VLM config
    parser.add_argument("--use-vlm", action="store_true")
    parser.add_argument("--llm-port", type=int, default=8000)
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    parser.add_argument("--verify-rule", choices=["any", "majority", "all"], default="any")

    args = parser.parse_args()

    dataset = args.dataset
    retrieval_dir = _ecml_dir / f"{dataset.lower()}_retrieval"
    seed_path = retrieval_dir / f"seed_events_{dataset}.json"
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_path}")

    seed_data = json.loads(seed_path.read_text())
    if not isinstance(seed_data, list):
        raise ValueError(f"Expected list in {seed_path}, got {type(seed_data)}")

    # Load GT/questions for Method B (expand_and_metrics signature + selective grounding)
    gt = load_gt()
    questions = None
    grounding_detector = None
    llm_for_grounding = None

    if args.fb_mode == "selective":
        questions = load_questions()
        from expansion_od.grounding import GroundingDetector
        grounding_detector = GroundingDetector()
        llm_for_grounding = QwenLM()

    # Optional VLM for Method A
    vlm = None
    if args.use_vlm:
        if indus_prompts is None:
            raise RuntimeError("indus_prompts not available; cannot use --use-vlm")
        from llms.init_model import init_model
        vlm = init_model("qwenvl_vllm", num_gpus=1, model_type=args.llm_model, port=args.llm_port)
        print(f"VLM enabled (port={args.llm_port}, model={args.llm_model}), verify_rule={args.verify_rule}")
    else:
        print("No-VLM mode for Method A (all sent candidates treated as verified).")

    # Embedding model + per-video vdb cache
    from embeddings.JinaCLIP import JinaCLIP
    embedding_model = JinaCLIP("jinaai/jina-clip-v1")

    current_vk: Optional[str] = None
    events_vdb = None
    entities_vdb = None
    valid_event_ids: Set[str] = set()

    out_entries: List[Dict[str, Any]] = []

    for idx, entry in enumerate(seed_data):
        vk = entry.get("video_key")
        qid = entry.get("question_id")
        if vk is None or qid is None:
            continue

        print(f"[{idx+1}/{len(seed_data)}] video={vk} qid={qid}")

        # init VDB per video
        if vk != current_vk:
            current_vk = vk
            kg_dir = od_resolve_kg_dir(vk, dataset)
            if not kg_dir:
                print(f"  [Warn] No KG dir for video={vk}; skip.")
                continue
            events_vdb, entities_vdb, _ = od_initialize_vdbs(kg_dir, embedding_model)
            valid_event_ids = _event_ids_in_vdb(events_vdb)

        if events_vdb is None or entities_vdb is None:
            print("  [Warn] VDB not ready; skip.")
            continue

        # Method A
        verified_A, trace_A = method_a_indus_batch1(
            entry=entry,
            events_vdb=events_vdb,
            valid_event_ids=valid_event_ids,
            top_k_seeds=args.top_k_seeds,
            budget=args.budget,
            use_vlm=args.use_vlm,
            llm=vlm,
            verify_rule=args.verify_rule,
        )

        # Method B
        kept_B, trace_B = method_b_boundary_fb(
            entry=entry,
            vk=str(vk),
            qid=qid,
            gt=gt,
            events_vdb=events_vdb,
            entities_vdb=entities_vdb,
            valid_event_ids=valid_event_ids,
            fb_mode=args.fb_mode,
            questions=questions,
            grounding_detector=grounding_detector,
            grounding_threshold=args.grounding_threshold,
            llm_for_grounding=llm_for_grounding,
            expand_k=args.expand_k,
            expand_t=args.expand_t,
            dataset=dataset,
        )

        # Merge
        if args.merge_mode == "intersection":
            final_ids = (verified_A & kept_B) & valid_event_ids
        else:
            final_ids = (verified_A | kept_B) & valid_event_ids

        final_events = fetch_event_data(final_ids, events_vdb)

        out_entry = {
            "dataset": dataset,
            "video_key": vk,
            "question_id": qid,
            "question": entry.get("question", ""),
            "options": entry.get("options", ""),
            "localization_time_seconds": entry.get("localization_time_seconds"),
            "seed_event_ids": sorted(final_ids),
            "seed_events": final_events,
            "merge_meta": {
                "merge_mode": args.merge_mode,
                "methodA_verified_n": len(verified_A),
                "methodB_kept_n": len(kept_B),
                "final_n": len(final_ids),
                "top_k_seeds": args.top_k_seeds,
                "budget": args.budget,
                "fb_mode": args.fb_mode,
                "expand_k": args.expand_k,
                "expand_t": args.expand_t,
            },
            # Keep traces for debugging (you can remove these fields if you want smaller JSON)
            "trace_methodA": trace_A,
            "trace_methodB": trace_B,
        }
        out_entries.append(out_entry)

    out_dir = retrieval_dir / "indus_explore"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_events_{dataset}_merged_final_{args.merge_mode}_{args.fb_mode}.json"
    out_path.write_text(json.dumps(_jsonify(out_entries), indent=2, ensure_ascii=False))
    print(f"\n✓ Saved merged output: {out_path} ({len(out_entries)} entries)")


if __name__ == "__main__":
    main()