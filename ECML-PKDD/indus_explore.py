#!/usr/bin/env python3
"""
Budgeted INDUS exploration: Batch 1 (base ∪ FB(base)), Batch 2 (temporal ∪ FB(temporal)), Batch 3 (EOE from base).

Pipeline per query:
  Batch 1: base seeds (top-K Borda) ∪ FB(base). Send combined set to VLM → verified_batch1.
           exploration_history = base ∪ FB(base).
  Batch 2: temporal events from log ∪ FB(temporal). Dedupe vs history; cap/sample to total_budget.
           Send to VLM → verified_batch2. exploration_history += temporal ∪ FB(temporal) (full set after dedup).
  Batch 3: EOE from base seeds only (mutual objects + IDF + clustering). Cap at total_budget.
           Send to VLM → verified_batch3. exploration_history += raw EOE candidates.

  Each batch may send at most total_budget events to the VLM (per-batch budget, not shared).

  Final output = verified_batch1 ∪ verified_batch2 ∪ verified_batch3.

Modes:
  - No VLM (default): no LLM calls; every candidate is treated as verified. Evaluation uses
    exactly the events sent (up to budget) per batch.
  - VLM (--use-vlm): Prompt 1 (query_event_extraction) once per query; Prompt 2
    (event_relevance_verification) per candidate per batch. Verified set = subset passing
    the chosen rule (any/majority/all yes). Evaluation uses this smaller verified set.

Reads:
  - ECML-PKDD/{dataset}_retrieval/seed_events_{dataset}.json
  - ECML-PKDD/indus_outputs/temporal_retrieval_log_{dataset}.json

Outputs under ECML-PKDD/{dataset}_retrieval/indus_explore/:
  - seed_events_{dataset}_indus_explore_temporal_prefilter.json  (Batch 2 prefilter stats)
  - seed_events_{dataset}_indus_explore_eoe_prefilter.json        (Batch 3 prefilter)
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
from fb_with_kg import kg_aware_fb_expansion as kg_fb  # type: ignore[attr-defined]

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
    This recreates the "base" seed set from run_seed_budget.py.
    """
    seed_events = entry.get("seed_events") or []
    if not seed_events or top_k <= 0:
        return []

    with_score = [e for e in seed_events if e.get("borda_score") is not None]
    sorted_scored = sorted(with_score, key=_sort_key_event)

    seen_ids: Set[str] = set()
    top_scored: List[Dict[str, Any]] = []
    for ev in sorted_scored:
        eid = ev.get("id") or ev.get("__id__")
        if not eid or eid in seen_ids:
            continue
        seen_ids.add(eid)
        top_scored.append(ev)
        if len(top_scored) >= top_k:
            break

    return [ev.get("id") or ev.get("__id__") for ev in top_scored if ev.get("id") or ev.get("__id__")]


# ── helpers: temporal parsing candidates (from precomputed logs) ─────────────────

def load_temporal_log_lookup(dataset: str) -> Dict[QueryKey, Dict[str, Any]]:
    """
    Load temporal_retrieval_log_{dataset}.json and index by (video_key, question_id).
    """
    log_path = _ecml_dir / "ava100_analysis" / f"temporal_retrieval_log_{dataset}.json"
    if not log_path.exists():
        print(f"[Warning] Temporal retrieval log not found: {log_path}")
        return {}
    data = json.loads(log_path.read_text())
    if not isinstance(data, list):
        print(f"[Warning] Expected list in {log_path}, got {type(data)}")
        return {}

    lookup: Dict[QueryKey, Dict[str, Any]] = {}
    for entry in data:
        vk = entry.get("video_key")
        qid = entry.get("question_id")
        if vk is None or qid is None:
            continue
        try:
            qid_int = int(qid)
        except (TypeError, ValueError):
            continue
        lookup[(str(vk), qid_int)] = entry
    return lookup


def temporal_candidates_from_log(
    video_key: str,
    question_id: int,
    temporal_log_lookup: Dict[QueryKey, Dict[str, Any]],
) -> Set[str]:
    """
    Reconstruct naive temporal candidates from temporal_retrieval_log:
    union of event_ids in all temporal_segments[*]['events_retrieved'].
    """
    entry = temporal_log_lookup.get((video_key, question_id))
    if not entry:
        return set()

    segs = entry.get("temporal_segments")
    if not segs:
        return set()

    out: Set[str] = set()
    for seg in segs:
        events = seg.get("events_retrieved") or []
        for ev in events:
            eid = ev.get("event_id")
            if eid:
                out.add(eid)
    return out


def _event_start_time(events_vdb, event_id: str) -> Optional[float]:
    """Return event start time in seconds, or None."""
    try:
        data = events_vdb.get_data(event_id)
    except (IndexError, KeyError):
        return None
    dur = data.get("duration")
    if not isinstance(dur, (list, tuple)) or len(dur) < 2:
        return None
    try:
        return float(dur[0])
    except (TypeError, ValueError):
        return None


def _uniform_sample_k(sorted_ids: List[str], k: int) -> List[str]:
    """
    Uniformly sample k indices from a list that is already sorted (e.g. by time).
    If len(list) <= k, returns the original list.
    """
    n = len(sorted_ids)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return sorted_ids
    indices = [i * n // k for i in range(k)]
    return [sorted_ids[i] for i in indices]


# ── helpers: EOE mutual objects + IDF + clustering ───────────────────────────────

def _history_midpoints(events_vdb, history_ids: Set[str]) -> List[float]:
    """Precompute midpoints for exploration_history (for exclusion checks)."""
    mids: List[float] = []
    for eid in history_ids:
        mid = kg_fb._event_midpoint(events_vdb, eid)  # type: ignore[attr-defined]
        if mid is not None:
            mids.append(mid)
    mids.sort()
    return mids


def _cluster_is_near_history(
    events_vdb,
    cluster: List[str],
    history_mids: List[float],
    history_exclusion_radius: float,
) -> bool:
    """Return True if any event in cluster is within radius of any history midpoint."""
    if not history_mids:
        return False
    for eid in cluster:
        mid = kg_fb._event_midpoint(events_vdb, eid)  # type: ignore[attr-defined]
        if mid is None:
            continue
        # Quick linear scan (history_mids is small per video); binary search not necessary.
        for h in history_mids:
            if abs(mid - h) <= history_exclusion_radius:
                return True
            if h > mid + history_exclusion_radius:
                break
    return False


# ── VLM verification (no-VLM stub vs real LLM) ────────────────────────────────────

def vlm_verify_events_stub(
    stage: str,
    video_key: str,
    question_id: int,
    question: str,
    candidate_event_ids: List[str],
) -> Set[str]:
    """
    No-VLM mode: return all candidates as verified (no LLM call).
    Number of events for evaluation = exactly the batch size (up to budget).
    """
    _ = (stage, video_key, question_id, question)
    return set(candidate_event_ids)


def _parse_query_events_from_llm(llm_output: str) -> List[str]:
    """Parse query_event_extraction LLM output into list of expected event descriptions."""
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
    """Format expected events as numbered list for event_relevance_verification prompt."""
    return "\n".join(f"{i+1}. {e}" for i, e in enumerate(query_events))


def _get_event_description(events_vdb, event_id: str) -> str:
    """Get a single string description for an event (for relevance prompt)."""
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
    """Parse event_relevance_verification output into list of 'yes'/'no' (one per expected event)."""
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
    question: str,
    options: str,
    stage: str,
    verify_rule: str = "any",
    max_batch_size: int = 64,
    out_matrix: Optional[Dict[str, Any]] = None,
) -> Set[str]:
    """
    Run Prompt 2 (event_relevance_verification) for each candidate; build M×N matrix;
    return set of verified event IDs according to verify_rule.

    If out_matrix is provided, fill it with: candidate_ids, matrix (list of rows of "yes"/"no"), verified_ids.
    """
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

    # Build N prompts (one per candidate)
    prompts: List[str] = []
    for eid in candidate_event_ids:
        desc = _get_event_description(events_vdb, eid)
        prompt = indus_prompts.INDUS_PROMPT["event_relevance_verification"].format(
            query_events_formatted=query_events_formatted,
            candidate_event_description=desc,
        )
        prompts.append(prompt)

    # Batch in chunks to respect max_batch_size
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
    entities_vdb,
    event_to_entities: Dict[str, List[str]],
    valid_event_ids: Set[str],
    temporal_log_lookup: Dict[QueryKey, Dict[str, Any]],
    top_k_seeds: int,
    total_budget: int,
    min_seeds_mutual: int,
    top_n_objects: int,
    cluster_gap: float,
    history_exclusion_radius: float,
    min_cluster_size: int,
    max_new_regions: int,
    use_vlm: bool = False,
    llm: Any = None,
    verify_rule: str = "any",
) -> Tuple[
    Dict[str, Any],  # temporal_prefilter_entry (may be empty)
    Dict[str, Any],  # eoe_prefilter_entry (may be empty)
    Dict[str, Any],  # final_entry
]:
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

    # ── Query event extraction (Prompt 1) when VLM mode: once per query ───────────
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

    # ── Step 0: Initial seeds (top-K Borda). No VLM here; seeds go in Batch 1. ─────
    seed_ids = select_top_k_borda_seed_ids(entry, top_k_seeds)
    exploration_history: Set[str] = set(seed_ids)
    # Per-batch budget: each batch may send at most total_budget events to the VLM (no shared remainder).

    vlm_trace_batches: List[Dict[str, Any]] = []

    def _verify_batch(candidate_event_ids: List[str], stage: str) -> Set[str]:
        if use_vlm and llm is not None and query_events and candidate_event_ids:
            trace_entry: Dict[str, Any] = {"stage": stage}
            verified = _verify_batch_with_vlm(
                llm=llm,
                query_events=query_events,
                candidate_event_ids=candidate_event_ids,
                events_vdb=events_vdb,
                question=question,
                options=options or "",
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

    # ── Batch 1: base ∪ FB(base) — send together to VLM (cap at total_budget per batch) ─
    verified_batch1: Set[str] = set()
    if seed_ids:
        fb_all = expand_forward_backward(set(seed_ids), events_vdb) & valid_event_ids
        fb_candidates = fb_all - exploration_history
        batch1_ids = set(seed_ids) | fb_candidates
        batch1_to_verify = list(seed_ids) + sorted(batch1_ids - set(seed_ids))
        print("Quangggggggggggggggggggggggggggggggggggggggg")
        print(len(batch1_to_verify))
        if len(batch1_to_verify) > total_budget:
            batch1_to_verify = batch1_to_verify[:total_budget]
        if batch1_to_verify:
            verified_batch1 = _verify_batch(batch1_to_verify, "batch1")
            exploration_history |= fb_candidates  # base ∪ FB(base)

    # ── Batch 2: temporal ∪ FB(temporal) — dedupe vs history, cap/sample to total_budget ─
    temporal_prefilter_entry: Dict[str, Any] = {}
    verified_batch2: Set[str] = set()

    temporal_raw = temporal_candidates_from_log(str(video_key), qid_int, temporal_log_lookup)
    temporal_raw = temporal_raw & valid_event_ids
    print("Quangggggggggggggggggggggggggggggggggggggggg")
    print(len(temporal_raw))
    if temporal_raw:
        # FB expansion from ALL temporal events
        fb_temporal = expand_forward_backward(temporal_raw, events_vdb) & valid_event_ids
        temporal_union_fb = temporal_raw | fb_temporal
        # Deduplicate against exploration_history
        batch2_candidates = temporal_union_fb - exploration_history

        if batch2_candidates:
            batch2_sorted = sorted(
                batch2_candidates,
                key=lambda eid: (_event_start_time(events_vdb, eid) or float("inf"), eid),
            )
            if len(batch2_sorted) <= total_budget:
                batch2_to_verify = batch2_sorted
                uniform_sampling_applied = False
            else:
                batch2_to_verify = _uniform_sample_k(batch2_sorted, total_budget)
                uniform_sampling_applied = True

            verified_batch2 = _verify_batch(batch2_to_verify, "batch2")
            # Full combined set (after dedup) goes into history so EOE doesn't rediscover
            exploration_history |= batch2_candidates

            # Temporal prefilter JSON with Batch 2 structure (Issue 5)
            batch2_events = fetch_event_data(set(batch2_to_verify), events_vdb)
            for ev in batch2_events:
                if "borda_score" in ev:
                    ev["borda_score"] = None
            temporal_prefilter_entry = {
                "dataset": dataset,
                "video_key": video_key,
                "question_id": question_id,
                "question": question,
                "options": options,
                "localization_time_seconds": entry.get("localization_time_seconds"),
                "seed_event_ids": batch2_to_verify,
                "seed_events": batch2_events,
                "indus_explore_stage": "temporal_prefilter",
                "n_temporal_from_log": len(temporal_raw),
                "n_fb_temporal": len(fb_temporal),
                "total_combined_temporal_union_fb": len(temporal_union_fb),
                "size_after_dedup_vs_history": len(batch2_candidates),
                "uniform_sampling_applied": uniform_sampling_applied,
                "n_sent_to_vlm": len(batch2_to_verify),
                "n_full_combined_set": len(batch2_candidates),
                "all_batch2_candidate_ids": sorted(batch2_candidates),
            }

    # ── Batch 3: EOE(base) — mutual objects + IDF + clustering, cap at total_budget ─────
    eoe_prefilter_entry: Dict[str, Any] = {}
    verified_batch3: Set[str] = set()

    # if seed_ids and event_to_entities:
    #     # 3a–3c: mutual objects + IDF + EOE from top-N objects
    #     top_objects, eoe_expanded, idf_trace = kg_fb._mutual_objects_idf_top_n_and_eoe(  # type: ignore[attr-defined]
    #         seed_event_ids=seed_ids,
    #         entities_vdb=entities_vdb,
    #         event_to_entities=event_to_entities,
    #         valid_event_ids=valid_event_ids,
    #         top_n_objects=top_n_objects,
    #         min_seeds_mutual=min_seeds_mutual,
    #     )

    #     if top_objects:
    #         raw_eoe = eoe_expanded
    #         raw_eoe_candidates = (raw_eoe & valid_event_ids) - exploration_history

    #         if raw_eoe_candidates:
    #             # 3d: temporal clustering over raw_eoe_candidates
    #             clusters = kg_fb._temporal_clusters(  # type: ignore[attr-defined]
    #                 event_ids=raw_eoe_candidates,
    #                 events_vdb=events_vdb,
    #                 gap_seconds=cluster_gap,
    #             )

    #             # 3e: filter clusters near history and by min_cluster_size
    #             history_mids = _history_midpoints(events_vdb, exploration_history & valid_event_ids)
    #             kept_clusters: List[List[str]] = []
    #             dropped_near_history_sizes: List[int] = []

    #             for cluster in clusters:
    #                 if len(cluster) < min_cluster_size:
    #                     continue
    #                 if _cluster_is_near_history(events_vdb, cluster, history_mids, history_exclusion_radius):
    #                     dropped_near_history_sizes.append(len(cluster))
    #                     continue
    #                 kept_clusters.append(cluster)

    #             # 3f: select clusters within per-batch budget, capped at max_new_regions
    #             kept_clusters.sort(key=len, reverse=True)
    #             selected_events: List[str] = []
    #             regions_used = 0
    #             budget_left = int(total_budget)

    #             for cluster in kept_clusters:
    #                 if budget_left <= 0 or regions_used >= max_new_regions:
    #                     break
    #                 take = min(len(cluster), budget_left)
    #                 if take <= 0:
    #                     break
    #                 # cluster already ordered by time
    #                 selected_events.extend(cluster[:take])
    #                 budget_left -= take
    #                 regions_used += 1

    #             eoe_to_verify = selected_events

    #             if eoe_to_verify:
    #                 verified_batch3 = _verify_batch(eoe_to_verify, "batch3")
    #                 used_here = len(eoe_to_verify)
    #             else:
    #                 used_here = 0

    #             # Add ALL raw_eoe_candidates to exploration_history
    #             exploration_history.update(raw_eoe_candidates)

    #             # Build EOE prefilter JSON entry
    #             eoe_events = fetch_event_data(set(eoe_to_verify), events_vdb)
    #             for ev in eoe_events:
    #                 if "borda_score" in ev:
    #                     ev["borda_score"] = None

    #             eoe_prefilter_entry = {
    #                 "dataset": dataset,
    #                 "video_key": video_key,
    #                 "question_id": question_id,
    #                 "question": question,
    #                 "options": options,
    #                 "localization_time_seconds": entry.get("localization_time_seconds"),
    #                 "seed_event_ids": eoe_to_verify,
    #                 "seed_events": eoe_events,
    #                 "indus_explore_stage": "eoe_prefilter",
    #                 "all_raw_eoe_candidate_ids": sorted(raw_eoe_candidates),
    #                 "cluster_gap": cluster_gap,
    #                 "history_exclusion_radius": history_exclusion_radius,
    #                 "min_cluster_size": min_cluster_size,
    #                 "max_new_regions": max_new_regions,
    #                 "n_clusters_total": len(clusters),
    #                 "n_clusters_kept_after_filters": len(kept_clusters),
    #                 "dropped_near_history_sizes": dropped_near_history_sizes,
    #                 "budget_used_step3": used_here,
    #                 "idf_trace": idf_trace,
    #             }

    # ── Final assembly: verified_batch1 ∪ verified_batch2 ∪ verified_batch3 ───────
    final_ids: Set[str] = verified_batch1 | verified_batch2 | verified_batch3
    # Ensure we only keep IDs that actually exist in the VDB
    final_ids &= valid_event_ids

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
            "verified_batch2_count": len(verified_batch2),
            "verified_batch3_count": len(verified_batch3),
            "final_event_count": len(final_ids),
        },
    }

    if use_vlm and vlm_trace_batches:
        final_entry["vlm_verification_trace"] = {
            "query_events": query_events,
            "verify_rule": verify_rule,
            "batches": vlm_trace_batches,
        }

    return temporal_prefilter_entry, eoe_prefilter_entry, final_entry


# ── main dataset-level driver ────────────────────────────────────────────────────

def process_dataset(
    dataset: str,
    top_k_seeds: int,
    total_budget: int,
    min_seeds_mutual: int,
    top_n_objects: int,
    cluster_gap: float,
    history_exclusion_radius: float,
    min_cluster_size: int,
    max_new_regions: int,
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

    temporal_log_lookup = load_temporal_log_lookup(dataset)

    out_dir = retrieval_dir / "indus_explore"
    out_dir.mkdir(parents=True, exist_ok=True)

    temporal_prefilters: List[Dict[str, Any]] = []
    eoe_prefilters: List[Dict[str, Any]] = []
    final_entries: List[Dict[str, Any]] = []

    current_video_key: Optional[str] = None
    events_vdb = entities_vdb = None
    event_to_entities: Optional[Dict[str, List[str]]] = None
    valid_event_ids: Set[str] = set()

    for idx, entry in enumerate(seed_data):
        video_key = entry.get("video_key")
        question_id = entry.get("question_id")
        dataset_entry = entry.get("dataset", dataset)
        print(f"\n[{idx + 1}/{len(seed_data)}] {dataset} video={video_key} Q{question_id}")

        if video_key is None:
            print("  [Warning] Missing video_key; skipping")
            continue

        # Initialize / reuse VDBs per video
        if video_key != current_video_key:
            current_video_key = video_key
            kg_dir = resolve_kg_dir(project_root, str(video_key), dataset_entry)
            if not kg_dir:
                print(f"  [Warning] KG directory not found for video_key={video_key!r}; skipping.")
                continue
            print(f"  Loading VDBs from {kg_dir}")
            from embeddings.JinaCLIP import JinaCLIP  # local import to avoid global dependency at import time

            embedding_model = JinaCLIP("jinaai/jina-clip-v1")
            events_vdb, entities_vdb, _ = initialize_vdbs(kg_dir, embedding_model)
            valid_event_ids = _event_ids_in_vdb(events_vdb)
            event_to_entities = _build_event_to_entities(entities_vdb)
            print(f"  Built event→entities index: {len(event_to_entities)} events")

        if not events_vdb or not entities_vdb or event_to_entities is None:
            print("  [Warning] VDBs not initialized; skipping entry.")
            continue

        try:
            temporal_entry, eoe_entry, final_entry = process_query(
                entry=entry,
                dataset=dataset,
                events_vdb=events_vdb,
                entities_vdb=entities_vdb,
                event_to_entities=event_to_entities,
                valid_event_ids=valid_event_ids,
                temporal_log_lookup=temporal_log_lookup,
                top_k_seeds=top_k_seeds,
                total_budget=total_budget,
                min_seeds_mutual=min_seeds_mutual,
                top_n_objects=top_n_objects,
                cluster_gap=cluster_gap,
                history_exclusion_radius=history_exclusion_radius,
                min_cluster_size=min_cluster_size,
                max_new_regions=max_new_regions,
                use_vlm=use_vlm,
                llm=llm,
                verify_rule=verify_rule,
            )
        except Exception as e:
            print(f"  [Error] Failed to process query: {e}")
            import traceback

            traceback.print_exc()
            continue

        if temporal_entry:
            temporal_prefilters.append(temporal_entry)
        if eoe_entry:
            eoe_prefilters.append(eoe_entry)
        final_entries.append(final_entry)

    # Write outputs
    temporal_out = out_dir / f"seed_events_{dataset}_indus_explore_temporal_prefilter.json"
    eoe_out = out_dir / f"seed_events_{dataset}_indus_explore_eoe_prefilter.json"
    final_out = out_dir / f"seed_events_{dataset}_indus_explore_final.json"

    with temporal_out.open("w") as f:
        json.dump(temporal_prefilters, f, indent=2)
    print(f"\n✓ Wrote temporal prefilter: {temporal_out} ({len(temporal_prefilters)} entries)")

    with eoe_out.open("w") as f:
        json.dump(eoe_prefilters, f, indent=2)
    print(f"✓ Wrote EOE prefilter: {eoe_out} ({len(eoe_prefilters)} entries)")

    with final_out.open("w") as f:
        json.dump(final_entries, f, indent=2)
    print(f"✓ Wrote final outputs: {final_out} ({len(final_entries)} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Budgeted INDUS exploration (FB, temporal parsing, EOE) with prefilter logging."
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
        help="Top-K Borda seeds per query (initial seeds, default: 80).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Per-batch VLM budget: each of Batch 1, 2, 3 may send at most this many events to the VLM (required).",
    )
    parser.add_argument(
        "--min-seeds-mutual",
        type=int,
        default=2,
        help="Minimum number of distinct seeds an object must appear in to be considered mutual (default: 2).",
    )
    parser.add_argument(
        "--top-n-objects",
        type=int,
        default=10,
        help="Top-N mutual objects by IDF for EOE expansion (default: 10).",
    )
    parser.add_argument(
        "--cluster-gap",
        type=float,
        default=60.0,
        help="Temporal gap (seconds) for grouping EOE events into clusters (default: 60).",
    )
    parser.add_argument(
        "--history-exclusion-radius",
        type=float,
        default=120.0,
        help="Radius (seconds) around any explored event to exclude EOE clusters (default: 120).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum number of events in an EOE cluster to keep (default: 2).",
    )
    parser.add_argument(
        "--max-new-regions",
        type=int,
        default=3,
        help="Maximum number of new temporal regions (clusters) to include (default: 3).",
    )
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="Enable VLM verification: run query_event_extraction (Prompt 1) and event_relevance_verification (Prompt 2) per batch. Default: no VLM (all candidates treated as verified).",
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
        min_seeds_mutual=args.min_seeds_mutual,
        top_n_objects=args.top_n_objects,
        cluster_gap=args.cluster_gap,
        history_exclusion_radius=args.history_exclusion_radius,
        min_cluster_size=args.min_cluster_size,
        max_new_regions=args.max_new_regions,
        use_vlm=args.use_vlm,
        llm_port=args.llm_port,
        llm_model=args.llm_model,
        verify_rule=args.verify_rule,
    )


if __name__ == "__main__":
    main()

