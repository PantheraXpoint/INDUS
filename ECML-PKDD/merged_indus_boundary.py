#!/usr/bin/env python3
"""
MERGED (Method A + Method B) with optional JSON cache.

Method A:
  - Top-K Borda seeds (borda_score None treated as +inf)
  - candidates = seeds ∪ FB(seeds)
  - budget cap
  - optional VLM verify => verified_A (else verified_A = sent candidates)

Method B (NO clustering-based expansion):
  - uses SAME Top-K seeds as Method A
  - cache step:
      * cluster Top-K seeds (time-gap) ONLY to detect cache hits
      * for hit clusters: reuse cached cluster events (no expansion)
      * remaining seeds: expand_and_metrics once from global boundary(first/last)
  - kept_B = (Top-K seeds) ∪ (cached reused events) ∪ (expanded events from remaining seeds)

Merge:
  - default: intersection (selective of both)
  - optional: union

Cache:
  - --cache <path> enables cache JSON
  - After final_ids computed: cluster final_ids and store to cache.
  - Cache match uses Jaccard over seed-cluster ids vs cached.seed_event_ids.

Outputs:
  ECML-PKDD/{dataset}_retrieval/indus_explore/seed_events_{dataset}_merged_final_{merge_mode}_{fb_mode}_cache.json
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

from expand_indus_seed_events import (  # type: ignore[attr-defined]
    expand_forward_backward,
    fetch_event_data,
    _event_ids_in_vdb,
)

from expansion_od.expansion import (  # type: ignore[attr-defined]
    initialize_vdbs as od_initialize_vdbs,
    resolve_kg_dir as od_resolve_kg_dir,
    get_video_path,
    expand_and_metrics,
)
from expansion_od.gt import load_gt, load_questions
from llms.QwenLM import QwenLM

try:
    import indus_prompts  # used only when --use-vlm
except ImportError:
    indus_prompts = None  # type: ignore[assignment]


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


# ── Cache helpers ─────────────────────────────────────────────────────────────────
def load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "videos": {}}
    try:
        obj = json.loads(path.read_text())
        if not isinstance(obj, dict):
            return {"version": 1, "videos": {}}
        obj.setdefault("version", 1)
        obj.setdefault("videos", {})
        if not isinstance(obj["videos"], dict):
            obj["videos"] = {}
        return obj
    except Exception:
        return {"version": 1, "videos": {}}


def save_cache(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(cache), indent=2, ensure_ascii=False))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def best_cache_match(
    cache_video_clusters: List[Dict[str, Any]],
    seed_cluster_ids: Set[str],
) -> Tuple[Optional[Dict[str, Any]], float]:
    best = None
    best_score = 0.0
    for c in cache_video_clusters:
        ids = set(c.get("seed_event_ids") or [])
        score = jaccard(seed_cluster_ids, ids)
        if score > best_score:
            best_score = score
            best = c
    return best, best_score


# ── Time helpers / clustering by time-gap ────────────────────────────────────────
def _event_interval(events_vdb: Any, eid: str) -> Optional[List[float]]:
    try:
        d = events_vdb.get_data(eid)
    except Exception:
        return None
    dur = d.get("duration")
    if isinstance(dur, (list, tuple)) and len(dur) >= 2:
        try:
            return [float(dur[0]), float(dur[1])]
        except Exception:
            return None
    return None


def merge_gap_intervals(intervals: List[List[float]], gap: float) -> List[List[float]]:
    """
    Equivalent spirit to expansion_od.clustering.merge_gap but independent:
    intervals: [[s,e], ...]
    Returns merged intervals list.
    """
    if not intervals:
        return []
    ivs = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: List[List[float]] = [ivs[0][:]]
    for s, e in ivs[1:]:
        last = merged[-1]
        if s <= last[1] + gap:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return merged


def cluster_ids_by_timegap(events_vdb: Any, ids: List[str], gap: float) -> List[List[str]]:
    """
    Cluster event ids by time-gap using their [start,end] intervals.
    Output clusters are lists of ids (roughly ordered by start time).
    """
    id_iv: List[Tuple[str, List[float]]] = []
    for eid in ids:
        iv = _event_interval(events_vdb, eid)
        if iv is not None:
            id_iv.append((eid, iv))
    if not id_iv:
        return []

    intervals = [iv for _, iv in id_iv]
    merged = merge_gap_intervals(intervals, gap)

    clusters: List[List[str]] = [[] for _ in merged]
    for eid, iv in id_iv:
        s, e = iv
        for i, (ms, me) in enumerate(merged):
            if not (e < ms or s > me):
                clusters[i].append(eid)
                break

    # drop empties + order ids by start time
    def _start(eid: str) -> float:
        iv = _event_interval(events_vdb, eid)
        return iv[0] if iv else float("inf")

    out = []
    for c in clusters:
        if c:
            out.append(sorted(list(dict.fromkeys(c)), key=_start))
    return out


def boundary_ids_global(events_vdb: Any, ids: List[str]) -> List[str]:
    """Return [first,last] by start time (or 1 id if only one)."""
    def _start(eid: str) -> float:
        iv = _event_interval(events_vdb, eid)
        return iv[0] if iv else float("inf")
    xs = [x for x in ids if x]
    xs = sorted(xs, key=_start)
    if not xs:
        return []
    if len(xs) == 1:
        return [xs[0]]
    return [xs[0], xs[-1]]


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


# ── VLM for Method A ─────────────────────────────────────────────────────────────
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


def method_a_indus_batch1_using_given_topk(
    entry: Dict[str, Any],
    events_vdb: Any,
    valid_event_ids: Set[str],
    topk_seed_ids: List[str],
    budget: int,
    use_vlm: bool,
    llm: Any,
    verify_rule: str,
) -> Tuple[Set[str], Dict[str, Any]]:
    """
    Method A but uses precomputed Top-K seeds (topk_seed_ids) to ensure both methods share same Top-K.
    """
    trace: Dict[str, Any] = {"seed_ids": list(topk_seed_ids)}

    seed_ids = [x for x in topk_seed_ids if x in valid_event_ids]
    if not seed_ids:
        trace["batch1_sent"] = []
        trace["verified_A_ids"] = []
        return set(), trace

    fb_all = expand_forward_backward(set(seed_ids), events_vdb) & valid_event_ids
    fb_candidates = fb_all - set(seed_ids)

    candidates = list(seed_ids) + sorted(list(fb_candidates))
    if len(candidates) > budget:
        candidates = candidates[:budget]
    trace["batch1_sent"] = list(candidates)

    if not use_vlm:
        verified = set(candidates)
        trace["verified_A_ids"] = sorted(verified)
        return verified, trace

    if indus_prompts is None:
        raise RuntimeError("indus_prompts not available; cannot use --use-vlm")

    question = entry.get("question", "")
    options = entry.get("options", "")

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
        candidate_ids=candidates,
        events_vdb=events_vdb,
        verify_rule=verify_rule,
        out_matrix=matrix_trace,
    )
    trace["vlm_matrix"] = matrix_trace
    trace["verified_A_ids"] = sorted(verified)
    return verified, trace


# ── Method B with cache hit on clustered Top-K, but NO clustering in expansion ────
def method_b_using_topk_with_cache(
    entry: Dict[str, Any],
    vk: str,
    qid: Any,
    dataset: str,
    events_vdb: Any,
    entities_vdb: Any,
    valid_event_ids: Set[str],
    topk_seed_ids: List[str],
    fb_mode: str,
    gt: Dict,
    questions: Optional[Dict],
    grounding_detector: Any,
    grounding_threshold: float,
    llm_for_grounding: Optional[QwenLM],
    cache: Optional[Dict[str, Any]],
    cache_overlap_threshold: float,
    cache_cluster_gap: float,
    final_cluster_gap: float,
) -> Tuple[Set[str], Dict[str, Any], Dict[str, Any], float]:
    """
    Steps:
      1) cluster Top-K seeds to find cache-hit clusters
      2) hit clusters -> reuse cached cluster_event_ids
      3) remaining seeds -> expand_and_metrics ONCE from global boundary(first/last)
      4) kept_B = Top-K seeds ∪ reused ∪ expanded
    Returns:
      kept_B_ids, trace_B, cache_meta, time_fb_used (for caching final clusters)
    """
    trace: Dict[str, Any] = {}
    cache_meta = {
        "cache_enabled": bool(cache is not None),
        "reused_clusters": 0,
        "reused_events": 0,
        "avg_time_reused_sec": 0.0,
        "new_clusters_saved": 0,
        "final_clusters_saved": 0,
    }
    reused_times: List[float] = []

    seed_ids = [x for x in topk_seed_ids if x in valid_event_ids]
    trace["topk_seed_n"] = len(seed_ids)

    if not seed_ids:
        return set(), {"reason": "no_topk_seeds"}, cache_meta, 0.0

    # cache per video
    cache_video_clusters: List[Dict[str, Any]] = []
    if cache is not None:
        videos = cache.setdefault("videos", {})
        cache_video_clusters = videos.setdefault(str(vk), [])
        if not isinstance(cache_video_clusters, list):
            cache_video_clusters = []
            videos[str(vk)] = cache_video_clusters

    # 1) cluster topK ONLY for cache hit detection
    seed_clusters = cluster_ids_by_timegap(events_vdb, seed_ids, gap=cache_cluster_gap)
    trace["seed_clusters_n"] = len(seed_clusters)

    reused_event_ids: Set[str] = set()
    hit_seed_ids: Set[str] = set()

    for cl in seed_clusters:
        cl_set = set(cl)
        best, score = (None, 0.0)
        if cache is not None and cache_video_clusters:
            best, score = best_cache_match(cache_video_clusters, cl_set)

        if best is not None and score >= cache_overlap_threshold:
            # reuse cached cluster events
            used = set(best.get("cluster_event_ids") or best.get("expanded_ids") or [])
            used &= valid_event_ids
            reused_event_ids |= used
            hit_seed_ids |= cl_set

            cache_meta["reused_clusters"] += 1
            cache_meta["reused_events"] += len(used)
            t = best.get("time_sec")
            if isinstance(t, (int, float)):
                reused_times.append(float(t))

    if reused_times:
        cache_meta["avg_time_reused_sec"] = sum(reused_times) / max(1, len(reused_times))

    # 2) remaining seeds expand normally (NO clustering in expansion)
    remaining_seeds = [x for x in seed_ids if x not in hit_seed_ids]
    trace["remaining_seeds_n"] = len(remaining_seeds)

    kept_B: Set[str] = set(seed_ids) | set(reused_event_ids)
    time_fb_used = 0.0

    if remaining_seeds:
        boundary_ids = set(remaining_seeds)
        trace["boundary_ids"] = list(boundary_ids)

        fb_selective = (fb_mode == "selective")
        video_path = get_video_path(vk, dataset) if fb_selective else None

        query_objects = []
        if fb_selective:
            from expansion_od.grounding import extract_query_objects
            qinfo = (questions or {}).get((vk, str(qid))) or {}
            query_objects = extract_query_objects(
                qinfo.get("question", ""),
                qinfo.get("answer_statements", []),
                llm_for_grounding,
            ) or []

        gt_seg = gt.get((vk, str(qid))) if gt else None
        try:
            _hit_fb, time_fb, _hit_evo, _time_evo, fb_events_kept = expand_and_metrics(
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
                original_merged=None,
            )
            time_fb_used = float(time_fb) if isinstance(time_fb, (int, float)) else 0.0
            kept_B |= (set(fb_events_kept or []) & valid_event_ids)
        except Exception as e:
            print(e)
            trace["expand_error"] = str(e)

    trace["kept_B_n_before_final_cache"] = len(kept_B)

    # 3) After everything, cluster FINAL B events and save to cache
    if cache is not None:
        final_b_clusters = cluster_ids_by_timegap(events_vdb, sorted(kept_B), gap=final_cluster_gap)
        cache_meta["final_clusters_saved"] = len(final_b_clusters)

        for cl in final_b_clusters:
            cl_set = set(cl)
            # use the subset of topk seeds inside this final cluster as seed signature
            seed_sig = sorted(list(cl_set & set(seed_ids)))
            cache_video_clusters.append({
                "seed_event_ids": seed_sig,                 # used for matching
                "cluster_event_ids": sorted(list(cl_set)),  # reused directly
                "time_sec": time_fb_used,                   # rough; from expansion call if any
            })

    return kept_B, trace, cache_meta, time_fb_used


# ── main driver ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Merged MethodA + MethodB with cache reuse on TopK clusters + final clustering saved to cache.")
    p.add_argument("--dataset", choices=["AVA100", "LVBench"], default="AVA100")
    p.add_argument("--top-k-seeds", type=int, default=20)
    p.add_argument("--budget", type=int, required=True)

    # Method B config
    p.add_argument("--fb-mode", choices=["full", "selective"], default="full")
    p.add_argument("--grounding-threshold", type=float, default=0.1)

    # Merge behavior
    p.add_argument("--merge-mode", choices=["intersection", "union"], default="union")

    # VLM for Method A
    p.add_argument("--use-vlm", action="store_true")
    p.add_argument("--llm-port", type=int, default=8000)
    p.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    p.add_argument("--verify-rule", choices=["any", "majority", "all"], default="any")

    # Cache
    p.add_argument("--cache", type=str, default="", help="Cache JSON path (optional).")
    p.add_argument("--cache-overlap-threshold", type=float, default=0.6, help="Jaccard overlap to reuse (default 0.6).")
    p.add_argument("--cache-cluster-gap", type=float, default=0.0, help="Gap sec used to cluster TopK seeds for cache hit (default 0).")
    p.add_argument("--final-cluster-gap", type=float, default=0.0, help="Gap sec used to cluster FINAL events before saving to cache (default 0).")

    args = p.parse_args()
    dataset = args.dataset

    retrieval_dir = _ecml_dir / f"{dataset.lower()}_retrieval"
    seed_path = retrieval_dir / f"seed_events_{dataset}.json"
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_path}")

    seed_data = json.loads(seed_path.read_text())
    if not isinstance(seed_data, list):
        raise ValueError(f"Expected list in {seed_path}, got {type(seed_data)}")

    # cache load
    cache_obj: Optional[Dict[str, Any]] = None
    cache_path: Optional[Path] = None
    if args.cache.strip():
        cache_path = Path(args.cache).expanduser().resolve()
        cache_obj = load_cache(cache_path)
        print(f"[Cache] Enabled: {cache_path}")
    else:
        print("[Cache] Disabled")

    # GT/questions for Method B
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
        print("No-VLM mode for Method A.")

    # embedding model + vdb cache per video
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
            kg_dir = od_resolve_kg_dir(str(vk), dataset)
            if not kg_dir:
                print(f"  [Warn] No KG dir for video={vk}; skip.")
                continue
            events_vdb, entities_vdb, _ = od_initialize_vdbs(kg_dir, embedding_model)
            valid_event_ids = _event_ids_in_vdb(events_vdb)

        if events_vdb is None or entities_vdb is None:
            print("  [Warn] VDB not ready; skip.")
            continue

        # SAME Top-K seeds for both methods
        topk_seed_ids = select_top_k_borda_seed_ids(entry, args.top_k_seeds)
        topk_seed_ids = [x for x in topk_seed_ids if x in valid_event_ids]

        # Method A using that Top-K
        verified_A, trace_A = method_a_indus_batch1_using_given_topk(
            entry=entry,
            events_vdb=events_vdb,
            valid_event_ids=valid_event_ids,
            topk_seed_ids=topk_seed_ids,
            budget=args.budget,
            use_vlm=args.use_vlm,
            llm=vlm,
            verify_rule=args.verify_rule,
        )

        # Method B with cache hit (cluster only for cache detection), no clustering expansion
        kept_B, trace_B, cache_meta, _time_fb = method_b_using_topk_with_cache(
            entry=entry,
            vk=str(vk),
            qid=qid,
            dataset=dataset,
            events_vdb=events_vdb,
            entities_vdb=entities_vdb,
            valid_event_ids=valid_event_ids,
            topk_seed_ids=topk_seed_ids,
            fb_mode=args.fb_mode,
            gt=gt,
            questions=questions,
            grounding_detector=grounding_detector,
            grounding_threshold=args.grounding_threshold,
            llm_for_grounding=llm_for_grounding,
            cache=cache_obj,
            cache_overlap_threshold=args.cache_overlap_threshold,
            cache_cluster_gap=args.cache_cluster_gap,
            final_cluster_gap=args.final_cluster_gap,
        )

        # Merge
        if args.merge_mode == "intersection":
            final_ids = (verified_A & kept_B) & valid_event_ids
        else:
            final_ids = (verified_A | kept_B) & valid_event_ids

        final_events = fetch_event_data(final_ids, events_vdb)

        out_entries.append({
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
                "top_k_seeds": args.top_k_seeds,
                "budget": args.budget,
                "fb_mode": args.fb_mode,
                "methodA_verified_n": len(verified_A),
                "methodB_kept_n": len(kept_B),
                "final_n": len(final_ids),
            },
            "cache_meta": cache_meta,
            "trace_methodA": trace_A,
            "trace_methodB": trace_B,
        })

    # Save cache
    if cache_obj is not None and cache_path is not None:
        save_cache(cache_path, cache_obj)
        print(f"[Cache] Saved: {cache_path}")

    # Save final merged output
    out_dir = retrieval_dir / "indus_explore"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_events_{dataset}_merged_final_{args.merge_mode}_{args.fb_mode}_cache.json"
    out_path.write_text(json.dumps(_jsonify(out_entries), indent=2, ensure_ascii=False))
    print(f"\n✓ Saved merged output: {out_path} ({len(out_entries)} entries)")


if __name__ == "__main__":
    main()