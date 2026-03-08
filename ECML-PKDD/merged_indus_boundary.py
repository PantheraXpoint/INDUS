#!/usr/bin/env python3
"""
MERGED METHOD A + METHOD B with SHARED cache (no A/B separation), reuse by TIME OVERLAP.

Key points (per your requests):
- Method B has NO boundary clustering logic.
- Both methods share the SAME Top-K seeds (selected once).
- Cache reuse is applied for BOTH methods, using a SINGLE cache pool per video_key.
- Cache reuse decision is based on TIME overlap (interval IoU), not id overlap.
- After producing final output, cluster final events and save clusters into the cache.
- Final JSON includes cache reuse stats: reused events count and avg time reused.

Method A (INDUS-like):
  - seeds = TopK
  - candidates = seeds ∪ FB(seeds)
  - budget cap (deterministic: seeds first then extras)
  - optional VLM verification (if --use-vlm)
  - output_A = reused_ids_from_cache ∪ verified_A

Method B (expansion_od expand_and_metrics):
  - seeds = TopK
  - cache-hit clusters => reuse cached cluster_event_ids
  - remaining seeds => expand_and_metrics ONCE using global boundary [first,last]
  - output_B = reused_ids_from_cache ∪ expanded_B ∪ seeds

Merge:
  - default: intersection (selective of both)
  - optional: union

Cache:
  cache.json structure:
    {
      "version": 1,
      "videos": {
        "<video_key>": [
          {"interval": [s,e], "cluster_event_ids": [...], "time_sec": 0.12},
          ...
        ]
      }
    }
"""

import sys
import json
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── path setup ────────────────────────────────────────────────────────────────────
_ecml_dir = Path(__file__).resolve().parent
project_root = _ecml_dir.parent
sys.path.insert(0, str(_ecml_dir))
sys.path.insert(0, str(project_root))

# Method A helpers (fb + fetch)
from expand_indus_seed_events import (  # type: ignore[attr-defined]
    expand_forward_backward,
    fetch_event_data,
    _event_ids_in_vdb,
)

# Method B helpers
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


# ── Cache IO ─────────────────────────────────────────────────────────────────────
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


# ── Time / interval utilities ─────────────────────────────────────────────────────
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


def interval_iou(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b:
        return 0.0
    a0, a1 = float(a[0]), float(a[1])
    b0, b1 = float(b[0]), float(b[1])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-9, max(a1, b1) - min(a0, b0))
    return inter / union


def best_cache_match_by_time(cache_video_clusters: List[Dict[str, Any]], cand_interval: List[float]) -> Tuple[Optional[Dict[str, Any]], float]:
    best = None
    best_score = 0.0
    for c in cache_video_clusters:
        iv = c.get("interval")
        if not isinstance(iv, (list, tuple)) or len(iv) < 2:
            continue
        score = interval_iou(cand_interval, [float(iv[0]), float(iv[1])])
        if score > best_score:
            best_score = score
            best = c
    return best, best_score


def merge_gap_intervals(intervals: List[List[float]], gap: float) -> List[List[float]]:
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


def cluster_ids_by_timegap(events_vdb: Any, ids: List[str], gap: float) -> List[Dict[str, Any]]:
    """
    Return list of clusters: [{"interval":[s,e], "ids":[...]}]
    """
    id_iv: List[Tuple[str, List[float]]] = []
    for eid in ids:
        iv = _event_interval(events_vdb, eid)
        if iv is not None:
            id_iv.append((eid, iv))
    if not id_iv:
        return []

    id_iv.sort(key=lambda x: (x[1][0], x[1][1]))
    merged = merge_gap_intervals([iv for _, iv in id_iv], gap)

    clusters: List[Dict[str, Any]] = [{"interval": iv, "ids": []} for iv in merged]

    for eid, iv in id_iv:
        s, e = iv
        for c in clusters:
            cs, ce = c["interval"]
            if not (e < cs or s > ce):
                c["ids"].append(eid)
                break

    # de-dup inside cluster
    for c in clusters:
        seen: Set[str] = set()
        c["ids"] = [x for x in c["ids"] if not (x in seen or seen.add(x))]
    return [c for c in clusters if c["ids"]]


def boundary_ids_global(events_vdb: Any, ids: List[str]) -> List[str]:
    """Return [first,last] by start time (or 1 id)."""
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


# ── Cache reuse / write helpers (shared for A and B) ─────────────────────────────
def reuse_from_cache_for_seed_clusters(
    cache_video_clusters: List[Dict[str, Any]],
    seed_clusters: List[Dict[str, Any]],
    thr: float,
    valid_event_ids: Set[str],
) -> Tuple[Set[str], Set[str], int, int, float]:
    """
    For each seed cluster interval, reuse cached cluster with best interval IoU if >= thr.
    Returns:
      reused_event_ids,
      covered_seed_ids (seeds that belong to clusters that hit),
      reused_clusters_count,
      reused_events_count,
      avg_time_reused_sec
    """
    reused_event_ids: Set[str] = set()
    covered_seed_ids: Set[str] = set()
    reused_times: List[float] = []
    reused_clusters = 0

    for sc in seed_clusters:
        cand_iv = sc["interval"]
        best, score = best_cache_match_by_time(cache_video_clusters, cand_iv)
        if best is not None and score >= thr:
            used = set(best.get("cluster_event_ids") or [])
            used &= valid_event_ids
            reused_event_ids |= used
            covered_seed_ids |= set(sc["ids"])
            reused_clusters += 1
            t = best.get("time_sec")
            if isinstance(t, (int, float)):
                reused_times.append(float(t))

    avg_time = sum(reused_times) / len(reused_times) if reused_times else 0.0
    return reused_event_ids, covered_seed_ids, reused_clusters, len(reused_event_ids), avg_time


def save_clusters_to_cache(
    cache_video_clusters: List[Dict[str, Any]],
    events_vdb: Any,
    ids_to_save: Set[str],
    gap: float,
    time_sec: float,
) -> int:
    """
    Cluster ids_to_save by time-gap and append clusters to cache.
    Each cache entry stores interval + cluster_event_ids + time_sec.
    Returns number of clusters saved.
    """
    clusters = cluster_ids_by_timegap(events_vdb, sorted(ids_to_save), gap)
    for c in clusters:
        cache_video_clusters.append({
            "interval": c["interval"],
            "cluster_event_ids": sorted(set(c["ids"])),
            "time_sec": float(time_sec) if isinstance(time_sec, (int, float)) else 0.0,
        })
    return len(clusters)


# ── Method A: Top-K Borda selection (None treated as +inf) ───────────────────────
def _sort_key_event(e: Dict[str, Any]) -> Tuple[int, float]:
    score = e.get("borda_score")
    if score is None:
        return (1, 0.0)
    try:
        return (0, -float(score))
    except Exception:
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

    sorted_scored = sorted(seed_events, key=lambda ev: (-_borda_score_max_none(ev), _sort_key_event(ev)))

    out: List[str] = []
    seen: Set[str] = set()
    for ev in sorted_scored:
        eid = ev.get("id") or ev.get("__id__")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        out.append(eid)
        if len(out) >= top_k:
            break
    return out


# ── VLM support (Method A optional) ──────────────────────────────────────────────
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


def _get_event_description(events_vdb: Any, event_id: str) -> str:
    try:
        data = events_vdb.get_data(event_id)
    except Exception:
        return ""
    if not data:
        return ""
    desc = data.get("description") or data.get("name") or ""
    if isinstance(desc, list):
        desc = " ".join(str(d) for d in desc)
    dur = data.get("duration")
    if isinstance(dur, (list, tuple)) and len(dur) >= 2:
        try:
            return f"[{float(dur[0])}-{float(dur[1])}] {str(desc).strip()}"
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
    result: List[str] = []
    for v in arr[:num_expected]:
        s = (str(v).strip().lower() if v is not None else "no")
        result.append("yes" if s == "yes" else "no")
    while len(result) < num_expected:
        result.append("no")
    return result[:num_expected]


def verify_candidates_with_vlm(
    llm: Any,
    query_events: List[str],
    candidate_ids: List[str],
    events_vdb: Any,
    verify_rule: str,
    max_batch_size: int = 64,
) -> Tuple[Set[str], Dict[str, Any]]:
    """
    Returns verified_ids, and a matrix trace.
    """
    trace: Dict[str, Any] = {"candidate_ids": list(candidate_ids), "matrix": [], "verified_ids": []}
    if not query_events or not candidate_ids:
        verified = set(candidate_ids) if verify_rule == "any" else set()
        trace["verified_ids"] = sorted(verified)
        return verified, trace

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

    trace["matrix"] = matrix_rows
    trace["verified_ids"] = sorted(verified)
    return verified, trace


# ── Method A runner with cache reuse ─────────────────────────────────────────────
def run_method_a(
    entry: Dict[str, Any],
    events_vdb: Any,
    valid_event_ids: Set[str],
    topk_seed_ids: List[str],
    budget: int,
    use_vlm: bool,
    vlm: Any,
    verify_rule: str,
    cache_video_clusters: Optional[List[Dict[str, Any]]],
    cache_overlap_threshold: float,
    cache_cluster_gap: float,
) -> Tuple[Set[str], Dict[str, Any], Dict[str, Any], float]:
    """
    Returns:
      out_A_ids,
      trace_A,
      cache_meta_A,
      time_spent_sec (from expansion, rough; we only have time from VLM / expand_and_metrics, so keep 0 here)
    """
    trace: Dict[str, Any] = {"seed_ids": list(topk_seed_ids)}
    cache_meta = {
        "reused_clusters": 0,
        "reused_events": 0,
        "avg_time_reused_sec": 0.0,
        "cache_enabled": bool(cache_video_clusters is not None),
    }

    seeds = [x for x in topk_seed_ids if x in valid_event_ids]
    if not seeds:
        return set(), {"reason": "no_seeds"}, cache_meta, 0.0

    # Cluster seeds (for cache hit detection)
    seed_clusters = cluster_ids_by_timegap(events_vdb, seeds, gap=cache_cluster_gap)
    trace["seed_clusters_n"] = len(seed_clusters)

    reused_ids: Set[str] = set()
    covered_seed_ids: Set[str] = set()
    if cache_video_clusters is not None:
        reused_ids, covered_seed_ids, rc, re, avg_t = reuse_from_cache_for_seed_clusters(
            cache_video_clusters, seed_clusters, cache_overlap_threshold, valid_event_ids
        )
        cache_meta["reused_clusters"] = rc
        cache_meta["reused_events"] = re
        cache_meta["avg_time_reused_sec"] = avg_t

    remaining_seeds = [x for x in seeds if x not in covered_seed_ids]
    trace["remaining_seeds_n"] = len(remaining_seeds)

    # Expand remaining seeds with Method A logic (FB)
    # candidates = remaining_seeds ∪ FB(remaining_seeds)
    candidates: List[str] = []
    if remaining_seeds:
        fb_all = expand_forward_backward(set(remaining_seeds), events_vdb) & valid_event_ids
        fb_candidates = fb_all - set(remaining_seeds)

        # deterministic ordering: seeds first, then extras
        candidates = list(remaining_seeds) + sorted(list(fb_candidates))
        if len(candidates) > budget:
            candidates = candidates[:budget]

    trace["batch_sent"] = list(candidates)

    if not use_vlm:
        verified = set(candidates)
        out_A = (reused_ids | verified) & valid_event_ids
        trace["verified_A_ids"] = sorted(verified)
        trace["out_A_n"] = len(out_A)
        return out_A, trace, cache_meta, 0.0

    if indus_prompts is None:
        raise RuntimeError("indus_prompts not available; cannot use --use-vlm")
    if vlm is None:
        raise RuntimeError("VLM is None but --use-vlm was set")

    question = entry.get("question", "")
    options = entry.get("options", "")

    # Prompt 1
    prompt_p1 = indus_prompts.INDUS_PROMPT["query_event_extraction"].format(
        question=question,
        options=options or "",
    )
    out_p1 = vlm.batch_generate_response([{"text": prompt_p1}])[0]
    query_events = _parse_query_events_from_llm(out_p1) or ["Relevant to the question or options"]
    trace["query_events"] = list(query_events)

    verified, matrix_trace = verify_candidates_with_vlm(
        llm=vlm,
        query_events=query_events,
        candidate_ids=candidates,
        events_vdb=events_vdb,
        verify_rule=verify_rule,
    )
    trace["vlm_matrix"] = matrix_trace
    trace["verified_A_ids"] = sorted(verified)

    out_A = (reused_ids | verified) & valid_event_ids
    trace["out_A_n"] = len(out_A)
    return out_A, trace, cache_meta, 0.0


# ── Method B runner with cache reuse (no boundary clustering logic) ──────────────
def run_method_b(
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
    cache_video_clusters: Optional[List[Dict[str, Any]]],
    cache_overlap_threshold: float,
    cache_cluster_gap: float,
) -> Tuple[Set[str], Dict[str, Any], Dict[str, Any], float]:
    trace: Dict[str, Any] = {"seed_ids": list(topk_seed_ids)}
    cache_meta = {
        "reused_clusters": 0,
        "reused_events": 0,
        "avg_time_reused_sec": 0.0,
        "cache_enabled": bool(cache_video_clusters is not None),
    }

    seeds = [x for x in topk_seed_ids if x in valid_event_ids]
    if not seeds:
        return set(), {"reason": "no_seeds"}, cache_meta, 0.0

    # Cluster seeds (for cache hit detection)
    seed_clusters = cluster_ids_by_timegap(events_vdb, seeds, gap=cache_cluster_gap)
    trace["seed_clusters_n"] = len(seed_clusters)

    reused_ids: Set[str] = set()
    covered_seed_ids: Set[str] = set()
    if cache_video_clusters is not None:
        reused_ids, covered_seed_ids, rc, re, avg_t = reuse_from_cache_for_seed_clusters(
            cache_video_clusters, seed_clusters, cache_overlap_threshold, valid_event_ids
        )
        cache_meta["reused_clusters"] = rc
        cache_meta["reused_events"] = re
        cache_meta["avg_time_reused_sec"] = avg_t

    remaining_seeds = [x for x in seeds if x not in covered_seed_ids]
    trace["remaining_seeds_n"] = len(remaining_seeds)

    kept_B: Set[str] = set(seeds) | reused_ids
    time_fb_used = 0.0

    if remaining_seeds:
        boundary_ids = boundary_ids_global(events_vdb, remaining_seeds)
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
            trace["expand_error"] = str(e)

    trace["out_B_n"] = len(kept_B)
    return kept_B & valid_event_ids, trace, cache_meta, time_fb_used


# ── main driver ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Merged A+B with shared cache reuse by time overlap (interval IoU).")

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
    p.add_argument("--cache-overlap-threshold", type=float, default=0.5, help="Interval IoU threshold (default 0.5).")
    p.add_argument("--cache-cluster-gap", type=float, default=0.0, help="Gap sec to cluster TopK seeds for cache detection.")
    p.add_argument("--final-cluster-gap", type=float, default=0.0, help="Gap sec to cluster FINAL merged output before saving to cache.")
    p.add_argument("--cache-save-source",
                   choices=["final", "A", "B", "all"],
                   default="final",
                   help="What to save into cache: final only (default), A only, B only, or all three.")

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

        # shared cache pool per video_key
        cache_video_clusters: Optional[List[Dict[str, Any]]] = None
        if cache_obj is not None:
            videos = cache_obj.setdefault("videos", {})
            cache_video_clusters = videos.setdefault(str(vk), [])
            if not isinstance(cache_video_clusters, list):
                cache_video_clusters = []
                videos[str(vk)] = cache_video_clusters

        # SAME Top-K seeds for both methods
        topk_seed_ids = select_top_k_borda_seed_ids(entry, args.top_k_seeds)
        topk_seed_ids = [x for x in topk_seed_ids if x in valid_event_ids]

        # Method A (with cache reuse)
        out_A, trace_A, cache_meta_A, time_A = run_method_a(
            entry=entry,
            events_vdb=events_vdb,
            valid_event_ids=valid_event_ids,
            topk_seed_ids=topk_seed_ids,
            budget=args.budget,
            use_vlm=args.use_vlm,
            vlm=vlm,
            verify_rule=args.verify_rule,
            cache_video_clusters=cache_video_clusters,
            cache_overlap_threshold=args.cache_overlap_threshold,
            cache_cluster_gap=args.cache_cluster_gap,
        )

        # Method B (with cache reuse)
        out_B, trace_B, cache_meta_B, time_B = run_method_b(
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
            cache_video_clusters=cache_video_clusters,
            cache_overlap_threshold=args.cache_overlap_threshold,
            cache_cluster_gap=args.cache_cluster_gap,
        )

        # Merge
        if args.merge_mode == "intersection":
            final_ids = (out_A & out_B) & valid_event_ids
        else:
            final_ids = (out_A | out_B) & valid_event_ids

        # Save to cache (shared pool) after handling everything
        clusters_saved = 0
        if cache_video_clusters is not None:
            if args.cache_save_source in ("A", "all"):
                clusters_saved += save_clusters_to_cache(
                    cache_video_clusters, events_vdb, out_A, args.final_cluster_gap, time_A
                )
            if args.cache_save_source in ("B", "all"):
                clusters_saved += save_clusters_to_cache(
                    cache_video_clusters, events_vdb, out_B, args.final_cluster_gap, time_B
                )
            if args.cache_save_source in ("final", "all"):
                clusters_saved += save_clusters_to_cache(
                    cache_video_clusters, events_vdb, final_ids, args.final_cluster_gap, max(time_A, time_B)
                )

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
                "methodA_n": len(out_A),
                "methodB_n": len(out_B),
                "final_n": len(final_ids),
            },
            "cache_meta": {
                "A": cache_meta_A,
                "B": cache_meta_B,
                "clusters_saved_this_query": clusters_saved,
                "cache_overlap_threshold_iou": args.cache_overlap_threshold,
                "cache_cluster_gap": args.cache_cluster_gap,
                "final_cluster_gap": args.final_cluster_gap,
                "cache_save_source": args.cache_save_source,
            },
            "trace_methodA": trace_A,
            "trace_methodB": trace_B,
        })

    # Save cache
    if cache_obj is not None and cache_path is not None:
        save_cache(cache_path, cache_obj)
        print(f"[Cache] Saved: {cache_path}")

    # Save output json
    out_dir = retrieval_dir / "indus_explore"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_events_{dataset}_merged_final_{args.merge_mode}_{args.fb_mode}_sharedcache_timeiou.json"
    out_path.write_text(json.dumps(_jsonify(out_entries), indent=2, ensure_ascii=False))
    print(f"\n✓ Saved merged output: {out_path} ({len(out_entries)} entries)")


if __name__ == "__main__":
    main()