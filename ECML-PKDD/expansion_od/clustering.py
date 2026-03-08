"""Temporal clustering: top-k seeds, merge within T sec, hit/total-time."""
from typing import List, Tuple, Dict, Any, Optional

Interval = Tuple[int, int]
MAX_SCORE = 1000000


def intervals_from_events(events: List[Dict[str, Any]]) -> List[Interval]:
    """(start, end) from seed_events durations, sorted by start."""
    out = []
    for ev in events:
        d = ev.get("duration")
        if not isinstance(d, (list, tuple)) or len(d) < 2:
            continue
        a, b = int(d[0]), int(d[1])
        if b > a:
            out.append((a, b))
    out.sort(key=lambda x: x[0])
    return out


def top_k_intervals(
    intervals: List[Interval], k: int, events: Optional[List[Dict[str, Any]]] = None
) -> List[Interval]:
    """First k by borda_score desc if present, else by start time."""
    if k >= len(intervals):
        return list(intervals)
    if events and any(isinstance(e.get("borda_score"), (int, float)) for e in events):
        id2int = {}
        id2sc = {}
        for e in events:
            d = e.get("duration")
            if not isinstance(d, (list, tuple)) or len(d) < 2:
                continue
            a, b = int(d[0]), int(d[1])
            if b <= a:
                continue
            iid = e.get("id") or e.get("__id__") or ""
            id2int[iid] = (a, b)
            sc = e.get("borda_score")
            if sc is None:
                sc = MAX_SCORE
            id2sc[iid] = float(sc)
        order = sorted(id2sc.keys(), key=lambda i: id2sc[i], reverse=True)[:k]
        out = [id2int[i] for i in order]
        out.sort(key=lambda x: x[0])
        return out
    return intervals[:k]


def merge_gap(intervals: List[Interval], T: int) -> List[Interval]:
    """Merge intervals with gap <= T (e.g. [200,210] and [220,250] with T=10 -> [200,250])."""
    if not intervals:
        return []
    srt = sorted(intervals, key=lambda x: x[0])
    out = [list(srt[0])]
    for s, e in srt[1:]:
        if s - out[-1][1] <= T:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def hits_gt(intervals: List[Interval], gt: List[Interval]) -> bool:
    """Any interval overlaps any GT segment."""
    for (a, b) in intervals:
        if b <= a:
            continue
        for (gs, ge) in gt:
            if ge < gs:
                continue
            if gs == ge:
                if a <= gs < b:
                    return True
            elif max(a, gs) < min(b, ge):
                return True
    return False


def total_sec(intervals: List[Interval]) -> float:
    return sum(e - s for s, e in intervals if e > s)


def events_to_id_intervals(events: List[Dict[str, Any]]) -> List[Tuple[str, Interval]]:
    """(event_id, (start_sec, end_sec)) from seed_events, sorted by start."""
    out = []
    for ev in events:
        d = ev.get("duration")
        if not isinstance(d, (list, tuple)) or len(d) < 2:
            continue
        a, b = int(d[0]), int(d[1])
        if b <= a:
            continue
        eid = (ev.get("id") or ev.get("__id__") or "").strip()
        if eid:
            out.append((eid, (a, b)))
    out.sort(key=lambda x: x[1][0])
    return out


def top_k_id_intervals(
    id_intervals: List[Tuple[str, Interval]], k: int, events: Optional[List[Dict[str, Any]]] = None
) -> List[Tuple[str, Interval]]:
    """First k (event_id, interval) by borda_score desc if present, else by start."""
    if k >= len(id_intervals):
        return list(id_intervals)
    if events:
        id2sc = {}
        for e in events:
            eid = e.get("id") or e.get("__id__") or ""
            sc = e.get("borda_score")
            id2sc[eid] = float(sc) if isinstance(sc, (int, float)) else MAX_SCORE
        order = sorted(
            [x for x in id_intervals if x[0] in id2sc],
            key=lambda x: id2sc[x[0]],
            reverse=True,
        )[:k]
        if len(order) < k:
            by_start = sorted(id_intervals, key=lambda x: x[1][0])
            seen = {x[0] for x in order}
            for x in by_start:
                if x[0] in seen:
                    continue
                order.append(x)
                if len(order) >= k:
                    break
        return sorted(order, key=lambda x: x[1][0])
    return id_intervals[:k]


def cluster_boundary_event_ids(
    id_intervals: List[Tuple[str, Interval]], merged: List[Interval]
) -> set:
    """For each merged interval, pick start event (min start) and end event (max end) among overlapping events. Return set of all such event ids."""
    out = set()
    for (S, E) in merged:
        if E <= S:
            continue
        overlapping = [(eid, (a, b)) for eid, (a, b) in id_intervals if max(a, S) < min(b, E)]
        if not overlapping:
            continue
        start_eid = min(overlapping, key=lambda x: x[1][0])[0]
        end_eid = max(overlapping, key=lambda x: x[1][1])[0]
        out.add(start_eid)
        out.add(end_eid)
    return out
