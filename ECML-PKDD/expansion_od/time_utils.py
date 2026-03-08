"""Time parsing and segment conversion."""
import re
from pathlib import Path
from typing import List, Tuple, Optional


def time_to_seconds(s: str) -> int:
    """Parse HH:MM:SS or MM:SS to seconds."""
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty time string")
    parts = [p.strip() for p in re.split(r"[:.]", s) if p.strip().replace(" ", "").isdigit()]
    if not parts:
        raise ValueError(f"Invalid: {s}")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) >= 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(parts[0])


def time_ref_to_segments(time_ref: str) -> Optional[List[Tuple[int, int]]]:
    """time_reference string -> [(start_sec, end_sec)]. None if invalid."""
    if not time_ref or time_ref.strip() in ("N/A", "", "None", "None-None"):
        return None
    time_ref = time_ref.strip()
    try:
        if "-" in time_ref:
            a, b = time_ref.split("-", 1)
            a, b = a.strip() or b.strip(), b.strip() or a.strip()
            return [(time_to_seconds(a), time_to_seconds(b))] if a and b else None
        if "," in time_ref:
            pts = [p.strip() for p in time_ref.split(",") if p.strip()]
            return [(time_to_seconds(p), time_to_seconds(p)) for p in pts] if pts else None
        t = time_to_seconds(time_ref)
        return [(t, t)]
    except (ValueError, IndexError):
        return None
