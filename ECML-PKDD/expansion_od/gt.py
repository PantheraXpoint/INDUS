"""Ground truth loading for AVA100."""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

from .time_utils import time_ref_to_segments
from .config import GT_BASE, GT_FILES


def load_questions(base: Path = GT_BASE) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """(video_key, qid_str) -> {question, options, answer, answer_statements} for object extraction."""
    out = {}
    if not base.exists():
        return out
    for name in GT_FILES:
        p = base / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            if not isinstance(data, list):
                continue
            for v in data:
                vk = v.get("video_key", v.get("key"))
                if not vk:
                    continue
                for qa in v.get("qa", []):
                    qid = qa.get("uid", qa.get("question_id"))
                    if qid is None:
                        continue
                    query = qa.get("query", qa.get("question", ""))
                    options = qa.get("options", [])
                    answer = qa.get("answer", "")
                    # answer_statements: chosen option text for object extraction
                    answer_statements = []
                    if answer and len(answer) == 1 and options:
                        idx = ord(answer.upper()) - ord("A")
                        if 0 <= idx < len(options):
                            opt = options[idx]
                            if isinstance(opt, str) and opt.startswith(tuple("ABCD.")):
                                opt = opt.split(".", 1)[-1].strip()
                            answer_statements.append(opt)
                    out[(vk, str(qid))] = {
                        "question": query,
                        "options": options,
                        "answer": answer,
                        "answer_statements": answer_statements,
                    }
        except Exception:
            continue
    return out


def load_gt(base: Path = GT_BASE) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    """(video_key, qid_str) -> list of (start_sec, end_sec)."""
    out = {}
    if not base.exists():
        return out
    for name in GT_FILES:
        p = base / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            if not isinstance(data, list):
                continue
            for v in data:
                vk = v.get("video_key", v.get("key"))
                if not vk:
                    continue
                for qa in v.get("qa", []):
                    qid = qa.get("uid", qa.get("question_id"))
                    if qid is None:
                        continue
                    segs = time_ref_to_segments(qa.get("time_reference"))
                    if segs:
                        out[(vk, str(qid))] = segs
        except Exception:
            continue
    return out
