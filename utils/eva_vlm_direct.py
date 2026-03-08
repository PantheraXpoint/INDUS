#!/usr/bin/env python3
"""
Evaluate accuracy of VLM direct output file.
Compares ground truth "answer" with predicted answer extracted from "response".
"""
import json
import re
import sys


def extract_predicted_answer(response):
    """Extract predicted letter (A/B/C/D) from response. Returns None if not found."""
    if not response or not isinstance(response, str):
        return None
    # Try to parse JSON (response may be wrapped in ```json ... ```)
    text = response.strip()
    # Remove markdown code block if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Try JSON first
    try:
        data = json.loads(text)
        ans = data.get("Answer") or data.get("answer")
        if ans and str(ans).upper() in ("A", "B", "C", "D"):
            return str(ans).upper()
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: look for "Answer": "X" or '"Answer":"X"'
    m = re.search(r'"Answer"\s*:\s*"([ABCD])"', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b([ABCD])\s*[\.\)]\s*$', text)
    if m:
        return m.group(1).upper()
    print(f"No predicted answer found in response: {response}")
    return None


def evaluate(output_path):
    with open(output_path, "r") as f:
        results = json.load(f)
    correct = 0
    total = 0
    no_pred = 0
    not_processed = []
    existing_questions = set()
    for item in results:
        if (item.get("video_id"), item.get("question_id")) in existing_questions:
            continue
        gt = (item.get("answer") or "").strip().upper()
        if gt not in ("A", "B", "C", "D"):
            continue
        pred = extract_predicted_answer(item.get("response"))
        if pred is None:
            no_pred += 1
            not_processed.append((item.get("video_id"), item.get("question_id")))
            continue
        total += 1
        existing_questions.add((item.get("video_id"), item.get("question_id")))
        if pred == gt:
            correct += 1
    acc = (correct / total * 100) if total else 0
    print(f"File: {output_path}")
    print(f"Total:  {total}")
    print(f"Correct: {correct}")
    print(f"No valid prediction: {no_pred}")
    print(f"Accuracy: {acc:.2f}%")
    return acc, not_processed


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/query_VLM_direct_ava100_qwenvl_process1.json"
    acc, not_processed = evaluate(path)
    # remove the not processed questions from the file
    # with open(path, "r") as f:
    #     results = json.load(f)
    # for video_id, question_id in not_processed:
    #     results = [item for item in results if item.get("video_id") != video_id or item.get("question_id") != question_id]
    # with open(path, "w") as f:
    #     json.dump(results, f, indent=4)