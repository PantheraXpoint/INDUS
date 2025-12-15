#!/usr/bin/env python3
"""
Evaluation script for video QA results
Supports two evaluation modes:
1. Highest confidence: Uses the answer with the highest confidence score
2. Most popular: Uses the most frequently occurring answer across all results
"""

import os
import json
import argparse
from collections import Counter
from typing import Dict, List, Tuple
from dataset.init_dataset import init_dataset, get_video_idx


def load_answers(answers_path: str) -> List[Dict]:
    """Load answers from JSON file."""
    if not os.path.exists(answers_path):
        return []
    with open(answers_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_answer(answer_str: str) -> str:
    """
    Extract answer letter (A, B, C, D) from answer string.
    Handles cases like "A", "[]", "", or JSON strings.
    """
    if not answer_str or answer_str.strip() == "":
        return None
    
    # Remove quotes and brackets
    answer_str = answer_str.strip().strip('"').strip("'").strip('[]')
    
    # Check if it's a valid answer (A, B, C, or D)
    if answer_str.upper() in ['A', 'B', 'C', 'D']:
        return answer_str.upper()
    
    return None


def get_highest_confidence_answer(answers: List[Dict]) -> str:
    """
    Get the answer from the result with the highest confidence score.
    
    Args:
        answers: List of answer dictionaries with 'Answer' and 'score' fields
        
    Returns:
        Answer letter (A, B, C, D) or None if no valid answer found
    """
    if not answers:
        return None
    
    # Find the answer with the highest score
    best_answer = max(answers, key=lambda x: x.get('score', 0.0))
    return extract_answer(best_answer.get('Answer', ''))


def get_most_popular_answer(answers: List[Dict]) -> str:
    """
    Get the most frequently occurring answer across all results.
    In case of ties, returns the first one alphabetically.
    
    Args:
        answers: List of answer dictionaries with 'Answer' field
        
    Returns:
        Most popular answer letter (A, B, C, D) or None if no valid answer found
    """
    if not answers:
        return None
    
    # Extract all valid answers
    valid_answers = []
    for ans in answers:
        extracted = extract_answer(ans.get('Answer', ''))
        if extracted:
            valid_answers.append(extracted)
    
    if not valid_answers:
        return None
    
    # Count occurrences
    answer_counts = Counter(valid_answers)
    
    # Get the most common answer
    # In case of tie, use the one that appears first (alphabetically)
    most_common = answer_counts.most_common(1)[0][0]
    
    # If there's a tie, prefer the alphabetically first one
    max_count = answer_counts[most_common]
    tied_answers = [ans for ans, count in answer_counts.items() if count == max_count]
    return sorted(tied_answers)[0]


def evaluate_dataset(
    dataset_name: str,
    method: str = "both"
) -> Dict[str, Dict]:
    """
    Evaluate all videos and questions in a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ava100')
        method: Evaluation method - 'highest_conf', 'most_popular', or 'both'
        
    Returns:
        Dictionary with evaluation results
    """
    # Initialize dataset
    dataset = init_dataset(dataset_name)
    video_start, video_end = get_video_idx(dataset_name)
    
    # Statistics
    total_questions = 0
    highest_conf_correct = 0
    most_popular_correct = 0
    
    # Per-video statistics
    video_stats = {}
    
    print(f"Evaluating dataset: {dataset_name}")
    print(f"Video range: {video_start} to {video_end}")
    print(f"Method: {method}")
    print("-" * 80)
    
    # Iterate through all videos
    for video_id in range(video_start, video_end + 1):
        try:
            video_info = dataset.get_video_info(video_id)
            video_key = video_info.get('video_key', f'video{video_id}')
            qa_pairs = video_info.get('qa', [])
            
            video_highest_conf_correct = 0
            video_most_popular_correct = 0
            video_total = 0
            
            # Iterate through all questions
            for qa in qa_pairs:
                question_id = qa.get('question_id', -1)
                ground_truth = qa.get('answer', '').strip().upper()
                
                if not ground_truth or ground_truth not in ['A', 'B', 'C', 'D']:
                    continue
                
                # Load answers
                answers_path = os.path.join('database', video_key, str(question_id), 'answers.json')
                answers = load_answers(answers_path)
                
                if not answers:
                    continue
                
                video_total += 1
                total_questions += 1
                
                # Evaluate using highest confidence method
                if method in ['highest_conf', 'both']:
                    predicted_highest = get_highest_confidence_answer(answers)
                    if predicted_highest == ground_truth:
                        highest_conf_correct += 1
                        video_highest_conf_correct += 1
                
                # Evaluate using most popular method
                if method in ['most_popular', 'both']:
                    predicted_popular = get_most_popular_answer(answers)
                    if predicted_popular == ground_truth:
                        most_popular_correct += 1
                        video_most_popular_correct += 1
            
            # Store video statistics
            if video_total > 0:
                video_stats[video_id] = {
                    'video_key': video_key,
                    'total_questions': video_total,
                    'highest_conf_correct': video_highest_conf_correct,
                    'most_popular_correct': video_most_popular_correct,
                    'highest_conf_accuracy': video_highest_conf_correct / video_total if method in ['highest_conf', 'both'] else None,
                    'most_popular_accuracy': video_most_popular_correct / video_total if method in ['most_popular', 'both'] else None
                }
                
                print(f"Video {video_id} ({video_key}): {video_total} questions")
                if method in ['highest_conf', 'both']:
                    print(f"  Highest Conf: {video_highest_conf_correct}/{video_total} = {video_stats[video_id]['highest_conf_accuracy']:.4f}")
                if method in ['most_popular', 'both']:
                    print(f"  Most Popular: {video_most_popular_correct}/{video_total} = {video_stats[video_id]['most_popular_accuracy']:.4f}")
        
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue
    
    # Calculate overall accuracy
    results = {
        'dataset': dataset_name,
        'total_questions': total_questions,
        'video_stats': video_stats
    }
    
    if method in ['highest_conf', 'both']:
        results['highest_conf'] = {
            'correct': highest_conf_correct,
            'total': total_questions,
            'accuracy': highest_conf_correct / total_questions if total_questions > 0 else 0.0
        }
    
    if method in ['most_popular', 'both']:
        results['most_popular'] = {
            'correct': most_popular_correct,
            'total': total_questions,
            'accuracy': most_popular_correct / total_questions if total_questions > 0 else 0.0
        }
    
    return results


def print_results(results: Dict):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Dataset: {results['dataset']}")
    print(f"Total Questions: {results['total_questions']}")
    print("-" * 80)
    
    if 'highest_conf' in results:
        hc = results['highest_conf']
        print(f"\nHighest Confidence Method:")
        print(f"  Correct: {hc['correct']}/{hc['total']}")
        print(f"  Accuracy: {hc['accuracy']:.4f} ({hc['accuracy']*100:.2f}%)")
    
    if 'most_popular' in results:
        mp = results['most_popular']
        print(f"\nMost Popular Method:")
        print(f"  Correct: {mp['correct']}/{mp['total']}")
        print(f"  Accuracy: {mp['accuracy']:.4f} ({mp['accuracy']*100:.2f}%)")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate video QA results')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., ava100, lvbench, videomme)')
    parser.add_argument('--method', type=str, default='both',
                       choices=['highest_conf', 'most_popular', 'both'],
                       help='Evaluation method: highest_conf, most_popular, or both (default: both)')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional: Path to save results as JSON file')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_dataset(args.dataset, args.method)
    
    # Print results
    print_results(results)
    
    # Save results if output path is provided
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

