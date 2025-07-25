#!/usr/bin/env python3
"""
Script to combine relevant articles from two JSON files:
- Ground truth file (alqac25_private_test_task2.json)
- Prediction file (entry01.json)

The output file will have the same format as the ground truth file
but with extended relevant articles.
"""

import json
import os
from collections import defaultdict


def load_json_file(file_path):
    """Load and parse a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data, file_path):
    """Save data to a JSON file with proper formatting."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def combine_relevant_articles(ground_truth_file, prediction_file, output_file):
    """
    Combine relevant articles from ground truth and prediction files.

    Args:
        ground_truth_file: Path to the ground truth file
        prediction_file: Path to the prediction file
        output_file: Path where the combined file will be saved
    """
    # Load both files
    ground_truth = load_json_file(ground_truth_file)
    predictions = load_json_file(prediction_file)

    # Create a mapping from question_id to predicted relevant articles
    prediction_map = {}
    for item in predictions:
        question_id = item.get('question_id')
        relevant_articles = item.get('relevant_articles', [])
        prediction_map[question_id] = relevant_articles

    # Create the combined output
    combined_output = []

    # Process each question in the ground truth file
    for question in ground_truth:
        question_id = question.get('question_id')

        # Skip empty entries if any
        if not question_id:
            combined_output.append(question)
            continue

        # Get the relevant articles from both sources
        gt_relevant_articles = question.get('relevant_articles', [])
        pred_relevant_articles = prediction_map.get(question_id, [])

        # Create a dictionary to track unique articles (by law_id and article_id combination)
        unique_articles = {}

        # Add ground truth articles first
        for article in gt_relevant_articles:
            key = (article.get('law_id', ''), article.get('article_id', ''))
            unique_articles[key] = article

        # Add prediction articles if not already present
        for article in pred_relevant_articles:
            key = (article.get('law_id', ''), article.get('article_id', ''))
            if key not in unique_articles:
                unique_articles[key] = article

        # Create a new question object with the combined relevant articles
        new_question = question.copy()
        new_question['relevant_articles'] = list(unique_articles.values())

        combined_output.append(new_question)

    # Save the combined output to a file
    save_json_file(combined_output, output_file)
    print(f"Combined file saved to: {output_file}")


def main():
    # Define file paths
    ground_truth_file = 'dataset/alqac25_private_test_task2.json'
    prediction_file = 'temp/entry_task1/entry01.json'
    output_file = 'dataset/alqac25_private_test_task2_combined.json'

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Combine the relevant articles
    combine_relevant_articles(ground_truth_file, prediction_file, output_file)

    # Print statistics about the merge
    gt_data = load_json_file(ground_truth_file)
    pred_data = load_json_file(prediction_file)
    combined_data = load_json_file(output_file)

    # Count total articles before and after
    gt_articles = sum(len(q.get('relevant_articles', []))
                      for q in gt_data if q.get('question_id'))
    pred_articles = sum(len(q.get('relevant_articles', []))
                        for q in pred_data if q.get('question_id'))
    combined_articles = sum(len(q.get('relevant_articles', []))
                            for q in combined_data if q.get('question_id'))

    print(f"Statistics:")
    print(f"- Ground truth articles: {gt_articles}")
    print(f"- Prediction articles: {pred_articles}")
    print(f"- Combined unique articles: {combined_articles}")
    print(
        f"- Additional unique articles from predictions: {combined_articles - gt_articles}")


if __name__ == "__main__":
    main()
