from typing import List, Dict
import numpy as np


class Evaluator:
    """Evaluate predictions for both tasks."""
    @staticmethod
    def evaluate_retrieval(predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Evaluate Task 1 using precision, recall, and F2."""
        f2_scores = []
        for pred, gt in zip(predictions, ground_truth):
            pred_articles = {(art['law_id'], art['article_id']) for art in pred['relevant_articles']}
            gt_articles = {(art['law_id'], art['article_id']) for art in gt['relevant_articles']}
            
            correct = len(pred_articles & gt_articles)
            precision = correct / len(pred_articles) if pred_articles else 0
            recall = correct / len(gt_articles) if gt_articles else 0
            f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
            f2_scores.append(f2)
        
        return {"F2": np.mean(f2_scores)}

    @staticmethod
    def evaluate_qa(predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Evaluate Task 2 using accuracy."""
        correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred['answer'] == gt['answer'])
        return {"accuracy": correct / len(ground_truth)}