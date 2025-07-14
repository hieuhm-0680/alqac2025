from typing import List, Dict, Union, Tuple
import numpy as np
import math
import json
import datetime
import os


class Evaluator:
    """Evaluate predictions for both tasks."""

    @staticmethod
    def evaluate_retrieval(
        predictions: List[Dict],
        ground_truth: List[Dict],
        k_values: List[int] = [1, 3, 5, 10],
        save_to_file: str = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation for Task 1 (Legal Document Retrieval).

        Args:
            predictions: List of predictions with 'question_id' and 'relevant_articles'
            ground_truth: List of ground truth with 'question_id' and 'relevant_articles'
            k_values: List of k values for top-k metrics
            save_to_file: Path to save the results as JSON file

        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {}

        # Convert predictions and ground truth to dictionaries for easier lookup
        pred_dict = {item['question_id']: item for item in predictions}
        gt_dict = {item['question_id']: item for item in ground_truth}

        # Ensure we have matching question IDs
        common_ids = set(pred_dict.keys()) & set(gt_dict.keys())
        if len(common_ids) != len(ground_truth):
            print(
                f"Warning: Only {len(common_ids)} out of {len(ground_truth)} questions have predictions")

        # Initialize metric collections
        all_precisions = {k: [] for k in k_values}
        all_recalls = {k: [] for k in k_values}
        all_accuracies = {k: [] for k in k_values}
        all_f2_scores_k = {k: [] for k in k_values}
        all_f2_scores = []
        all_ap_scores = []  # For MAP calculation
        all_rr_scores = []  # For MRR calculation
        all_dcg_scores = []  # For NDCG calculation

        for question_id in common_ids:
            pred = pred_dict[question_id]
            gt = gt_dict[question_id]

            # Extract article sets
            pred_articles = [(art['law_id'], art['article_id'])
                             for art in pred['relevant_articles']]
            gt_articles = {(art['law_id'], art['article_id'])
                           for art in gt['relevant_articles']}

            # Calculate metrics for different k values
            for k in k_values:
                pred_k = pred_articles[:k]
                pred_k_set = set(pred_k)

                # Precision@k
                if pred_k:
                    precision_k = len(
                        pred_k_set & gt_articles) / len(pred_k_set)
                else:
                    precision_k = 0.0
                all_precisions[k].append(precision_k)

                # Recall@k
                if gt_articles:
                    recall_k = len(pred_k_set & gt_articles) / len(gt_articles)
                else:
                    recall_k = 0.0
                all_recalls[k].append(recall_k)

                # Accuracy@k (Hit@k - whether at least one relevant item is in top-k)
                accuracy_k = 1.0 if (pred_k_set & gt_articles) else 0.0
                all_accuracies[k].append(accuracy_k)

                # F2 Score@k
                if pred_k_set and gt_articles:
                    correct = len(pred_k_set & gt_articles)
                    precision = correct / len(pred_k_set)
                    recall = correct / len(gt_articles)
                    if precision + recall > 0:
                        f2 = (5 * precision * recall) / \
                            (4 * precision + recall)
                    else:
                        f2 = 0.0
                else:
                    f2 = 0.0
                all_f2_scores_k[k].append(f2)

            # F2 Score (using all predictions, not limited to k)
            pred_all_set = set(pred_articles)
            if pred_all_set and gt_articles:
                correct = len(pred_all_set & gt_articles)
                precision = correct / len(pred_all_set)
                recall = correct / len(gt_articles)
                if precision + recall > 0:
                    f2 = (5 * precision * recall) / (4 * precision + recall)
                else:
                    f2 = 0.0
            else:
                f2 = 0.0
            all_f2_scores.append(f2)

            # Average Precision (for MAP@10)
            ap = Evaluator._calculate_average_precision(
                pred_articles[:10], gt_articles)
            all_ap_scores.append(ap)

            # Reciprocal Rank (for MRR@10)
            rr = Evaluator._calculate_reciprocal_rank(
                pred_articles[:10], gt_articles)
            all_rr_scores.append(rr)

            # DCG@10 (for NDCG@10)
            dcg = Evaluator._calculate_dcg(pred_articles[:10], gt_articles)
            idcg = Evaluator._calculate_ideal_dcg(gt_articles, 10)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            all_dcg_scores.append(ndcg)

        # Aggregate results
        for k in k_values:
            results[f'accuracy@{k}'] = np.mean(all_accuracies[k])
            results[f'precision@{k}'] = np.mean(all_precisions[k])
            results[f'recall@{k}'] = np.mean(all_recalls[k])
            results[f'f2_score@{k}'] = np.mean(all_f2_scores_k[k])

        results['f2_score'] = np.mean(all_f2_scores)
        results['map@10'] = np.mean(all_ap_scores)
        results['mrr@10'] = np.mean(all_rr_scores)
        results['ndcg@10'] = np.mean(all_dcg_scores)

        # Save results to file if path is provided
        if save_to_file:
            Evaluator.save_results_to_file(
                results, save_to_file, task_type="retrieval")

        return results

    @staticmethod
    def _calculate_average_precision(pred_articles: List[Tuple], gt_articles: set) -> float:
        """Calculate Average Precision for a single query."""
        if not gt_articles:
            return 0.0

        relevant_count = 0
        precision_sum = 0.0

        for i, article in enumerate(pred_articles, 1):
            if article in gt_articles:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i

        return precision_sum / len(gt_articles) if gt_articles else 0.0

    @staticmethod
    def _calculate_reciprocal_rank(pred_articles: List[Tuple], gt_articles: set) -> float:
        """Calculate Reciprocal Rank for a single query."""
        for i, article in enumerate(pred_articles, 1):
            if article in gt_articles:
                return 1.0 / i
        return 0.0

    @staticmethod
    def _calculate_dcg(pred_articles: List[Tuple], gt_articles: set) -> float:
        """Calculate Discounted Cumulative Gain."""
        dcg = 0.0
        for i, article in enumerate(pred_articles, 1):
            relevance = 1.0 if article in gt_articles else 0.0
            dcg += relevance / math.log2(i + 1)
        return dcg

    @staticmethod
    def _calculate_ideal_dcg(gt_articles: set, k: int) -> float:
        """Calculate Ideal DCG (best possible DCG)."""
        num_relevant = min(len(gt_articles), k)
        idcg = 0.0
        for i in range(1, num_relevant + 1):
            idcg += 1.0 / math.log2(i + 1)
        return idcg

    @staticmethod
    def evaluate_qa(predictions: List[Dict], ground_truth: List[Dict], save_to_file: str = None) -> Dict[str, float]:
        """
        Evaluate Task 2 using accuracy.

        Args:
            predictions: List of predictions with 'question_id' and 'answer'
            ground_truth: List of ground truth with 'question_id' and 'answer'
            save_to_file: Optional path to save results to JSON file

        Returns:
            Dictionary containing evaluation metrics
        """
        pred_dict = {item['question_id']: item for item in predictions}
        gt_dict = {item['question_id']: item for item in ground_truth}

        common_ids = set(pred_dict.keys()) & set(gt_dict.keys())

        correct = 0
        total = len(common_ids)

        for question_id in common_ids:
            if pred_dict[question_id]['answer'] == gt_dict[question_id]['answer']:
                correct += 1

        results = {"accuracy": correct / total if total > 0 else 0.0}

        # Save results to file if path is provided
        if save_to_file:
            Evaluator.save_results_to_file(
                results, save_to_file, task_type="qa")

        return results

    @staticmethod
    def save_results_to_file(
        results: Dict[str, float],
        output_path: str,
        predictions_path: str = None,
        ground_truth_path: str = None,
        task_type: str = None,
        additional_info: Dict = None
    ) -> None:
        """
        Save evaluation results to a JSON file with metadata.

        Args:
            results: Dictionary containing evaluation metrics
            output_path: Path where to save the results JSON file
            predictions_path: Path to the predictions file (optional)
            ground_truth_path: Path to the ground truth file (optional)
            task_type: Type of task ('retrieval' or 'qa')
            additional_info: Additional information to include in the output
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Prepare the output data structure
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "task_type": task_type,
            "predictions_file": predictions_path,
            "ground_truth_file": ground_truth_path,
            "metrics": results,
            "summary": {
                "total_metrics": len(results),
                "best_metric": max(results.items(), key=lambda x: x[1]) if results else None,
                "worst_metric": min(results.items(), key=lambda x: x[1]) if results else None
            }
        }

        # Add additional information if provided
        if additional_info:
            output_data["additional_info"] = additional_info

        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving results to {output_path}: {e}")
            raise

    @staticmethod
    def evaluate(
        predictions: List[Dict],
        ground_truth: List[Dict],
        task_type: str = "retrieval",
        k_values: List[int] = [1, 3, 5, 10],
        save_to_file: str = None,
        predictions_path: str = None,
        ground_truth_path: str = None
    ) -> Dict[str, float]:
        """
        Main evaluation function that automatically detects task type and evaluates accordingly.

        Args:
            predictions: List of predictions
            ground_truth: List of ground truth
            task_type: Either "retrieval" or "qa"
            k_values: List of k values for retrieval metrics
            save_to_file: Optional path to save results to JSON file
            predictions_path: Path to predictions file (for metadata)
            ground_truth_path: Path to ground truth file (for metadata)

        Returns:
            Dictionary containing evaluation metrics
        """
        if task_type == "retrieval":
            results = Evaluator.evaluate_retrieval(
                predictions, ground_truth, k_values)
        elif task_type == "qa":
            results = Evaluator.evaluate_qa(predictions, ground_truth)
        else:
            raise ValueError(
                f"Unknown task type: {task_type}. Must be 'retrieval' or 'qa'")

        if save_to_file:
            Evaluator.save_results_to_file(
                results,
                save_to_file,
                predictions_path=predictions_path,
                ground_truth_path=ground_truth_path,
                task_type=task_type
            )

        return results
