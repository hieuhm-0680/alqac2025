#!/usr/bin/env python3
"""
Script để lọc relevant articles dựa trên rerank_score và tính toán các metrics
"""

import json
import argparse
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict


def load_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def filter_by_top_k(relevant_articles: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    # Lọc các articles có đầy đủ thông tin (có rerank_score)
    valid_articles = [
        article for article in relevant_articles if 'rerank_score' in article]

    # Sắp xếp theo rerank_score giảm dần và lấy top-k
    sorted_articles = sorted(
        valid_articles, key=lambda x: x['rerank_score'], reverse=True)
    return sorted_articles[:k]


def filter_by_threshold(relevant_articles: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    # Lọc các articles có rerank_score >= threshold
    filtered_articles = [
        article for article in relevant_articles
        if 'rerank_score' in article and article['rerank_score'] >= threshold
    ]

    # Sắp xếp theo rerank_score giảm dần
    return sorted(filtered_articles, key=lambda x: x['rerank_score'], reverse=True)


def filter_data(data: List[Dict[str, Any]], method: str, value: float) -> List[Dict[str, Any]]:
    """
    Lọc dữ liệu theo phương pháp được chọn

    Args:
        data: Dữ liệu gốc
        method: Phương pháp lọc ('top_k' hoặc 'threshold')
        value: Giá trị k hoặc threshold

    Returns:
        Dữ liệu đã được lọc
    """
    count = 0
    count2 = 0
    filtered_data = []

    for question in data:
        question_id = question['question_id']
        relevant_articles = question.get('relevant_articles', [])
        count += len(relevant_articles)
        # relevant_articles = [
        #     {
        #         'law_id': article.get('law_id'),
        #         'article_id': article.get('article_id')
        #     }
        #     for article in question['relevant_articles']
        # ]
        count2 += len([ a for a in relevant_articles if a['rerank_score'] < value])

        if method == 'top_k':
            filtered_articles = filter_by_top_k(relevant_articles, int(value))
        elif method == 'threshold':
            filtered_articles = filter_by_threshold(relevant_articles, value)
        else:
            raise ValueError(f"Phương pháp không hỗ trợ: {method}")
        
        filtered_articles = [
            {
                'law_id': article.get('law_id'),
                'article_id': article.get('article_id'),
                'text': article.get('text'),
                'rerank_score': article.get('rerank_score', 0.0),
                'index': article.get('index', 0)
            }
            for article in filtered_articles
        ]

        filtered_question = {
            'question_id': question_id,
            'relevant_articles': filtered_articles,
        }

        filtered_data.append(filtered_question)
    print(f"Đã lọc {len(filtered_data)} câu hỏi với tổng số articles: {count}")
    print(f"Tổng số articles lọc: {count2}")
    return filtered_data


def extract_article_ids(question: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """
    Trích xuất các (law_id, article_id) từ một question

    Args:
        question: Dictionary chứa thông tin question

    Returns:
        Set các tuple (law_id, article_id)
    """
    article_ids = set()
    for article in question.get('relevant_articles', []):
        if 'law_id' in article and 'article_id' in article:
            article_ids.add((article['law_id'], article['article_id']))
    return article_ids


def calculate_metrics(predicted_data: List[Dict[str, Any]],
                      ground_truth_data: List[Dict[str, Any]]) -> Dict[str, float]:
    # Tạo mapping từ question_id đến ground truth articles
    gt_mapping = {}
    for question in ground_truth_data:
        question_id = question['question_id']
        gt_mapping[question_id] = extract_article_ids(question)

    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives

    question_metrics = []

    for question in predicted_data:
        question_id = question['question_id']
        predicted_articles = extract_article_ids(question)

        if question_id not in gt_mapping:
            print(
                f"Warning: Question {question_id} không có trong ground truth")
            continue

        gt_articles = gt_mapping[question_id]

        # Tính TP, FP, FN cho question này
        tp = len(predicted_articles.intersection(gt_articles))
        fp = len(predicted_articles - gt_articles)
        fn = len(gt_articles - predicted_articles)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Tính metrics cho từng question
        precision = tp / \
            len(predicted_articles) if len(predicted_articles) > 0 else 0
        recall = tp / len(gt_articles) if len(gt_articles) > 0 else 0
        f2 = (5 * precision * recall) / (4 * precision +
                                         recall) if (precision + recall) > 0 else 0

        question_metrics.append({
            'question_id': question_id,
            'precision': precision,
            'recall': recall,
            'f2': f2,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'predicted_count': len(predicted_articles),
            'gt_count': len(gt_articles)
        })

    # Tính macro và micro metrics
    macro_precision = sum([m['precision'] for m in question_metrics]) / \
        len(question_metrics) if question_metrics else 0
    macro_recall = sum([m['recall'] for m in question_metrics]) / \
        len(question_metrics) if question_metrics else 0
    macro_f2 = sum([m['f2'] for m in question_metrics]) / \
        len(question_metrics) if question_metrics else 0

    micro_precision = total_tp / \
        (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / \
        (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f2 = (5 * micro_precision * micro_recall) / (4 * micro_precision +
                                                       micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f2': macro_f2,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f2': micro_f2,
        'total_questions': len(question_metrics),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'question_metrics': question_metrics
    }


def print_summary_stats(data: List[Dict[str, Any]]):
    """
    In thống kê tóm tắt về dữ liệu

    Args:
        data: Dữ liệu đã được lọc
    """
    total_questions = len(data)
    total_original = sum([q['original_count'] for q in data])
    total_filtered = sum([q['filtered_count'] for q in data])

    avg_original = total_original / total_questions if total_questions > 0 else 0
    avg_filtered = total_filtered / total_questions if total_questions > 0 else 0

    print(f"\n=== THỐNG KÊ TÓM TẮT ===")
    print(f"Tổng số câu hỏi: {total_questions}")
    print(f"Tổng số articles gốc: {total_original}")
    print(f"Tổng số articles sau lọc: {total_filtered}")
    print(f"Trung bình articles/câu hỏi (gốc): {avg_original:.2f}")
    print(f"Trung bình articles/câu hỏi (sau lọc): {avg_filtered:.2f}")
    print(f"Tỉ lệ giữ lại: {(total_filtered/total_original*100):.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Lọc relevant articles và tính toán metrics')
    parser.add_argument('input_file', help='Đường dẫn file JSON đầu vào')
    parser.add_argument('--method', choices=['top_k', 'threshold'], required=True,
                        help='Phương pháp lọc: top_k hoặc threshold')
    parser.add_argument('--value', type=float, required=True,
                        help='Giá trị k (cho top_k) hoặc threshold (cho threshold)')
    parser.add_argument(
        '--output', help='Đường dẫn file JSON đầu ra (tùy chọn)')
    parser.add_argument(
        '--ground_truth', help='Đường dẫn file JSON ground truth để tính metrics')

    args = parser.parse_args()

    print(f"Đang load dữ liệu từ: {args.input_file}")
    data = load_data(args.input_file)
    print(f"Đã load {len(data)} câu hỏi")

    print(
        f"\nĐang lọc dữ liệu bằng phương pháp: {args.method} với giá trị: {args.value}")
    filtered_data = filter_data(data, args.method, args.value)

    # Lưu kết quả nếu có đường dẫn output
    if args.output:
        print(f"\nĐang lưu kết quả vào: {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        print("Đã lưu thành công!")

    if args.ground_truth:
        print(f"\nĐang tính metrics với ground truth: {args.ground_truth}")
        gt_data = load_data(args.ground_truth)
        metrics = calculate_metrics(filtered_data, gt_data)

        print(f"\n=== METRICS ===")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F2: {metrics['macro_f2']:.4f}")
        print(f"Micro Precision: {metrics['micro_precision']:.4f}")
        print(f"Micro Recall: {metrics['micro_recall']:.4f}")
        print(f"Micro F2: {metrics['micro_f2']:.4f}")
        print(f"Total TP: {metrics['total_tp']}")
        print(f"Total FP: {metrics['total_fp']}")
        print(f"Total FN: {metrics['total_fn']}")

        # Lưu metrics nếu có output file
        if args.output:
            metrics_file = args.output.replace('.json', '_metrics.json')
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"Đã lưu metrics vào: {metrics_file}")


if __name__ == "__main__":
    main()
