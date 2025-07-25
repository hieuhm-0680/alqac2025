#!/usr/bin/env python3
"""
Advanced script to expand retrieval results by adding neighboring articles.

This script provides detailed statistics and handles various edge cases:
- Non-consecutive article numbering
- Mixed numeric/non-numeric article IDs
- Article gaps in laws
- Detailed expansion statistics

Example usage:
    python expand_retrieval_results.py --input_path retrieval_results.json --output_dir expanded_results --top_k 1 --left_expand 3 --right_expand 3 --stats
"""

import json
import os
import argparse
from typing import Dict, List, Any, Set, Tuple, Optional
from pathlib import Path
import logging
from collections import defaultdict, Counter

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArticleExpander:
    """Class to handle article expansion with detailed statistics."""

    def __init__(self, law_path: str):
        """Initialize with law data."""
        self.law_map = self._load_law_data(law_path)
        self.stats = {
            'questions_processed': 0,
            'articles_expanded': 0,
            'expansion_by_law': defaultdict(int),
            'expansion_failures': 0,
            'non_numeric_articles': 0,
            'missing_articles': 0
        }

    def _load_law_data(self, law_path: str) -> Dict[str, Dict[str, Any]]:
        """Load law data and create mapping."""
        logger.info(f"Loading law data from: {law_path}")

        with open(law_path, 'r', encoding='utf-8') as f:
            laws = json.load(f)

        law_map = {}
        total_articles = 0

        for law in laws:
            law_id = law['id']
            law_map[law_id] = {}

            for article in law['articles']:
                article_id = article['id']
                law_map[law_id][article_id] = {
                    'law_id': law_id,
                    'article_id': article_id,
                    'text': article['text']
                }
                total_articles += 1

        logger.info(
            f"Loaded {len(law_map)} laws with {total_articles} total articles")
        return law_map

    def _get_numeric_article_map(self, law_id: str) -> Dict[int, str]:
        """Get mapping of numeric positions to article IDs for a law."""
        if law_id not in self.law_map:
            return {}

        numeric_map = {}
        for article_id in self.law_map[law_id].keys():
            try:
                numeric_id = int(article_id)
                numeric_map[numeric_id] = article_id
            except ValueError:
                continue

        return numeric_map

    def get_article_neighbors(self,
                              law_id: str,
                              article_id: str,
                              left_expand: int,
                              right_expand: int) -> List[Dict[str, Any]]:
        """
        Get neighboring articles within the same law.

        Handles both consecutive and non-consecutive article numbering.
        """
        if law_id not in self.law_map:
            logger.warning(f"Law {law_id} not found in law data")
            self.stats['missing_articles'] += 1
            return []

        law_articles = self.law_map[law_id]

        # Try to convert article ID to integer
        try:
            current_id = int(article_id)
        except ValueError:
            logger.warning(
                f"Article ID {article_id} is not numeric, cannot expand")
            self.stats['non_numeric_articles'] += 1
            # Return original article if it exists
            if article_id in law_articles:
                return [law_articles[article_id]]
            return []

        # Get numeric mapping for this law
        numeric_map = self._get_numeric_article_map(law_id)

        if not numeric_map:
            logger.warning(f"No numeric article IDs found in law {law_id}")
            return []

        # Get sorted list of available numeric IDs
        available_ids = sorted(numeric_map.keys())

        # Find the current article's position in the sorted list
        if current_id not in available_ids:
            logger.warning(f"Article {article_id} not found in law {law_id}")
            self.stats['missing_articles'] += 1
            return []

        current_idx = available_ids.index(current_id)

        # Calculate expansion range based on available articles
        start_idx = max(0, current_idx - left_expand)
        end_idx = min(len(available_ids) - 1, current_idx + right_expand)

        # Collect articles in the expanded range
        expanded_articles = []
        for idx in range(start_idx, end_idx + 1):
            numeric_id = available_ids[idx]
            article_id_str = numeric_map[numeric_id]
            expanded_articles.append(law_articles[article_id_str])

        # -1 for original
        self.stats['expansion_by_law'][law_id] += len(expanded_articles) - 1
        return expanded_articles

    def expand_question_results(self,
                                question_data: Dict[str, Any],
                                top_k: int,
                                left_expand: int,
                                right_expand: int) -> Dict[str, Any]:
        """Expand the retrieval results for a single question."""
        self.stats['questions_processed'] += 1

        if 'relevant_articles' not in question_data:
            return question_data

        relevant_articles = question_data['relevant_articles']
        if not relevant_articles:
            return question_data

        # Start with ALL original articles
        expanded_articles = relevant_articles.copy()
        seen_articles = set()  # To avoid duplicates

        # Track which articles we've seen from original list
        for article in relevant_articles:
            if 'law_id' in article and 'article_id' in article:
                article_key = f"{article['law_id']}#{article['article_id']}"
                seen_articles.add(article_key)

        # Take top-k articles for expansion
        top_articles = relevant_articles[:top_k]
        original_count = len(relevant_articles)
        newly_added_count = 0

        for article in top_articles:
            if 'law_id' not in article or 'article_id' not in article:
                # If article doesn't have proper structure, skip expansion
                continue

            law_id = article['law_id']
            article_id = article['article_id']

            # Get neighbors for this article
            neighbors = self.get_article_neighbors(
                law_id, article_id, left_expand, right_expand)

            if not neighbors:
                self.stats['expansion_failures'] += 1
                continue

            # Add NEW neighbors to expanded list (avoid duplicates)
            for neighbor in neighbors:
                article_key = f"{neighbor['law_id']}#{neighbor['article_id']}"
                if article_key not in seen_articles:
                    expanded_articles.append(neighbor)
                    seen_articles.add(article_key)
                    newly_added_count += 1

        # Update statistics
        self.stats['articles_expanded'] += newly_added_count

        # Create new question data with expanded articles
        expanded_question = question_data.copy()
        expanded_question['relevant_articles'] = expanded_articles

        # Add metadata about expansion
        if 'expansion_info' not in expanded_question:
            expanded_question['expansion_info'] = {
                'original_count': original_count,
                'top_k_used': min(top_k, len(relevant_articles)),
                'expanded_count': len(expanded_articles),
                'newly_added_count': newly_added_count,
                'left_expand': left_expand,
                'right_expand': right_expand
            }

        return expanded_question

    def expand_retrieval_results(self,
                                 input_data: List[Dict[str, Any]],
                                 top_k: int = 1,
                                 left_expand: int = 3,
                                 right_expand: int = 3) -> List[Dict[str, Any]]:
        """Expand retrieval results for all questions."""
        expanded_results = []

        logger.info(f"Expanding {len(input_data)} questions with top_k={top_k}, "
                    f"left_expand={left_expand}, right_expand={right_expand}")

        for i, question_data in enumerate(input_data):
            expanded_question = self.expand_question_results(
                question_data, top_k, left_expand, right_expand
            )
            expanded_results.append(expanded_question)

        return expanded_results

    def print_statistics(self):
        """Print detailed expansion statistics."""
        logger.info("=== Expansion Statistics ===")
        logger.info(
            f"Questions processed: {self.stats['questions_processed']}")
        logger.info(f"Articles expanded: {self.stats['articles_expanded']}")
        logger.info(f"Expansion failures: {self.stats['expansion_failures']}")
        logger.info(
            f"Non-numeric articles encountered: {self.stats['non_numeric_articles']}")
        logger.info(f"Missing articles: {self.stats['missing_articles']}")

        if self.stats['expansion_by_law']:
            logger.info("\nTop 10 laws by expansion count:")
            for law_id, count in Counter(self.stats['expansion_by_law']).most_common(10):
                logger.info(f"  {law_id}: {count} articles")

    def save_statistics(self, output_path: str):
        """Save statistics to JSON file."""
        stats_output = {
            'summary': dict(self.stats),
            'expansion_by_law': dict(self.stats['expansion_by_law'])
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_output, f, ensure_ascii=False, indent=2)

        logger.info(f"Statistics saved to: {output_path}")

with open('dataset/alqac25_law.json', 'r', encoding='utf-8') as f:
    law_data = json.load(f)

article_mapping = {
    item['id']: {
      a['id']: a['text'] for a in item['articles'] 
    }
    for item in law_data
}


def process_single_file(input_path: str,
                        output_path: str,
                        expander: ArticleExpander,
                        top_k: int,
                        left_expand: int,
                        right_expand: int,
                        save_stats: bool = False):
    """Process a single retrieval results file."""
    logger.info(f"Processing file: {input_path}")

    # Load input data
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    for item in input_data:
        for rel_doc in item['relevant_articles']:   
            law_id = rel_doc.get('law_id')
            article_id = rel_doc.get('article_id')
            if law_id and article_id:
                rel_doc['text'] = article_mapping[law_id][article_id]

    

    # Expand results
    expanded_data = expander.expand_retrieval_results(
        input_data, top_k, left_expand, right_expand
    )

    # Save expanded results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(expanded_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved expanded results to: {output_path}")

    # Save statistics if requested
    if save_stats:
        stats_path = output_path.replace('.json', '_expansion_stats.json')
        expander.save_statistics(stats_path)


def process_directory(input_dir: str,
                      output_dir: str,
                      expander: ArticleExpander,
                      top_k: int,
                      left_expand: int,
                      right_expand: int,
                      save_stats: bool = False):
    """Process all JSON files in a directory recursively."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all JSON files
    json_files = list(input_path.rglob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return

    logger.info(f"Found {len(json_files)} JSON files to process")

    for i, json_file in enumerate(json_files):
        logger.info(
            f"Processing file {i+1}/{len(json_files)}: {json_file.name}")

        # Calculate relative path and output path
        relative_path = json_file.relative_to(input_path)
        output_file = output_path / relative_path

        try:
            # Reset stats for each file
            expander.stats = {
                'questions_processed': 0,
                'articles_expanded': 0,
                'expansion_by_law': defaultdict(int),
                'expansion_failures': 0,
                'non_numeric_articles': 0,
                'missing_articles': 0
            }

            process_single_file(
                str(json_file), str(output_file), expander,
                top_k, left_expand, right_expand, save_stats
            )

            if save_stats:
                expander.print_statistics()

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Advanced expansion of retrieval results')

    # Input/output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_path', type=str,
                             help='Path to single JSON file')
    input_group.add_argument('--input_dir', type=str,
                             help='Directory containing JSON files')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for expanded results')

    # Law data
    parser.add_argument('--law_path', type=str, default='dataset/alqac25_law.json',
                        help='Path to law data JSON file')

    # Expansion parameters
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of top articles to expand per question')
    parser.add_argument('--left_expand', type=int, default=3,
                        help='Number of articles to expand to the left')
    parser.add_argument('--right_expand', type=int, default=3,
                        help='Number of articles to expand to the right')

    # Statistics
    parser.add_argument('--stats', action='store_true',
                        help='Save detailed expansion statistics')

    args = parser.parse_args()

    # Initialize expander
    expander = ArticleExpander(args.law_path)

    # Process input
    if args.input_path:
        # Single file processing
        output_file = os.path.join(
            args.output_dir, os.path.basename(args.input_path))
        process_single_file(
            args.input_path, output_file, expander,
            args.top_k, args.left_expand, args.right_expand, args.stats
        )
        if args.stats:
            expander.print_statistics()
    else:
        # Directory processing
        process_directory(
            args.input_dir, args.output_dir, expander,
            args.top_k, args.left_expand, args.right_expand, args.stats
        )

    logger.info("Processing completed!")


if __name__ == "__main__":
    main()
