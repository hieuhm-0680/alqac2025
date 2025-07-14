#!/usr/bin/env python3
"""
Results analyzer for ALQAC2025 experiments.

This script analyzes and compares results across all experiments.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Try to import optional dependencies for advanced analysis
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def load_experiment_results(experiments_dir: str) -> List[Dict]:
    """Load results from all experiments."""
    results = []

    if not os.path.exists(experiments_dir):
        print(f"❌ Experiments directory not found: {experiments_dir}")
        return results

    experiment_dirs = [d for d in os.listdir(experiments_dir)
                       if os.path.isdir(os.path.join(experiments_dir, d)) and 'config' in d]

    for exp_dir in experiment_dirs:
        exp_path = os.path.join(experiments_dir, exp_dir)

        # Load experiment summary
        summary_file = os.path.join(exp_path, "experiment_summary.json")
        eval_file = os.path.join(exp_path, "evaluation_results.json")

        if os.path.exists(summary_file) and os.path.exists(eval_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            with open(eval_file, 'r', encoding='utf-8') as f:
                evaluation = json.load(f)

            # Combine data
            result = {
                "experiment_name": exp_dir,
                "config_details": summary.get("config_details", {}),
                "metrics": evaluation.get("metrics", {}),
                "summary_stats": evaluation.get("summary", {}),
                "status": summary.get("status", "unknown")
            }
            results.append(result)
        else:
            print(f"⚠️  Incomplete results for {exp_dir}")

    return results


def create_results_dataframe(results: List[Dict]) -> Optional[object]:
    """Create a pandas DataFrame from experiment results."""
    if not PANDAS_AVAILABLE:
        print("⚠️  pandas not available. Install with: pip install pandas")
        return None

    rows = []

    for result in results:
        if result["status"] != "completed":
            continue

        config = result["config_details"]
        metrics = result["metrics"]

        row = {
            "experiment_name": result["experiment_name"],
            "enable_local": config.get("enable_local", False),
            "enable_global": config.get("enable_global", False),
            "enable_reranker": config.get("enable_reranker", False),
            "local_lexical": config.get("local_lexical", False),
            "local_semantic": config.get("local_semantic", False),
            "global_lexical": config.get("global_lexical", False),
            "global_semantic": config.get("global_semantic", False),
        }

        # Add lexical config details
        lexical_config = config.get("lexical_config", {})
        row.update({
            "bm25_enabled": lexical_config.get("enable_bm25", False),
            "tfidf_enabled": lexical_config.get("enable_tfidf", False),
            "qld_enabled": lexical_config.get("enable_qld", False),
        })

        # Add metrics
        for metric_name, metric_value in metrics.items():
            row[metric_name] = metric_value

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_results(df, output_dir: str = "analysis") -> None:
    """Analyze experiment results and generate reports."""
    if df is None:
        return

    Path(output_dir).mkdir(exist_ok=True)

    print(f"📊 Analyzing {len(df)} completed experiments...")

    # Basic statistics
    print("\n📈 BASIC STATISTICS")
    print("=" * 50)

    key_metrics = ['recall@1', 'recall@3', 'recall@5', 'recall@10',
                   'precision@1', 'precision@3', 'precision@5', 'precision@10']

    available_metrics = [m for m in key_metrics if m in df.columns]

    if available_metrics:
        stats = df[available_metrics].describe()
        print(stats)

        # Save statistics
        stats.to_csv(os.path.join(output_dir, "basic_statistics.csv"))

        # Top performers
        print(f"\n🏆 TOP PERFORMERS")
        print("=" * 50)

        for metric in available_metrics[:4]:  # Focus on recall metrics
            if metric in df.columns:
                top_3 = df.nlargest(3, metric)[['experiment_name', metric]]
                print(f"\nTop 3 for {metric}:")
                for _, row in top_3.iterrows():
                    print(f"  {row['experiment_name']}: {row[metric]:.4f}")

    # Configuration impact analysis
    print(f"\n🔍 CONFIGURATION IMPACT ANALYSIS")
    print("=" * 50)

    config_columns = ['enable_local', 'enable_global', 'enable_reranker',
                      'local_lexical', 'local_semantic', 'global_lexical', 'global_semantic']

    for config_col in config_columns:
        if config_col in df.columns and 'recall@5' in df.columns:
            grouped = df.groupby(config_col)[
                'recall@5'].agg(['mean', 'std', 'count'])
            print(f"\n{config_col}:")
            print(grouped)

    # Save detailed results
    df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)

    # Generate visualizations if matplotlib is available
    try:
        create_visualizations(df, output_dir)
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")


def create_visualizations(df, output_dir: str) -> None:
    """Create visualizations of the results."""
    if not PLOTTING_AVAILABLE:
        print("⚠️  matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")
        return

    plt.style.use('default')

    # Recall comparison
    if 'recall@5' in df.columns:
        plt.figure(figsize=(12, 8))

        # Bar plot of recall@5 by experiment
        top_20 = df.nlargest(20, 'recall@5')
        plt.subplot(2, 2, 1)
        plt.bar(range(len(top_20)), top_20['recall@5'])
        plt.title('Top 20 Experiments by Recall@5')
        plt.xlabel('Experiment Rank')
        plt.ylabel('Recall@5')
        plt.xticks(range(0, len(top_20), 5))

        # Configuration impact on recall@5
        config_cols = ['enable_local', 'enable_global', 'enable_reranker']
        for i, col in enumerate(config_cols, 2):
            if col in df.columns:
                plt.subplot(2, 2, i)
                grouped = df.groupby(col)['recall@5'].mean()
                grouped.plot(kind='bar')
                plt.title(f'Impact of {col} on Recall@5')
                plt.ylabel('Mean Recall@5')
                plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'results_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Correlation heatmap if multiple metrics exist
    metric_cols = [col for col in df.columns if any(
        metric in col for metric in ['recall', 'precision', 'f1'])]
    if len(metric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[metric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Metric Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metric_correlations.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def find_best_configurations(df, metric: str = 'recall@5', top_n: int = 10):
    """Find the best performing configurations."""
    if df is None:
        return None

    if metric not in df.columns:
        print(f"❌ Metric {metric} not found in results")
        return None

    top_configs = df.nlargest(top_n, metric)

    print(f"\n🥇 TOP {top_n} CONFIGURATIONS BY {metric.upper()}")
    print("=" * 80)

    for i, (_, row) in enumerate(top_configs.iterrows(), 1):
        print(
            f"{i:2d}. {row['experiment_name']} ({metric}: {row[metric]:.4f})")
        print(
            f"     Local: {row['enable_local']} (lex: {row['local_lexical']}, sem: {row['local_semantic']})")
        print(
            f"     Global: {row['enable_global']} (lex: {row['global_lexical']}, sem: {row['global_semantic']})")
        print(f"     Reranker: {row['enable_reranker']}")
        if row['global_lexical']:
            lexical_methods = []
            if row.get('bm25_enabled', False):
                lexical_methods.append('BM25')
            if row.get('tfidf_enabled', False):
                lexical_methods.append('TF-IDF')
            if row.get('qld_enabled', False):
                lexical_methods.append('QLD')
            print(
                f"     Lexical methods: {', '.join(lexical_methods) if lexical_methods else 'None'}")
        print()

    return top_configs


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ALQAC2025 experiment results")
    parser.add_argument("--experiments-dir", default="experiments",
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", default="analysis",
                        help="Directory to save analysis results")
    parser.add_argument("--metric", default="recall@5",
                        help="Primary metric for ranking configurations")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top configurations to show")

    args = parser.parse_args()

    print(f"🔍 ALQAC2025 Experiment Results Analysis")
    print(f"📁 Experiments directory: {args.experiments_dir}")
    print(f"📁 Output directory: {args.output_dir}")

    # Load results
    results = load_experiment_results(args.experiments_dir)

    if not results:
        print("❌ No valid experiment results found.")
        return 1

    print(f"✅ Loaded {len(results)} experiment results")

    # Create DataFrame
    df = create_results_dataframe(results)

    if df is None or df.empty:
        print("❌ No completed experiments found or pandas not available.")
        return 1

    print(f"📊 {len(df)} completed experiments ready for analysis")

    # Analyze results
    analyze_results(df, args.output_dir)

    # Find best configurations
    find_best_configurations(df, args.metric, args.top_n)

    print(f"\n💾 Analysis results saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
