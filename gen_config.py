#!/usr/bin/env python3
"""
Configuration generator for multiple retrieval scenarios.

This script generates various configuration files for testing different combinations of:
- Local/Global retriever enable/disable
- Lexical/Semantic search enable/disable (with constraints)
- Reranker enable/disable
- Lexical ensemble configurations (BM25 only vs BM25 + others)
"""

import yaml
import os
from itertools import product
from typing import Dict, Any, List, Tuple


def get_base_config() -> Dict[str, Any]:
    return {
        "system": {
            "model_cache_dir": "/path/to/model/cache",
            "output_dir": "output",
            "log_level": "INFO"
        },
        "data": {
            "law_path": "dataset/alqac25_law.json",
            "queries_path": "dataset/alqac25_train.json",
            "wseg_law_path": "dataset/wseg_alqac25_law.json",
            "wseg_queries_path": "dataset/wseg_alqac25_train.json",
            "output_path": "data/output.json"
        },
        "pipeline": {
            "enable_local_retriever": True,
            "enable_global_retriever": True,
            "enable_reranker": False,
            "save_local_path": "output/local_retriever_results.json",
            "save_global_path": "output/global_retriever_results.json",
            "save_fused_path": "output/fused_results.json",
            "save_reranked_path": "output/reranked_results.json",
            "save_results_path": "output/final_results.json",
            "local_retriever": {
                "classifier": {
                    "model_name_or_path": "hiuminee/alqac25-classifier-phobert-base",
                    "top_k": 1
                },
                "embedding_model_name_or_path": "AITeamVN/Vietnamese_Embedding",
                "top_k_lexical": 15,
                "top_k_semantic": 15,
                "enable_lexical_search": True,
                "enable_semantic_search": True,
                "indexes": {
                    "index_dir": "data/local_indexes",
                    "chroma_db_path": "data/local_indexes/chroma_db",
                    "bm25_path": "data/local_indexes/bm25_local.pkl"
                }
            },
            "global_retriever": {
                "embedding_model_name": "AITeamVN/Vietnamese_Embedding",
                "top_k_semantic": 20,
                "enable_lexical_search": True,
                "enable_semantic_search": True,
                "indexes": {
                    "index_dir": "data/global_indexes",
                    "chroma_db_path": "data/global_indexes/chroma_db",
                    "lexical_path": "data/global_indexes/bm25_global.pkl"
                },
                "chroma_collection_name": "global_documents",
                "lexical_ensemble_config": {
                    "k": 100,
                    "weights": [0.5, 0.2, 0.3],
                    "enable_bm25": True,
                    "enable_tfidf": True,
                    "enable_qld": True
                }
            },
            "rank_fusion": {
                "method": "rrf",
                "top_n_candidates": 10
            },
            "reranker": {
                "cross_encoder_model": "AITeamVN/Vietnamese_Reranker",
                "batch_size": 2
            }
        }
    }


def get_local_search_combinations() -> List[Tuple[bool, bool]]:
    """Get valid combinations for local retriever (lexical, semantic)."""
    return [(True, True), (True, False), (False, True)]


def get_global_search_combinations() -> List[Tuple[bool, bool, Dict[str, bool]]]:
    """Get valid combinations for global retriever (lexical, semantic, lexical_ensemble_config)."""
    combinations = []

    # Semantic only
    combinations.append(
        (False, True, {"enable_bm25": False, "enable_tfidf": False, "enable_qld": False}))

    # Lexical only - BM25 only
    combinations.append(
        (True, False, {"enable_bm25": True, "enable_tfidf": False, "enable_qld": False}))

    # Lexical only - BM25 + others
    combinations.append(
        (True, False, {"enable_bm25": True, "enable_tfidf": True, "enable_qld": True}))

    # Both lexical and semantic - BM25 only
    combinations.append(
        (True, True, {"enable_bm25": True, "enable_tfidf": False, "enable_qld": False}))

    # Both lexical and semantic - BM25 + others
    combinations.append(
        (True, True, {"enable_bm25": True, "enable_tfidf": True, "enable_qld": True}))

    return combinations


def generate_config_name(enable_local: bool, enable_global: bool, enable_reranker: bool,
                         local_lexical: bool, local_semantic: bool,
                         global_lexical: bool, global_semantic: bool,
                         lexical_config: Dict[str, bool]) -> str:
    """Generate a descriptive configuration name."""
    parts = []

    # Main components
    if enable_local:
        parts.append("local")
    if enable_global:
        parts.append("global")
    if enable_reranker:
        parts.append("rerank")

    # Local retriever details
    if enable_local:
        local_parts = []
        if local_lexical:
            local_parts.append("lex")
        if local_semantic:
            local_parts.append("sem")
        if local_parts:
            parts.append(f"L({'+'.join(local_parts)})")

    # Global retriever details
    if enable_global:
        global_parts = []
        if global_lexical:
            if lexical_config["enable_bm25"] and not lexical_config["enable_tfidf"] and not lexical_config["enable_qld"]:
                global_parts.append("bm25")
            elif lexical_config["enable_bm25"] and (lexical_config["enable_tfidf"] or lexical_config["enable_qld"]):
                global_parts.append("ensemble")
        if global_semantic:
            global_parts.append("sem")
        if global_parts:
            parts.append(f"G({'+'.join(global_parts)})")

    return "config_" + "_".join(parts)


def create_config(enable_local: bool, enable_global: bool, enable_reranker: bool,
                  local_lexical: bool, local_semantic: bool,
                  global_lexical: bool, global_semantic: bool,
                  lexical_config: Dict[str, bool]) -> Dict[str, Any]:
    """Create a configuration with the specified parameters."""
    config = get_base_config()

    # Set main pipeline flags
    config["pipeline"]["enable_local_retriever"] = enable_local
    config["pipeline"]["enable_global_retriever"] = enable_global
    config["pipeline"]["enable_reranker"] = enable_reranker

    # Configure local retriever
    if enable_local:
        config["pipeline"]["local_retriever"]["enable_lexical_search"] = local_lexical
        config["pipeline"]["local_retriever"]["enable_semantic_search"] = local_semantic

    # Configure global retriever
    if enable_global:
        config["pipeline"]["global_retriever"]["enable_lexical_search"] = global_lexical
        config["pipeline"]["global_retriever"]["enable_semantic_search"] = global_semantic
        config["pipeline"]["global_retriever"]["lexical_ensemble_config"].update(
            lexical_config)

    return config


def generate_all_configs(output_dir: str = "src/config/generated_configs") -> None:
    os.makedirs(output_dir, exist_ok=True)

    configs_generated = []

    # local, global, reranker
    main_combinations = list(product([True, False], repeat=3))

    valid_main_combinations = [
        (local, global_, reranker)
        for local, global_, reranker in main_combinations
        if local or global_
    ]
    id = 1
    for enable_local, enable_global, enable_reranker in valid_main_combinations:
        local_combinations = get_local_search_combinations() if enable_local else [
            (False, False)]
        global_combinations = get_global_search_combinations() if enable_global else [
            (False, False, {})]

        if not enable_local:
            local_combinations = [(False, False)]

        if not enable_global:
            global_combinations = [(False, False, {})]

        for local_lexical, local_semantic in local_combinations:
            for global_lexical, global_semantic, lexical_config in global_combinations:

                if enable_local and not local_lexical and not local_semantic:
                    continue

                if enable_global and not global_lexical and not global_semantic:
                    continue

                config = create_config(
                    enable_local, enable_global, enable_reranker,
                    local_lexical, local_semantic,
                    global_lexical, global_semantic,
                    lexical_config
                )

                config_name = generate_config_name(
                    enable_local, enable_global, enable_reranker,
                    local_lexical, local_semantic,
                    global_lexical, global_semantic,
                    lexical_config
                )

                filename = f"{id}_{config_name}.yaml"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False,
                              sort_keys=False, indent=2)

                configs_generated.append({
                    'id': id,
                    'filename': filename,
                    'enable_local': enable_local,
                    'enable_global': enable_global,
                    'enable_reranker': enable_reranker,
                    'local_lexical': local_lexical,
                    'local_semantic': local_semantic,
                    'global_lexical': global_lexical,
                    'global_semantic': global_semantic,
                    'lexical_config': lexical_config
                })
                id += 1

    summary_file = os.path.join(output_dir, "config_summary.yaml")
    with open(summary_file, 'w', encoding='utf-8') as f:
        yaml.dump({
            'total_configs': len(configs_generated),
            'configs': configs_generated
        }, f, default_flow_style=False, sort_keys=False, indent=2)

    print(
        f"Generated {len(configs_generated)} configuration files in {output_dir}")
    print(f"Summary saved to {summary_file}")

    print("\nConfiguration Statistics:")
    print(
        f"- Configs with local retriever: {sum(1 for c in configs_generated if c['enable_local'])}")
    print(
        f"- Configs with global retriever: {sum(1 for c in configs_generated if c['enable_global'])}")
    print(
        f"- Configs with reranker: {sum(1 for c in configs_generated if c['enable_reranker'])}")
    print(
        f"- Configs with both retrievers: {sum(1 for c in configs_generated if c['enable_local'] and c['enable_global'])}")


def list_configs(config_dir: str = "src/config/generated_configs") -> None:
    summary_file = os.path.join(config_dir, "config_summary.yaml")

    if not os.path.exists(summary_file):
        print(f"No summary file found at {summary_file}")
        print("Run generate_all_configs() first.")
        return

    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = yaml.safe_load(f)

    print(f"Total configurations: {summary['total_configs']}")
    print("\nConfiguration Details:")
    print("-" * 80)

    for i, config in enumerate(summary['configs'], 1):
        print(f"{i:2d}. {config['filename']}")
        print(
            f"    Local: {config['enable_local']} (lex: {config['local_lexical']}, sem: {config['local_semantic']})")
        print(
            f"    Global: {config['enable_global']} (lex: {config['global_lexical']}, sem: {config['global_semantic']})")
        print(f"    Reranker: {config['enable_reranker']}")
        if config['global_lexical'] and config['lexical_config']:
            ensemble = config['lexical_config']
            active_methods = [k.replace('enable_', '')
                              for k, v in ensemble.items() if v]
            print(
                f"    Lexical ensemble: {', '.join(active_methods) if active_methods else 'none'}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate configuration files for retrieval experiments")
    parser.add_argument("--output-dir", default="src/config/generated_configs",
                        help="Output directory for generated configs")
    parser.add_argument("--list", action="store_true",
                        help="List existing configurations instead of generating new ones")

    args = parser.parse_args()

    if args.list:
        list_configs(args.output_dir)
    else:
        generate_all_configs(args.output_dir)
