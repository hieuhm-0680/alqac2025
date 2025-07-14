import json
import argparse
import os
from typing import Dict, Tuple
from git import List
from src.config.config import load_config
from src.core.retriever import Retriever
from src.retrievers.global_retriever import build_global_indexes
from src.retrievers.local_retriever import build_local_indexes
from src.models.schemas import Document, QuestionBase
from src.utils.logging import get_logger

logger = get_logger("alqac25")


def load_data(
    laws_path: str, question_path: str
) -> Tuple[List[Document], Dict[str, List[Document]], List[QuestionBase]]:
    """Load documents from a JSON file."""
    global_docs = []
    local_docs = {}
    questions = []
    with open(laws_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for law in data:
            law_id = law["id"]
            local_docs[law_id] = [
                Document(law_id=law_id, article_id=doc["id"], text=doc["text"])
                for doc in law["articles"]
            ]
    for law_id, articles in local_docs.items():
        global_docs += articles

    with open(question_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
        questions = [QuestionBase(**q) for q in questions_data]

    return global_docs, local_docs, questions


def ensure_directories_exist(config):
    directories_to_create = set()

    if hasattr(config.pipeline, 'save_local_path') and config.pipeline.save_local_path:
        directories_to_create.add(
            os.path.dirname(config.pipeline.save_local_path))

    if hasattr(config.pipeline, 'save_global_path') and config.pipeline.save_global_path:
        directories_to_create.add(
            os.path.dirname(config.pipeline.save_global_path))

    if hasattr(config.pipeline, 'save_fused_path') and config.pipeline.save_fused_path:
        directories_to_create.add(
            os.path.dirname(config.pipeline.save_fused_path))

    if hasattr(config.pipeline, 'save_reranked_path') and config.pipeline.save_reranked_path:
        directories_to_create.add(os.path.dirname(
            config.pipeline.save_reranked_path))

    if hasattr(config.pipeline, 'save_results_path') and config.pipeline.save_results_path:
        directories_to_create.add(
            os.path.dirname(config.pipeline.save_results_path))

    for directory in directories_to_create:
        if directory and directory != '': 
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")


def main(config_path: str, build_indexes: bool):
    # 1. Load the master configuration object
    config = load_config(config_path)

    # 2. Ensure all required directories exist
    ensure_directories_exist(config)

    # 3. Instantiate components
    local_retriever_config = None
    global_retriever_config = None
    reranker_config = None
    rank_fusion_config = None

    if config.pipeline.enable_local_retriever:
        local_retriever_config = config.pipeline.local_retriever

    if config.pipeline.enable_global_retriever:
        global_retriever_config = config.pipeline.global_retriever

    if config.pipeline.enable_reranker:
        reranker_config = config.pipeline.reranker
        rank_fusion_config = config.pipeline.rank_fusion

    all_docs, local_docs, queries = load_data(
        config.data.law_path, config.data.queries_path
    )

    # ======================================================================
    # STEP 3: OFFLINE INDEXING (Run this once or when data changes)
    # ======================================================================
    if build_indexes:
        logger.info("--- Running Offline Indexing for All Components ---")
        build_local_indexes(local_retriever_config, local_docs)
        build_global_indexes(global_retriever_config, all_docs)
        logger.info("\n--- All Indexing Complete ---\n")

    # ======================================================================
    # STEP 4: INITIALIZE AND RUN THE PIPELINE (Online Application)
    # ======================================================================
    pipeline = Retriever(
        local_retriever_config=local_retriever_config,
        global_retriever_config=global_retriever_config,
        rank_fusion_config=rank_fusion_config,
        reranker_config=reranker_config,

        save_local_path=config.pipeline.save_local_path,
        save_global_path=config.pipeline.save_global_path,
        save_fused_path=config.pipeline.save_fused_path,
        save_reranked_path=config.pipeline.save_reranked_path,
    )
    results = []
    queries = queries[:2]
    for query in queries:
        logger.info(f"Processing query {query.question_id}: {query.text}")
        final_ranked_docs = pipeline.retrieve(query.text)
        q_result = {
            "question_id": query.question_id,
            "relevant_articles": []
        }
        for doc in final_ranked_docs:
            law_id, article_id = doc["id"].split("|")
            q_result["relevant_articles"].append({
                "law_id": law_id,
                "article_id": article_id,
                "text": doc["text"]
            })
        results.append(q_result)
    with open(config.pipeline.save_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--build_indexes",
        type=str,
        default="False",
        help="Whether to build indexes before running the pipeline (True/False).",
    )
    args = parser.parse_args()
    build_indexes = args.build_indexes.lower() in ["true", "1", "yes", "on"]
    main(args.config, build_indexes)
