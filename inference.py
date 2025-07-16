import json
import argparse
import os
from typing import Dict, Tuple, List, Optional
from src.config.config import load_config
from src.core.retriever import Retriever
from src.retrievers.global_retriever import build_global_indexes
from src.retrievers.local_retriever import build_local_indexes
from src.models.schemas import Document, QuestionBase
from src.utils.logging import get_logger
from tqdm import tqdm

logger = get_logger("alqac25")


def load_laws_data(laws_path: str) -> Tuple[List[Document], Dict[str, List[Document]]]:
    """Load law documents from a JSON file."""
    global_docs = []
    local_docs = {}
    
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

    return global_docs, local_docs


def load_test_questions(question_path: str) -> List[QuestionBase]:
    """Load test questions from a JSON file (without ground truth)."""
    questions = []
    
    with open(question_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
        for q in questions_data:
            question = QuestionBase(
                question_id=q["question_id"],
                text=q["text"]
            )
            questions.append(question)

    return questions


def ensure_directories_exist(config):
    """Ensure all output directories exist."""
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


def save_task1_predictions(results: List[Dict], output_path: str):
    """Save Task 1 (Document Retrieval) predictions in competition format."""
    # Convert to competition format (without text field)
    formatted_results = []
    for result in results:
        formatted_result = {
            "question_id": result["question_id"],
            "relevant_articles": []
        }
        for article in result["relevant_articles"]:
            formatted_result["relevant_articles"].append({
                "law_id": article["law_id"],
                "article_id": article["article_id"],
                "text": article["text"]  
            })
        formatted_results.append(formatted_result)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Task 1 predictions saved to: {output_path}")


def save_task2_predictions(results: List[Dict], output_path: str):
    """Save Task 2 (Question Answering) predictions in competition format."""
    # For now, this is a placeholder - you would need to integrate a QA model
    formatted_results = []
    for result in results:
        formatted_result = {
            "question_id": result["question_id"],
            "answer": "Placeholder"  # Replace with actual QA model prediction
        }
        formatted_results.append(formatted_result)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Task 2 predictions saved to: {output_path}")


def main(config_path: str, test_questions_path: str, wseg_test_questions_path: str, build_indexes: bool, 
         task1_output: Optional[str] = None, task2_output: Optional[str] = None):
    """Main inference function."""
    
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

    # 4. Load law data (same as training)
    all_docs, local_docs = load_laws_data(config.data.law_path)
    wseg_all_docs, wseg_local_docs = load_laws_data(config.data.wseg_law_path)

    # 5. Load test questions (without ground truth)
    test_questions = load_test_questions(test_questions_path)
    wseg_test_questions = load_test_questions(wseg_test_questions_path)
    logger.info(f"Loaded {len(test_questions)} test questions")

    # ======================================================================
    # STEP 6: OFFLINE INDEXING (Run this once or when data changes)
    # ======================================================================
    if build_indexes:
        logger.info("--- Running Offline Indexing for All Components ---")
        if local_retriever_config:
            build_local_indexes(local_retriever_config, local_docs, wseg_local_docs)
        if global_retriever_config:
            build_global_indexes(global_retriever_config, all_docs, wseg_all_docs)
        logger.info("\n--- All Indexing Complete ---\n")

    # ======================================================================
    # STEP 7: INITIALIZE AND RUN THE PIPELINE (Online Application)
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

    # ======================================================================
    # STEP 8: INFERENCE
    # ======================================================================
    results = []
    logger.info("Starting inference on test questions...")
    
    for query, wseg_query in tqdm(zip(test_questions, wseg_test_questions), 
                                  total=len(test_questions), desc="Processing test questions"):
        final_ranked_docs = pipeline.retrieve(query.text, wseg_query.text)
        
        q_result = {
            "question_id": query.question_id,
            "question_type": query.question_type,
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

    # ======================================================================
    # STEP 9: SAVE RESULTS
    # ======================================================================
    
    # Save detailed results (with text) for analysis
    detailed_output = config.pipeline.save_results_path
    with open(detailed_output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Detailed results saved to: {detailed_output}")

    # Save Task 1 predictions in competition format
    if task1_output:
        save_task1_predictions(results, task1_output)
    
    # Save Task 2 predictions in competition format (placeholder)
    if task2_output:
        save_task2_predictions(results, task2_output)

    logger.info(f"Inference completed successfully for {len(results)} questions")
    
    # Print summary statistics
    total_retrieved = sum(len(r["relevant_articles"]) for r in results)
    avg_retrieved = total_retrieved / len(results) if results else 0
    logger.info(f"Average articles retrieved per question: {avg_retrieved:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ALQAC2025 Inference Script - Run inference on test set without evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/generated_configs/5_config_local_global_rerank_L(lex+sem)_G(ensemble+sem).yaml",
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--test-questions",
        type=str,
        required=True,
        help="Path to test questions JSON file (without ground truth)"
    )
    parser.add_argument(
        "--wseg-test-questions",
        type=str,
        required=True,
        help="Path to wseg test questions JSON file (without ground truth)"
    )
    parser.add_argument(
        "--build-indexes",
        type=str,
        default="False",
        help="Whether to build indexes before running inference (True/False)."
    )
    parser.add_argument(
        "--task1-output",
        type=str,
        help="Output path for Task 1 (Document Retrieval) predictions in competition format"
    )
    parser.add_argument(
        "--task2-output", 
        type=str,
        help="Output path for Task 2 (Question Answering) predictions in competition format"
    )
    
    args = parser.parse_args()
    build_indexes = args.build_indexes.lower() in ["true", "1", "yes", "on"]
    
    if not os.path.exists(args.test_questions):
        logger.error(f"Test questions file not found: {args.test_questions}")
        exit(1)
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        exit(1)
    
    main(
        config_path=args.config,
        test_questions_path=args.test_questions,
        wseg_test_questions_path=args.wseg_test_questions,
        build_indexes=build_indexes,
        task1_output=args.task1_output,
        task2_output=args.task2_output
    )
