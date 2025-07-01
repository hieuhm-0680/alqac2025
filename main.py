# import argparse
# from src.config import load_config
# from src.retrieval import GlobalRetriever, LocalRetriever, Reranker 

# def main(query: str, config_path: str):
#     # 1. Load the master configuration object once
#     config = load_config(config_path)

#     print(f"Running query: '{query}'")
#     print(f"Reranker model: {config.pipeline.reranker.cross_encoder_model}")
#     print(f"Vector DB Host: {config.pipeline.global_retrieval.semantic.vector_db.host}")
    
#     # 2. Instantiate components, injecting their specific configurations
#     # This makes each component self-contained and easier to test.
#     if config.pipeline.enable_global_retrieval:
#         global_retriever = GlobalRetriever(
#             config=config.pipeline.global_retrieval
#         )
#         # ... use it
    
#     if config.pipeline.enable_reranker:
#         reranker = Reranker(
#             config=config.pipeline.reranker,
#             cache_dir=config.system.model_cache_dir
#         )
#         # ... use it

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query", type=str, help="The search query.")
#     parser.add_argument(
#         "--config", 
#         type=str, 
#         default="config/config.yaml", 
#         help="Path to the configuration file."
#     )
#     args = parser.parse_args()
#     main(args.query, args.config)

from src.core.reranker import RerankerConfig
from src.core.retriever import Retriever
from src.retrievers.global_retriever import GlobalRetrieverConfig, build_global_indexes
from src.retrievers.local_retriever import LocalRetrieverConfig, build_local_indexes


if __name__ == '__main__':
    all_docs = [
        {'id': 'tech001', 'text': 'NVIDIA announced a new GPU for deep learning.'},
        {'id': 'tech002', 'text': 'Quantum computing aims to solve complex problems.'},
        {'id': 'fin001', 'text': 'Central banks are raising interest rates to combat inflation.'},
        {'id': 'fin002', 'text': 'The stock market shows high volatility.'},
        {'id': 'fin003', 'text': 'Investors are concerned about rising bond yields.'},
        {'id': 'health001', 'text': 'A new study shows benefits of a Mediterranean diet.'},
    ]

    local_doc_store = {
        "LABEL_0": [all_docs[0], all_docs[1]],
        "LABEL_1": [all_docs[2], all_docs[3], all_docs[4]],
        "LABEL_2": [all_docs[5]]
    }
    
    local_config = LocalRetrieverConfig(
        classifier=({"candidate_labels": list(local_doc_store.keys())})
    )
    global_config = GlobalRetrieverConfig()
    reranker_config = RerankerConfig()
    
    # ======================================================================
    # STEP 1: OFFLINE INDEXING (Run this once or when data changes)
    # ======================================================================
    print("--- Running Offline Indexing for All Components ---")
    build_local_indexes(local_config, local_doc_store)
    build_global_indexes(global_config, all_docs)
    print("\n--- All Indexing Complete ---\n")
    
    # ======================================================================
    # STEP 2: INITIALIZE AND RUN THE PIPELINE (Online Application)
    # ======================================================================
    # Initialize the full pipeline
    pipeline = Retriever(
        local_retriever_config=local_config,
        global_retriever_config=global_config,
        reranker_config=reranker_config
    )

    # Execute a search
    search_query = "What are the financial implications of new technology?"
    final_ranked_docs = pipeline.retrieve(search_query)

    # Print the final results
    print("\n\n========= FINAL RESULTS ==========")
    if final_ranked_docs:
        for i, doc in enumerate(final_ranked_docs):
            print(f"Rank {i+1}: ID: {doc['id']}, Score: {doc['rerank_score']:.4f}, Text: \"{doc['text']}\"")
    else:
        print("No relevant documents found.")
