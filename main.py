# from src.core.reranker import RerankerConfig
# from src.core.retriever import Retriever
# from src.retrievers.global_retriever import GlobalRetrieverConfig, build_global_indexes
# from src.retrievers.local_retriever import LocalRetrieverConfig, build_local_indexes


# if __name__ == '__main__':
#     all_docs = [
#         {'id': 'tech001', 'text': 'NVIDIA announced a new GPU for deep learning.'},
#         {'id': 'tech002', 'text': 'Quantum computing aims to solve complex problems.'},
#         {'id': 'fin001', 'text': 'Central banks are raising interest rates to combat inflation.'},
#         {'id': 'fin002', 'text': 'The stock market shows high volatility.'},
#         {'id': 'fin003', 'text': 'Investors are concerned about rising bond yields.'},
#         {'id': 'health001', 'text': 'A new study shows benefits of a Mediterranean diet.'},
#     ]

#     local_doc_store = {
#         "LABEL_0": [all_docs[0], all_docs[1]],
#         "LABEL_1": [all_docs[2], all_docs[3], all_docs[4]],
#         "LABEL_2": [all_docs[5]]
#     }
    
#     local_config = LocalRetrieverConfig(
#         classifier=({"candidate_labels": list(local_doc_store.keys())})
#     )
#     global_config = GlobalRetrieverConfig()
#     reranker_config = RerankerConfig()
    
#     # ======================================================================
#     # STEP 1: OFFLINE INDEXING (Run this once or when data changes)
#     # ======================================================================
#     print("--- Running Offline Indexing for All Components ---")
#     build_local_indexes(local_config, local_doc_store)
#     build_global_indexes(global_config, all_docs)
#     print("\n--- All Indexing Complete ---\n")
    
#     # ======================================================================
#     # STEP 2: INITIALIZE AND RUN THE PIPELINE (Online Application)
#     # ======================================================================
#     # Initialize the full pipeline
#     pipeline = Retriever(
#         local_retriever_config=local_config,
#         global_retriever_config=global_config,
#         reranker_config=reranker_config
#     )

#     # Execute a search
#     search_query = "What are the financial implications of new technology?"
#     final_ranked_docs = pipeline.retrieve(search_query)

#     # Print the final results
#     print("\n\n========= FINAL RESULTS ==========")
#     if final_ranked_docs:
#         for i, doc in enumerate(final_ranked_docs):
#             print(f"Rank {i+1}: ID: {doc['id']}, Score: {doc['rerank_score']:.4f}, Text: \"{doc['text']}\"")
#     else:
#         print("No relevant documents found.")

import json
import argparse
from src.config.config import load_config
from src.core.retriever import Retriever
from src.retrievers.global_retriever import build_global_indexes
from src.retrievers.local_retriever import build_local_indexes

def main(config_path: str):
    # all_docs = [
    #     {'id': 'tech001', 'text': 'NVIDIA announced a new GPU for deep learning.'},
    #     {'id': 'tech002', 'text': 'Quantum computing aims to solve complex problems.'},
    #     {'id': 'fin001', 'text': 'Central banks are raising interest rates to combat inflation.'},
    #     {'id': 'fin002', 'text': 'The stock market shows high volatility.'},
    #     {'id': 'fin003', 'text': 'Investors are concerned about rising bond yields.'},
    #     {'id': 'health001', 'text': 'A new study shows benefits of a Mediterranean diet.'},
    # ]

    # local_doc_store = {
    #     "LABEL_0": [all_docs[0], all_docs[1]],
    #     "LABEL_1": [all_docs[2], all_docs[3], all_docs[4]],
    #     "LABEL_2": [all_docs[5]]
    # }

    # 1. Load the master configuration object
    config = load_config(config_path)

    #print(f"Running query: '{query}'")
    print(f"Reranker model: {config.pipeline.reranker.cross_encoder_model}")
    print(f"Vector DB Host: {config.pipeline.global_retriever.chroma_collection_name}")




    # 2. Instantiate components, injecting their specific configurations
    # This makes each component self-contained and easier to test.
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

    
    all_docs = json.load(open(config.data.all_doc_path, 'r', encoding='utf-8'))
    local_doc = json.load(open(config.data.local_doc_path, 'r', encoding='utf-8'))
    queries = json.load(open(config.data.queries_path, 'r', encoding='utf-8'))
    texts = [item['text'] for item in queries] # Might have format error



    # ======================================================================
    # STEP 3: OFFLINE INDEXING (Run this once or when data changes)
    # ======================================================================
    print("--- Running Offline Indexing for All Components ---")
    build_local_indexes(local_retriever_config, local_doc)
    build_global_indexes(global_retriever_config, all_docs)
    print("\n--- All Indexing Complete ---\n")
    
    # ======================================================================
    # STEP 4: INITIALIZE AND RUN THE PIPELINE (Online Application)
    # ======================================================================
    # Initialize the full pipeline
    pipeline = Retriever(
        local_retriever_config=local_retriever_config,
        global_retriever_config=global_retriever_config,
        rank_fusion_config = rank_fusion_config,
        reranker_config=reranker_config
    )

    # Execute a search
    for text in texts:
        final_ranked_docs = pipeline.retrieve(text)

        # Print the final results
        print(f"\n\nRunning query: '{text}'")
        print("========= FINAL RESULTS ==========")
        if final_ranked_docs:
            for i, doc in enumerate(final_ranked_docs):
                print(f"Rank {i+1}: ID: {doc['id']}, Score: {doc['rerank_score']:.4f}, Text: \"{doc['text']}\"")
        else:
            print("No relevant documents found.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    #parser.add_argument("query", type=str, help="The search query.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/config/config.yaml", 
        help="Path to the configuration file."
    )
    args = parser.parse_args()
    main(args.config)
