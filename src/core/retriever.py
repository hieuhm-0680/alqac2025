from collections import defaultdict
from typing import Callable, Dict, Iterable, Iterator, List, Optional, TypeVar, Union
from collections.abc import Hashable
from itertools import chain


# from torch import T

from src.core.reranker import Reranker, RerankerConfig
from src.models.schemas import Document
from src.retrievers.global_retriever import GlobalRetriever, GlobalRetrieverConfig, build_global_indexes
from src.retrievers.local_retriever import LocalRetriever, LocalRetrieverConfig, build_local_indexes
from ..retrievers.base import BaseRetriever

# T = TypeVar("T")
# H = TypeVar("H", bound=Hashable)


# def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
#     seen = set()
#     for e in iterable:
#         if (k := key(e)) not in seen:
#             seen.add(k)
#             yield e

class RankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, ranked_lists: List[List[Dict]]) -> List[Dict]:
        scores = {}
        # Aggregate scores from all lists
        for doc_list in ranked_lists:
            for i, doc in enumerate(doc_list):
                doc_id = doc['id']
                if doc_id not in scores:
                    scores[doc_id] = {'score': 0, 'doc': doc}
                # The RRF formula: 1 / (k + rank)
                scores[doc_id]['score'] += 1 / (self.k + i + 1)

        # Sort documents by their new aggregated RRF score
        sorted_docs_with_scores = sorted(
            scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # Return just the document objects in the new order
        return [item['doc'] for item in sorted_docs_with_scores]


    # def weighted_reciprocal_rank(
    #     self, doc_lists: list[list[Document]], weights: list[float]
    # ) -> list[Document]:
    #     if len(doc_lists) != len(weights):
    #         raise ValueError(
    #             "Number of rank lists must be equal to the number of weights."
    #         )

    #     # Associate each doc's content with its RRF score for later sorting by it
    #     # Duplicated contents across retrievers are collapsed & scored cumulatively
    #     rrf_score: dict[str, float] = defaultdict(float)
    #     for doc_list, weight in zip(doc_lists, weights):
    #         for rank, doc in enumerate(doc_list, start=1):
    #             rrf_score[
    #                 (
    #                     doc.page_content
    #                     if self.id_key is None
    #                     else doc.metadata[self.id_key]
    #                 )
    #             ] += weight / (rank + self.c)

    #     # Docs are deduplicated by their contents then sorted by their scores
    #     all_docs = chain.from_iterable(doc_lists)
    #     sorted_docs = sorted(
    #         unique_by_key(
    #             all_docs,
    #             lambda doc: (
    #                 doc.page_content
    #                 if self.id_key is None
    #                 else doc.metadata[self.id_key]
    #             ),
    #         ),
    #         reverse=True,
    #         key=lambda doc: rrf_score[
    #             doc.page_content if self.id_key is None else doc.metadata[self.id_key]
    #         ],
    #     )
    #     return sorted_docs


class Retriever:
    def __init__(self, 
                 local_retriever_config: LocalRetrieverConfig, 
                 global_retriever_config: GlobalRetrieverConfig, 
                 reranker_config: RerankerConfig):
        
        print("--- Initializing Full Search Pipeline ---")
        # Initialize each component
        self.local_retriever = LocalRetriever(local_retriever_config)
        self.global_retriever = GlobalRetriever(global_retriever_config)
        self.rank_fusion = RankFusion()
        self.reranker = Reranker(reranker_config)

    def retrieve(self, query: str) -> List[Dict[str, Union[str, float]]]:
        # Step 1: Retrieve from Local and Global Retrievers
        local_results = self.local_retriever.retrieve(query)
        global_results = self.global_retriever.retrieve(query)

        # Step 2: Rank Fusion
        fused_results = self.rank_fusion.fuse([local_results, global_results])

        # Step 3: Reranking
        top_candidates_for_reranking = fused_results[:100]
        final_results = self.reranker.rerank(query, top_candidates_for_reranking)

        return final_results

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
        "technology": [all_docs[0], all_docs[1]],
        "finance": [all_docs[2], all_docs[3], all_docs[4]],
        "health": [all_docs[5]]
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
