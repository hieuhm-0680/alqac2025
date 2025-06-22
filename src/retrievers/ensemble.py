from typing import List
from .base import BaseRetriever

class EnsembleRetriever:
    """
    A class that combines multiple retrievers to provide a unified retrieval interface.
    """

    def __init__(self, retrievers: List[BaseRetriever]):
        self._retrievers = retrievers

    def retrieve(self, query):
        results = []
        for retriever in self._retrievers:
            results.extend(retriever.retrieve(query))
        # TODO: rank fusion, reranking, etc.
        return results