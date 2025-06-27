from collections import defaultdict
from typing import Callable, Iterable, Iterator, List, Optional, TypeVar
from collections.abc import Hashable
from itertools import chain


from torch import T

from src.models.schemas import Document
from ..retrievers.base import BaseRetriever

T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e


class Retriever:
    """
    A class that combines multiple retrievers to provide a unified retrieval interface.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        c: int = 60,
        id_key: Optional[str] = None,
    ):
        self._retrievers = retrievers
        self.c = c
        self.id_key = id_key
        if weights is None:
            self.weights = [1 / len(retrievers)] * len(retrievers)
        else:
            if len(weights) != len(retrievers):
                raise ValueError("Number of weights must match number of retrievers.")
            self.weights = weights

    def retrieve(self, query: str) -> List[Document]:
        """Perform retrieval from all base retrievers and apply rank fusion."""
        return self.rank_fusion(query)

         
    def rank_fusion(self, query: str) -> List[Document]:
        """
        Calls all retrievers and combines their results using weighted RRF.
        """
        doc_lists = [
            retriever.retrieve(query) for retriever in self._retrievers
        ]

        # Ensure all items are Documents
        for i in range(len(doc_lists)):
            doc_lists[i] = [
                doc if isinstance(doc, Document) else Document(page_content=str(doc))
                for doc in doc_lists[i]
            ]

        return self.weighted_reciprocal_rank(doc_lists)

    def weighted_reciprocal_rank(
        self, doc_lists: list[list[Document]]
    ) -> list[Document]:
        if len(doc_lists) != len(self.weights):
            raise ValueError(
                "Number of rank lists must be equal to the number of weights."
            )

        # Associate each doc's content with its RRF score for later sorting by it
        # Duplicated contents across retrievers are collapsed & scored cumulatively
        rrf_score: dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[
                    (
                        doc.page_content
                        if self.id_key is None
                        else doc.metadata[self.id_key]
                    )
                ] += weight / (rank + self.c)

        # Docs are deduplicated by their contents then sorted by their scores
        all_docs = chain.from_iterable(doc_lists)
        sorted_docs = sorted(
            unique_by_key(
                all_docs,
                lambda doc: (
                    doc.page_content
                    if self.id_key is None
                    else doc.metadata[self.id_key]
                ),
            ),
            reverse=True,
            key=lambda doc: rrf_score[
                doc.page_content if self.id_key is None else doc.metadata[self.id_key]
            ],
        )
        return sorted_docs
