from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from pydantic import Field

from src.models.schemas import Document
from .base import BaseRetriever

_DEFAULT_TOP_K_BM25 = 10


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


class BM25Retriever(BaseRetriever):
    vectorizer: Any = None
    docs: List[Document] = Field(repr=False)
    k: int = _DEFAULT_TOP_K_BM25
    preprocess_func: Callable[[str], List[str]] = default_preprocessing_func

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        scores = self.vectorizer.get_scores(processed_query)
        top_indices = scores.argsort()[::-1][:self.k]
        results = [self.docs[i] for i in top_indices]
        return results

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]
                                  ] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BaseRetriever:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )

        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)

        docs = [
            Document(
                text=text,
                law_id=metadata.get("law_id", ""),
                article_id=metadata.get("article_id", ""),
            )
            for text, metadata in zip(texts, metadatas)
        ]

        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]
                                  ] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BaseRetriever:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )
        texts_preprocessed = [preprocess_func(doc.text) for doc in documents]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_preprocessed, **bm25_params)
        return cls(
            vectorizer=vectorizer,
            docs=documents,
            preprocess_func=preprocess_func,
            **kwargs
        )
