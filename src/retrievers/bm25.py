from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import Field

from src.models.schemas import Document
from .base import BaseRetriever

_DEFAULT_TOP_K_BM25 = 10

def default_preprocessing_func(text: str) -> List[str]:
    return text.split()

class BM25Retriever(BaseRetriever):
    docs: List[Document] = Field(repr=False)
    k: int = _DEFAULT_TOP_K_BM25
    preprocess_func: Callable[[str], List[str]] = default_preprocessing_func

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        return return_docs

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BaseRetriever:
        """
        Create a BM25Retriever from a list of Document objects.
        Args:
            documents: A list of Document objects to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever. 
        Returns:
            A BM25Retriever instance.
        """
        texts = [doc.text for doc in documents]
        metadatas = [
            {"law_id": doc.law_id, "article_id": doc.article_id} for doc in documents
        ]
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            bm25_params=bm25_params,
            preprocess_func=preprocess_func,
            **kwargs,
        )
        

