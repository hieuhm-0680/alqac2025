from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import Field

from src.models.schemas import Document
import re
from .base import BaseRetriever

_DEFAULT_TOP_K_TFIDF = 10


def default_preprocessing_func(text: str) -> str:
    text = text.replace("\xAD", "")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", "<url>", text)
    text = " ".join(text.split())
    text = text.lower()
    return text


class TFIDFRetriever(BaseRetriever):
    vectorizer: Any = None
    docs: List[Document] = Field(repr=False)
    tfidf_array: Any = None
    k: int = _DEFAULT_TOP_K_TFIDF
    preprocess_func: Callable[[str], str] = default_preprocessing_func

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError(
                "Could not import scikit-learn. Please install it with `pip install scikit-learn`."
            )

        processed_query = self.preprocess_func(query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(self.tfidf_array, query_vec).flatten()
        top_indices = scores.argsort()[::-1][:self.k]
        return [self.docs[i] for i in top_indices]
    
    def _get_relevant_documents_with_scores(self, query):
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError(
                "Could not import scikit-learn. Please install it with `pip install scikit-learn`."
            )

        processed_query = self.preprocess_func(query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(self.tfidf_array, query_vec).flatten()
        top_indices = scores.argsort()[::-1][:self.k]
        return [(self.docs[i], scores[i]) for i in top_indices]

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        tfidf_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], str] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BaseRetriever:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError(
                "Could not import scikit-learn. Please install it with `pip install scikit-learn`."
            )

        tfidf_params = tfidf_params or {}
        vectorizer = TfidfVectorizer(**tfidf_params)
        processed_texts = [preprocess_func(t) for t in texts]
        tfidf_array = vectorizer.fit_transform(processed_texts)

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
            vectorizer=vectorizer,
            docs=docs,
            tfidf_array=tfidf_array,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        tfidf_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], str] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BaseRetriever:
        texts = [doc.text for doc in documents]
        metadatas = [{"law_id": doc.law_id, "article_id": doc.article_id} for doc in documents]
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            tfidf_params=tfidf_params,
            preprocess_func=preprocess_func,
            **kwargs,
        )
