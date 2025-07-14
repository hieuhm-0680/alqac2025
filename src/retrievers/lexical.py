from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from pydantic import BaseModel, Field

from src.models.schemas import Document
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.tfidf import TFIDFRetriever
from src.retrievers.qld import QLDRetriever
from src.retrievers.base import BaseRetriever


class LexicalEnsembleConfig(BaseModel):
    """Configuration for LexicalEnsembleRetriever"""
    k: int = Field(default=10, ge=1,
                   description="Number of top documents to return")
    weights: Tuple[float, float, float] = Field(
        default=(1.0, 1.0, 1.0),
        description="Weights for BM25, TF-IDF, and QLD respectively"
    )
    enable_bm25: bool = Field(
        default=True, description="Enable/disable BM25 retriever")
    enable_tfidf: bool = Field(
        default=True, description="Enable/disable TF-IDF retriever")
    enable_qld: bool = Field(
        default=True, description="Enable/disable QLD retriever")

    class Config:
        extra = 'allow'


class LexicalEnsembleRetriever(BaseModel):
    """
    Ensemble retriever that combines BM25, TF-IDF, and QLD retrievers
    using weighted rank fusion.
    """

    bm25: BM25Retriever
    tfidf: TFIDFRetriever
    qld: QLDRetriever
    k: int = Field(default=10, ge=1)
    weights: Tuple[float, float, float] = Field(default=(1.0, 1.0, 1.0))
    enable_bm25: bool = Field(default=True)
    enable_tfidf: bool = Field(default=True)
    enable_qld: bool = Field(default=True)

    def __init__(
        self,
        bm25: BM25Retriever,
        tfidf: TFIDFRetriever,
        qld: QLDRetriever,
        k: int = 10,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        config: Optional[LexicalEnsembleConfig] = None,
    ):
        # If config is provided, use its values
        if config is not None:
            k = config.k
            weights = config.weights
            enable_bm25 = config.enable_bm25
            enable_tfidf = config.enable_tfidf
            enable_qld = config.enable_qld
        else:
            enable_bm25 = True
            enable_tfidf = True
            enable_qld = True

        super().__init__(
            bm25=bm25,
            tfidf=tfidf,
            qld=qld,
            k=k,
            weights=weights,
            enable_bm25=enable_bm25,
            enable_tfidf=enable_tfidf,
            enable_qld=enable_qld
        )

        # Validate that at least one retriever is enabled
        if not any([self.enable_bm25, self.enable_tfidf, self.enable_qld]):
            raise ValueError(
                "At least one lexical retriever (BM25, TF-IDF, or QLD) must be enabled")

        print(
            f"LexicalEnsembleRetriever initialized with k={self.k}, weights={self.weights}, "
            f"enabled: BM25={self.enable_bm25}, TF-IDF={self.enable_tfidf}, QLD={self.enable_qld}")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using ensemble of lexical retrievers.

        Args:
            query: Search query string

        Returns:
            List of top-k documents ranked by weighted ensemble score
        """
        score_map: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        retrieve_k = min(2 * self.k, 100)

        # Only retrieve from enabled retrievers
        bm25_docs = self.bm25._get_relevant_documents_with_scores(
            query) if self.enable_bm25 else []
        tfidf_docs = self.tfidf._get_relevant_documents_with_scores(
            query) if self.enable_tfidf else []
        qld_docs = self.qld._get_relevant_documents_with_scores(
            query) if self.enable_qld else []

        rrf_k = 60

        # Process BM25 results if enabled
        if self.enable_bm25 and bm25_docs:
            bm25_scores = [score for _, score in bm25_docs]
            max_score = max(bm25_scores) if bm25_scores else 1.0
            min_score = min(bm25_scores) if bm25_scores else 0.0
            score_range = max_score - min_score if max_score != min_score else 1.0

            for rank, (doc, raw_score) in enumerate(bm25_docs, 1):
                rrf_score = 1.0 / (rrf_k + rank)
                normalized_score = (raw_score - min_score) / score_range
                combined_score = 0.6 * rrf_score + 0.4 * normalized_score

                score_map[doc.text] += self.weights[0] * combined_score
                doc_map[doc.text] = doc

        # Process TF-IDF results if enabled
        if self.enable_tfidf and tfidf_docs:
            tfidf_scores = [score for _, score in tfidf_docs]
            max_score = max(tfidf_scores) if tfidf_scores else 1.0
            min_score = min(tfidf_scores) if tfidf_scores else 0.0
            score_range = max_score - min_score if max_score != min_score else 1.0

            for rank, (doc, raw_score) in enumerate(tfidf_docs, 1):
                rrf_score = 1.0 / (rrf_k + rank)
                normalized_score = (raw_score - min_score) / score_range
                combined_score = 0.6 * rrf_score + 0.4 * normalized_score

                score_map[doc.text] += self.weights[1] * combined_score
                doc_map[doc.text] = doc

        # Process QLD results if enabled
        if self.enable_qld and qld_docs:
            qld_scores = [score for _, score in qld_docs]
            max_score = max(qld_scores) if qld_scores else 1.0
            min_score = min(qld_scores) if qld_scores else 0.0
            score_range = max_score - min_score if max_score != min_score else 1.0

            for rank, (doc, raw_score) in enumerate(qld_docs, 1):
                rrf_score = 1.0 / (rrf_k + rank)
                normalized_score = (raw_score - min_score) / score_range
                combined_score = 0.6 * rrf_score + 0.4 * normalized_score

                score_map[doc.text] += self.weights[2] * combined_score
                doc_map[doc.text] = doc

        # Rank documents by combined scores
        ranked_texts = sorted(
            score_map.items(), key=lambda x: x[1], reverse=True)
        top_texts = [doc_map[text] for text, _ in ranked_texts[:self.k]]
        return top_texts

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        config: Optional[LexicalEnsembleConfig] = None,
        k: int = 10,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> "LexicalEnsembleRetriever":
        """
        Create LexicalEnsembleRetriever from documents.

        Args:
            documents: List of documents to index
            config: Configuration object (optional)
            k: Number of documents to return (used if config is None)
            weights: Retriever weights (used if config is None)

        Returns:
            Initialized LexicalEnsembleRetriever
        """
        # Create individual retrievers
        bm25 = BM25Retriever.from_documents(documents, k=2 * k)
        tfidf = TFIDFRetriever.from_documents(documents, k=2 * k)
        qld = QLDRetriever.from_documents(documents, k=2 * k)

        return cls(bm25=bm25, tfidf=tfidf, qld=qld, k=k, weights=weights, config=config)

    def get_config(self) -> LexicalEnsembleConfig:
        """Get current configuration"""
        return LexicalEnsembleConfig(
            k=self.k,
            weights=self.weights,
            enable_bm25=self.enable_bm25,
            enable_tfidf=self.enable_tfidf,
            enable_qld=self.enable_qld
        )

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        if 'k' in kwargs:
            self.k = kwargs['k']
        if 'weights' in kwargs:
            self.weights = kwargs['weights']
        if 'enable_bm25' in kwargs:
            self.enable_bm25 = kwargs['enable_bm25']
        if 'enable_tfidf' in kwargs:
            self.enable_tfidf = kwargs['enable_tfidf']
        if 'enable_qld' in kwargs:
            self.enable_qld = kwargs['enable_qld']

        # Validate that at least one retriever is enabled after update
        if not any([self.enable_bm25, self.enable_tfidf, self.enable_qld]):
            raise ValueError(
                "At least one lexical retriever (BM25, TF-IDF, or QLD) must be enabled")

        # Log any unknown parameters
        known_params = {'k', 'weights', 'enable_bm25',
                        'enable_tfidf', 'enable_qld'}
        for key in kwargs:
            if key not in known_params:
                print(f"Warning: Unknown config parameter '{key}'")


if __name__ == '__main__':
    pass
