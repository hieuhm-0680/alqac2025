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
    k: int = Field(default=10, ge=1, description="Number of top documents to return")
    weights: Tuple[float, float, float] = Field(
        default=(1.0, 1.0, 1.0), 
        description="Weights for BM25, TF-IDF, and QLD respectively"
    )
    
    class Config:
        extra = 'allow'


class LexicalEnsembleRetriever(BaseRetriever):
    """
    Ensemble retriever that combines BM25, TF-IDF, and QLD retrievers
    using weighted rank fusion.
    """
    
    bm25: BM25Retriever
    tfidf: TFIDFRetriever
    qld: QLDRetriever
    k: int = Field(default=10, ge=1)
    weights: Tuple[float, float, float] = Field(default=(1.0, 1.0, 1.0))

    def __init__(
        self,
        bm25: BM25Retriever,
        tfidf: TFIDFRetriever,
        qld: QLDRetriever,
        k: int = 10,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        config: Optional[LexicalEnsembleConfig] = None,
    ):
        # Nếu có config, ưu tiên sử dụng
        if config is not None:
            k = config.k
            weights = config.weights
        
        # Gọi parent constructor với tất cả parameters
        super().__init__(
            bm25=bm25,
            tfidf=tfidf,
            qld=qld,
            k=k,
            weights=weights
        )
        
        print(f"LexicalEnsembleRetriever initialized with k={self.k}, weights={self.weights}")

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

        # --- BM25 ---
        print(f"Retrieving from BM25...")
        bm25_docs = self.bm25._get_relevant_documents(query)
        for i, doc in enumerate(bm25_docs):
            # Rank-based scoring
            score_map[doc.text] += self.weights[0] * (len(bm25_docs) - i)
            doc_map[doc.text] = doc

        # --- TF-IDF ---
        print(f"Retrieving from TF-IDF...")
        tfidf_docs = self.tfidf._get_relevant_documents(query)
        for i, doc in enumerate(tfidf_docs):
            score_map[doc.text] += self.weights[1] * (len(tfidf_docs) - i)
            doc_map[doc.text] = doc

        # --- QLD ---
        print(f"Retrieving from QLD...")
        qld_docs = self.qld._get_relevant_documents(query)
        for i, doc in enumerate(qld_docs):
            score_map[doc.text] += self.weights[2] * (len(qld_docs) - i)
            doc_map[doc.text] = doc

        # --- Sort by aggregated score ---
        ranked_texts = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        top_texts = [doc_map[text] for text, _ in ranked_texts[:self.k]]

        print(f"Retrieved {len(top_texts)} documents from ensemble")
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
        bm25 = BM25Retriever.from_documents(documents)
        tfidf = TFIDFRetriever.from_documents(documents)
        qld = QLDRetriever.from_documents(documents)
        
        return cls(bm25=bm25, tfidf=tfidf, qld=qld, k=k, weights=weights, config=config)

    def get_config(self) -> LexicalEnsembleConfig:
        """Get current configuration"""
        return LexicalEnsembleConfig(k=self.k, weights=self.weights)

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        if 'k' in kwargs:
            self.k = kwargs['k']
        if 'weights' in kwargs:
            self.weights = kwargs['weights']
        
        # Log any unknown parameters
        known_params = {'k', 'weights'}
        for key in kwargs:
            if key not in known_params:
                print(f"Warning: Unknown config parameter '{key}'")

if __name__ == '__main__':
    pass