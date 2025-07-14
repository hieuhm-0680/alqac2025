from typing import Any, Callable, Iterable, List
from collections import Counter
import math

from pydantic import Field, field_validator

from src.models.schemas import Document
from src.utils import preprocess_func_for_bm25
from .base import BaseRetriever


_DEFAULT_TOP_K_QLD = 10
_DEFAULT_MU = 1500

class QLDRetriever(BaseRetriever):
    """Query Likelihood with Dirichlet Smoothing Retriever.
    
    Implements the Query Likelihood Model with Dirichlet smoothing for document retrieval.
    The scoring function is: P(q|d) = ∏ P(t|d) where P(t|d) = (tf + μ * P(t|C)) / (|d| + μ)
    """
    
    docs: List[Document] = Field(repr=False)
    k: int = _DEFAULT_TOP_K_QLD
    mu: int = _DEFAULT_MU
    preprocess_func: Callable[[str], List[str]] = preprocess_func_for_bm25

    # Corpus-level statistics
    doc_tokens: List[List[str]] = Field(default_factory=list, repr=False)
    doc_lengths: List[int] = Field(default_factory=list, repr=False)
    doc_counters: List[Counter] = Field(default_factory=list, repr=False)
    vocab: set = Field(default_factory=set, repr=False)
    corpus_counter: Counter = Field(default_factory=Counter, repr=False)
    corpus_length: int = 0

    class Config:
        arbitrary_types_allowed = True

    @field_validator('k')
    @classmethod
    def validate_k(cls, v):
        if v <= 0:
            raise ValueError("k must be positive")
        return v

    @field_validator('mu')
    @classmethod
    def validate_mu(cls, v):
        if v <= 0:
            raise ValueError("mu must be positive")
        return v

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents using Query Likelihood with Dirichlet smoothing."""
        # Handle edge cases
        if not query or not query.strip():
            return []
        
        if not self.docs:
            return []

        query_tokens = self.preprocess_func(query)
        if not query_tokens:
            return []

        doc_scores = []

        for idx, (counter, length) in enumerate(zip(self.doc_counters, self.doc_lengths)):
            # Skip empty documents
            if length == 0:
                continue
                
            score = 0.0
            for token in query_tokens:
                tf = counter.get(token, 0)
                cf = self.corpus_counter.get(token, 0)
                
                # Collection probability with smoothing
                pwc = cf / self.corpus_length if self.corpus_length > 0 else 1e-10
                
                # Dirichlet smoothed probability
                prob = (tf + self.mu * pwc) / (length + self.mu)
                
                # Add log probability (with safety check)
                score += math.log(max(prob, 1e-10))
                
            doc_scores.append((idx, score))

        # Sort by score (descending) and return top-k
        if not doc_scores:
            return []
            
        top_indices = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:self.k]
        return [self.docs[i] for i, _ in top_indices]
    
    def _get_relevant_documents_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """Retrieve relevant documents and their scores using Query Likelihood with Dirichlet smoothing."""
        # Handle edge cases
        if not query or not query.strip():
            return []
        
        if not self.docs:
            return []

        query_tokens = self.preprocess_func(query)
        if not query_tokens:
            return []

        doc_scores = []

        for idx, (counter, length) in enumerate(zip(self.doc_counters, self.doc_lengths)):
            if length == 0:
                continue  # Skip empty documents
            
            score = 0.0
            for token in query_tokens:
                tf = counter.get(token, 0)
                cf = self.corpus_counter.get(token, 0)
                
                # Collection probability
                pwc = cf / self.corpus_length if self.corpus_length > 0 else 1e-10
                
                # Dirichlet smoothing
                prob = (tf + self.mu * pwc) / (length + self.mu)
                
                score += math.log(max(prob, 1e-10))  # Safe log

            doc_scores.append((idx, score))

        if not doc_scores:
            return []

        top_indices = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:self.k]

        return [(self.docs[i], score) for i, score in top_indices]


    def get_scores(self, query: str) -> List[tuple[int, float]]:
        """Get document indices and their scores for the query."""
        if not query or not query.strip() or not self.docs:
            return []

        query_tokens = self.preprocess_func(query)
        if not query_tokens:
            return []

        doc_scores = []
        for idx, (counter, length) in enumerate(zip(self.doc_counters, self.doc_lengths)):
            if length == 0:
                continue
                
            score = 0.0
            for token in query_tokens:
                tf = counter.get(token, 0)
                cf = self.corpus_counter.get(token, 0)
                pwc = cf / self.corpus_length if self.corpus_length > 0 else 1e-10
                prob = (tf + self.mu * pwc) / (length + self.mu)
                score += math.log(max(prob, 1e-10))
                
            doc_scores.append((idx, score))

        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        preprocess_func: Callable[[str], List[str]] = preprocess_func_for_bm25,
        k: int = _DEFAULT_TOP_K_QLD,
        mu: int = _DEFAULT_MU,
        **kwargs: Any
    ) -> "QLDRetriever":
        """Create QLDRetriever from a collection of documents."""
        docs = list(documents)
        
        if not docs:
            raise ValueError("Documents list cannot be empty")
        
        # Preprocess all documents
        doc_tokens = []
        for doc in docs:
            tokens = preprocess_func(doc.text) if doc.text else []
            doc_tokens.append(tokens)
        
        # Calculate statistics
        doc_lengths = [len(tokens) for tokens in doc_tokens]
        doc_counters = [Counter(tokens) for tokens in doc_tokens]
        
        # Build vocabulary and corpus counter
        vocab = set()
        corpus_counter = Counter()
        for tokens in doc_tokens:
            vocab.update(tokens)
            corpus_counter.update(tokens)
        
        corpus_length = sum(doc_lengths)
        
        if corpus_length == 0:
            raise ValueError("Corpus contains no tokens")

        return cls(
            docs=docs,
            doc_tokens=doc_tokens,
            doc_lengths=doc_lengths,
            doc_counters=doc_counters,
            vocab=vocab,
            corpus_counter=corpus_counter,
            corpus_length=corpus_length,
            k=k,
            mu=mu,
            preprocess_func=preprocess_func,
            **kwargs
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the retriever (requires rebuilding statistics)."""
        all_docs = self.docs + documents
        new_retriever = self.from_documents(
            all_docs,
            preprocess_func=self.preprocess_func,
            k=self.k,
            mu=self.mu
        )
        
        # Update current instance
        self.docs = new_retriever.docs
        self.doc_tokens = new_retriever.doc_tokens
        self.doc_lengths = new_retriever.doc_lengths
        self.doc_counters = new_retriever.doc_counters
        self.vocab = new_retriever.vocab
        self.corpus_counter = new_retriever.corpus_counter
        self.corpus_length = new_retriever.corpus_length

    def get_corpus_stats(self) -> dict:
        """Get corpus statistics for debugging/analysis."""
        return {
            "num_documents": len(self.docs),
            "vocab_size": len(self.vocab),
            "corpus_length": self.corpus_length,
            "avg_doc_length": self.corpus_length / len(self.docs) if self.docs else 0,
            "mu": self.mu,
            "k": self.k
        }