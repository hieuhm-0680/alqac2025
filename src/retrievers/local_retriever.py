import torch
from transformers import pipeline, Pipeline
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field
from typing import List, Dict, Union
import time
import pickle
from pathlib import Path
import chromadb
import numpy as np

from src.utils.preprocess_func_for_bm25 import preprocess_func_for_bm25, tokenize_text

class CategoryClassifierConfig(BaseModel):
    model_name_or_path: str = Field(default="vinai/phobert-base")
    top_k: int = 3
    model_cache_dir: str | None = None

class LocalIndexConfig(BaseModel):
    index_dir: Path = Field(default=Path("data/local_indexes"))
    chroma_db_path: Path = Field(default=Path("data/local_indexes/chroma_db"))
    bm25_path: Path = Field(default=Path("data/local_indexes/bm25_local.pkl"))

class LocalRetrieverConfig(BaseModel):
    classifier: CategoryClassifierConfig = Field(default_factory=CategoryClassifierConfig)
    embedding_model_name_or_path: str = Field(default="all-MiniLM-L6-v2")
    classification_threshold: float = Field(default=0.7)
    top_k_lexical: int = Field(default=5)
    top_k_semantic: int = Field(default=5)
    indexes: LocalIndexConfig = Field(default_factory=LocalIndexConfig)


def build_local_indexes(
    config: LocalRetrieverConfig, 
    document_store: Dict[str, List[Dict]]
):
    config.indexes.index_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(config.embedding_model_name_or_path, device=device)
    
    client = chromadb.PersistentClient(path=str(config.indexes.chroma_db_path))

    bm25_indexes = {}
    
    for category, docs in document_store.items():
        if not docs: continue
        print(f"Building VectorStore for '{category}'...")
        collection_name = f"{category}"
        collection = client.get_or_create_collection(name=collection_name)
        
        doc_texts = [doc['text'] for doc in docs]
        # doc_ids = None # TODO: fill here if needed
        doc_ids = [doc['id'] for doc in docs] if 'id' in docs[0] else [str(i) for i in range(len(doc_texts))]
        
        embeddings = embedding_model.encode(doc_texts, convert_to_numpy=True)
        collection.add(
            embeddings=embeddings,
            documents=doc_texts,
            ids=doc_ids
        )
        print(f"Building BM25 index for '{category}'...")
        tokenized_corpus = [tokenize_text(doc.lower()) for doc in doc_texts]
        bm25_indexes[category] = {
            'index': BM25Okapi(tokenized_corpus),
            'doc_ids': doc_ids 
        }

    with open(config.indexes.bm25_path, "wb") as f:
        pickle.dump(bm25_indexes, f)


class CategoryClassifier:
    def __init__(self, config: CategoryClassifierConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier: Pipeline = pipeline("text-classification", model=config.model_name_or_path, device=self.device)
    def classify(self, query: str) -> List[str]:
        results = self.classifier(query, top_k=self.config.top_k)
        return [
            res['label'] for res in results
        ]

class LocalRetriever:
    def __init__(self, config: LocalRetrieverConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("--- Initializing LocalRetriever for Inference ---")
        
        try:
            with open(config.indexes.bm25_path, "rb") as f:
                self.bm25_indexes = pickle.load(f)
                print(f"{self.bm25_indexes.keys()} BM25 indexes loaded successfully.")
        except FileNotFoundError as e:
            print(f"ERROR: BM25 index file not found: {e.filename}. Please run indexing script.")
            raise
            
        print(f"Connecting to ChromaDB at: {config.indexes.chroma_db_path}")
        self.chroma_client = chromadb.PersistentClient(path=str(config.indexes.chroma_db_path))

        self.classifier = CategoryClassifier(config.classifier)
        self.embedding_model = SentenceTransformer(config.embedding_model_name_or_path, device=self.device)
        print("LocalRetriver OK")

    def retrieve(self, query: str) -> List[Dict[str, Union[str, int, float]]]:
        selected_categories = self.classifier.classify(query)
        if not selected_categories: return []

        final_docs_by_id = {}
        query_embedding = self.embedding_model.encode(query)
        
        for category in selected_categories:
            print(f"Processing category: {category}")
            if category not in self.bm25_indexes: continue
            # Lexical Search
            bm25_data = self.bm25_indexes[category]
            tokenized_query = preprocess_func_for_bm25(query)

            # TODO: check from here
            bm25_scores = bm25_data['index'].get_scores(tokenized_query)
            top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.config.top_k_lexical]
            for i in top_bm25_indices:
                doc_id = bm25_data['doc_ids'][i]
                final_docs_by_id[doc_id] = {'id': doc_id} # We'll fetch text later if needed

            try:
                collection = self.chroma_client.get_collection(name=f"{category}")
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=self.config.top_k_semantic
                )
                # The result contains ids, documents, distances, etc.
                for i, doc_id in enumerate(results['ids'][0]):
                    final_docs_by_id[doc_id] = {'id': doc_id, 'text': results['documents'][0][i]}
            except ValueError:
                print(f"Warning: ChromaDB collection '{category}' not found. Skipping.")


        # ChromaDB already gives us the text, so we can return a richer object.
        return list(final_docs_by_id.values())

# --------------------------------------------------------------------------
# 5. Example Usage
# --------------------------------------------------------------------------

if __name__ == '__main__':
    # mock_document_store = {
    #     "technology": [{'id': 'tech001', 'text': 'NVIDIA ...'}, {'id': 'tech002', 'text': 'Quantum ...'}],
    #     "finance": [
    #         {'id': 'fin001', 'text': 'Central banks are raising interest rates to combat inflation.'},
    #         {'id': 'fin002', 'text': 'The stock market shows high volatility.'},
    #         {'id': 'fin003', 'text': 'Investors are concerned about rising bond yields.'},
    #     ],
    #     "health": [] 
    # }
    mock_document_store = {
        "LABEL_0": [{'id': 'tech001', 'text': 'NVIDIA ...'}, {'id': 'tech002', 'text': 'Quantum ...'}],
        "LABEL_1": [
            {'id': 'fin001', 'text': 'Central banks are raising interest rates to combat inflation.'},
            {'id': 'fin002', 'text': 'The stock market shows high volatility.'},
            {'id': 'fin003', 'text': 'Investors are concerned about rising bond yields.'},
        ],
    }
    config = LocalRetrieverConfig(
        classifier=CategoryClassifierConfig(candidate_labels=list(mock_document_store.keys()))
    )

    # STEP 1: OFFLINE INDEXING
    build_local_indexes(config, mock_document_store)
    
    # STEP 2: ONLINE RETRIEVAL
    print("\n\n--- Simulating Application Startup ---")
    retriever = LocalRetriever(config=config)
    
    search_query = "What is the latest news on equities?"
    print(f"\nPerforming retrieval for query: '{search_query}'")
    results = retriever.retrieve(search_query)

    print(f"\n--- Final Retrieved Documents ---")
    if results:
        for doc in results:
            print(f"  - ID: {doc.get('id')}, Text: \"{doc.get('text', 'N/A - from lexical search')}\"")
    
    print("\n--- Demo Finished ---")

