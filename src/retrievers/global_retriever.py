import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field
from typing import List, Dict, Union
import pickle
from pathlib import Path
import chromadb
import numpy as np

from src.utils.preprocess_func_for_bm25 import preprocess_func_for_bm25, tokenize_text


class GlobalIndexConfig(BaseModel):
    index_dir: Path = Field(default=Path("data/global_indexes"))
    chroma_db_path: Path = Field(default=Path("data/global_indexes/chroma_db"))
    bm25_path: Path = Field(default=Path("data/global_indexes/bm25_global.pkl"))

class GlobalRetrieverConfig(BaseModel):
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    top_k_lexical: int = Field(default=50)
    top_k_semantic: int = Field(default=50)
    indexes: GlobalIndexConfig = Field(default_factory=GlobalIndexConfig)
    chroma_collection_name: str = "global_documents"


def build_global_indexes(
    config: GlobalRetrieverConfig, 
    all_documents: List[Dict[str, str]]
):
    config.indexes.index_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ChromaDB data to: {config.indexes.chroma_db_path.resolve()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(config.embedding_model_name, device=device)
    
    # TODO: chroma client occurs more than once, consider refactoring
    client = chromadb.PersistentClient(path=str(config.indexes.chroma_db_path))
    client.delete_collection(name=config.chroma_collection_name)

    doc_texts = [doc['text'] for doc in all_documents]
    doc_ids = [doc['id'] for doc in all_documents]
    
    collection = client.get_or_create_collection(name=config.chroma_collection_name)
    
    embeddings = embedding_model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True)
    collection.add(
        embeddings=embeddings,
        documents=doc_texts,
        ids=doc_ids
    )

    print("\nBuilding global BM25 index...")
    tokenized_corpus = [tokenize_text(doc.lower()) for doc in doc_texts]
    bm25_index_data = {
        'index': BM25Okapi(tokenized_corpus),
        'doc_ids': doc_ids
    }
    with open(config.indexes.bm25_path, "wb") as f:
        pickle.dump(bm25_index_data, f)


class GlobalRetriever:
    def __init__(self, config: GlobalRetrieverConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            with open(config.indexes.bm25_path, "rb") as f:
                self.bm25_index_data = pickle.load(f)
        except FileNotFoundError as e:
            print(f"ERROR: Global BM25 index file not found: {e.filename}.")
            raise
            
        # TODO: chroma client occurs more than once, consider refactoring
        self.chroma_client = chromadb.PersistentClient(path=str(config.indexes.chroma_db_path))
        try:
            self.collection = self.chroma_client.get_collection(name=config.chroma_collection_name)
        except ValueError:
            print(f"ERROR: ChromaDB collection '{config.chroma_collection_name}' not found.")
            raise

        self.embedding_model = SentenceTransformer(config.embedding_model_name, device=self.device)
        print(f"GlobalRetriever OK")

    def retrieve(self, query: str) -> List[Dict[str, Union[str, float]]]:
        final_docs_by_id = {}
        query_embedding = self.embedding_model.encode(query)
        
        # --- Lexical Search ---
        print("\nPerforming global lexical search...")
        tokenized_query = preprocess_func_for_bm25(query)
        bm25_scores = self.bm25_index_data['index'].get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.config.top_k_lexical]
        for i in top_bm25_indices:
            doc_id = self.bm25_index_data['doc_ids'][i]
            final_docs_by_id[doc_id] = {'id': doc_id}
        print(f"  - Found {len(final_docs_by_id)} candidates via lexical search.")

        # --- Semantic Search with ChromaDB ---
        print("Performing global semantic search...")
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.config.top_k_semantic
        )
        for i, doc_id in enumerate(results['ids'][0]):
            final_docs_by_id[doc_id] = {'id': doc_id, 'text': results['documents'][0][i]}
        print(f"  - Total unique candidates after semantic search: {len(final_docs_by_id)}.")
        
        return list(final_docs_by_id.values())


if __name__ == '__main__':
    all_docs = [
        {'id': 'tech001', 'text': 'NVIDIA announced a new GPU for deep learning.'},
        {'id': 'tech002', 'text': 'Quantum computing aims to solve complex problems.'},
        {'id': 'fin001', 'text': 'Central banks are raising interest rates to combat inflation.'},
        {'id': 'fin002', 'text': 'The stock market shows high volatility.'},
        {'id': 'fin003', 'text': 'Investors are concerned about rising bond yields.'},
        {'id': 'health001', 'text': 'A new study shows benefits of a Mediterranean diet.'},
    ]
    
    config = GlobalRetrieverConfig()

    # STEP 1: OFFLINE INDEXING
    build_global_indexes(config, all_docs)
    
    # STEP 2: ONLINE RETRIEVAL
    print("\n\n--- Simulating Application Startup ---")
    global_retriever = GlobalRetriever(config=config)
    
    search_query = "What are the financial implications of new technology?"
    print(f"\nPerforming retrieval for query: '{search_query}'")
    retrieved_docs = global_retriever.retrieve(search_query)

    print(f"\n--- Final Retrieved Documents ---")
    if retrieved_docs:
        for doc in retrieved_docs:
            print(f"  - ID: {doc.get('id')}, Text: \"{doc.get('text', 'N/A')}\"")
