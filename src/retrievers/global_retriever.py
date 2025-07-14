from src.utils.logging import get_logger
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
from src.retrievers.lexical import LexicalEnsembleRetriever, LexicalEnsembleConfig
from src.utils.rrf_fusion import fuse
from src.models.schemas import Document


logger = get_logger("alqac25")


class GlobalIndexConfig(BaseModel):
    index_dir: Path = Field(default=Path("data/global_indexes"))
    chroma_db_path: Path = Field(default=Path("data/global_indexes/chroma_db"))
    lexical_path: Path = Field(default=Path(
        "data/global_indexes/lexical_global.pkl"))


class GlobalRetrieverConfig(BaseModel):
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    top_k_semantic: int = Field(default=50)
    indexes: GlobalIndexConfig = Field(default_factory=GlobalIndexConfig)
    chroma_collection_name: str = "global_documents"
    lexical_ensemble_config: LexicalEnsembleConfig = Field(
        default_factory=LexicalEnsembleConfig
    )
    enable_lexical_search: bool = Field(
        default=True, description="Enable/disable lexical search (BM25, TF-IDF, QLD)")
    enable_semantic_search: bool = Field(
        default=True, description="Enable/disable semantic search")


def build_global_indexes(
    config: GlobalRetrieverConfig, all_documents: List[Dict[str, str]]
):
    config.indexes.index_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Building global indexes with semantic={config.enable_semantic_search}, lexical={config.enable_lexical_search}")

    # Build semantic index if enabled
    if config.enable_semantic_search:
        print(
            f"Saving ChromaDB data to: {config.indexes.chroma_db_path.resolve()}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer(
            config.embedding_model_name, device=device)

        # TODO: chroma client occurs more than once, consider refactoring
        client = chromadb.PersistentClient(
            path=str(config.indexes.chroma_db_path))

        doc_texts = [doc.text for doc in all_documents]
        doc_ids = [doc.id for doc in all_documents]

        collection = client.get_or_create_collection(
            name=config.chroma_collection_name)

        # Semantic
        embeddings = embedding_model.encode(
            doc_texts, convert_to_numpy=True, show_progress_bar=True
        )
        collection.add(embeddings=embeddings, documents=doc_texts, ids=doc_ids)
        print("✓ Semantic index built successfully")
    else:
        print("✗ Semantic indexing skipped (disabled)")

    # Build lexical index if enabled
    if config.enable_lexical_search:
        print("\nBuilding global lexical index...")
        lexical_ensemble = LexicalEnsembleRetriever.from_documents(
            all_documents, config=config.lexical_ensemble_config, k=config.lexical_ensemble_config.k
        )

        with open(config.indexes.lexical_path, "wb") as f:
            pickle.dump(lexical_ensemble, f)
        print("✓ Lexical index built successfully")
    else:
        print("✗ Lexical indexing skipped (disabled)")

    print("Global indexing completed")


class GlobalRetriever:
    def __init__(self, config: GlobalRetrieverConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize lexical search components if enabled
        if config.enable_lexical_search:
            try:
                with open(config.indexes.lexical_path, "rb") as f:
                    self.lexical_ensemble_data = pickle.load(f)
            except FileNotFoundError as e:
                logger.error(
                    f"ERROR: Global lexical ensemble index file not found: {e.filename}."
                )
                raise
        else:
            self.lexical_ensemble_data = None
            logger.info("Lexical search disabled for GlobalRetriever")

        # Initialize semantic search components if enabled
        if config.enable_semantic_search:
            # TODO: chroma client occurs more than once, consider refactoring
            self.chroma_client = chromadb.PersistentClient(
                path=str(config.indexes.chroma_db_path)
            )
            try:
                self.collection = self.chroma_client.get_collection(
                    name=config.chroma_collection_name
                )
            except ValueError:
                logger.error(
                    f"ERROR: ChromaDB collection '{config.chroma_collection_name}' not found."
                )
                raise

            self.embedding_model = SentenceTransformer(
                config.embedding_model_name, device=self.device
            )
        else:
            self.chroma_client = None
            self.collection = None
            self.embedding_model = None
            logger.info("Semantic search disabled for GlobalRetriever")

        # Validate that at least one search method is enabled
        if not config.enable_lexical_search and not config.enable_semantic_search:
            raise ValueError(
                "At least one search method (lexical or semantic) must be enabled")

        logger.info(">>> GlobalRetriever OK")

    def retrieve(self, query: str) -> List[Dict[str, Union[str, float]]]:
        query_embedding = self.embedding_model.encode(
            query) if self.config.enable_semantic_search else None

        lexical_results = []
        semantic_results = []

        # Lexical search
        if self.config.enable_lexical_search:
            ensemble_docs = self.lexical_ensemble_data._get_relevant_documents(
                query)
            for doc in ensemble_docs:
                lexical_results.append({
                    "id": doc.id,
                    "text": doc.text,
                })

        # Semantic search
        if self.config.enable_semantic_search:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.config.top_k_semantic,
            )
            for i, doc_id in enumerate(results["ids"][0]):
                semantic_results.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                })

        # Combine results based on what's enabled
        search_results = []
        if self.config.enable_lexical_search and self.config.enable_semantic_search:
            fused_results = fuse([lexical_results, semantic_results])
            search_results = fused_results
        elif self.config.enable_lexical_search:
            search_results = lexical_results
        elif self.config.enable_semantic_search:
            search_results = semantic_results

        logger.info(
            f"  - Lexical search found {len(lexical_results)} candidates (enabled: {self.config.enable_lexical_search})."
        )
        logger.info(
            f"  - Semantic search found {len(semantic_results)} candidates (enabled: {self.config.enable_semantic_search})."
        )
        logger.info(
            f"  - Total fused candidates after global search: {len(search_results)}."
        )

        return search_results
