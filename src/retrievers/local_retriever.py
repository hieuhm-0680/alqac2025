import torch
from src.models.schemas import Document
from transformers import pipeline, Pipeline
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field
from typing import List, Dict, Union
import pickle
from pathlib import Path
import chromadb

from src.utils.logging import get_logger
from src.utils.rrf_fusion import fuse
from src.utils.convert_collection_name import convert_collection_name
from src.utils.preprocess_func_for_bm25 import preprocess_func_for_bm25, tokenize_text

logger = get_logger("alqac25")


class CategoryClassifierConfig(BaseModel):
    model_name_or_path: str = Field(default="vinai/phobert-base")
    top_k: int = 3
    model_cache_dir: str | None = None


class LocalIndexConfig(BaseModel):
    index_dir: Path = Field(default=Path("data/local_indexes"))
    chroma_db_path: Path = Field(default=Path("data/local_indexes/chroma_db"))
    bm25_path: Path = Field(default=Path("data/local_indexes/bm25_local.pkl"))


class LocalRetrieverConfig(BaseModel):
    classifier: CategoryClassifierConfig = Field(
        default_factory=CategoryClassifierConfig
    )
    embedding_model_name_or_path: str = Field(default="all-MiniLM-L6-v2")
    top_k_lexical: int = Field(default=5)
    top_k_semantic: int = Field(default=5)
    indexes: LocalIndexConfig = Field(default_factory=LocalIndexConfig)
    enable_lexical_search: bool = Field(
        default=True, description="Enable/disable lexical search (BM25)")
    enable_semantic_search: bool = Field(
        default=True, description="Enable/disable semantic search")


def build_local_indexes(
    config: LocalRetrieverConfig, document_store: Dict[str, List[Document]]
):
    config.indexes.index_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Building local indexes with semantic={config.enable_semantic_search}, lexical={config.enable_lexical_search}")

    # Initialize components based on what's enabled
    if config.enable_semantic_search:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer(
            config.embedding_model_name_or_path, device=device
        )
        client = chromadb.PersistentClient(
            path=str(config.indexes.chroma_db_path))

    if config.enable_lexical_search:
        bm25_indexes = {}

    for category, docs in document_store.items():
        if not docs:
            continue
        collection_name = convert_collection_name(category)

        doc_texts = [doc.text for doc in docs]
        doc_ids = [doc.id for doc in docs]

        # Build semantic index if enabled
        if config.enable_semantic_search:
            logger.info(f"Building VectorStore for '{collection_name}'...")
            collection = client.get_or_create_collection(
                name=collection_name, metadata={"category": category}
            )
            embeddings = embedding_model.encode(
                doc_texts, convert_to_numpy=True)
            collection.add(embeddings=embeddings,
                           documents=doc_texts, ids=doc_ids)

        # Build lexical index if enabled
        if config.enable_lexical_search:
            logger.info(f"Building BM25 index for '{category}'...")
            tokenized_corpus = [
                preprocess_func_for_bm25(doc) for doc in doc_texts]
            bm25_indexes[category] = {
                "index": BM25Okapi(tokenized_corpus),
                "doc_ids": doc_ids,
                "texts": doc_texts,
            }

    # Save lexical indexes if enabled
    if config.enable_lexical_search:
        with open(config.indexes.bm25_path, "wb") as f:
            pickle.dump(bm25_indexes, f)
        logger.info("✓ Lexical indexes saved successfully")
    else:
        logger.info("✗ Lexical indexing skipped (disabled)")

    if config.enable_semantic_search:
        logger.info("✓ Semantic indexes built successfully")
    else:
        logger.info("✗ Semantic indexing skipped (disabled)")

    logger.info("Local indexing completed")


class CategoryClassifier:
    def __init__(self, config: CategoryClassifierConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier: Pipeline = pipeline(
            "text-classification", model=config.model_name_or_path, device=self.device
        )

    def classify(self, query: str) -> List[str]:
        results = self.classifier(query, top_k=self.config.top_k)
        return [res["label"] for res in results]


class LocalRetriever:
    def __init__(self, config: LocalRetrieverConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("--- Initializing LocalRetriever ---")

        # Initialize lexical search components if enabled
        if config.enable_lexical_search:
            try:
                with open(config.indexes.bm25_path, "rb") as f:
                    self.bm25_indexes = pickle.load(f)
                    logger.info(
                        f"BM25 indexes loaded successfully.\n Found {len(self.bm25_indexes)} categories: {self.bm25_indexes.keys()} "
                    )
            except FileNotFoundError as e:
                logger.error(
                    f"ERROR: BM25 index file not found: {e.filename}. Please run indexing script."
                )
                raise
        else:
            self.bm25_indexes = None
            logger.info("Lexical search disabled for LocalRetriever")

        # Initialize semantic search components if enabled
        if config.enable_semantic_search:
            logger.info(
                f"Connecting to ChromaDB at: {config.indexes.chroma_db_path}")
            self.chroma_client = chromadb.PersistentClient(
                path=str(config.indexes.chroma_db_path)
            )
            self.embedding_model = SentenceTransformer(
                config.embedding_model_name_or_path, device=self.device
            )
        else:
            self.chroma_client = None
            self.embedding_model = None
            logger.info("Semantic search disabled for LocalRetriever")

        # Validate that at least one search method is enabled
        if not config.enable_lexical_search and not config.enable_semantic_search:
            raise ValueError(
                "At least one search method (lexical or semantic) must be enabled")

        self.classifier = CategoryClassifier(config.classifier)
        logger.info("LocalRetriever OK")

    def retrieve(self, query: str) -> List[Dict[str, Union[str, int, float]]]:
        selected_categories = self.classifier.classify(query)
        if not selected_categories:
            return []

        final_docs_by_id = {}
        query_embedding = self.embedding_model.encode(
            query) if self.config.enable_semantic_search else None
        bm25_results = []
        semantic_results = []

        for category in selected_categories:
            logger.info(f"Processing category: {category}")

            # Lexical search (BM25)
            if self.config.enable_lexical_search:
                if category not in self.bm25_indexes:
                    logger.debug(
                        f"Category '{category}' not found in BM25 indexes. Skipping."
                    )
                    continue
                bm25_data = self.bm25_indexes[category]
                tokenized_query = preprocess_func_for_bm25(query)

                bm25_scores = bm25_data["index"].get_scores(tokenized_query)
                top_bm25_indices = sorted(
                    range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True
                )[:4 * self.config.top_k_lexical]
                for i in top_bm25_indices:
                    doc_id = bm25_data["doc_ids"][i]
                    doc_text = bm25_data["texts"][i]
                    bm25_results.append(
                        {"id": doc_id, "text": doc_text,
                            "bm25_score": bm25_scores[i]}
                    )

            # Semantic search
            if self.config.enable_semantic_search:
                try:
                    collection = self.chroma_client.get_collection(
                        name=convert_collection_name(category)
                    )
                    results = collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=4 * self.config.top_k_semantic,
                    )

                    for i, doc_id in enumerate(results["ids"][0]):
                        semantic_results.append(
                            {
                                "id": doc_id,
                                "text": results["documents"][0][i],
                                "distance": results["distances"][0][i],
                            }
                        )
                except ValueError:
                    logger.error(
                        f"Warning: ChromaDB collection '{category}' not found. Skipping."
                    )

        # Process results based on what's enabled
        if self.config.enable_lexical_search:
            bm25_results.sort(key=lambda x: x["bm25_score"], reverse=True)
            bm25_results = bm25_results[:self.config.top_k_lexical]

        if self.config.enable_semantic_search:
            semantic_results.sort(key=lambda x: x["distance"])
            semantic_results = semantic_results[: self.config.top_k_semantic]

        # Combine results based on what's enabled
        if self.config.enable_lexical_search and self.config.enable_semantic_search:
            results = fuse([bm25_results, semantic_results])
        elif self.config.enable_lexical_search:
            results = bm25_results
        elif self.config.enable_semantic_search:
            results = semantic_results
        else:
            results = []

        logger.info(
            f"Retrieved {len(bm25_results)} BM25 results (enabled: {self.config.enable_lexical_search}) and {len(semantic_results)} semantic results (enabled: {self.config.enable_semantic_search})."
        )
        logger.info(f"Final results count: {len(results)}")
        return results
