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
from src.models.schemas import Document

class GlobalIndexConfig(BaseModel):
    index_dir: Path = Field(default=Path("data/global_indexes"))
    chroma_db_path: Path = Field(default=Path("data/global_indexes/chroma_db"))
    lexical_path: Path = Field(default=Path("data/global_indexes/lexical_global.pkl"))

class GlobalRetrieverConfig(BaseModel):
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    top_k_semantic: int = Field(default=50)
    indexes: GlobalIndexConfig = Field(default_factory=GlobalIndexConfig)
    chroma_collection_name: str = "global_documents"
    lexical_ensemble_config: LexicalEnsembleConfig = Field(default_factory=LexicalEnsembleConfig)


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

    doc_texts = [doc.text for doc in all_documents]
    doc_ids = [doc.id for doc in all_documents]
    
    collection = client.get_or_create_collection(name=config.chroma_collection_name)
    
    # Semantic 
    embeddings = embedding_model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True)
    collection.add(
        embeddings=embeddings,
        documents=doc_texts,
        ids=doc_ids
    )

    # Lexical
    print("\nBuilding global lexical index...")
    # ensemble_config = LexicalEnsembleConfig(
    #     k=config.top_k_lexical,
    #     weights=(1.0, 1.0, 1.0)  
    # )
    lexical_ensemble = LexicalEnsembleRetriever.from_documents(all_documents, config=config.lexical_ensemble_config)

    with open(config.indexes.lexical_path, "wb") as f:
        pickle.dump(lexical_ensemble, f)


class GlobalRetriever:
    def __init__(self, config: GlobalRetrieverConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            with open(config.indexes.lexical_path, "rb") as f:
                self.lexical_ensemble_data = pickle.load(f)
        except FileNotFoundError as e:
            print(f"ERROR: Global lexical ensemble index file not found: {e.filename}.")
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
        ensemble_docs = self.lexical_ensemble_data._get_relevant_documents(query)
        for doc in ensemble_docs:
            final_docs_by_id[doc.id] = {'id': doc.id, 'text': doc.text}
        print(f"  - Found {len(ensemble_docs)} candidates via ensemble lexical search.")

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
    # all_docs = [
    #     {'id': 'tech001', 'text': 'NVIDIA announced a new GPU for deep learning.'},
    #     {'id': 'tech002', 'text': 'Quantum computing aims to solve complex problems.'},
    #     {'id': 'fin001', 'text': 'Central banks are raising interest rates to combat inflation.'},
    #     {'id': 'fin002', 'text': 'The stock market shows high volatility.'},
    #     {'id': 'fin003', 'text': 'Investors are concerned about rising bond yields.'},
    #     {'id': 'health001', 'text': 'A new study shows benefits of a Mediterranean diet.'},
    # ]

    docs = [
        Document(
            law_id="pl001",
            article_id="1",
            text="Người lao động có quyền được hưởng chế độ bảo hiểm xã hội theo quy định của pháp luật."
        ),
        Document(
            law_id="pl002",
            article_id="2",
            text="Việc xử phạt vi phạm hành chính phải dựa trên nguyên tắc khách quan, công bằng."
        ),
        Document(
            law_id="pl003",
            article_id="3",
            text="Mọi công dân đều có quyền tự do ngôn luận, tự do báo chí theo Hiến pháp nước Cộng hoà xã hội chủ nghĩa Việt Nam."
        ),
        Document(
            law_id="cn001",
            article_id="1",
            text="Trí tuệ nhân tạo đang được ứng dụng rộng rãi trong lĩnh vực chăm sóc sức khỏe và giáo dục."
        ),
        Document(
            law_id="yt001",
            article_id="1",
            text="Bộ Y tế khuyến cáo người dân nên tiêm vắc-xin phòng bệnh theo đúng lịch trình để đảm bảo hiệu quả bảo vệ."
        ),
        Document(
            law_id="kt001",
            article_id="1",
            text="Tăng trưởng GDP quý 1 năm nay đạt mức 5,8% nhờ vào sự phục hồi mạnh mẽ của ngành du lịch và xuất khẩu."
        ),
        Document(
            law_id="mt001",
            article_id="1",
            text="Luật bảo vệ môi trường yêu cầu các doanh nghiệp phải đánh giá tác động môi trường trước khi triển khai dự án."
        ),
    ]
    config = GlobalRetrieverConfig()

    # STEP 1: OFFLINE INDEXING
    build_global_indexes(config, docs)
    
    # STEP 2: ONLINE RETRIEVAL
    print("\n\n--- Simulating Application Startup ---")
    global_retriever = GlobalRetriever(config=config)
    
    search_query = "Quyền tự do ngôn luận của công dân là gì?"
    print(f"\nPerforming retrieval for query: '{search_query}'")
    retrieved_docs = global_retriever.retrieve(search_query)

    print(f"\n--- Final Retrieved Documents ---")
    if retrieved_docs:
        for doc in retrieved_docs:
            print(f"  - ID: {doc.get('id')}, Text: \"{doc.get('text', 'N/A')}\"")
    

