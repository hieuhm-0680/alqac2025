import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional

from src.models.schemas import Document

class ChromaFacade:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path='./database/chroma',
            database='alqac2025',
        )

    def get_or_create_collection(self, name: str):
        for c in self.client.list_collections():
            if c.name == name:
                return self.client.get_collection(name)
        return self.client.create_collection(name)

    def add_texts(self, collection_name: str, texts: List[str], metadatas: Optional[List[dict]] = None):
        collection = self.get_or_create_collection(collection_name)
        ids = [f"{collection_name}_{i}" for i in range(len(texts))]
        collection.add(ids=ids, documents=texts, metadatas=metadatas or [{} for _ in texts])

    def add_documents(self, collection_name: str, documents: List[Document]):
        collection = self.get_or_create_collection(collection_name)
        ids = [f"{collection_name}_{i}" for i in range(len(documents))]
        texts = [doc.text for doc in documents]
        metadata = [{'law_id': doc.law_id, 'article_id': doc.article_id} for doc in documents]
        collection.add(ids=ids, documents=texts, metadatas=metadata)
  

    def query_texts(self, collection_name: str, query: str, top_k):
        collection = self.get_or_create_collection(collection_name)
        return collection.query(query_texts=[query], n_results=top_k)

