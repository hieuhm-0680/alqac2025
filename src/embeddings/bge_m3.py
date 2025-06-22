
from .embeddings import Embeddings
from .registry import emd_registry

@emd_registry.register("bge_m3")
class BGEM3Embedding(Embeddings):
    def __init__(self):
        pass
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    def embed_query(self, text: str) -> list[float]:
        pass