class EmbeddingRegistry:
    def __init__(self):
        self._embeddings = {}

    def register(self, name, embedding_class):
        if name in self._embeddings:
            raise ValueError(f"Embedding '{name}' is already registered.")
        self._embeddings[name] = embedding_class

    def get(self, name):
        if name not in self._embeddings:
            raise ValueError(f"Embedding '{name}' is not registered.")
        return self._embeddings[name]

    def list_embeddings(self):
        return list(self._embeddings.keys())

        
emd_registry = EmbeddingRegistry()