import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel, Field
from typing import List, Dict, Union
import time


class RerankerConfig(BaseModel):
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="The name of the Cross-Encoder model from Hugging Face."
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for inference to optimize performance."
    )
    model_cache_dir: str | None = Field(
        default=None,
        description="Directory to cache downloaded models. If None, uses default Hugging Face cache."
    )


class Reranker:
    def __init__(self, config: RerankerConfig):
        print(f"\nInitializing Reranker with config: \n{config.model_dump_json(indent=2)}")
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.cross_encoder_model,
            cache_dir=self.config.model_cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.cross_encoder_model,
            cache_dir=self.config.model_cache_dir
        ).to(self.device)
        self.model.eval()
        print(f"Reranker OK\n")

    def rerank(self, query: str, documents: List[Dict[str, Union[str, int, float]]]) -> List[Dict[str, Union[str, int, float]]]:
        if not documents:
            return []

        if not all('text' in doc for doc in documents):
            raise ValueError("All document dictionaries must contain a 'text' key.")

        start_time = time.time()
        
        pairs = [(query, doc['text']) for doc in documents]

        all_scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.config.batch_size):
                batch_pairs = pairs[i:i + self.config.batch_size]
                
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)

                logits = self.model(**inputs).logits
                scores = logits.view(-1, ).float()
                all_scores.extend(scores.cpu().numpy().tolist())

        for doc, score in zip(documents, all_scores):
            doc['rerank_score'] = score
            
        sorted_documents = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        end_time = time.time()
        print(f"Reranking completed for {len(documents)} documents in {end_time - start_time:.4f} seconds.")

        return sorted_documents
