from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class BGEReranker:
    _model = None
    _tokenizer = None

    def __init__(self, model_name='BAAI/bge-reranker-v2-m3', device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if BGEReranker._model is None or BGEReranker._tokenizer is None:
            BGEReranker._tokenizer = AutoTokenizer.from_pretrained(model_name)
            BGEReranker._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            BGEReranker._model.eval()

        self.tokenizer = BGEReranker._tokenizer
        self.model = BGEReranker._model

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        max_token_per_doc: Optional[int] = None
    ) -> List[Dict[str, float]]:
        
        pairs = []
        for doc in documents:
            if max_token_per_doc is not None:
                tokens = self.tokenizer.tokenize(doc)
                doc = self.tokenizer.convert_tokens_to_string(tokens[:max_token_per_doc])
            pairs.append((query, doc))

        # Tokenize batch
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()

        scored_docs = [
            {"index": idx, "relevance_score": score.item()}
            for idx, score in enumerate(scores)
        ]
        scored_docs = sorted(scored_docs, key=lambda x: x["relevance_score"], reverse=True)

        return scored_docs[:top_n] if top_n is not None else scored_docs