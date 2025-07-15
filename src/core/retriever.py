from pydantic import BaseModel, Field
from typing import List, Dict, Union

import json

from src.utils.logging import get_logger
from src.core.reranker import Reranker, RerankerConfig
from src.models.schemas import Document
from src.retrievers.global_retriever import (
    GlobalRetriever,
    GlobalRetrieverConfig,
    build_global_indexes,
)
from src.retrievers.local_retriever import (
    LocalRetriever,
    LocalRetrieverConfig,
    build_local_indexes,
)
from ..retrievers.base import BaseRetriever

logger = get_logger("alqac25")


class RankFusionConfig(BaseModel):
    k: int = 60
    method: str = Field(default="rrf")
    top_n_candidates: int = 100


class RankFusion:
    def __init__(self, config: RankFusionConfig):
        self.config = config

    def fuse(self, ranked_lists: List[List[Dict]]) -> List[Dict]:
        scores = {}
        for _, doc_list in enumerate(ranked_lists):
            for i, doc in enumerate(doc_list):
                doc_id = doc["id"]
                if doc_id not in scores:
                    scores[doc_id] = {"score": 0, "doc": doc}

                if self.config.method == "rrf":  # The RRF formula: 1 / (k + rank)
                    scores[doc_id]["score"] += 1 / (self.config.k + i + 1)
                else:
                    logger.debug(f"Fusion method `{self.config.method}` undefine")

        sorted_docs_with_scores = sorted(
            scores.values(), key=lambda x: x["score"], reverse=True
        )

        result = [item["doc"] for item in sorted_docs_with_scores]
        return result[:self.config.top_n_candidates]


class Retriever:
    def __init__(
        self,
        local_retriever_config: LocalRetrieverConfig,
        global_retriever_config: GlobalRetrieverConfig,
        rank_fusion_config: RankFusionConfig,
        reranker_config: RerankerConfig,
        save_local_path: str | None = None,
        save_global_path: str | None = None,
        save_fused_path: str | None = None,
        save_reranked_path: str | None = None,
    ):

        self.local_retriever = (
            LocalRetriever(local_retriever_config) if local_retriever_config is not None else None
        )
        self.global_retriever = (
            GlobalRetriever(global_retriever_config) if global_retriever_config is not None else None
        )
        self.rank_fusion = (
            RankFusion(rank_fusion_config) if rank_fusion_config is not None else None
        )
        self.reranker = (
            Reranker(reranker_config) if reranker_config is not None else None
        )

        self.save_local_path = save_local_path
        self.save_global_path = save_global_path
        self.save_fused_path = save_fused_path
        self.save_reranked_path = save_reranked_path

    def retrieve(self, query: str, wseg_query: str) -> List[Dict[str, Union[str, float]]]:
        local_results = []
        global_results = []

        if self.local_retriever is not None:
            local_results = self.local_retriever.retrieve(query, wseg_query)
            if self.save_local_path:
                with open(self.save_local_path, "w", encoding="utf-8") as f:
                    json.dump(local_results, f, ensure_ascii=False, indent=2)

        if self.global_retriever is not None:
            global_results = self.global_retriever.retrieve(query, wseg_query)
            if self.save_global_path:
                with open(self.save_global_path, "w", encoding="utf-8") as f:
                    json.dump(global_results, f, ensure_ascii=False, indent=2)

        candidates = []
        if self.rank_fusion is not None:
            candidates = self.rank_fusion.fuse([local_results, global_results])
            if self.save_fused_path:
                with open(self.save_fused_path, "w", encoding="utf-8") as f:
                    json.dump(candidates, f, ensure_ascii=False, indent=2)
        else:
            candidates = local_results + global_results

        final_results = candidates
        if self.reranker is not None:
            final_results = self.reranker.rerank(query, candidates)
            if self.save_reranked_path:
                with open(self.save_reranked_path, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, ensure_ascii=False, indent=2)

        return final_results
