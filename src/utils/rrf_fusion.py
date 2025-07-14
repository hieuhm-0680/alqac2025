from typing import Dict, List
from src.utils.logging import get_logger

logger = get_logger('alqac25')

C = 60

def fuse(ranked_lists: List[List[Dict]]) -> List[Dict]:
    scores = {}
    for _, doc_list in enumerate(ranked_lists):
        for i, doc in enumerate(doc_list):
            doc_id = doc["id"]
            if doc_id not in scores:
                scores[doc_id] = {"score": 0, "doc": doc}

            scores[doc_id]["score"] += 1 / (C + i + 1)

    sorted_docs_with_scores = sorted(
        scores.values(), key=lambda x: x["score"], reverse=True
    )

    result = [item["doc"] for item in sorted_docs_with_scores]
    return result