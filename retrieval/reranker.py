"""
retrieval/reranker.py

Optional cross-modal re-ranking: given query and retrieved candidates,
re-score using a stronger cross-encoder or by cosine similarity in embedding space.

This implementation uses SentenceTransformer for a fast re-rank by computing
cosine similarity between query embedding and candidate text embeddings.
"""
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reranker")

DEFAULT_RERANK_MODEL = "all-mpnet-base-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        logger.info("Loading rerank model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        candidates: list of dicts with 'text' fields
        returns candidates sorted by similarity desc
        """
        texts = [c.get("text", "") for c in candidates]
        if not texts:
            return candidates
        q_emb = self.model.encode([query], convert_to_numpy=True)
        c_embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # normalize
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        c_embs = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-9)
        sims = (c_embs @ q_emb.T).squeeze(axis=1)
        for ci, s in zip(candidates, sims.tolist()):
            ci["rerank_score"] = float(s)
        candidates = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return candidates


if __name__ == "__main__":
    # small demo
    r = Reranker()
    cand = [{"text": "Qatar GDP growth is 2% in 2024"}, {"text": "Inflation was 1.2% in 2024"}]
    print(r.rerank("What is Qatar GDP growth in 2024?", cand))
