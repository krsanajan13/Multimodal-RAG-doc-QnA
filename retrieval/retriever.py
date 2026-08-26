"""
retrieval/retriever.py

Loads FAISS index and id2meta mapping and performs retrieval for a given query.
Supports retrieving top_k results with scores and metadata.
"""
import faiss
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retriever")

from embeddings.embed_text import TextEmbedder


class Retriever:
    def __init__(self, index_path: str = "data/processed/vector_index.faiss",
                 id2meta_path: str = "data/processed/id2meta.json",
                 embeddings_path: str = "data/processed/embeddings.npy",
                 text_model: str = None):
        self.index_path = index_path
        self.id2meta_path = id2meta_path
        self.embeddings_path = embeddings_path
        self.text_embedder = TextEmbedder(model_name=text_model) if text_model else TextEmbedder()
        self._load_index()

    def _load_index(self):
        p = Path(self.index_path)
        if not p.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        self.index = faiss.read_index(str(p))
        self.id2meta = json.loads(Path(self.id2meta_path).read_text(encoding="utf8"))
        logger.info("Loaded FAISS index and %d metadata entries", len(self.id2meta))

    def query(self, text_query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.text_embedder.embed([text_query])[0].astype("float32")
        # normalize (should already be normalized)
        norm = np.linalg.norm(q_emb)
        if norm != 0:
            q_emb = q_emb / norm
        q_emb = q_emb.reshape(1, -1)
        scores, idxs = self.index.search(q_emb, top_k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()
        results = []
        for sc, idx in zip(scores, idxs):
            if idx < 0:
                continue
            meta = self.id2meta.get(str(idx)) or self.id2meta.get(idx)
            results.append({
                "score": float(sc),
                "index": int(idx),
                "chunk_id": meta.get("id") if meta else None,
                "page": meta.get("page") if meta else None,
                "type": meta.get("type") if meta else None,
                "title": meta.get("title") if meta else None,
                "text": meta.get("text") if meta else None,
                "meta": meta.get("meta") if meta else None
            })
        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    r = Retriever()
    out = r.query(args.q, top_k=args.k)
    import json
    print(json.dumps(out, indent=2, ensure_ascii=False))
