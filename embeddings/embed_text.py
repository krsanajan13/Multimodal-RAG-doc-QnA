"""
embeddings/embed_text.py

Creates text embeddings using SentenceTransformers.
Saves and returns numpy arrays. Provides a simple wrapper with caching.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embed_text")

# default model - small, fast and good for RAG
DEFAULT_TEXT_MODEL = "all-MiniLM-L6-v2"


class TextEmbedder:
    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, device: str = None):
        self.model_name = model_name
        logger.info("Loading text embedding model: %s", model_name)
        if device:
            self.model = SentenceTransformer(model_name, device=device)
        else:
            self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of texts and return normalized embeddings (L2 normalized).
        """
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype="float32")
        embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        # L2 normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        return embs.astype("float32")

    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()


def save_embeddings(path: str, embeddings: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    logger.info("Saved embeddings to %s (shape=%s)", path, embeddings.shape)


if __name__ == "__main__":
    # quick local test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embed_text.py path/to/texts.json")
        sys.exit(1)
    import json
    texts = json.load(open(sys.argv[1], "r"))
    # expect a list of strings
    if isinstance(texts, list) and isinstance(texts[0], str):
        emb = TextEmbedder().embed(texts)
        save_embeddings("data/processed/text_embeddings.npy", emb)
        print("Saved:", emb.shape)
    else:
        print("Input should be a list of strings")
