"""
embeddings/build_index.py

Loads final_chunks.json, builds unified multimodal embeddings:
- text chunks -> text embeddings
- image chunks -> combine text (OCR) embedding and image visual embedding (if image exists)
Saves:
 - FAISS index (.index)
 - id2meta.json (mapping index->chunk metadata)
 - embeddings.npy (optional)
"""
import json
from pathlib import Path
import numpy as np
import faiss
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_index")

from embeddings.embed_text import TextEmbedder
from embeddings.embed_image import ImageEmbedder


def load_chunks(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    return json.loads(p.read_text(encoding="utf8"))


def build_index(chunks_path: str,
                out_dir: str = "data/processed",
                text_model: str = None,
                clip_model: str = None,
                weight_image: float = 0.6):
    """
    Build FAISS index for all chunks.
    If a chunk includes an image_path in meta, we compute image embedding and combine with text embedding:
      final_emb = normalize( w_text * text_emb + w_img * img_emb )
    weight_image: ratio for visual features when present (0..1)
    """
    chunks = load_chunks(chunks_path)
    logger.info("Loaded %d chunks", len(chunks))
    texts = [c.get("text","") for c in chunks]

    text_embedder = TextEmbedder(model_name=text_model) if text_model else TextEmbedder()
    text_embs = text_embedder.embed(texts)

    # prepare image embeddings where available
    img_paths = []
    img_indices = []  # map to chunk idx
    for idx, c in enumerate(chunks):
        meta = c.get("meta", {})
        img_path = meta.get("image_path") or meta.get("image") or meta.get("image_path_rel")
        if img_path:
            img_paths.append(img_path)
            img_indices.append(idx)

    image_embedder = None
    img_embs = None
    if img_paths:
        image_embedder = ImageEmbedder(model_name=clip_model) if clip_model else ImageEmbedder()
        img_embs = image_embedder.embed(img_paths)
        logger.info("Computed image embeddings shape %s for %d images", img_embs.shape, len(img_paths))

    # combine embeddings
    final_embs = []
    for i in range(len(chunks)):
        t = text_embs[i]
        if i in img_indices:
            img_pos = img_indices.index(i)
            v = img_embs[img_pos]
            # align dims if needed (CLIP dim may differ from text dim) -> project or pad/truncate
            if v.shape[0] != t.shape[0]:
                # simple projection: if CLIP larger, truncate; if smaller, pad with zeros
                if v.shape[0] > t.shape[0]:
                    v = v[:t.shape[0]]
                else:
                    v = np.pad(v, (0, t.shape[0] - v.shape[0]), mode="constant")
            # weighted average
            w_img = float(weight_image)
            w_text = 1.0 - w_img
            comb = w_text * t + w_img * v
            # normalize
            norm = np.linalg.norm(comb)
            if norm == 0:
                comb = comb
            else:
                comb = comb / norm
            final_embs.append(comb.astype("float32"))
        else:
            final_embs.append(t.astype("float32"))

    final_embs = np.vstack(final_embs)
    logger.info("Final embeddings shape: %s", final_embs.shape)

    # build FAISS index (inner product on normalized vectors = cosine)
    d = final_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    # if number of vectors large, consider IndexIVFFlat with training.
    index.add(final_embs)
    logger.info("FAISS index built with %d vectors. dim=%d", index.ntotal, d)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir_p / "vector_index.faiss"))
    np.save(str(out_dir_p / "embeddings.npy"), final_embs)
    # save id->meta mapping
    id2meta = {i: {"id": chunks[i].get("id"), "page": chunks[i].get("page"),
                   "type": chunks[i].get("type"), "title": chunks[i].get("title"),
                   "text": chunks[i].get("text"), "meta": chunks[i].get("meta", {})}
               for i in range(len(chunks))}
    (out_dir_p / "id2meta.json").write_text(json.dumps(id2meta, ensure_ascii=False, indent=2), encoding="utf8")

    logger.info("Saved index, embeddings.npy, id2meta.json into %s", out_dir_p)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/processed/final_chunks.json")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--weight_image", type=float, default=0.6)
    args = parser.parse_args()
    build_index(args.chunks, out_dir=args.out, weight_image=args.weight_image)
