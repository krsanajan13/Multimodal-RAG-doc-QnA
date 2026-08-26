"""
embeddings/embed_image.py

Creates image embeddings using HuggingFace CLIP (ViT-B/32) via transformers.
Outputs normalized feature vectors.
"""
from pathlib import Path
from typing import List
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embed_image")

DEFAULT_CLIP = "openai/clip-vit-base-patch32"


class ImageEmbedder:
    def __init__(self, model_name: str = DEFAULT_CLIP, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CLIP model (%s) on device %s", model_name, self.device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, image_paths: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Embed images into normalized vectors.
        """
        all_emb = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i: i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                emb = outputs.cpu().numpy()
                # normalize
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                emb = emb / norms
                all_emb.append(emb.astype("float32"))
        if all_emb:
            return np.vstack(all_emb)
        return np.zeros((0, self.model.config.projection_dim), dtype="float32")

    def dim(self) -> int:
        return self.model.config.projection_dim


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embed_image.py img1.png img2.jpg ...")
        sys.exit(1)
    emb = ImageEmbedder().embed(sys.argv[1:])
    print("Saved image embeddings shape:", emb.shape)
