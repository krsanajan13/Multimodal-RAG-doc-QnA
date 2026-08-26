"""
config.py
Central configuration file for Multi-Modal RAG QA System.
Contains all paths, model names, chunk settings, and vector database configs.
Used by ingestion, embeddings, retrieval, generation, and the Streamlit app.
"""

from pathlib import Path

# -----------------------------
# PROJECT ROOT
# -----------------------------
ROOT = Path(__file__).resolve().parent

# -----------------------------
# DATA PATHS
# -----------------------------
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# Raw PDF expected by assignment
DEFAULT_PDF = DATA_RAW / "qatar_test_doc.pdf"

# Processed output files
TEXT_CHUNKS_FILE = DATA_PROCESSED / "text_chunks.json"
TABLE_CHUNKS_FILE = DATA_PROCESSED / "table_chunks.json"
IMAGE_CHUNKS_FILE = DATA_PROCESSED / "image_chunks.json"
FINAL_CHUNKS_FILE = DATA_PROCESSED / "final_chunks.json"

# Embeddings + FAISS index
EMBEDDINGS_FILE = DATA_PROCESSED / "embeddings.npy"
FAISS_INDEX_FILE = DATA_PROCESSED / "vector_index.faiss"
ID2META_FILE = DATA_PROCESSED / "id2meta.json"

# -----------------------------
# CHUNK SETTINGS
# -----------------------------
TEXT_CHUNK_SIZE = 250           # fine-grained chunks improve retrieval
TEXT_CHUNK_OVERLAP = 40

TABLE_AS_TEXT = True            # Convert extracted tables into textual rows

OCR_IMAGE_DPI = 300             # For better OCR quality

# -----------------------------
# EMBEDDING MODELS
# -----------------------------
# Text embedding model — strong, fast, multilingual, perfect for assignments
TEXT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Image embedding model — CLIP ViT ensures multi-modal compatibility
IMAGE_EMBED_MODEL = "openai/clip-vit-base-patch32"

# Embedding vector size (must match model output)
TEXT_EMBED_DIM = 768
IMAGE_EMBED_DIM = 512

# Whether to normalize embeddings
NORMALIZE_EMBEDDINGS = True

# -----------------------------
# VECTOR DATABASE SETTINGS
# -----------------------------
FAISS_METRIC = "L2"             # cosine or L2 (cosine works via normalized vectors)
TOP_K_RETRIEVAL = 5             # default number of chunks to retrieve

# -----------------------------
# LLM GENERATION SETTINGS
# -----------------------------
USE_OPENAI = False              # If True → use OpenAI API, else local model
OPENAI_MODEL = "gpt-3.5-turbo"  # For remote generation

# Local fallback model (for assignment)
LOCAL_GEN_MODEL = "sentence-transformers/all-mpnet-base-v2"

MAX_ANSWER_TOKENS = 512

# -----------------------------
# LOGGING SETTINGS
# -----------------------------
LOG_FILE = ROOT / "logs.txt"
LOG_LEVEL = "INFO"

# -----------------------------
# APP SETTINGS
# -----------------------------
APP_TITLE = "Big AIR Lab | Multi-Modal RAG QA System"
APP_DESCRIPTION = (
    "Assignment Demo — Multi-modal Retrieval-Augmented Generation for IMF Qatar Document."
)
