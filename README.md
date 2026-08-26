# Multi-Modal RAG Q&A System (IMF Qatar Document)

An end-to-end Multi-Modal Retrieval-Augmented Generation (RAG) system built specifically to process complex multi-modal documents (PDFs containing text, tables, charts, and figures, such as the IMF Qatar Report). 

The system extracts text, tables, and images, performs OCR, creates multi-modal embeddings using Sentence-Transformers and CLIP, indexes them in FAISS, and provides an interactive web interface powered by Flask and a custom frontend.

---

## 🏗️ Architecture & Pipeline

```
                       ┌─────────────────────────┐
                       │  qatar_test_doc.pdf     │
                       └────────────┬────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    [Text Extraction]      [Table Extraction]      [Image Extraction]
      (PyMuPDF fitz)          (pdfplumber)          (PyMuPDF + Tesseract OCR)
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
                         [Multi-Modal Chunker]
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
  [Text/Table Embeddings]                          [Image Embeddings]
(all-mpnet-base-v2, 768d)                     (clip-vit-base-patch32, 512d)
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    ▼
                          [FAISS Vector Index]
                                    │
                                    ▼
                    [Flask Web Application (app.py)]
                                    │
                                    ▼
                       [Interactive Web UI /ask]
```

---

## 🛠️ Project Structure

```
multi_modal_rag_qna/
├── app.py                      # Flask web server (serves frontend & /ask endpoint on port 3000)
├── config.py                   # Configuration for paths, embedding models, chunk size, FAISS metric
├── requirements.txt            # Python dependencies (PyMuPDF, pdfplumber, pytesseract, FAISS, etc.)
├── .gitignore                  # Excludes caches, processed vector files, logs, and venvs
├── ingestion/                  # Document parsing module
│   ├── extract_text.py         # Extracts raw text using PyMuPDF (fitz)
│   ├── extract_tables.py       # Extracts tables using pdfplumber into text rows
│   ├── extract_images.py       # Extracts embedded figures and images from PDF pages
│   ├── ocr_images.py           # Performs OCR on images via PyTesseract
│   └── chunker.py              # Combines text, tables, and OCR output into structured JSON chunks
├── embeddings/                 # Embeddings & Vector Store builder
│   ├── embed_text.py           # Encodes text & tables using sentence-transformers/all-mpnet-base-v2
│   ├── embed_image.py          # Encodes images using openai/clip-vit-base-patch32
│   └── build_index.py          # Builds FAISS vector_index.faiss and id2meta.json
├── retrieval/                  # Search & Context Retrieval
│   ├── retriever.py            # Similarity search against FAISS index
│   ├── reranker.py             # Context re-ranking logic
│   └── evaluator.py            # Evaluates retrieval accuracy & precision
├── generation/                 # Answer Generation
│   └── answer_generator.py     # Generates responses from retrieved context
├── utils/                      # Common utilities
│   ├── pdf_utils.py            # PDF loading & validation utilities
│   ├── file_utils.py           # JSON and directory helpers
│   └── logger.py               # Application logging setup
├── frontend/                   # Web Interface
│   ├── index.html              # Custom Q&A search interface
│   └── style.css               # Frontend styling
└── data/                       # Data storage
    ├── raw/                    # Raw input documents (e.g., qatar_test_doc.pdf) & extracted images
    └── processed/              # Processed chunks, embeddings.npy, vector_index.faiss, id2meta.json
```

---

## ⚡ How to Run

### 1. Install Dependencies

Ensure Python 3.9+ and Tesseract OCR are installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Build Embeddings & FAISS Index

To process the document (`data/raw/qatar_test_doc.pdf`), extract multi-modal content, and build the FAISS index:

```bash
python -m embeddings.build_index
```

This generates `data/processed/vector_index.faiss` and `data/processed/id2meta.json`.

### 3. Launch the Web Application

Run the Flask server:

```bash
python app.py
```

Open your browser and navigate to **`http://127.0.0.1:3000`** to start asking questions!

---

## ⚙️ Configuration (`config.py`)

Key settings used across the pipeline:

- **Text Embeddings**: `sentence-transformers/all-mpnet-base-v2` (768 dimensions)
- **Image Embeddings**: `openai/clip-vit-base-patch32` (512 dimensions)
- **Chunk Size**: 250 words (overlap: 40 words)
- **Vector Metric**: FAISS L2 metric (normalized cosine)
- **Default Top-K Retrieval**: 6 relevant chunks (combining text, table, and image modalities)
