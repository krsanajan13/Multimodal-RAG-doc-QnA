"""
ingestion/chunker.py

Smart chunker that:
 - accepts page-wise text, table chunks, and OCR image text
 - performs structural chunking (split by headings/sections) and semantic chunking (fixed char size with overlap)
 - outputs a unified list of chunks ready for embedding/storage

Outputs JSON files under data/processed by default (text_chunks.json, table_chunks.json, image_chunks.json)
"""
from pathlib import Path
from typing import List, Dict, Optional
import json
import re
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chunker")

# default chunk size and overlap; keep these small/efficient for RAG
DEFAULT_CHUNK_SIZE = 900  # characters
DEFAULT_OVERLAP = 200


def is_heading(line: str) -> bool:
    """
    Heuristic: consider a line a heading if it's short and either ALL CAPS or ends with ':' or starts with a digit and dot.
    """
    if not line:
        return False
    s = line.strip()
    if len(s) < 120 and (s.isupper() and len(s.split()) <= 8):
        return True
    if s.endswith(":") and len(s.split()) <= 12:
        return True
    if re.match(r"^\d+(\.\d+)*\s+", s):
        return True
    return False


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split by double newline into paragraphs.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return paras


def structural_chunk_text(page_texts: List[Dict]) -> List[Dict]:
    """
    Use headings and paragraphs to create structural chunks.
    page_texts: [{'page': int, 'text': str, 'page_label': str}, ...]
    Returns list of chunks: [{'id', 'type','page','text','meta'}]
    """
    chunks = []
    for p in page_texts:
        page_no = p.get("page")
        txt = p.get("text", "")
        if not txt:
            continue
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        current_block = []
        current_title = None
        for ln in lines:
            if is_heading(ln):
                # flush previous
                if current_block:
                    chunks.append({
                        "id": f"p{page_no}_block{len(chunks)+1}",
                        "type": "text",
                        "page": page_no,
                        "title": current_title,
                        "text": "\n".join(current_block),
                        "meta": {}
                    })
                    current_block = []
                current_title = ln
            else:
                current_block.append(ln)
        if current_block:
            chunks.append({
                "id": f"p{page_no}_block{len(chunks)+1}",
                "type": "text",
                "page": page_no,
                "title": current_title,
                "text": "\n".join(current_block),
                "meta": {}
            })
    return chunks


def semantic_split(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks of approx chunk_size characters.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start += step
    return chunks


def create_final_chunks(text_chunks: List[Dict],
                        table_chunks: Optional[List[Dict]] = None,
                        image_ocr_chunks: Optional[List[Dict]] = None,
                        chunk_size: int = DEFAULT_CHUNK_SIZE,
                        overlap: int = DEFAULT_OVERLAP) -> List[Dict]:
    """
    Combine structural chunks and further split large blocks semantically.
    Returns unified list of chunk dicts with metadata.
    """
    final = []
    # process text structural blocks
    for t in text_chunks:
        text = t.get("text","").strip()
        if not text:
            continue
        sems = semantic_split(text, chunk_size=chunk_size, overlap=overlap)
        for idx, s in enumerate(sems, start=1):
            final.append({
                "id": f"{t['id']}_s{idx}",
                "source_id": t['id'],
                "type": "text",
                "page": t.get("page"),
                "title": t.get("title"),
                "text": s,
                "meta": t.get("meta", {})
            })

    # process tables (each table as one or multiple chunks)
    if table_chunks:
        for table in table_chunks:
            table_text = table.get("csv","")
            # small tables -> single chunk; big tables split by rows
            rows = table.get("rows") or []
            raw = table_text
            if len(raw) <= chunk_size:
                final.append({
                    "id": f"{table['table_id']}_s1",
                    "source_id": table['table_id'],
                    "type": "table",
                    "page": table.get("page"),
                    "title": table.get("table_id"),
                    "text": raw,
                    "meta": {"rows": rows}
                })
            else:
                parts = semantic_split(raw, chunk_size=chunk_size, overlap=overlap)
                for idx, p in enumerate(parts, start=1):
                    final.append({
                        "id": f"{table['table_id']}_s{idx}",
                        "source_id": table['table_id'],
                        "type": "table",
                        "page": table.get("page"),
                        "title": table.get("table_id"),
                        "text": p,
                        "meta": {"rows_sample": rows[:20]}
                    })

    # process OCR image texts
    if image_ocr_chunks:
        for img in image_ocr_chunks:
            txt = img.get("text","").strip()
            if not txt:
                continue
            parts = semantic_split(txt, chunk_size=chunk_size, overlap=overlap)
            for idx,p in enumerate(parts, start=1):
                final.append({
                    "id": f"img_{Path(img.get('image')).stem}_s{idx}",
                    "source_id": Path(img.get("image")).name,
                    "type": "image_ocr",
                    "page": img.get("page") if img.get("page") else None,
                    "title": Path(img.get("image")).name,
                    "text": p,
                    "meta": {"image_path": img.get("image")}
                })
    return final


def save_json(obj, out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Example flow: expects processed inputs at data/raw and will write to data/processed
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--images_dir", required=False, help="Directory where extract_images saved images")
    parser.add_argument("--out_dir", default="data/processed", help="Directory to write processed json")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()

    # Minimal pipeline invocation to demonstrate chunking:
    from ingestion.extract_text import extract_text_by_page
    from ingestion.extract_tables import extract_tables
    from ingestion.ocr_images import ocr_images
    from ingestion.extract_images import extract_images

    pdf_path = args.pdf
    out_dir = args.out_dir

    # 1. extract text
    pages = extract_text_by_page(pdf_path)
    text_blocks = structural_chunk_text(pages)
    save_json(text_blocks, f"{out_dir}/text_blocks.json")
    logger.info("Saved text_blocks.json (%d blocks)", len(text_blocks))

    # 2. extract tables
    tables = extract_tables(pdf_path)
    save_json(tables, f"{out_dir}/table_chunks_raw.json")
    logger.info("Saved table_chunks_raw.json (%d tables)", len(tables))

    # 3. extract images
    images_dir = args.images_dir or "data/raw/images"
    images_meta = extract_images(pdf_path, images_dir)
    save_json(images_meta, f"{out_dir}/images_meta.json")
    logger.info("Saved images_meta.json (%d images)", len(images_meta))

    # 4. OCR images (if any)
    img_paths = [m["path"] for m in images_meta]
    ocr_results = ocr_images(img_paths) if img_paths else []
    save_json(ocr_results, f"{out_dir}/image_ocr.json")
    logger.info("Saved image_ocr.json (%d results)", len(ocr_results))

    # 5. final chunk creation
    final_chunks = create_final_chunks(text_blocks, table_chunks=tables, image_ocr_chunks=ocr_results,
                                       chunk_size=args.chunk_size, overlap=args.overlap)
    save_json(final_chunks, f"{out_dir}/final_chunks.json")
    logger.info("Saved final_chunks.json (%d chunks)", len(final_chunks))
