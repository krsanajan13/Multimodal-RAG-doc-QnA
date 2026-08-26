"""
ingestion/extract_text.py

Extracts textual content from a PDF using PyMuPDF (fitz).
Provides a clean page-wise text output with basic cleaning and simple metadata.
"""
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_text")


def clean_text(text: str) -> str:
    """Basic cleaning: normalize whitespace and remove repeated newlines."""
    text = text.replace("\r", "\n")
    # collapse multiple newlines
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # strip trailing/leading whitespace on each line
    text = "\n".join([line.strip() for line in text.splitlines()])
    # collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text_by_page(pdf_path: str) -> List[Dict]:
    """
    Extract text from each page of the PDF.

    Returns:
        List of dicts: [{'page': int, 'text': str, 'page_label': str}, ...]
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, start=1):
        try:
            raw = page.get_text("text")  # simple plain text
            text = clean_text(raw)
            pages.append({"page": i, "text": text, "page_label": str(i)})
        except Exception as e:
            logger.exception("Failed to extract text from page %s: %s", i, e)
            pages.append({"page": i, "text": "", "page_label": str(i)})
    doc.close()
    return pages


if __name__ == "__main__":
    # quick local test (won't run in your environment automatically)
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py path/to/doc.pdf")
        sys.exit(1)
    out = extract_text_by_page(sys.argv[1])
    print(f"Extracted {len(out)} pages")
