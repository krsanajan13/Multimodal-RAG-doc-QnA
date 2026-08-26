"""
utils/pdf_utils.py

Small helpers around PDF path handling and simple checks.
"""
from pathlib import Path
import fitz
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_utils")


def validate_pdf(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        logger.error("PDF not found: %s", path)
        return False
    try:
        doc = fitz.open(str(p))
        n = len(doc)
        doc.close()
        logger.info("PDF OK: %s (pages=%d)", p, n)
        return True
    except Exception as e:
        logger.exception("Invalid PDF or cannot open: %s", e)
        return False
