"""
ingestion/extract_tables.py

Attempts to extract tables from a PDF. Preferred method: pdfplumber if installed.
Fallback: heuristics that detect lines with repeated separators.

Output: list of table dicts:
[ { 'page': int, 'table_id': str, 'csv': str, 'rows': List[List[str]] }, ... ]
"""
from pathlib import Path
from typing import List, Dict
import logging
import io
import csv
import json

logger = logging.getLogger("extract_tables")
logging.basicConfig(level=logging.INFO)

# try to import pdfplumber (recommended)
try:
    import pdfplumber

    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False


def _tables_with_pdfplumber(pdf_path: str) -> List[Dict]:
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                page_tables = page.extract_tables()
                for tidx, tab in enumerate(page_tables):
                    # tab is list of rows (each row a list of cells)
                    # convert to CSV string
                    buf = io.StringIO()
                    writer = csv.writer(buf)
                    for row in tab:
                        # normalize None to empty string
                        writer.writerow([("" if c is None else str(c)).strip() for c in row])
                    csv_text = buf.getvalue()
                    tables.append({
                        "page": i,
                        "table_id": f"p{i}_t{tidx+1}",
                        "csv": csv_text,
                        "rows": tab
                    })
            except Exception as e:
                logger.exception("pdfplumber table extraction failed on page %s: %s", i, e)
    return tables


def _heuristic_table_extraction(pdf_path: str) -> List[Dict]:
    """
    Very simple fallback: extract text and find lines with many separators (| or multiple spaces)
    Groups consecutive such lines as a table.
    """
    import fitz
    doc = fitz.open(pdf_path)
    tables = []
    for i, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text")
            lines = text.splitlines()
            candidate = []
            table_count = 0
            for ln in lines:
                # heuristics: pipe-separated or long sequences of multiple spaces indicating columns
                if ("|" in ln and ln.count("|") >= 2) or (len(ln.split()) >= 4 and "  " in ln):
                    candidate.append(ln)
                else:
                    if candidate:
                        # convert to rows by splitting on pipes or multiple spaces
                        rows = []
                        for r in candidate:
                            if "|" in r:
                                cells = [c.strip() for c in r.split("|")]
                            else:
                                # split on 2+ spaces
                                import re
                                cells = [c.strip() for c in re.split(r"\s{2,}", r)]
                            rows.append(cells)
                        buf = io.StringIO()
                        writer = csv.writer(buf)
                        for row in rows:
                            writer.writerow(row)
                        table_count += 1
                        tables.append({
                            "page": i,
                            "table_id": f"p{i}_h{table_count}",
                            "csv": buf.getvalue(),
                            "rows": rows
                        })
                        candidate = []
            # flush end
            if candidate:
                rows = []
                for r in candidate:
                    if "|" in r:
                        cells = [c.strip() for c in r.split("|")]
                    else:
                        import re
                        cells = [c.strip() for c in re.split(r"\s{2,}", r)]
                    rows.append(cells)
                buf = io.StringIO()
                writer = csv.writer(buf)
                for row in rows:
                    writer.writerow(row)
                table_count += 1
                tables.append({
                    "page": i,
                    "table_id": f"p{i}_h{table_count}",
                    "csv": buf.getvalue(),
                    "rows": rows
                })
        except Exception as e:
            logger.exception("Heuristic table extraction failed on page %s: %s", i, e)
    doc.close()
    return tables


def extract_tables(pdf_path: str) -> List[Dict]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if _HAS_PDFPLUMBER:
        logger.info("Using pdfplumber for table extraction")
        tables = _tables_with_pdfplumber(str(pdf_path))
    else:
        logger.warning("pdfplumber not available — using heuristic fallback")
        tables = _heuristic_table_extraction(str(pdf_path))
    logger.info("Extracted %d tables", len(tables))
    return tables


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python extract_tables.py path/to/doc.pdf")
        sys.exit(1)
    t = extract_tables(sys.argv[1])
    print(json.dumps(t[:3], indent=2))
