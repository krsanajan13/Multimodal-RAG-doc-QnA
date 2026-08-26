"""
retrieval/evaluator.py

Simple evaluation harness. Provide a CSV with columns: query, expected_page (optional), expected_snippet (optional)
Computes recall@k (if expected_chunk_id or expected_snippet present) and prints results.
"""
import csv
from typing import List, Dict
from retrieval.retriever import Retriever
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")


def load_bench(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def recall_at_k(retriever: Retriever, query: str, expected_snippet: str = None, k: int = 5) -> int:
    results = retriever.query(query, top_k=k)
    if expected_snippet:
        expected_snippet = expected_snippet.strip().lower()
        for r in results:
            txt = (r.get("text") or "").strip().lower()
            if expected_snippet in txt:
                return 1
        return 0
    # fallback: if no expected snippet, return 0
    return 0


def evaluate(csv_path: str, top_k: int = 5):
    bench = load_bench(csv_path)
    r = Retriever()
    total = len(bench)
    hits = 0
    for row in bench:
        q = row.get("query")
        expected = row.get("expected_snippet") or row.get("expected_text")
        h = recall_at_k(r, q, expected_snippet=expected, k=top_k)
        hits += h
        logger.info("Q: %s -> hit=%s", q[:80], h)
    logger.info("Recall@%d = %.3f (%d/%d)", top_k, hits / max(total, 1), hits, total)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True, help="CSV with columns: query, expected_snippet")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.bench, top_k=args.k)
