"""
answer_generator.py (Improved Version)
Accurate answer generation using FLAN-T5-LARGE with a strict grounding prompt.
This version solves issues where the answer is too short or irrelevant.
"""

import config
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------------------------------
# Helper: merge chunks into context
# ----------------------------------------------------
def format_context(chunks: List[Dict]) -> str:
    merged = []
    for c in chunks:
        merged.append(
            f"[ID: {c['chunk_id']} | Page {c['page']} | Type {c['type']}]\n{c['text']}\n"
        )
    return "\n".join(merged)


# ----------------------------------------------------
# Improved Local Generator Using FLAN-T5-Large
# ----------------------------------------------------
class LocalGenerator:
    def __init__(self):
        self.model_name = "google/flan-t5-large"  # much stronger model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def generate(self, query: str, chunks: List[Dict]) -> str:
        context = format_context(chunks)

        prompt = (
            "You are a precise economic analyst. Your job is to answer the user’s "
            "question STRICTLY using the retrieved context below.\n\n"
            "RULES:\n"
            "- DO NOT hallucinate.\n"
            "- Use facts ONLY from the context.\n"
            "- Provide a clear 4–6 line answer.\n"
            "- Include macroeconomic concepts if mentioned.\n"
            "- Absolutely avoid repeating table titles.\n"
            "- Focus on the main economic themes.\n\n"
            f"QUESTION:\n{query}\n\n"
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            "FINAL ANSWER:"
        )

        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(
            encoded["input_ids"],
            max_new_tokens=220,
            num_beams=5,
            temperature=0.0,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


_local_gen = None


def get_local_generator():
    global _local_gen
    if _local_gen is None:
        _local_gen = LocalGenerator()
    return _local_gen


# ----------------------------------------------------
# Main interface
# ----------------------------------------------------
def generate_answer(query: str, chunks: List[Dict], prefer_openai=False):
    if not chunks:
        return {
            "answer": "No relevant information found.",
            "sources": [],
            "generated_by": "none"
        }

    generator = get_local_generator()
    answer = generator.generate(query, chunks)

    return {
        "answer": answer,
        "sources": [c["chunk_id"] for c in chunks],
        "generated_by": "local-flan-large"
    }
