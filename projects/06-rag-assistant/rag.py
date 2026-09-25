"""Local retrieval demo: sentence-transformer embeddings + FAISS.

This module demonstrates retrieval only. Generation is intentionally separated so
retrieved evidence can be inspected before an LLM is introduced.
"""
import argparse
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def chunks(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size) if text[i:i+size].strip()]

p = argparse.ArgumentParser()
p.add_argument("document")
p.add_argument("query")
p.add_argument("--top-k", type=int, default=3)
args = p.parse_args()
parts = chunks(Path(args.document).read_text(encoding="utf-8"))
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = encoder.encode(parts, normalize_embeddings=True).astype("float32")
index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
q = encoder.encode([args.query], normalize_embeddings=True).astype("float32")
scores, ids = index.search(q, min(args.top_k, len(parts)))
for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
    print(f"\n#{rank} score={score:.4f}\n{parts[idx]}")
