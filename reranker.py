"""
reranker.py
- Hybrid reranker: combine vector-based score (from FAISS) with BM25 keyword score
- Exposes hybrid_rerank(query, top_k_results)
"""
from rank_bm25 import BM25Okapi
from search import texts, baseline_search
import numpy as np

# build BM25 on all texts (simple tokenization by whitespace)
tokenized = [t.split() for t in texts]
bm25 = BM25Okapi(tokenized)

def normalize(x, xmin, xmax):
    if xmax == xmin:
        return 0.0
    return (x - xmin) / (xmax - xmin)

# hybrid_rerank.py (fixed)
def hybrid_rerank(query, top_k_results, alpha=0.7):
    q_tokens = query.split()
    bm25_scores = []

    vec_scores = [r["score"] for r in top_k_results]

    for r in top_k_results:
        # Compute BM25 score for this single chunk
        doc_tokens = r["text"].split()
        score = bm25.get_scores(q_tokens)[texts.index(r["text"])]  # get_scores returns scores for all docs
        bm25_scores.append(score)

    # normalize vector and BM25 scores
    vs_min, vs_max = min(vec_scores), max(vec_scores)
    bm_min, bm_max = min(bm25_scores), max(bm25_scores)

    reranked = []
    for r, v, b in zip(top_k_results, vec_scores, bm25_scores):
        nv = (v - vs_min) / (vs_max - vs_min) if vs_max != vs_min else 0.0
        nb = (b - bm_min) / (bm_max - bm_min) if bm_max != bm_min else 0.0
        combined = alpha * nv + (1 - alpha) * nb
        rr = r.copy()
        rr["bm25_score"] = float(b)
        rr["nv"] = float(nv)
        rr["nb"] = float(nb)
        rr["reranked_score"] = float(combined)
        reranked.append(rr)

    reranked.sort(key=lambda x: x["reranked_score"], reverse=True)
    return reranked
