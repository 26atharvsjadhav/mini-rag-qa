"""
search.py
- Loads chunks from db.sqlite
- Builds embeddings with sentence-transformers all-MiniLM-L6-v2 (CPU)
- Keeps FAISS Index in memory (fast, simple)
- Exposes baseline_search(query, k)
"""
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import pickle
import os

DB_PATH = "db.sqlite"
SEED = 42
np.random.seed(SEED)

# model (CPU)
model = SentenceTransformer("all-MiniLM-L6-v2")

# load chunks
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT rowid, doc_id, chunk_id, text, title, url FROM chunks")
rows = cur.fetchall()
conn.close()

if not rows:
    raise RuntimeError("No chunks found in DB. Run ingest.py first.")

rowids = [r[0] for r in rows]
doc_ids = [r[1] for r in rows]
chunk_ids = [r[2] for r in rows]
texts = [r[3] for r in rows]
titles = [r[4] for r in rows]
urls = [r[5] for r in rows]

# Compute embeddings once and keep in memory
EMB_PATH = Path("embeddings.npy")
IDS_PATH = Path("rowids.pkl")

if EMB_PATH.exists() and IDS_PATH.exists():
    embeddings = np.load(str(EMB_PATH))
    with open(IDS_PATH, "rb") as f:
        rowids = pickle.load(f)
    if embeddings.shape[0] != len(texts):
        print("[info] embeddings shape mismatch -> recomputing")
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        np.save(str(EMB_PATH), embeddings)
        with open(IDS_PATH, "wb") as f:
            pickle.dump(rowids, f)
else:
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    np.save(str(EMB_PATH), embeddings)
    with open(IDS_PATH, "wb") as f:
        pickle.dump(rowids, f)

# normalize for cosine via inner product
faiss.normalize_L2(embeddings)
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

def baseline_search(query, k=5):
    q_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "rowid": int(rowids[idx]),
            "doc_id": doc_ids[idx],
            "chunk_id": int(chunk_ids[idx]),
            "title": titles[idx],
            "url": urls[idx],
            "score": float(score),
            "text": texts[idx]
        })
    return results
