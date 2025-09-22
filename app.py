# app.py - FastAPI version of Mini-RAG Q&A
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from search import baseline_search
from reranker import hybrid_rerank
import textwrap

app = FastAPI(title="Mini-RAG Q&A Service")

THRESHOLD = 0.45  # abstain threshold

# ----------------------------
# Request model
# ----------------------------
class AskRequest(BaseModel):
    q: str
    k: int = 5
    mode: str = "baseline"  # 'baseline' or 'rerank'

# ----------------------------
# Helper
# ----------------------------
def make_excerpt(text: str, max_len: int = 400) -> str:
    return textwrap.shorten(text.replace("\n", " "), width=max_len, placeholder="...")

# ----------------------------
# API endpoint
# ----------------------------
@app.post("/ask")
async def ask(req: AskRequest):
    if not req.q.strip():
        raise HTTPException(status_code=400, detail="Missing 'q' in request")

    baseline = baseline_search(req.q, req.k)

    if req.mode == "rerank":
        contexts = hybrid_rerank(req.q, baseline)
        reranker_used = "hybrid_bm25"
    else:
        contexts = baseline
        for c in contexts:
            c["reranked_score"] = c.get("score")
        reranker_used = None

    top = contexts[0]
    top_score = top.get("reranked_score", top.get("score", 0.0))

    if top_score < THRESHOLD:
        answer = None
        reason = f"Top score {top_score:.3f} below threshold {THRESHOLD}"
    else:
        answer = make_excerpt(top["text"], max_len=500)
        reason = None

    # Build output contexts with excerpts
    out_contexts = []
    for c in contexts:
        out_contexts.append({
            "title": c.get("title"),
            "url": c.get("url"),
            "doc_id": c.get("doc_id"),
            "chunk_id": c.get("chunk_id"),
            "score": c.get("score"),
            "reranked_score": c.get("reranked_score"),
            "bm25_score": c.get("bm25_score", None),
            "text_excerpt": make_excerpt(c.get("text", ""), 400)
        })

    return {
        "answer": answer,
        "abstained": answer is None,
        "abstain_reason": reason,
        "contexts": out_contexts,
        "reranker_used": reranker_used
    }

# ----------------------------
# Run (for direct python execution)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
