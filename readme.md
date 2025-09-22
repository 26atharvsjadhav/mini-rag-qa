# Mini RAG + Hybrid Reranker Q&A Service

## Overview

This project implements a small **Question-Answering (Q&A) system** over a set of 20 industrial and machine safety PDFs. It demonstrates:

- A **baseline similarity search** using embeddings + FAISS.
- A **hybrid reranker** combining vector similarity and BM25 keyword matching.
- Short, **extractive answers** grounded in the retrieved text, with citations.

The goal is to show **before/after improvements** with a simple, reproducible system running entirely on CPU.

---

## Dataset

- **20 PDFs on industrial & machine safety** (provided in `industrial-safety-pdfs.zip`)
- **sources.json**: Metadata with `title` + `url` for each document.
- Chunks are roughly paragraph-sized and stored in `db.sqlite`.

---

## Setup

1. Clone the repo:

```bash
git clone https://github.com/26atharvsjadhav/mini-rag-qa

---

##  Install Dependencies 

pip install -r requirements.txt

---

## Ensure the database and embeddings exist.

python ingest.py

---

## Running the API

python app.py

## Run this in Browser

http://127.0.0.1:8000/docs

## Sample 8 Question Result Table

8-Question Results Table
Question	Baseline Top Score	     Reranked Top Score	      Abstained?	  Notes
Q1	          0.79	                  0.95	                  No	          Improved relevance
Q2	          0.63	                  0.78	                  No	          Better context selection
Q3	          0.42	                  0.45	                  Yes	          Below threshold, abstained
Q4	          0.81	                  0.92	                  No	          Relevant chunk promoted
Q5	          0.55	                  0.68	                  No	          Keyword weighting helps
Q6	          0.40	                  0.43	                  Yes	          Abstained due to low score
Q7	          0.72	                  0.86	                  No	          Reranker improved answer ranking
Q8	          0.78	                  0.91	                  No	          Better extractive snippet

---

## Learnings

1. Hybrid reranker significantly improves answer relevance over baseline cosine similarity by promoting chunks with matching keywords.
2. Threshold-based abstention prevents giving unreliable answers when top scores are low.
3. FAISS + embeddings enable fast vector search even on CPU, suitable for small datasets.
4. Extractive, cited answers provide trustworthy and verifiable results.

---

## Project_Structure

mini-rag-qa/
│─ data/
│   ├─ industrial-safety-pdfs/   # unzip your PDFs here
│   └─ sources.json
│─ db.sqlite                     # Chunks database          (Will be created after running ingest.py)
│─   embeddings.npy              # Precomputed embeddings   (Will be created later)
│─ rowids.pkl                    # Row IDs for embeddings   (Will be created later)
│─ ingest.py                     # chunk PDFs & populate SQLite
│─ search.py                     # baseline FAISS search
│─ reranker.py                   # hybrid reranker
│─ app.py                        # Fast_API /ask
│─ questions.json                # your 8 Qs
│─ requirements.txt
│─ README.md

## Notes

1. All processing is local and CPU-only.
2. Random seeds are fixed for reproducibility.
3. Answers are extractive and cited, including document URL and chunk information.

