"""
ingest.py
- Reads data/sources.json
- Downloads PDFs (if missing) into data/industrial-safety-pdfs/
- Extracts text, splits into paragraph-size chunks
- Stores chunks (doc_id, chunk_id, text, title, url) into db.sqlite
"""
import os
import json
import sqlite3
from pathlib import Path
import requests
from tqdm import tqdm
from PyPDF2 import PdfReader

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "industrial-safety-pdfs"
SOURCES = DATA_DIR / "sources.json"
DB_PATH = Path("db.sqlite")
MIN_CHUNK_LEN = 50  # characters
PDF_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, dest: Path):
    """Download with streaming if missing."""
    if dest.exists():
        return
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=f"dl {dest.name}", total=total, unit="B", unit_scale=True
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            f.write(chunk)
            pbar.update(len(chunk))

def sanitize_filename(title):
    bad = r'<>:"/\|?*'
    s = "".join("_" if c in bad else c for c in title).strip()
    s = "_".join(s.split())
    return s[:180]  # keep short

def extract_text_from_pdf(path: Path):
    reader = PdfReader(str(path))
    pages_text = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        pages_text.append(t)
    return "\n\n".join(pages_text)

def chunk_text(text):
    # very simple paragraph-splitting by blank lines
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= MIN_CHUNK_LEN]
    return paras

def main():
    with open(SOURCES, "r", encoding="utf-8") as f:
        sources = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT,
        chunk_id INTEGER,
        text TEXT,
        title TEXT,
        url TEXT
    )
    """)
    conn.commit()

    for src in sources:
        title = src.get("title", "no-title")
        url = src.get("url")
        file_hint = src.get("file")  # optional
        fname = sanitize_filename(title) + ".pdf"
        pdf_path = PDF_DIR / fname

        # If a `file` key was given in sources.json (some flows have it), prefer that name
        if file_hint:
            alt = PDF_DIR / file_hint
            if alt.exists():
                pdf_path = alt

        if url:
            try:
                download_file(url, pdf_path)
            except Exception as e:
                print(f"[warn] could not download {url} -> {e}. Skipping download, continue if file exists locally.")
                if not pdf_path.exists():
                    print(f"[error] file {pdf_path} missing, skipping source: {title}")
                    continue

        # Extract & chunk
        try:
            full_text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"[error] failed to extract text from {pdf_path}: {e}")
            continue

        paras = chunk_text(full_text)
        if not paras:
            print(f"[warn] no chunks found for {pdf_path}")
            continue

        for i, para in enumerate(paras):
            cur.execute(
                "INSERT INTO chunks (doc_id, chunk_id, text, title, url) VALUES (?, ?, ?, ?, ?)",
                (pdf_path.name, i, para, title, url)
            )
        conn.commit()
        print(f"Ingsted {len(paras)} chunks from {title}")

    conn.close()
    print("Ingest completed. DB:", DB_PATH)

if __name__ == "__main__":
    main()
