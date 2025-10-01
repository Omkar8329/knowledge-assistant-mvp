import os
import io
import re
import json
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import streamlit as st
from bs4 import BeautifulSoup
import pdfplumber
from docx import Document as DocxDocument

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# ---------- Config ----------
APP_TITLE = "Knowledge Assistant (MVP)"
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(exist_ok=True, parents=True)

CHUNK_WORDS = 800
CHUNK_OVERLAP = 150
TOP_K_CHUNKS = 6
TOP_SENTENCES = 4
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9“"(])')

# FIRST Streamlit call
st.set_page_config(page_title=APP_TITLE, layout="wide")


# ---------- Auth ----------
def check_auth():
    required = os.getenv("APP_PASSWORD")
    if not required:
        return True
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if st.session_state.authed:
        return True
    st.title(APP_TITLE)
    st.info("Enter password to continue.")
    pwd = st.text_input("Password", type="password")
    if st.button("Let me in"):
        if pwd == required:
            st.session_state.authed = True
            return True
        else:
            st.error("Wrong password.")
    st.stop()


# ---------- Readers/Utils ----------
def read_pdf(file_bytes: bytes) -> List[Tuple[str, int]]:
    out = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            out.append((text, i))
    return out

def read_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")

def clean_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = BeautifulSoup(s, "lxml").get_text()
    lines = [ln.strip() for ln in s.splitlines()]
    return "\n".join([ln for ln in lines if ln])

def chunk_text(text: str, size_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    if not text: return []
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        end = min(len(words), start + size_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words): break
        start = max(0, end - overlap)
    return chunks

def split_sentences(text: str):
    sentences = []
    for para in text.split("\n"):
        para = para.strip()
        if not para: continue
        parts = SENTENCE_SPLIT_RE.split(para)
        for s in parts:
            s = s.strip()
            if s: sentences.append(s)
    return sentences


# ---------- TF-IDF Store ----------
class Store:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.dir = DATA_DIR / workspace
        self.dir.mkdir(exist_ok=True, parents=True)
        self.meta_path = self.dir / "meta.json"
        self.meta = {"docs": [], "chunks": []}
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self._load()

    def _load(self):
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.meta["chunks"]:
            self._rebuild_index()

    def _save(self):
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self):
        for p in self.dir.glob("*"):
            try: p.unlink()
            except: pass
        self.meta = {"docs": [], "chunks": []}
        self.vectorizer = None
        self.matrix = None

    def _rebuild_index(self):
        texts = [c["text"] for c in self.meta["chunks"]]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1,2),
            max_features=100_000, norm="l2"
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def add(self, filename: str, file_bytes: bytes) -> int:
        ext = filename.lower().split(".")[-1]
        entries = []
        if ext == "pdf":
            for page_text, page_no in read_pdf(file_bytes):
                text = clean_text(page_text)
                for ch in chunk_text(text):
                    entries.append({"id": str(uuid.uuid4()), "filename": filename, "page": page_no, "text": ch})
        elif ext == "docx":
            text = clean_text(read_docx(file_bytes))
            for ch in chunk_text(text):
                entries.append({"id": str(uuid.uuid4()), "filename": filename, "page": None, "text": ch})
        elif ext == "txt":
            text = clean_text(read_txt(file_bytes))
            for ch in chunk_text(text):
                entries.append({"id": str(uuid.uuid4()), "filename": filename, "page": None, "text": ch})
        else:
            raise ValueError(f"Unsupported file type: .{ext}")
        if not entries: return 0
        self.meta["chunks"].extend(entries)
        self.meta["docs"].append({"filename": filename, "size": len(file_bytes)})
        self._save()
        self._rebuild_index()
        return len(entries)

    def search(self, query: str, top_k: int = TOP_K_CHUNKS):
        if not self.vectorizer or self.matrix is None or not self.meta["chunks"]:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        top_idx = np.argsort(-sims)[:top_k]
        return [{"score": float(sims[i]), "chunk": self.meta["chunks"][i]} for i in top_idx]


# ---------- Answer composer ----------
def answer_with_sentences(question: str, retrieved: List[Dict], top_sentences: int = TOP_SENTENCES) -> Dict:
    if not retrieved:
        return {"answer": "No documents indexed yet or no relevant passages found.", "citations": []}
    sentences = []
    for r in retrieved:
        ch = r["chunk"]
        for s in split_sentences(ch["text"]):
            if len(s) >= 5:
                sentences.append({"sentence": s, "filename": ch["filename"], "page": ch.get("page")})
    if not sentences:
        return {"answer": "I couldn't extract meaningful sentences from the retrieved passages.", "citations": []}
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), norm="l2")
    sent_matrix = vec.fit_transform([s["sentence"] for s in sentences])
    q_vec = vec.transform([question])
    sims = cosine_similarity(q_vec, sent_matrix)[0]
    order = np.argsort(-sims)
    picks, seen = [], set()
    for idx in order:
        s = sentences[idx]
        key = s["sentence"].lower()[:200]
        if key in seen:
            continue
        seen.add(key)
        picks.append((float(sims[idx]), s))
        if len(picks) >= top_sentences:
            break
    answer_text = " ".join([p[1]["sentence"] for p in picks])
    citations = [{
        "filename": p[1]["filename"],
        "page": p[1]["page"],
        "score": p[0],
        "preview": p[1]["sentence"][:200] + ("..." if len(p[1]["sentence"]) > 200 else "")
    } for p in picks]
    return {"answer": answer_text, "citations": citations}


# ---------- PDF Export ----------
def export_answer_pdf(path: Path, question: str, answer: str, citations: List[Dict], workspace: str):
    c = canvas.Canvas(str(path), pagesize=letter)
    w, h = letter
    m = 0.75 * inch
    x, y = m, h - m
    import textwrap
    def line(t, size=11, leading=14):
        nonlocal y
        c.setFont("Helvetica", size)
        for row in t.split("\n"):
            c.drawString(x, y, row)
            y -= leading
            if y < m:
                c.showPage(); y = h - m; c.setFont("Helvetica", size)
    c.setTitle("Knowledge Assistant — Answer Export")
    line(f"{APP_TITLE} — Export", 14, 18); line(f"Workspace: {workspace}", 10, 14); line("")
    line("Question:", 12); line(question); line("")
    line("Answer:", 12)
    for chunk in textwrap.wrap(answer, width=100): line(chunk)
    line(""); line("Citations:"
