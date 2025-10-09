# app.py — Knowledge Assistant (MVP, Fast extractive Q&A only — no external LLMs)

import os
import io
import re
import json
import uuid
from datetime import datetime
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

# ----------------------------
# App config
# ----------------------------
APP_TITLE = "Knowledge Assistant (MVP)"
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(exist_ok=True, parents=True)

CHUNK_WORDS = 800
CHUNK_OVERLAP = 150
TOP_K_CHUNKS = 6
TOP_SENTENCES = 4
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9“"(])')

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ----------------------------
# Optional Auth (single password via env)
#   - To disable, ensure APP_PASSWORD is not set
# ----------------------------
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

# ----------------------------
# Readers & text utils
# ----------------------------
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
    if not text:
        return []
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(len(words), start + size_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

def split_sentences(text: str) -> List[str]:
    sentences = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts = SENTENCE_SPLIT_RE.split(para)
        for s in parts:
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences

# ----------------------------
# TF-IDF store (per workspace)
# ----------------------------
class Store:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.dir = DATA_DIR / workspace
        self.dir.mkdir(exist_ok=True, parents=True)
        self.meta_path = self.dir / "meta.json"
        self.raw_path = self.dir / "raw.json"  # raw pages for Explain/Explore

        self.meta = {"docs": [], "chunks": []}
        self.raw_pages: Dict[str, List[Dict]] = {}

        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self._load()

    def _load(self):
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.raw_path.exists():
            self.raw_pages = json.loads(self.raw_path.read_text(encoding="utf-8"))
        if self.meta.get("chunks"):
            self._rebuild_index()

    def _save(self):
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.raw_path.write_text(json.dumps(self.raw_pages, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self):
        for p in self.dir.glob("*"):
            try:
                p.unlink()
            except:
                pass
        self.meta = {"docs": [], "chunks": []}
        self.raw_pages = {}
        self.vectorizer = None
        self.matrix = None   # ← fixed indentation (aligned with the line above)

    def _rebuild_index(self):
        texts = [c["text"] for c in self.meta["chunks"]]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            max_features=100_000, norm="l2"
        )
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def add(self, filename: str, file_bytes: bytes) -> int:
        ext = filename.lower().split(".")[-1]
        entries = []

        if ext == "pdf":
            pages = read_pdf(file_bytes)
            self.raw_pages[filename] = [{"page": i, "text": clean_text(t)} for (t, i) in pages]
            for page_text, page_no in pages:
                text = clean_text(page_text)
                for ch in chunk_text(text):
                    entries.append({"id": str(uuid.uuid4()), "filename": filename, "page": page_no, "text": ch})
        elif ext == "docx":
            text = clean_text(read_docx(file_bytes))
            self.raw_pages[filename] = [{"page": None, "text": text}]
            for ch in chunk_text(text):
                entries.append({"id": str(uuid.uuid4()), "filename": filename, "page": None, "text": ch})
        elif ext == "txt":
            text = clean_text(read_txt(file_bytes))
            self.raw_pages[filename] = [{"page": None, "text": text}]
            for ch in chunk_text(text):
                entries.append({"id": str(uuid.uuid4()), "filename": filename, "page": None, "text": ch})
        else:
            raise ValueError(f"Unsupported file type: .{ext}")

        if not entries:
            return 0

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

# ----------------------------
# Extractive answer (fast)
# ----------------------------
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

    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), norm="l2")
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

# ----------------------------
# Summarize & key facts (heuristics)
# ----------------------------
def extract_key_facts(raw_pages: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    facts = {
        "Dates": [],
        "Notice Periods": [],
        "Fees & Payment": [],
        "Percentages (SLA/others)": [],
        "Governing Law / Venue": [],
        "Renewal / Termination": [],
    }
    date_re = re.compile(r"\b(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b", re.I)
    notice_re = re.compile(r"\b(\d{1,3})\s*(day|days|business days|calendar days)\b.*(notice|non[- ]?renewal|termination)", re.I)
    fee_re = re.compile(r"\$?\s?\b(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?)\b.*(per\s*(month|year|annum)|subscription|fee|invoice|net\s*\d+)", re.I)
    pct_re = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*%\b", re.I)
    law_re = re.compile(r"\b(governing law|jurisdiction|venue|laws of)\b.*", re.I)
    renew_re = re.compile(r"\b(renew|renewal|auto[- ]?renew|terminate|termination|initial term)\b.*", re.I)

    for fname, pages in raw_pages.items():
        for p in pages:
            text = p.get("text") or ""
            page = p.get("page")
            for ln in [ln.strip() for ln in text.split("\n") if ln.strip()]:
                if date_re.search(ln):   facts["Dates"].append({"text": ln, "filename": fname, "page": page})
                if notice_re.search(ln): facts["Notice Periods"].append({"text": ln, "filename": fname, "page": page})
                if fee_re.search(ln):    facts["Fees & Payment"].append({"text": ln, "filename": fname, "page": page})
                if pct_re.search(ln):    facts["Percentages (SLA/others)"].append({"text": ln, "filename": fname, "page": page})
                if law_re.search(ln):    facts["Governing Law / Venue"].append({"text": ln, "filename": fname, "page": page})
                if renew_re.search(ln):  facts["Renewal / Termination"].append({"text": ln, "filename": fname, "page": page})
    return facts

def summarize_workspace(raw_pages: Dict[str, List[Dict]], max_sentences: int = 8) -> str:
    all_sentences = []
    for _, pages in raw_pages.items():
        for p in pages:
            for s in split_sentences(p.get("text") or ""):
                if len(s) >= 40:
                    all_sentences.append(s)
    if not all_sentences:
        return "No content available to summarize."
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(all_sentences)
    scores = (X.sum(axis=1)).A.ravel()
    idx = np.argsort(-scores)[:max_sentences]
    return " ".join([all_sentences[i] for i in idx])

# ----------------------------
# PDF export (answer + citations)
# ----------------------------
def export_answer_pdf(path: Path, question: str, answer: str, citations: List[Dict], workspace: str):
    import textwrap
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    margin = 0.75 * inch
    x = margin
    y = height - margin

    def line(t, size=11, leading=14):
        nonlocal y
        c.setFont("Helvetica", size)
        c.drawString(x, y, t)
        y -= leading
        if y < margin:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", size)

    c.setTitle("Knowledge Assistant — Answer Export")
    line(f"{APP_TITLE} — Export", 14, 18)
    line(f"Workspace: {workspace}", 10, 14)
    line(f"Generated: {datetime.utcnow().isoformat()}Z", 8, 12)
    line("")
    line("Question:", 12)
    for row in textwrap.wrap(question or "", 100): line(row)
    line("")
    line("Answer:", 12)
    for row in textwrap.wrap(answer or "", 100): line(row)
    line("")
    line("Citations:", 12)
    for cit in citations or []:
        page = f"(p. {cit.get('page')})" if cit.get("page") else ""
        line(f"- {cit.get('filename','')} {page} — score {cit.get('score',0):.3f}")
        for row in textwrap.wrap(cit.get('preview',''), 100): line(f"  {row}")
    c.save()

# ----------------------------
# UI
# ----------------------------
def main():
    check_auth()
    st.title(APP_TITLE)

    # Sidebar / Workspace
    st.sidebar.header("Workspace")
    existing = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
    ws = st.sidebar.text_input("Name", value=(existing[0] if existing else "default"))
    if st.sidebar.button("Create/Use"):
        st.session_state.workspace = ws
    if "workspace" not in st.session_state:
        st.session_state.workspace = ws
    ws = st.session_state.workspace
    st.sidebar.write(f"Active: **{ws}**")

    store = Store(ws)

    if st.sidebar.button("Reset workspace"):
        store.reset()
        st.success("Workspace reset. (All files & index cleared)")

    # Upload & Ingest
    st.subheader("Upload documents")
    files = st.file_uploader("Add PDF / DOCX / TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.button("Ingest & Index") and files:
        with st.spinner("Indexing…"):
            total = 0
            for f in files:
                try:
                    total += store.add(f.name, f.getvalue())
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            st.success(f"Indexed chunks: {total}")

    # Tabs
    tab_ask, tab_explain, tab_chunks = st.tabs(["❓ Ask", "📘 Explain", "🧩 Chunks"])

    # --- Ask Tab ---
    with tab_ask:
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            question = st.text_input("Ask a question", placeholder="e.g., What is the renewal date and notice period?")
        with c2:
            top_k = st.number_input("Top-K chunks", min_value=1, max_value=20, value=TOP_K_CHUNKS, step=1)
        with c3:
            top_sents = st.number_input("Sentences in answer", min_value=1, max_value=10, value=TOP_SENTENCES, step=1)

        if st.button("Answer"):
            with st.spinner("Working…"):
                results = store.search(question, top_k=top_k)
                out = answer_with_sentences(question, results, top_sentences=top_sents)

            st.markdown("### Answer")
            st.write(out["answer"] or "_No answer_")

            st.markdown("### Citations")
            if out["citations"]:
                for citem in out["citations"]:
                    page = f"(p. {citem['page']})" if citem.get("page") else ""
                    st.markdown(f"- **{citem['filename']}** {page} — score `{citem['score']:.3f}`\n\n> {citem['preview']}")
            else:
                st.write("_No citations_")

            # Export PDF
            export_name = f"answer_{uuid.uuid4().hex[:8]}.pdf"
            export_path = DATA_DIR / ws / export_name
            export_answer_pdf(export_path, question, out["answer"], out["citations"], ws)
            with open(export_path, "rb") as fh:
                st.download_button("⬇️ Download answer as PDF", data=fh, file_name=export_name, mime="application/pdf")

    # --- Explain Tab ---
    with tab_explain:
        st.write("Get a quick overview and key facts extracted from your documents.")
        if not store.raw_pages:
            st.info("Upload and ingest documents first.")
        else:
            colA, colB = st.columns([2, 1])
            with colA:
                if st.button("Explain this workspace"):
                    with st.spinner("Summarizing…"):
                        summary = summarize_workspace(store.raw_pages, max_sentences=8)
                    st.markdown("### Summary")
                    st.write(summary)
            with colB:
                st.markdown("### Key facts")
                facts = extract_key_facts(store.raw_pages)
                for k, items in facts.items():
                    if not items:
                        continue
                    with st.expander(k, expanded=(k in ["Renewal / Termination", "Fees & Payment", "Notice Periods"])):
                        for it in items[:10]:
                            page = f"(p. {it.get('page')})" if it.get("page") else ""
                            st.markdown(f"- {it['text']}  \n  _{it['filename']} {page}_")

            st.caption("Tip: Use the Ask tab to dig deeper into any of these facts.")

    # --- Chunks Tab ---
    with tab_chunks:
        if not store.meta["chunks"]:
            st.info("No chunks indexed yet.")
        else:
            st.write(f"**Documents:** {len(store.meta['docs'])} • **Chunks:** {len(store.meta['chunks'])}")
            filenames = ["All"] + [d["filename"] for d in store.meta["docs"]]
            pick = st.selectbox("Filter by file", filenames)
            shown = [c for c in store.meta["chunks"] if pick == "All" or c["filename"] == pick]
            start = st.number_input("Start at chunk #", 0, max(0, len(shown) - 1), 0, 10)
            for c in shown[start:start+30]:
                st.markdown(f"**{c['filename']}**  (page {c.get('page')}) — id `{c['id'][:8]}`")
                st.code(c["text"][:1500] + ("..." if len(c["text"]) > 1500 else ""))

    st.caption("Fast mode only: answers are extracted & cited from your documents.")

if __name__ == "__main__":
    main()
