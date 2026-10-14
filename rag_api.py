"""
DocuChat API — wires YOUR existing RAG pipeline to the docuchat_ui.html frontend.

It reuses your own modules (src/retrieve.py and src/generate.py) and adds
runtime document upload (/upload): new files are saved to data/uploads/ and
added straight into the live FAISS index, so the chatbot can answer from
documents the *user* provides. ingest.py also reads data/uploads/, so
re-running it keeps them.

WHERE TO PUT THIS FILE
  Place rag_api.py in the project root — the same folder that contains `src/`
  and `embeddings/`.  (So: "Rag Based AI Advanced/rag_api.py")

RUN
  pip install -r requirements.txt
  uvicorn rag_api:app --port 8000

Then open http://localhost:8000 — the API serves docuchat_ui.html itself.

To host the UI somewhere else, list that site's address in the
ALLOWED_ORIGINS environment variable (comma-separated), e.g.
  set ALLOWED_ORIGINS=https://my-site.com
"""

import os
import re
import sys
import tempfile

# Make your src/ modules importable (they use imports like `from retrieve import ...`)
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# These imports load your models + FAISS index ONCE at startup:
#   retrieve.py  -> SentenceTransformer + the FAISS index, docs, filenames
#   generate.py  -> TinyLlama + small talk + answer_question
import retrieve
import generate
from documents import UPLOADS_DIR, SUPPORTED_EXTENSIONS, extract_text, chunk_text

UI_PATH = os.path.join(ROOT, "docuchat_ui.html")
MAX_UPLOAD_MB = 20

app = FastAPI(title="DocuChat API")

# The UI is served from this same server, so no cross-origin access is needed.
# Allowing every origin ("*") would let ANY website you visit while this runs
# upload documents into your index and read answers from it.
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def safe_filename(name):
    """Just the file name (no folders, so it can't escape data/uploads), with odd characters replaced."""
    name = re.split(r"[\\/]", name or "")[-1].strip()
    return re.sub(r"[^\w.\- ()]", "_", name)


def index_upload(upload):
    """Saves one uploaded file to data/uploads and indexes it. Returns the result for the UI."""
    name = safe_filename(upload.filename)
    if not name.lower().endswith(SUPPORTED_EXTENSIONS):
        ext = os.path.splitext(name)[1].lower()
        if ext == ".doc":
            return {"name": name, "chunks": 0, "error": "old .doc files aren't supported - save it as .docx"}
        return {"name": name, "chunks": 0, "error": "unsupported file type - use PDF, DOCX or TXT"}

    data = upload.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return {"name": name, "chunks": 0, "error": f"file is larger than {MAX_UPLOAD_MB} MB"}

    # Read the text from a temp copy first, so a broken file never lands in data/uploads
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1]) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        chunks = chunk_text(extract_text(tmp_path))
    except Exception as e:
        return {"name": name, "chunks": 0, "error": f"couldn't read the file ({e})"}
    finally:
        os.remove(tmp_path)

    if not chunks:
        hint = " - a scanned PDF needs OCR" if name.lower().endswith(".pdf") else ""
        return {"name": name, "chunks": 0, "error": "no text found" + hint}

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    with open(os.path.join(UPLOADS_DIR, name), "wb") as f:
        f.write(data)
    return {"name": name, "chunks": retrieve.add_document(name, chunks)}


# ----------------------------------------------------------------------
#  /upload — users add their own documents at runtime
#  (plain `def` endpoints run on a worker thread, so the slow embedding and
#  generation work doesn't freeze the whole server the way `async def` did)
# ----------------------------------------------------------------------
@app.post("/upload")
def upload(files: list[UploadFile] = File(...)):
    # One result per file; a bad file gets an "error" instead of failing the batch
    return {"documents": [index_upload(f) for f in files]}


@app.get("/documents")
def documents():
    """Everything currently in the index, so the UI can show the knowledge base on load."""
    return {"documents": retrieve.list_documents()}


# ----------------------------------------------------------------------
#  /chat — your retrieval + generation, now also returning sources
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    chat_id: str | None = None


@app.post("/chat")
def chat(req: ChatRequest):
    # "documents" mode answers only from the indexed documents (see generate.PERSONAS)
    answer, used = generate.answer_question(req.message, k=5, mode="documents")

    # Sources = the chunks the answer was actually built from, one chip per file
    # (your chunks have no page numbers)
    seen, sources = set(), []
    for r in used:
        fn = r["filename"]
        if fn not in seen:
            seen.add(fn)
            sources.append({"document": fn})

    return {"answer": answer, "sources": sources}


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(UI_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": retrieve.index.ntotal}
