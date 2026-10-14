"""
Shared document handling: where documents and the index live, how text is
pulled out of files, and how it is chunked. Used by ingest.py (bulk indexing)
and rag_api.py (runtime uploads) so both index documents the same way.
"""
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Downloaded models (TinyLlama, all-MiniLM-L6-v2) live in the project's models/
# folder instead of the user cache on C:. Hugging Face reads this when it is first
# imported, so every script imports this module BEFORE transformers /
# sentence_transformers. An HF_HOME you set yourself still wins.
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "models"))
SAMPLE_DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "sample_docs")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")   # files added through the API
EMBED_DIR = os.path.join(PROJECT_ROOT, "embeddings")
INDEX_PATH = os.path.join(EMBED_DIR, "index.faiss")
DOCS_PATH = os.path.join(EMBED_DIR, "docs.pkl")

SUPPORTED_EXTENSIONS = (".txt", ".docx", ".pdf")

CHUNK_SIZE, OVERLAP = 300, 50


def extract_text(path):
    """Plain text from a .txt / .docx / .pdf file. Raises ValueError for anything else."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(path).paragraphs)

    if ext == ".pdf":
        from pypdf import PdfReader
        text = "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
        # --- OCR fallback for scanned PDFs (optional) ---------------------
        # A scanned PDF has no text layer, so extract_text() returns almost
        # nothing. To support those, install pytesseract + pdf2image and add:
        #
        # if len(text.strip()) < 30:
        #     import pytesseract
        #     from pdf2image import convert_from_path
        #     text = "\n".join(pytesseract.image_to_string(img)
        #                      for img in convert_from_path(path))
        # ------------------------------------------------------------------
        return text

    if ext == ".doc":
        raise ValueError("old .doc files aren't supported - save it as .docx and upload again")
    raise ValueError(f"unsupported file type '{ext or 'none'}' - use PDF, DOCX or TXT")


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Overlapping chunks of about `chunk_size` characters; overlap keeps sentences
    that cross a boundary findable. Cuts land on whitespace, so no word is split
    in half (a half-word embeds badly and reads badly as a source)."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + chunk_size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        # step back by `overlap`, then forward to a word start
        start = max(end - overlap, start + 1)
        while 0 < start < end and text[start - 1] != " ":
            start += 1
    return chunks
