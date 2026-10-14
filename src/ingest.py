import os
import pickle

# First, so the models folder is set before sentence_transformers loads
from documents import (
    SAMPLE_DOCS_DIR, UPLOADS_DIR, EMBED_DIR, INDEX_PATH, DOCS_PATH,
    SUPPORTED_EXTENSIONS, extract_text, chunk_text,
)

from sentence_transformers import SentenceTransformer
import faiss

# Create the embeddings directory if it doesn't already exist
os.makedirs(EMBED_DIR, exist_ok=True)

# Load a pre-trained sentence transformer model for converting text to vectors
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
filenames = []

# Index the sample docs AND anything uploaded through rag_api.py, so
# re-running this script never drops documents users added
for folder in (SAMPLE_DOCS_DIR, UPLOADS_DIR):
    if not os.path.isdir(folder):
        continue
    for file in sorted(os.listdir(folder)):
        path = os.path.join(folder, file)
        if not os.path.isfile(path):
            continue
        if not file.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"Skipping {file}: not a PDF, DOCX or TXT file")
            continue

        # Extract the text (txt / docx / pdf), split it into chunks and store them
        try:
            text = extract_text(path)
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue
        chunks = chunk_text(text)
        if not chunks:
            hint = " (a scanned PDF needs OCR)" if file.lower().endswith(".pdf") else ""
            print(f"Skipping {file}: no text found{hint}")
            continue
        docs.extend(chunks)
        filenames.extend([file] * len(chunks))

# If no documents were processed, raise an error
if not docs:
    raise RuntimeError("No documents found!")

# Convert text chunks into embeddings, build FAISS similarity index, and store it for fast search
embeddings = embed_model.encode(docs, show_progress_bar=True)
embeddings = embeddings.astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index and raw document data for later retrieval in search or RAG pipeline
faiss.write_index(index, INDEX_PATH)

with open(DOCS_PATH, "wb") as f:
    pickle.dump({"docs": docs, "filenames": filenames}, f)

print(f"Ingested {len(docs)} chunks from {len(set(filenames))} files")
