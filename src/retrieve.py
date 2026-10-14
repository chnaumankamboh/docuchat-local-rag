import os
import pickle
import threading

# First, so the models folder is set before sentence_transformers loads
from documents import EMBED_DIR, INDEX_PATH, DOCS_PATH

import faiss
from sentence_transformers import SentenceTransformer

# # Load embedding model once to convert queries into vector representations (IMPORTANT)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Kept for code that refers to these names
index_path = INDEX_PATH
docs_path = DOCS_PATH

# Load FAISS index (for similarity search) and stored document chunks. With no
# index yet, start empty so the API can still run and accept uploads.
if os.path.exists(index_path) and os.path.exists(docs_path):
    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        data = pickle.load(f)
    docs = data["docs"]
    filenames = data["filenames"]
else:
    print("No index found - starting empty. Run ingest.py or upload documents through the API.")
    index = faiss.IndexFlatL2(embed_model.get_sentence_embedding_dimension())
    docs = []
    filenames = []

# FAISS isn't safe to search while another thread changes the index (the API
# serves several requests at once), so every read and write goes through this.
_lock = threading.RLock()


# Convert query into embedding, search similar vectors, and return top matching document chunks
def retrieve_docs(query, k=5, threshold=1.5):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")

    with _lock:
        # Search top-k similar documents using FAISS
        D, I = index.search(q_emb, k)

        results = []

        # Filter results based on distance threshold (relevance control).
        # FAISS returns -1 for empty slots when the index has fewer than k chunks.
        for dist, i in zip(D[0], I[0]):
            if 0 <= i < len(docs) and dist < threshold:
                results.append({
                    "filename": filenames[i],
                    "content": docs[i]
                })

    return results


def add_document(filename, chunks):
    """
    Embeds chunks for one file and adds them to the live index, replacing any
    chunks previously indexed under the same filename (so re-uploading a file
    updates it instead of duplicating it). Saves the index to disk.
    """
    global index, docs, filenames
    embeddings = embed_model.encode(chunks, convert_to_numpy=True).astype("float32")

    with _lock:
        keep = [i for i, name in enumerate(filenames) if name != filename]
        if len(keep) < len(filenames):
            # IndexFlatL2 can hand back its stored vectors, so rebuild it without the old version
            vectors = index.reconstruct_n(0, index.ntotal)[keep]
            new_index = faiss.IndexFlatL2(index.d)
            new_index.add(vectors)
            index = new_index
            docs = [docs[i] for i in keep]
            filenames = [filenames[i] for i in keep]

        index.add(embeddings)
        docs = docs + list(chunks)
        filenames = filenames + [filename] * len(chunks)
        _save()

    return len(chunks)


def list_documents():
    """[{"name": ..., "chunks": ...}] for every indexed file, in the order they were added."""
    with _lock:
        counts = {}
        for name in filenames:
            counts[name] = counts.get(name, 0) + 1
    return [{"name": name, "chunks": n} for name, n in counts.items()]


def _save():
    # Write to temp files and swap them in, so a crash mid-write can't leave a corrupt index
    os.makedirs(EMBED_DIR, exist_ok=True)
    faiss.write_index(index, index_path + ".tmp")
    with open(docs_path + ".tmp", "wb") as f:
        pickle.dump({"docs": docs, "filenames": filenames}, f)
    os.replace(index_path + ".tmp", index_path)
    os.replace(docs_path + ".tmp", docs_path)
