# RAG AI Assistant

This project is a **Retrieval-Augmented Generation (RAG) AI assistant** that can answer questions using your documents (PDF, DOCX, TXT). It combines **vector retrieval** with a **local language model** — everything runs on your own machine, no API key needed.

## Features
- Index documents and generate embeddings
- Search for relevant content using FAISS
- Generate answers locally with TinyLlama (TinyLlama-1.1B-Chat)
- Streamlit chat app (computer science tutor)
- DocuChat: web UI + API where users upload their own documents and get answers with sources

## Tech Stack
- Python, Streamlit, FastAPI
- FAISS for vector search
- Sentence-Transformers (all-MiniLM-L6-v2) for embeddings
- TinyLlama via Hugging Face Transformers for text generation
- pypdf & python-docx for document parsing

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```
   The models (TinyLlama + all-MiniLM-L6-v2, about 2.3 GB) are kept in the
   project's models/ folder; if it's missing, the first run downloads them there.
   Leave models/ out when you zip or share the project.

2. Build the index from data/sample_docs (and data/uploads):
```bash
python src/ingest.py
```

3. Run the Streamlit app:
```bash
streamlit run src/app.py
```

   Or run DocuChat, then open http://localhost:8000:
```bash
uvicorn rag_api:app --port 8000
```
