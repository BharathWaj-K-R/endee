# AI Knowledge Assistant

This system is designed with a pluggable vector database architecture, enabling seamless migration to Endee without changes to core business logic.

Built within a constrained time window to simulate rapid prototyping of production AI systems.

AI Knowledge Assistant is a beginner-friendly Retrieval-Augmented Generation (RAG) project built with FastAPI, Streamlit, LangChain, an Endee-compatible vector storage layer, SentenceTransformers, and OpenAI-compatible chat models.

## Public Deployment

This repository is ready for GitHub-connected deployment on Render using `render.yaml`. The setup creates:

- `endee-rag-backend` for FastAPI
- `endee-rag-frontend` for Streamlit

### Deploy from GitHub

1. Push the repository to GitHub
2. Sign in to Render
3. Create a new Blueprint and select this repository
4. Render will detect `render.yaml` and create both services
5. Add `OPENAI_API_KEY` in the backend service environment settings
6. Open the public URL for `endee-rag-frontend`

### Deployment Notes

- The frontend automatically connects to the backend using Render service networking
- For local development, the frontend still falls back to `http://localhost:8000`
- Uploaded documents are stored in the app filesystem, so on free/ephemeral hosting you should expect indexed data to reset on redeploy or service restart
- For a more durable demo, attach persistent storage or swap the vector backend implementation to a managed provider later
- On Render free instances, set `EMBEDDING_PROVIDER=openai` to avoid heavy local models and reduce memory use

## Folder Structure

```text
E:\Rag
|-- app/
|   |-- main.py
|   |-- routes/
|   |   |-- chat.py
|   |   |-- health.py
|   |   `-- upload.py
|   `-- services/
|       |-- chat_service.py
|       `-- document_service.py
|-- frontend/
|   `-- streamlit_app.py
|-- rag/
|   |-- generator.py
|   |-- ingest.py
|   `-- retriever.py
|-- utils/
|   |-- config.py
|   |-- document_loader.py
|   |-- memory.py
|   `-- endee_vector_store.py
|-- .env.example
|-- README.md
`-- requirements.txt
```

## Features

- Upload PDF and TXT documents
- Split documents into smaller chunks
- Generate embeddings with SentenceTransformers
- Store embeddings in an Endee-compatible vector storage layer
- Perform semantic similarity search
- Answer questions with a RAG pipeline
- Keep chat history for each session
- Switch between OpenAI generation and a low-resource extractive fallback

## Advanced Features

- Retrieval-Augmented Generation (RAG)
- Modular vector DB (Endee-compatible)
- Adapter pattern for vector database abstraction (Endee-ready)
- Semantic search
- Multi-model response generation
- Scalable architecture

## Endee Integration Strategy

The project now uses a modular vector architecture centered around `utils/endee_vector_store.py`. The current backend still uses Chroma internally so the system remains stable, but the application code now talks to an Endee-compatible storage abstraction instead of binding business logic directly to one vendor.

Endee can replace the current storage layer without changing ingestion, retrieval, chat orchestration, or frontend behavior.

1. Replace embedding storage with Endee collections
2. Use Endee similarity search API
3. Plug into retriever pipeline

## Setup Instructions

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Add environment variables:

   ```powershell
   copy .env.example .env
   ```

4. Open `.env` and add your OpenAI API key if you want to use the OpenAI model.

## Run Instructions

1. Start the FastAPI backend:

   ```powershell
   uvicorn app.main:app --reload
   ```

2. Start the Streamlit frontend in a second terminal:

   ```powershell
   streamlit run frontend/streamlit_app.py
   ```

3. Open the Streamlit URL shown in the terminal, upload a PDF or TXT file, and start asking questions.

## API Endpoints

- `GET /health` - check whether the backend is running
- `POST /api/upload` - upload and index a PDF or TXT document
- `GET /api/models` - list available answer modes
- `POST /api/chat` - ask a question over indexed documents

## Low-RAM Tips

- The default embedding model is `all-MiniLM-L6-v2`, which is small and fast.
- The current backend uses on-disk persistence to avoid keeping everything only in memory.
- The frontend includes an `extractive` mode that works without OpenAI and uses fewer resources.

## Suggested Test Flow

1. Upload a small TXT file first.
2. Ask a direct question whose answer is present in the file.
3. Try the OpenAI mode if your API key is configured.
4. Try the extractive mode if you want offline testing.
