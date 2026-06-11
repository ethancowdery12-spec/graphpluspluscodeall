# GraphRAG++

Hybrid Retrieval-Augmented Generation system combining a NetworkX knowledge graph with BM25 sparse retrieval, BGE-M3 dense embeddings, cross-encoder reranking, and LLM answer synthesis.

## Quick start

```powershell
.\start.ps1
```

Launches llama-server (port 8080), backend (port 8000), and frontend (port 3000) in order.

## Requirements

- **Python 3.9+** (3.11 recommended)
- **Node.js 18+**
- Copy `.env.example` to `.env` and fill in at minimum `GEMINI_API_KEY` or `HF_TOKEN`

## Backend setup

```powershell
cd backend
pip install -r requirements.txt
```

### Apple Silicon (M1/M2/M3) — faiss-cpu

The `faiss-cpu` pip wheel is not available for Apple Silicon. Install via conda instead:

```bash
conda install -c conda-forge faiss-cpu
pip install -r requirements.txt --no-deps faiss-cpu  # skip faiss from pip
```

Or use the conda environment for the full install:

```bash
conda create -n graphrag python=3.11
conda activate graphrag
conda install -c conda-forge faiss-cpu
pip install -r requirements.txt
```

### tree-sitter version

`tree-sitter` and `tree-sitter-languages` are exact-pinned in `requirements.txt` and must stay that way — the versions are API-incompatible with each other. Do not upgrade them independently.

## Frontend setup

```powershell
cd frontend
npm install
npm run dev
```

## Architecture

```
frontend/    Next.js 16 + React 19 + D3.js (port 3000)
backend/     FastAPI + Uvicorn (port 8000)
  pipeline/  Query engine, graph builder, chunk store, embedder, reranker
  eval/      RAGAS evaluation harness + golden set
  data/      Persisted graph and chunk store
```

See `CLAUDE.md` for full architecture details, endpoint reference, and development notes.
