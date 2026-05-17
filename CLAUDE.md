# GraphRAG++ — Claude Context File

## What This Project Is

GraphRAG++ is a hybrid Retrieval-Augmented Generation system that combines a **knowledge graph** (NetworkX) with **dense passage retrieval** (FAISS + sentence-transformers) and feeds both into an LLM for final answer synthesis. It ingests documents (PDF, code, text, images, ZIP), extracts entity-relation triples, stores raw passage chunks, and exposes a streaming query API with a Next.js frontend.

---

## Architecture Overview

```
frontend/           Next.js 16 + React 19 + D3.js + Chart.js (port 3000)
backend/            FastAPI + Uvicorn (port 8000)
  main.py           REST + WebSocket entry point; 13 endpoints
  pipeline/
    query_engine.py  Query pipeline: intent → RAG-Fusion → hybrid retrieval → rerank → graph → generate
    llm.py           LLM adapter with 4-tier fallback
    graph_builder.py NetworkX MultiDiGraph + PageRank + Louvain
    embedder.py      BAAI/bge-m3 (1024-dim) + FAISS index
    extractor.py     LLM-based triple extraction from raw text
    chunk_store.py   Hybrid chunk store: BM25 + cosine + RRF
    reranker.py      Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
    contextualizer.py Anthropic Contextual Retrieval — LLM context prepend + re-embed
    file_router.py   Routes files to correct extractor (PDF/code/image/ZIP)
    persistence.py   Graph serialisation to disk
    sample_data.py   Seeds graph with AI/ML domain data on first run
  eval/
    ragas_eval.py    RAGAS evaluation harness
    golden_set.json  20 golden Q&A pairs for pipeline scoring
  data/
    chunk_store.json Persisted passage chunks (~993 chunks)
    graph.json       Persisted knowledge graph
graphify-out/       Codebase knowledge graph (1055 nodes, 1280 edges, 42 communities)
inference/          Standalone GPU inference servers (Modal, RunPod, local)
docker-compose.yml  Multi-service local orchestration
start.ps1           One-shot PowerShell startup script
```

---

## Running the Project

### Quick start (PowerShell)
```powershell
.\start.ps1
```
Launches llama-server (port 8080), backend (port 8000), frontend (port 3000) in order, then opens the browser.

### Manual startup
```powershell
# Terminal 1 — local LLM (optional; skip if using HF API)
llama-server -m .\graphrag-plus-plus-qwen35-4b-q3_k_m-fixed.gguf --port 8080 --n-gpu-layers 99

# Terminal 2 — backend
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 3 — frontend
cd frontend
npm run dev
```

### Docker Compose
```bash
docker-compose up --build
```

### Environment variables (copy `.env.example` → `.env`)
| Variable | Purpose |
|---|---|
| `HF_TOKEN` | HuggingFace API key for hosted inference |
| `HF_MODEL_ID` | HF model repo for fallback inference |
| `LOCAL_LLAMA_CPP_URL` | Local llama.cpp server (e.g. `http://localhost:8080`) |
| `CLOUD_RUN_URL` | Optional Google Cloud Run / Modal endpoint |
| `USE_SIMULATION` | `true` → skip all real LLMs (deterministic mock) |
| `EMBED_MODEL` | Sentence Transformers model (default: `BAAI/bge-m3`; use `all-MiniLM-L6-v2` for fast dev) |
| `NEXT_PUBLIC_API_URL` | Frontend → backend URL (`http://localhost:8000`) |

---

## Key Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Uptime, node/edge counts |
| `GET` | `/graph` | Full graph as JSON (for D3) |
| `GET` | `/graph/stats` | Node/edge/community stats |
| `POST` | `/graph/reset` | Wipe and re-seed |
| `POST` | `/ingest` | Raw text → triple extraction → graph |
| `POST` | `/ingest/file` | Upload any file (PDF/code/image/ZIP) |
| `POST` | `/ingest/chunks` | Fast PDF/text → chunk store only (no LLM) |
| `POST` | `/ingest/directory` | Walk a server-side directory |
| `POST` | `/ingest/contextualize` | Run Anthropic Contextual Retrieval on all chunks (background task) |
| `GET` | `/chunks/stats` | Chunk store statistics |
| `POST` | `/query` | Run the full hybrid pipeline |
| `GET` | `/metrics` | System metrics + LLM source info |
| `WS` | `/ws` | Real-time pipeline step + token events |

---

## Query Pipeline (`pipeline/query_engine.py`)

The `query()` coroutine runs the following steps:

1. **Intent classification** — extracts entities and classifies the query type (factual / multi-hop / aggregative / comparative) using the LLM
2. **RAG-Fusion** *(multi-hop / comparative only)* — generates 3 query variants via LLM, embeds all variants in parallel, runs `hybrid_search` for each concurrently, RRF-merges all candidate lists. Skipped for factual queries.
3. **Semantic node matching** — embeds the query, searches FAISS index, returns top-k graph nodes
4. **Hybrid passage retrieval + reranking** — `hybrid_search` (BM25 sparse + BGE-M3 dense, merged via RRF) fetches 20 candidates; cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) re-scores and returns top 5
5. **Graph traversal** — multi-hop BFS/PageRank across the NetworkX graph (up to 3 hops for multi-hop, 2 for factual)
6. **Context fusion** — stitches graph paths + reranked passage chunks into a grounded prompt
7. **LLM generation** — sends to LLM; falls back to direct passage extraction if LLM output is corrupted

### Fallback logic (Step 7)

When the LLM fails or its output is corrupted, the pipeline falls back to **direct passage extraction**:

- **"Continue this section: [quote]" queries** bypass the LLM entirely (shortcut path)
- The hybrid retrieval top result is used directly (no keyword-boost heuristic — RRF already handles exact matches)
- Collect chunks from `top_page`, `top_page + 1`, `top_page + 2` (3-page span) to handle content that spans page boundaries
- **Quote-continuation trim**: for "Continue" queries, find where the user's quoted text ends in the stitched passage and return only what follows
  - Uses `_norm_quotes()` (quote-only normalisation, position-preserving) so `m.end()` maps directly back to the original string
  - Ellipsis-aware: text after the last `...` / `…` in the query is used as the anchor so user abbreviations don't break matching

### Corruption detection (`_answer_is_corrupted`)

An LLM answer is flagged as corrupted if:
- Empty / whitespace only
- Contains `### Response:` or `--[...]-->` artefacts (Alpaca template bleed-through)
- The first 60 chars appear verbatim in the passage text, path text, or instruction (instruction-echo from local model)

---

## LLM Adapter (`pipeline/llm.py`)

Five-tier fallback, tried in order:

1. **Gemini** — `GEMINI_API_KEY` + `GEMINI_MODEL` (default: `gemini-2.5-flash`). Primary path.
2. **Local llama.cpp** — `LOCAL_LLAMA_CPP_URL` (e.g. AMD GPU via Vulkan, port 8080)
3. **Cloud Run / Modal** — `CLOUD_RUN_URL` optional endpoint
4. **HuggingFace Inference API** — `HF_TOKEN` + `HF_MODEL_ID`
5. **Simulation** — deterministic mock; detected via `gen_result["source"] == "simulation"`

### Gemini integration (`_call_gemini`)

- Uses `google-genai` SDK (`pip install google-genai`)
- Comprehensive `GRAPHRAG_SYSTEM_PROMPT` covers all 4 pipeline tasks with exact JSON schemas:
  - **Task 1** — Triple extraction → JSON array `[{subject, predicate, object, confidence, confidence_tier}]`
  - **Task 2** — Intent classification → `{intent, entities, confidence}` where `intent` ∈ `factual|multi-hop|aggregative|comparative`
  - **Task 3** — Query variant generation → JSON array of 3 strings (for RAG-Fusion)
  - **Task 4** — Answer synthesis → plain prose 2–6 paragraphs
- `thinking_budget=0` for synthesis tasks (disables thinking so all tokens go to the answer)
- `max_output_tokens` floored at 4096 to prevent truncation
- Source tag: `"gemini"` or `"gemini+refined"`

### Intent response format

The `/query` endpoint returns `intent` as a dict:
```json
{ "intent": "factual", "entities": ["FAISS"], "confidence": 0.98 }
```
Access the type with `result["intent"]["intent"]` (not `result["intent"]["query_type"]`).

### Alpaca template (fine-tuned local models only)
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
```

---

## Chunk Store (`pipeline/chunk_store.py`)

- Persisted to `backend/data/chunk_store.json` (~993 chunks from the Claude Mythos Preview System Card PDF)
- Each chunk: `{ id, text, source, page, embedding, context? }`
  - `context` is present after running `/ingest/contextualize` (Anthropic Contextual Retrieval)
  - `embedding` encodes `context + text` if context exists, `text` alone otherwise
- **`hybrid_search(query_text, query_embedding, top_k)`** — BM25 sparse + BGE-M3 dense, merged via RRF (k=60). Replaces the old cosine-only search + keyword-boost heuristic.
- **`bm25_search(query_text, top_k)`** — BM25 sparse retrieval only (lazy-built index, rebuilt when chunks change)
- **`search(query_embedding, top_k)`** — pure cosine search (used internally)
- **`ensure_embeddings_valid(save_path)`** — called at startup; detects dim mismatch (e.g. old 384-dim vs new 1024-dim BGE-M3) and re-embeds all chunks automatically
- Loaded at startup via `lifespan`; saved on clean shutdown and after every file ingest

## Reranker (`pipeline/reranker.py`)

- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `rerank(query, chunks, top_k)` — re-scores (query, passage) pairs with full attention; +33–40% accuracy over bi-encoder retrieval alone
- Degrades silently if the model fails to load (returns `chunks[:top_k]` unchanged)

## Contextualizer (`pipeline/contextualizer.py`)

- Implements Anthropic's Contextual Retrieval: generates a 1-2 sentence LLM context blurb per chunk describing what it's about, then re-embeds using `context + text` as input
- Triggered via `POST /ingest/contextualize` (runs as background task, broadcasts progress over WebSocket)
- Skip already-contextualized chunks unless `?force=true`
- Expected improvement: 35–67% retrieval failure reduction (per Anthropic benchmarks)
- `CONTEXT_CONCURRENCY = 4` parallel LLM calls — keep low for local GPU

## Embedder (`pipeline/embedder.py`)

- Default model: `BAAI/bge-m3` (MTEB ~70, 1024-dim). Override with `EMBED_MODEL` env var.
- BGE retrieval query prefix applied automatically for bge-* models
- Downloads ~2 GB on first run; re-embeds all chunks if dim mismatch detected at startup
- Use `EMBED_MODEL=all-MiniLM-L6-v2` for fast development (384-dim, no download needed)

## Evaluation (`eval/`)

Run RAGAS evaluation against the live backend:
```powershell
cd backend
python -m eval.ragas_eval --no-ragas    # pipeline dry-run only (no LLM judge needed)
python -m eval.ragas_eval               # full RAGAS scoring (needs HF_TOKEN or OPENAI_API_KEY)
python -m eval.ragas_eval --max 5       # quick test with first 5 questions
```
Golden set: `eval/golden_set.json` — 20 Q&A pairs covering pipeline mechanics, retrieval concepts, and system architecture. Edit ground truths to match your actual corpus.

---

## Windows-Specific Gotchas

### Killing stale Python server processes
`kill` in Git Bash does not work on Windows processes. Use PowerShell:
```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

### Stale `.pyc` bytecode cache
Python 3.11 on Windows can serve a stale `.pyc` even after the `.py` is edited (especially if a previous server process compiled it with a newer mtime). Always delete `.pyc` files before restarting during development:
```powershell
Get-ChildItem -Path "C:\Users\ethan\OneDrive\Desktop\GraphRAG" -Recurse -Filter "*.pyc" | Remove-Item -Force
```

### Smart / curly quotes in Python source
The Edit tool can silently introduce Unicode curly quotes (U+2018, U+2019, U+201C, U+201D) into Python string literals, causing `SyntaxError: invalid character` on Python 3.11. **Never use literal curly quotes in Python source.** Instead:
- Use `chr(0x2018)` etc. for quote characters
- Use `‘` escape sequences inside strings
- After any suspicious edit, run `python -m py_compile backend/pipeline/query_engine.py` before restarting

### Port conflicts
If port 8000 is already bound after a crash, the new server silently fails. Check with:
```powershell
netstat -ano | findstr :8000
```
Then kill the owning PID with `Stop-Process -Id <PID> -Force`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI 0.115 + Uvicorn |
| Graph engine | NetworkX `MultiDiGraph` + PageRank + Louvain |
| Embeddings | `BAAI/bge-m3` (MTEB ~70, 1024-dim) via Sentence Transformers |
| Vector search | FAISS `IndexFlatIP` (CPU) |
| Sparse retrieval | BM25 via `rank-bm25` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` via Sentence Transformers |
| Codebase graph | Graphify (`graphifyy`) — Tree-sitter AST, 1055 nodes / 1280 edges |
| Evaluation | RAGAS (`ragas`) — faithfulness, answer relevancy, context precision/recall |
| LLM (local) | Llama.cpp (`llama-server`) with Qwen 3.5B GGUF, AMD Vulkan |
| LLM (hosted) | HuggingFace Inference API — `Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2` |
| PDF parsing | pdfplumber |
| Frontend | Next.js 16 + React 19 + TypeScript |
| Graph viz | D3.js 7 |
| Charts | Chart.js 4 + react-chartjs-2 |
| Deployment | Docker Compose (local), Modal / RunPod (GPU cloud) |

---

## Frontend (`frontend/`)

- **`src/app/page.tsx`** — homepage: live graph stats, reseed button, performance comparison table
- **`src/app/query/page.tsx`** — query interface with real-time pipeline step display; graph is **opt-in** (hidden by default via `showGraph` state to avoid fetching 2000+ nodes on every load)
- **`src/app/graph/page.tsx`** — D3.js interactive graph visualisation
- **`src/app/ingest/page.tsx`** — three-tab ingestion UI: **Upload Files** (multi-file, all types), **Paste Text** (raw text → triples), **Directory** (server-side path); per-file error isolation
- **`src/app/analytics/page.tsx`** — system metrics and performance charts
- **`src/lib/api.ts`** — typed API client + self-reconnecting WebSocket manager; 10-minute query timeout for slow GPUs; `ingestFiles` accepts `onError` callback for per-file failure surfacing
- **`src/components/IngestPanel.tsx`** — sidebar ingest widget (used on other pages); delegates to `ingestFiles`

### Supported ingest file types
```
.py .js .jsx .ts .tsx .go .rs .java .c .cpp .cc .rb
.md .txt .rst .pdf .zip .json .yaml .yml .toml .csv
.png .jpg .jpeg .webp .gif .ipynb
```

### Graph opt-in on query page
The graph visualisation is hidden by default and only fetched when the user clicks **Show Graph**. This avoids the 2000+ node D3 render on every query. The toggle is top-right of the graph pane.

Frontend talks to backend at `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). WebSocket at `ws://localhost:8000/ws` broadcasts all pipeline events and token-level generation progress.

---

## Common Development Tasks

### Add a new pipeline step
Edit `backend/pipeline/query_engine.py`. Each step should call `on_step(name, data)` and append to the `steps` list that is returned in the final result dict.

### Ingest a new PDF into the chunk store
```bash
curl -X POST http://localhost:8000/ingest/chunks \
  -F "file=@your_document.pdf"
```
Or use the Ingest page in the frontend. Chunks are persisted to `backend/data/chunk_store.json` automatically.

### Re-seed the knowledge graph
```bash
curl -X POST http://localhost:8000/graph/reset
```
Or click "Re-seed Graph" on the homepage.

### Check what the LLM is actually sending/receiving
Look at `backend/pipeline/llm.py`. Enable `print()` statements or check the `thinking` field in the query response — it contains the pipeline's internal reasoning about which path was taken (LLM vs fallback, page range, chunk count).

### Run Contextual Retrieval on all chunks (one-time, high impact)
```bash
curl -X POST http://localhost:8000/ingest/contextualize
# Re-run if you add new chunks:
curl -X POST "http://localhost:8000/ingest/contextualize?force=true"
```
Watch WebSocket or `/chunks/stats` for progress. Takes ~5–15 minutes on local GPU for 993 chunks.

### Run RAGAS evaluation
```powershell
cd backend
python -m eval.ragas_eval --no-ragas    # pipeline dry-run
python -m eval.ragas_eval               # full scoring (needs HF_TOKEN)
```

### Update the Graphify codebase graph after code changes
```powershell
graphify update .
```

### Run a quick syntax check
```powershell
python -m py_compile backend/pipeline/query_engine.py
python -m py_compile backend/pipeline/chunk_store.py
python -m py_compile backend/pipeline/reranker.py
python -m py_compile backend/pipeline/contextualizer.py
python -m py_compile backend/pipeline/llm.py
```

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- **Windows encoding**: `graphify` output contains Unicode arrows (`→`) which crash in cp1252 terminals. Run via subprocess with `PYTHONIOENCODING=utf-8` or pipe through `encode('ascii','replace')`.

---

## Known Bugs Fixed (session log)

| Bug | File | Fix |
|---|---|---|
| `string index out of range` on directory ingest | `graph_builder.py` `_infer_entity_type` | Guard empty string: `last = parts[-1] if parts else ""`; add `and last` before indexing |
| `/metrics` 500 — `got multiple values for keyword argument 'file_count'` | `main.py` | `get_stats()` already returns `file_count`; remove duplicate explicit kwarg from `MetricsResponse(...)` |
| Gemini answer truncated to ~74 chars | `llm.py` + `query_engine.py` | Thinking tokens consumed the budget; fix: `thinking_budget=0` for synthesis, floor `max_output_tokens=4096` |
| `max_tokens=512` too small for generation | `query_engine.py` | Raised to `max_tokens=2048` for both generate and refine steps |

## `/ingest/directory` request schema

Field name is **`path`** (not `directory`):
```json
{ "path": "/absolute/path/to/dir", "recursive": true }
```

## JSON body gotcha

Do not use Unicode dashes (`—`, `–`) inside `curl` JSON bodies — they cause `"Invalid \escape"` / `"error parsing the body"` on the FastAPI side. Use ASCII hyphens instead.
