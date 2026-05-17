"""
GraphRAG++ FastAPI Backend
Serves the knowledge graph engine with REST + WebSocket APIs.
"""
import os
import time
import hashlib
import asyncio
import json
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from models import (
    IngestRequest, IngestURLRequest, IngestDirRequest,
    IngestResponse, IngestDirResponse,
    GraphResponse, QueryRequest, QueryResponse, MetricsResponse
)
from pipeline.graph_builder import get_graph
from pipeline.extractor import extract_triples
from pipeline.query_engine import query as run_query
from pipeline.embedder import embed_text, add_to_index
from pipeline.sample_data import seed_graph
from pipeline.file_router import route_file
from pipeline.dir_walker import walk_directory
from pipeline.persistence import save_graph
from pipeline.chunk_store import get_chunk_store

load_dotenv()

_start_time = time.time()
_ws_clients: List[WebSocket] = []

CHUNK_STORE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "chunk_store.json"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load persisted graph + chunk store, or seed with sample data."""
    print("[GraphRAG++] Starting up…")
    graph = get_graph()   # persistence.load_graph() is called inside get_graph()

    if graph.G.number_of_nodes() == 0:
        print("[GraphRAG++] No persisted graph found — seeding with sample data…")
        result = seed_graph()
        save_graph(graph)
        print(f"[GraphRAG++] Seeded: {result}")
    else:
        n = graph.G.number_of_nodes()
        e = graph.G.number_of_edges()
        print(f"[GraphRAG++] Restored from disk: {n} nodes, {e} edges")

    # Load passage chunk store
    cs = get_chunk_store()
    cs.load(CHUNK_STORE_PATH)
    print(f"[GraphRAG++] Chunk store: {len(cs)} passages ready")

    # Re-embed chunks if the embedding model changed (e.g. MiniLM → BGE-M3)
    asyncio.create_task(cs.ensure_embeddings_valid(CHUNK_STORE_PATH))
    asyncio.create_task(_embed_all_nodes())
    yield
    print("[GraphRAG++] Shutting down.")
    # Persist chunk store on clean shutdown
    get_chunk_store().save(CHUNK_STORE_PATH)


app = FastAPI(
    title="GraphRAG++ Knowledge Graph Fusion Engine",
    version="2.0.0",
    description="Hybrid knowledge graph with multi-hop reasoning, code extraction, and LLM-fused generation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    graph = get_graph()
    stats = graph.get_stats()
    return {
        "status":       "ok",
        "uptime_s":     round(time.time() - _start_time, 1),
        "graph_nodes":  stats["node_count"],
        "graph_edges":  stats["edge_count"],
        "file_count":   stats.get("file_count", 0),
        "version":      stats["version"],
    }


# ─── Graph ─────────────────────────────────────────────────────────────────────

@app.get("/graph", response_model=GraphResponse)
async def get_graph_data():
    return get_graph().to_json()


@app.get("/graph/stats")
async def get_stats():
    return get_graph().get_stats()


@app.post("/graph/reset")
async def reset_graph():
    """Reset to seed data (clears any custom ingestions)."""
    from pipeline.graph_builder import _graph_store
    import pipeline.graph_builder as gb
    gb._graph_store = None          # force fresh singleton
    graph = get_graph()             # creates new empty GraphStore
    result = seed_graph()
    save_graph(graph)
    asyncio.create_task(_embed_all_nodes())
    await _broadcast({"event": "graph_reset", "data": result})
    return {"status": "reset", **result}


# ─── Ingestion — Text ──────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_text(req: IngestRequest):
    """Ingest raw text: extract triples (LLM) and add to graph."""
    await _broadcast({"event": "ingest_start", "data": {"source": req.source}})

    triples = await extract_triples(req.text)
    for t in triples:
        await _broadcast({"event": "triple_extracted", "data": t})
        await asyncio.sleep(0.03)

    graph  = get_graph()
    result = graph.ingest_triples(triples, source=req.source)
    asyncio.create_task(_embed_all_nodes())
    save_graph(graph)

    await _broadcast({"event": "ingest_complete", "data": result})
    return IngestResponse(
        status="ok",
        triples_extracted=len(triples),
        nodes_added=result["nodes_added"],
        edges_added=result["edges_added"],
        version=result["version"],
        triples=[{
            "subject": t["subject"], "predicate": t["predicate"],
            "object": t["object"], "confidence": t.get("confidence", 0.85),
            "confidence_tier": t.get("confidence_tier", "INFERRED"),
        } for t in triples],
        source=req.source,
    )


# ─── Ingestion — File (any type) ───────────────────────────────────────────────

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...), force: bool = False):
    """
    Ingest any supported file: code (AST), PDF, image (vision), ZIP, text/markdown.
    Skips files that have already been ingested (by SHA256 hash) unless force=true.
    """
    content  = await file.read()
    filename = file.filename or "upload"

    # Dedup check
    sha256 = hashlib.sha256(content).hexdigest()
    graph  = get_graph()
    if not force and graph.has_file_hash(sha256):
        return {
            "status": "already_ingested",
            "sha256": sha256,
            "source": filename,
            "triples_extracted": 0,
            "nodes_added": 0,
            "edges_added": 0,
            "version": graph._version,
            "triples": [],
        }

    await _broadcast({"event": "ingest_start", "data": {"source": filename}})

    triples, sha256 = await route_file(content, filename)
    result = graph.ingest_triples(triples, source=filename)
    # Only lock out re-ingestion if we actually extracted something
    if triples:
        graph.register_file_hash(sha256, filename)
    asyncio.create_task(_embed_all_nodes())
    save_graph(graph)
    # Persist chunk store (route_file may have added passage chunks)
    get_chunk_store().save(CHUNK_STORE_PATH)

    await _broadcast({"event": "ingest_complete", "data": result})
    return IngestResponse(
        status="ok",
        triples_extracted=len(triples),
        nodes_added=result["nodes_added"],
        edges_added=result["edges_added"],
        version=result["version"],
        triples=[{
            "subject": t["subject"], "predicate": t["predicate"],
            "object": t["object"], "confidence": t.get("confidence", 0.85),
            "confidence_tier": t.get("confidence_tier", "INFERRED"),
        } for t in triples],
        source=filename,
    )


# ─── Ingestion — Directory ─────────────────────────────────────────────────────

@app.post("/ingest/directory", response_model=IngestDirResponse)
async def ingest_directory(req: IngestDirRequest):
    """
    Walk a local directory path (server-side), route each file to the appropriate
    extractor, and ingest all discovered triples.  Files are SHA256-deduplicated,
    so re-running on the same directory only processes changed files.
    """
    graph = get_graph()
    existing_hashes = set(graph._file_hashes.keys())

    total_triples = 0
    total_nodes   = 0
    total_edges   = 0
    files_done    = 0
    files_skipped = 0

    await _broadcast({"event": "ingest_start", "data": {"source": req.path}})

    try:
        async for triples, sha256, filepath in walk_directory(req.path, existing_hashes):
            if not triples:
                files_skipped += 1
                continue

            result = graph.ingest_triples(triples, source=filepath)
            graph.register_file_hash(sha256, filepath)
            existing_hashes.add(sha256)

            total_triples += len(triples)
            total_nodes   += result["nodes_added"]
            total_edges   += result["edges_added"]
            files_done    += 1

            await _broadcast({
                "event": "ingest_progress",
                "data": {
                    "file": os.path.basename(filepath),
                    "triples": len(triples),
                    "total_files": files_done,
                },
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Directory ingestion failed: {e}")

    asyncio.create_task(_embed_all_nodes())
    save_graph(graph)

    summary = {
        "status":          "ok",
        "files_processed": files_done,
        "files_skipped":   files_skipped,
        "triples_total":   total_triples,
        "nodes_added":     total_nodes,
        "edges_added":     total_edges,
        "version":         graph._version,
    }
    await _broadcast({"event": "ingest_complete", "data": summary})
    return IngestDirResponse(**summary)


# ─── Ingestion — Pre-extracted triples (no LLM) ───────────────────────────────

@app.post("/ingest/triples")
async def ingest_raw_triples(payload: dict):
    """Accept pre-extracted triples directly — skips LLM extraction entirely.

    Body: {"triples": [...], "source": "..."}
    Each triple: {subject, predicate, object, confidence, confidence_tier?}
    """
    triples = payload.get("triples", [])
    source  = payload.get("source", "direct")
    if not triples:
        raise HTTPException(status_code=400, detail="No triples provided")
    graph  = get_graph()
    result = graph.ingest_triples(triples, source=source)
    asyncio.create_task(_embed_all_nodes())
    save_graph(graph)
    await _broadcast({"event": "ingest_complete", "data": result})
    return {"status": "ok", "triples_received": len(triples), **result}


# ─── Ingestion — Chunks only (no LLM, no triples) ────────────────────────────

@app.post("/ingest/chunks")
async def ingest_chunks_only(file: UploadFile = File(...)):
    """
    Fast passage-only ingest: extract text and add to ChunkStore WITHOUT
    calling the LLM triple extractor.  Ideal for large PDFs where you want
    passage retrieval immediately (re-uploading an already-ingested file).

    Supports: PDF (pdfplumber page-by-page), plain text, markdown.
    Returns chunk count and source name.
    """
    import io as _io
    content  = await file.read()
    filename = file.filename or "upload"
    ext      = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    store    = get_chunk_store()

    added = 0
    try:
        if ext == "pdf":
            import pdfplumber
            with pdfplumber.open(_io.BytesIO(content)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    pt = page.extract_text()
                    if pt:
                        n = await store.add_text(pt.strip(), source=filename, page=page_num)
                        added += n
                        await asyncio.sleep(0)  # yield between pages
        else:
            text = content.decode("utf-8", errors="replace")
            added = await store.add_text(text, source=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunk extraction failed: {e}")

    store.save(CHUNK_STORE_PATH)
    await _broadcast({"event": "chunks_ingested",
                      "data": {"source": filename, "chunks_added": added}})
    return {
        "status":       "ok",
        "source":       filename,
        "chunks_added": added,
        "total_chunks": len(store),
    }


# ─── Chunk store stats ────────────────────────────────────────────────────────

@app.get("/chunks/stats")
async def chunk_stats():
    """Return chunk store statistics."""
    return get_chunk_store().get_stats()


# ─── Contextual Retrieval ─────────────────────────────────────────────────────

_contextualize_task = None
_contextualize_progress = {"done": 0, "total": 0, "running": False}


@app.post("/ingest/contextualize")
async def contextualize_chunks(force: bool = False):
    """
    Enrich every chunk with a short LLM-generated context blurb (Anthropic
    Contextual Retrieval approach), then re-embed using context + original text.
    Runs as a background task — returns immediately with task status.

    Query params:
        force=true  Re-contextualise chunks that already have context.
    """
    global _contextualize_task, _contextualize_progress

    if _contextualize_progress["running"]:
        return {
            "status":   "already_running",
            "progress": _contextualize_progress,
        }

    store = get_chunk_store()
    if len(store) == 0:
        return {"status": "error", "detail": "Chunk store is empty — ingest documents first"}

    from pipeline.contextualizer import contextualize_all

    async def _run():
        global _contextualize_progress
        _contextualize_progress = {"done": 0, "total": len(store), "running": True}

        async def _on_progress(done, total):
            _contextualize_progress["done"] = done
            _contextualize_progress["total"] = total
            if done % 50 == 0:
                await _broadcast({"event": "contextualize_progress",
                                  "data": {"done": done, "total": total}})

        result = await contextualize_all(
            store, save_path=CHUNK_STORE_PATH,
            on_progress=_on_progress, force=force,
        )
        _contextualize_progress["running"] = False
        await _broadcast({"event": "contextualize_complete", "data": result})
        print(f"[GraphRAG++] Contextualization complete: {result}")

    _contextualize_task = asyncio.create_task(_run())

    return {
        "status":       "started",
        "total_chunks": len(store),
        "force":        force,
        "note":         "Check /chunks/stats or WebSocket for progress",
    }


# ─── Ingestion — Reseed ────────────────────────────────────────────────────────

@app.post("/ingest/seed")
async def reseed():
    """Re-seed with sample AI/ML data (additive — doesn't clear custom nodes)."""
    result = seed_graph()
    graph  = get_graph()
    asyncio.create_task(_embed_all_nodes())
    save_graph(graph)
    await _broadcast({"event": "graph_reset", "data": result})
    return {"status": "seeded", **result}


# ─── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_graph(req: QueryRequest):
    await _broadcast({"event": "query_start", "data": {"question": req.question}})

    async def _on_step(step_name: str, step_data: dict):
        await _broadcast({
            "event": "pipeline_step",
            "data": {
                "step":  step_name,
                "label": step_data.get("label", step_name),
                "ms":    step_data.get("ms", 0),
            },
        })

    gen_state = {"start": None, "preview_parts": [], "preview_chars": 0, "last_emit": 0.0}
    PREVIEW_CAP      = 80
    EMIT_INTERVAL_S  = 0.2

    async def _on_token(delta: str, total_tokens: int):
        now = time.monotonic()
        if gen_state["start"] is None:
            gen_state["start"] = now
        if gen_state["preview_chars"] < PREVIEW_CAP:
            remain = PREVIEW_CAP - gen_state["preview_chars"]
            gen_state["preview_parts"].append(delta[:remain])
            gen_state["preview_chars"] += min(remain, len(delta))
        if now - gen_state["last_emit"] < EMIT_INTERVAL_S:
            return
        gen_state["last_emit"] = now
        elapsed = max(now - gen_state["start"], 1e-3)
        await _broadcast({
            "event": "generation_token",
            "data": {
                "tokens":         total_tokens,
                "tokens_per_sec": round(total_tokens / elapsed, 2),
                "preview":        "".join(gen_state["preview_parts"]),
                "elapsed_s":      round(elapsed, 2),
            },
        })

    result = await run_query(req.question, on_step=_on_step, on_token=_on_token)

    await _broadcast({"event": "query_complete", "data": {
        "question":       req.question,
        "total_ms":       result["total_ms"],
        "answer_preview": result["answer"][:100],
    }})
    return result


# ─── Metrics ───────────────────────────────────────────────────────────────────

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    graph  = get_graph()
    stats  = graph.get_stats()
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    llm_src = ("gemini"          if gemini_key else
               "llama_cpp_local" if os.getenv("LOCAL_LLAMA_CPP_URL") else
               "hf_api"          if os.getenv("HF_TOKEN") else "simulation")
    return MetricsResponse(
        **stats,
        uptime_s   = round(time.time() - _start_time, 1),
        llm_source = llm_src,
    )


# ─── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        graph = get_graph()
        await websocket.send_json({"event": "connected", "data": graph.get_stats()})
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_text("ping")
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ─── Helpers ───────────────────────────────────────────────────────────────────

async def _broadcast(msg: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


async def _embed_all_nodes():
    graph = get_graph()
    nodes = [(nid, data.get("label", nid)) for nid, data in graph.G.nodes(data=True)]
    batch_size = 32
    for i in range(0, len(nodes), batch_size):
        batch  = nodes[i:i + batch_size]
        labels = [b[1] for b in batch]
        try:
            embeddings = await embed_text(labels)
            for (nid, _), emb in zip(batch, embeddings):
                graph.set_embedding(nid, emb)
                add_to_index(nid, emb)
        except Exception as e:
            print(f"[Embedder] Batch failed: {e}")
        await asyncio.sleep(0)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
