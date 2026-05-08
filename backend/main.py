"""
GraphRAG++ FastAPI Backend
Serves the knowledge graph engine with REST + WebSocket APIs.
"""
import os
import time
import asyncio
import json
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from models import (
    IngestRequest, IngestURLRequest, QueryRequest,
    IngestResponse, GraphResponse, QueryResponse, MetricsResponse
)
from pipeline.graph_builder import get_graph
from pipeline.extractor import extract_triples
from pipeline.query_engine import query as run_query
from pipeline.embedder import embed_text, add_to_index
from pipeline.sample_data import seed_graph

load_dotenv()

_start_time = time.time()
_ws_clients: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: seed graph with sample data."""
    print("[GraphRAG++] Starting up...")
    result = seed_graph()
    print(f"[GraphRAG++] Graph seeded: {result}")
    
    # Pre-embed all nodes in background
    asyncio.create_task(_embed_all_nodes())
    yield
    print("[GraphRAG++] Shutting down.")


app = FastAPI(
    title="GraphRAG++ Knowledge Graph Fusion Engine",
    version="1.0.0",
    description="Hybrid knowledge graph with multi-hop reasoning and LLM-fused generation",
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
        "status": "ok",
        "uptime_s": round(time.time() - _start_time, 1),
        "graph_nodes": stats["node_count"],
        "graph_edges": stats["edge_count"],
        "version": stats["version"],
    }


# ─── Graph ─────────────────────────────────────────────────────────────────────

@app.get("/graph", response_model=GraphResponse)
async def get_graph_data():
    """Get full graph data for D3.js visualization."""
    graph = get_graph()
    return graph.to_json()


@app.get("/graph/stats")
async def get_stats():
    graph = get_graph()
    return graph.get_stats()


@app.post("/graph/reset")
async def reset_graph():
    """Reset and re-seed the graph."""
    result = seed_graph()
    asyncio.create_task(_embed_all_nodes())
    await _broadcast({"event": "graph_reset", "data": result})
    return {"status": "reset", **result}


# ─── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_text(req: IngestRequest):
    """Ingest raw text: extract triples and add to graph."""
    await _broadcast({"event": "ingest_start", "data": {"source": req.source}})
    
    triples = await extract_triples(req.text)
    
    # Stream each triple to WebSocket clients
    for triple in triples:
        await _broadcast({"event": "triple_extracted", "data": triple})
        await asyncio.sleep(0.05)  # small delay for streaming effect
    
    # Add to graph
    graph = get_graph()
    result = graph.ingest_triples(triples, source=req.source)
    
    # Embed new nodes
    asyncio.create_task(_embed_all_nodes())
    
    await _broadcast({"event": "ingest_complete", "data": result})
    
    return IngestResponse(
        status="ok",
        triples_extracted=len(triples),
        nodes_added=result["nodes_added"],
        edges_added=result["edges_added"],
        version=result["version"],
        triples=[{"subject": t["subject"], "predicate": t["predicate"], "object": t["object"], "confidence": t.get("confidence", 0.85)} for t in triples],
        source=req.source,
    )


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest uploaded file (.txt, .md, .pdf text)."""
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    
    req = IngestRequest(text=text[:10000], source=file.filename or "upload")
    return await ingest_text(req)


@app.post("/ingest/seed")
async def reseed():
    """Re-seed with sample AI/ML data."""
    result = seed_graph()
    asyncio.create_task(_embed_all_nodes())
    await _broadcast({"event": "graph_reset", "data": result})
    return {"status": "seeded", **result}


# ─── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_graph(req: QueryRequest):
    """Run a full GraphRAG++ query pipeline with real-time step broadcasting."""
    await _broadcast({"event": "query_start", "data": {"question": req.question}})

    async def _on_step(step_name: str, step_data: dict):
        """Broadcast each pipeline step to connected WebSocket clients immediately."""
        await _broadcast({
            "event": "pipeline_step",
            "data": {
                "step": step_name,
                "label": step_data.get("label", step_name),
                "ms": step_data.get("ms", 0),
            },
        })

    # Token-stream broadcaster — throttled to ~5 Hz so we don't flood the WS
    # with one frame per token. Each broadcast carries cumulative tokens,
    # tokens/sec since generation began, and the first ~80 chars as a preview.
    gen_state = {"start": None, "preview_parts": [], "preview_chars": 0,
                 "last_emit": 0.0}
    PREVIEW_CAP = 80
    EMIT_INTERVAL_S = 0.2

    async def _on_token(delta: str, total_tokens: int):
        now = time.monotonic()
        if gen_state["start"] is None:
            gen_state["start"] = now
        # Build a capped first-N-chars preview that grows then freezes
        if gen_state["preview_chars"] < PREVIEW_CAP:
            remain = PREVIEW_CAP - gen_state["preview_chars"]
            gen_state["preview_parts"].append(delta[:remain])
            gen_state["preview_chars"] += min(remain, len(delta))
        # Throttle outgoing frames
        if now - gen_state["last_emit"] < EMIT_INTERVAL_S:
            return
        gen_state["last_emit"] = now
        elapsed = max(now - gen_state["start"], 1e-3)
        await _broadcast({
            "event": "generation_token",
            "data": {
                "tokens": total_tokens,
                "tokens_per_sec": round(total_tokens / elapsed, 2),
                "preview": "".join(gen_state["preview_parts"]),
                "elapsed_s": round(elapsed, 2),
            },
        })

    result = await run_query(req.question, on_step=_on_step, on_token=_on_token)

    await _broadcast({"event": "query_complete", "data": {
        "question": req.question,
        "total_ms": result["total_ms"],
        "answer_preview": result["answer"][:100],
    }})

    return result


# ─── Metrics ───────────────────────────────────────────────────────────────────

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    graph = get_graph()
    stats = graph.get_stats()
    llm_source = os.getenv("HF_TOKEN", "") and "hf_api" or "simulation"
    return MetricsResponse(
        **stats,
        uptime_s=round(time.time() - _start_time, 1),
        llm_source=llm_source,
    )


# ─── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        # Send current graph state on connect
        graph = get_graph()
        await websocket.send_json({
            "event": "connected",
            "data": graph.get_stats(),
        })
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # Echo ping
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_text("ping")
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ─── Helpers ────────────────────────────────────────────────────────────────────

async def _broadcast(msg: dict):
    """Broadcast message to all connected WebSocket clients."""
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


async def _embed_all_nodes():
    """Background task: embed all graph nodes into FAISS index."""
    from pipeline.embedder import embed_text, add_to_index
    graph = get_graph()
    nodes = [(nid, data.get("label", nid)) for nid, data in graph.G.nodes(data=True)]
    
    batch_size = 32
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]
        labels = [b[1] for b in batch]
        try:
            embeddings = await embed_text(labels)
            for (nid, _), emb in zip(batch, embeddings):
                graph.set_embedding(nid, emb)
                add_to_index(nid, emb)
        except Exception as e:
            print(f"[Embedder] Batch embedding failed: {e}")
        await asyncio.sleep(0)  # yield to event loop


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
