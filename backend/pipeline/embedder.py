"""
Embedding Service — Sentence Transformers + FAISS hybrid index.
Default model: BAAI/bge-m3 (MTEB ~70, 1024-dim, hybrid dense+sparse).
Falls back to random embeddings if torch/sentence-transformers unavailable.
Set EMBED_MODEL env var to override (e.g. all-MiniLM-L6-v2 for fast dev).
"""
import os
import math
import random
import asyncio
from typing import List, Dict, Optional, Tuple

_model = None
_index = None
_index_ids: List[str] = []
_dim = 1024  # BAAI/bge-m3 dimension (updated from 384)

# HNSW tuning knobs — raise efSearch at query time for higher recall at cost of speed
_HNSW_M = 32               # bi-directional links per node (16–64; 32 is a safe default)
_HNSW_EF_CONSTRUCTION = 200  # beam width during graph construction (higher → better recall)
_HNSW_EF_SEARCH = 64        # beam width during search (can be overridden per query)

# BGE-M3 retrieval query prefix — improves dense retrieval recall
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")


def get_embedding_dim() -> int:
    """Return the current embedding dimension (depends on loaded model)."""
    return _dim


def _load_model():
    global _model, _dim
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        _dim = _model.get_sentence_embedding_dimension()
        print(f"[Embedder] Loaded {MODEL_NAME} (dim={_dim})")
    except Exception as e:
        print(f"[Embedder] Could not load ST model: {e}. Using random embeddings.")
        _model = None
    return _model


def _load_index():
    global _index
    if _index is not None:
        return _index
    try:
        import faiss
        _index = faiss.IndexHNSWFlat(_dim, _HNSW_M, faiss.METRIC_INNER_PRODUCT)
        _index.hnsw.efConstruction = _HNSW_EF_CONSTRUCTION
        print(f"[Embedder] FAISS HNSW index ready (dim={_dim}, M={_HNSW_M}, efC={_HNSW_EF_CONSTRUCTION})")
    except Exception as e:
        print(f"[Embedder] FAISS not available: {e}. Using brute-force search.")
        _index = None
    return _index


async def embed_text(texts: List[str]) -> List[List[float]]:
    """Embed a list of document/passage texts. Returns normalized vectors."""
    model = _load_model()
    if model is None:
        return [_random_embedding(_dim) for _ in texts]

    embeddings = await asyncio.to_thread(
        lambda: model.encode(texts, normalize_embeddings=True).tolist()
    )
    return embeddings


async def embed_query(query: str) -> List[float]:
    """Embed a single query string with the BGE retrieval prefix."""
    model = _load_model()
    if model is None:
        return _random_embedding(_dim)

    # Use retrieval prefix for BGE-M3 (no-op for other models that ignore it)
    prefixed = _BGE_QUERY_PREFIX + query if "bge" in MODEL_NAME.lower() else query
    results = await asyncio.to_thread(
        lambda: model.encode([prefixed], normalize_embeddings=True).tolist()
    )
    return results[0]


def add_to_index(node_id: str, embedding: List[float]):
    """Add an embedding to the FAISS index."""
    global _index_ids
    import numpy as np

    # Ensure model is loaded so _dim is set correctly before index creation
    _load_model()

    idx = _load_index()
    if idx is None:
        return

    # Reject mismatched embeddings (stale 384-dim from old model)
    if len(embedding) != _dim:
        return

    vec = np.array([embedding], dtype=np.float32)
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm > 0:
        vec = vec / norm

    idx.add(vec)
    _index_ids.append(node_id)


def search_index(query_embedding: List[float], k: int = 10) -> List[Tuple[str, float]]:
    """Search the FAISS index. Returns list of (node_id, score) tuples."""
    import numpy as np

    idx = _load_index()
    if idx is None or not _index_ids:
        return []

    if len(query_embedding) != _dim:
        return []

    idx.hnsw.efSearch = _HNSW_EF_SEARCH
    vec = np.array([query_embedding], dtype=np.float32)
    distances, indices = idx.search(vec, min(k, len(_index_ids)))

    results = []
    for dist, i in zip(distances[0], indices[0]):
        if i >= 0 and i < len(_index_ids):
            results.append((_index_ids[i], float(dist)))
    return results


def _random_embedding(dim: int) -> List[float]:
    """Generate random normalized embedding as fallback."""
    raw = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]
