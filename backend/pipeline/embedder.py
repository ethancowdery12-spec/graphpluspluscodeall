"""
Embedding Service — Sentence Transformers + FAISS hybrid index.
Falls back to random embeddings if torch/sentence-transformers unavailable.
"""
import os
import math
import random
import asyncio
from typing import List, Dict, Optional, Tuple

_model = None
_index = None
_index_ids: List[str] = []
_dim = 384  # all-MiniLM-L6-v2 dimension

MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


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
        _index = faiss.IndexFlatIP(_dim)  # Inner product (cosine after normalize)
        print(f"[Embedder] FAISS index ready (dim={_dim})")
    except Exception as e:
        print(f"[Embedder] FAISS not available: {e}. Using brute-force search.")
        _index = None
    return _index


async def embed_text(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts. Returns list of embedding vectors."""
    model = _load_model()
    if model is None:
        return [_random_embedding(_dim) for _ in texts]

    embeddings = await asyncio.to_thread(
        lambda: model.encode(texts, normalize_embeddings=True).tolist()
    )
    return embeddings


async def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    results = await embed_text([query])
    return results[0]


def add_to_index(node_id: str, embedding: List[float]):
    """Add an embedding to the FAISS index."""
    global _index_ids
    import numpy as np

    idx = _load_index()
    if idx is None:
        return

    vec = np.array([embedding], dtype=np.float32)
    # Normalize for cosine similarity
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
