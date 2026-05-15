"""
Cross-encoder reranker — re-scores (query, passage) pairs with full attention.
Provides +33-40% accuracy over bi-encoder retrieval alone.
Falls back silently if the cross-encoder model is unavailable.
"""
import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None
_reranker_failed = False  # don't retry after a load failure


def _load_reranker():
    global _reranker, _reranker_failed
    if _reranker is not None or _reranker_failed:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info(f"[Reranker] Loaded {RERANKER_MODEL}")
    except Exception as e:
        logger.warning(f"[Reranker] Could not load cross-encoder ({e}) — skipping reranking")
        _reranker_failed = True
    return _reranker


async def rerank(query: str, chunks: List[dict], top_k: int = 5) -> List[dict]:
    """Re-score chunks against *query* using a cross-encoder.

    Takes up to len(chunks) candidates, returns top_k sorted by reranker score.
    If the cross-encoder isn't available, returns chunks[:top_k] unchanged.
    """
    if not chunks:
        return []

    reranker = _load_reranker()
    if reranker is None:
        return chunks[:top_k]

    pairs = [(query, c["text"]) for c in chunks]
    try:
        scores = await asyncio.to_thread(reranker.predict, pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        ranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        return ranked[:top_k]
    except Exception as e:
        logger.warning(f"[Reranker] Prediction failed ({e}) — using original order")
        return chunks[:top_k]
