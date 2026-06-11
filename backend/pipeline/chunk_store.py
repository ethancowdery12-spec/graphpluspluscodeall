"""
Chunk Store — hybrid passage retrieval over raw text chunks.

Combines dense cosine search (FAISS-style) with sparse BM25 retrieval,
merged via Reciprocal Rank Fusion (RRF). This replaces the brittle keyword-boost
heuristic with proper sparse retrieval that handles exact-match queries correctly.

Each chunk: { id, text, source, page, embedding }
"""
import json
import math
import re
import hashlib
import logging
import os
import tempfile
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CHUNK_SIZE    = 600   # characters per chunk
CHUNK_OVERLAP = 100   # overlap between consecutive windows
MAX_CHUNKS_PER_CALL = 600

_store: Optional["ChunkStore"] = None
_RRF_K = 60  # RRF constant — higher = smoother fusion, lower = rank-1 dominates
DEDUP_COSINE_THRESHOLD = 0.97  # skip new chunks whose embedding is this similar to an existing one


def get_chunk_store() -> "ChunkStore":
    global _store
    if _store is None:
        _store = ChunkStore()
    return _store


def _tokenize(text: str) -> List[str]:
    """Simple whitespace+punctuation tokenizer for BM25."""
    return [t.lower() for t in re.split(r"\W+", text) if len(t) >= 2]


class ChunkStore:
    def __init__(self):
        self._chunks: List[dict] = []
        self._source_set: set = set()
        self._chunk_id_set: set = set()  # O(1) dedup across the full store
        self._bm25 = None          # lazy-built BM25Plus index
        self._bm25_dirty = True    # rebuild BM25 on next search

    # ── Text splitting ─────────────────────────────────────────────────────────

    @staticmethod
    def split_text(text: str,
                   chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split *text* into overlapping character windows."""
        text = text.strip()
        if not text:
            return []
        windows: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            windows.append(text[start:end])
            if end == len(text):
                break
            start += chunk_size - overlap
        return windows

    # ── Ingestion ──────────────────────────────────────────────────────────────

    async def add_text(self, text: str, source: str, page: int = 0) -> int:
        """Chunk *text*, embed each window, and store. Returns new chunks added."""
        from .embedder import embed_text

        windows = self.split_text(text)
        if not windows:
            return 0
        windows = windows[:MAX_CHUNKS_PER_CALL]

        try:
            embeddings = await embed_text(windows)
        except Exception as e:
            logger.warning(f"[ChunkStore] Embedding failed ({source} p{page}): {e}")
            return 0

        added = 0
        for window, emb in zip(windows, embeddings):
            chunk_id = hashlib.md5(
                f"{source}:{page}:{window[:60]}".encode()
            ).hexdigest()[:12]
            if chunk_id in self._chunk_id_set:
                continue
            if emb and self._is_near_duplicate(emb):
                continue
            self._chunk_id_set.add(chunk_id)
            self._chunks.append({
                "id":        chunk_id,
                "text":      window,
                "source":    source,
                "page":      page,
                "embedding": emb,
            })
            added += 1

        if added:
            self._source_set.add(source)
            self._bm25_dirty = True  # invalidate BM25 index
        return added

    def has_source(self, source: str) -> bool:
        return source in self._source_set

    # ── BM25 (sparse) ─────────────────────────────────────────────────────────

    def _ensure_bm25(self):
        """Rebuild BM25 index if dirty. Called lazily before any BM25 search."""
        if not self._bm25_dirty and self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Plus
            corpus = [_tokenize(c["text"]) for c in self._chunks]
            self._bm25 = BM25Plus(corpus)
            self._bm25_dirty = False
        except ImportError:
            logger.warning("[ChunkStore] rank_bm25 not installed — BM25 disabled")
            self._bm25 = None

    def bm25_search(self, query_text: str, top_k: int = 20) -> List[dict]:
        """BM25 sparse retrieval. Returns ranked chunks with 'bm25_score' added."""
        if not self._chunks:
            return []
        self._ensure_bm25()
        if self._bm25 is None:
            return []

        tokens = _tokenize(query_text)
        scores = self._bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_idx[:top_k]:
            if scores[idx] > 0:
                results.append({**self._chunks[idx], "bm25_score": float(scores[idx])})
        return results

    # ── Dense cosine search ────────────────────────────────────────────────────

    def search(self, query_embedding: List[float], top_k: int = 20) -> List[dict]:
        """Cosine-similarity search over all chunks with 'score' added."""
        if not self._chunks:
            return []

        import numpy as np
        q = np.array(query_embedding, dtype=np.float32)
        norm_q = float(math.sqrt(float(q @ q)))
        if norm_q == 0:
            return []
        q_unit = q / norm_q

        scored: List[dict] = []
        for chunk in self._chunks:
            emb = chunk.get("embedding")
            if not emb or len(emb) != len(query_embedding):
                continue
            v = np.array(emb, dtype=np.float32)
            norm_v = float(math.sqrt(float(v @ v)))
            if norm_v == 0:
                continue
            score = float(q_unit @ (v / norm_v))
            scored.append({**chunk, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate near-identical windows
        deduped: List[dict] = []
        seen_snippets: set = set()
        for c in scored:
            snippet = c["text"][:80].lower().strip()
            if snippet not in seen_snippets:
                seen_snippets.add(snippet)
                deduped.append(c)
            if len(deduped) >= top_k:
                break
        return deduped

    # ── Hybrid BM25 + cosine via RRF ──────────────────────────────────────────

    def hybrid_search(self,
                      query_text: str,
                      query_embedding: List[float],
                      top_k: int = 20) -> List[dict]:
        """Hybrid retrieval: BM25 sparse + dense cosine, merged with RRF.

        RRF score = Σ 1 / (RRF_K + rank_i(doc))
        Handles both semantic similarity and exact keyword matches without the
        brittle "≥3 keyword hit" heuristic.
        """
        n_candidates = top_k * 4  # fetch wide net from each retriever
        dense_results = self.search(query_embedding, top_k=n_candidates)
        bm25_results  = self.bm25_search(query_text, top_k=n_candidates)

        # Build rank maps: chunk_id → 1-based rank
        dense_rank = {c["id"]: i + 1 for i, c in enumerate(dense_results)}
        bm25_rank  = {c["id"]: i + 1 for i, c in enumerate(bm25_results)}

        # Collect all unique chunks from both retrievers
        all_ids: Dict[str, dict] = {}
        for c in dense_results + bm25_results:
            all_ids.setdefault(c["id"], c)

        # Compute RRF score for each chunk
        rrf_scores: List[tuple] = []
        for cid, chunk in all_ids.items():
            rrf = 0.0
            if cid in dense_rank:
                rrf += 1.0 / (_RRF_K + dense_rank[cid])
            if cid in bm25_rank:
                rrf += 1.0 / (_RRF_K + bm25_rank[cid])
            # Carry both scores for transparency
            rrf_scores.append((rrf, {
                **chunk,
                "score":      chunk.get("score", 0.0),
                "bm25_score": chunk.get("bm25_score", 0.0),
                "rrf_score":  rrf,
            }))

        rrf_scores.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in rrf_scores[:top_k]]

    # ── Direct word search / fast paths ──────────────────────────────────────

    def exact_phrase_search(self, phrase: str) -> List[dict]:
        """Return all chunks containing *phrase* as a verbatim substring (case-insensitive)."""
        needle = phrase.lower().strip()
        if not needle:
            return []
        return [c for c in self._chunks if needle in c["text"].lower()]

    def bm25_fast_path(self, query_text: str, z_threshold: float = 2.5) -> Optional[dict]:
        """Return the top BM25 result if its score is z_threshold std-devs above the mean.

        When one document contains the exact terminology of the query and the rest do not,
        the top BM25 score is a clear outlier — we can skip dense retrieval and reranking
        entirely and return this hit as the sole passage.  Returns None when the scores are
        too close together to indicate a confident single-document match.
        """
        if len(self._chunks) < 5:
            return None
        self._ensure_bm25()
        if self._bm25 is None:
            return None
        tokens = _tokenize(query_text)
        if not tokens:
            return None
        scores = self._bm25.get_scores(tokens)
        positive = [s for s in scores if s > 0.0]
        if not positive:
            return None
        top_idx = max(range(len(scores)), key=lambda i: scores[i])
        # With 1-2 positive scores the entire corpus matched on nothing except
        # this one document — that IS the strongest possible specificity signal.
        if len(positive) <= 2:
            return {**self._chunks[top_idx],
                    "bm25_score": float(scores[top_idx]),
                    "bm25_z": 99.0}
        mean = sum(positive) / len(positive)
        variance = sum((s - mean) ** 2 for s in positive) / len(positive)
        std = math.sqrt(variance)
        if std < 1e-6:
            return None
        z = (scores[top_idx] - mean) / std
        if z >= z_threshold:
            return {**self._chunks[top_idx],
                    "bm25_score": float(scores[top_idx]),
                    "bm25_z": round(z, 2)}
        return None

    def has_relevant_content(self, query_text: str, query_embedding: List[float],
                              bm25_min: float = 0.5, cosine_min: float = 0.25) -> bool:
        """Return False when the corpus has no content relevant to this query.

        Used as an early-exit gate before the full pipeline runs — saves all
        downstream LLM calls for out-of-domain questions.
        """
        if not self._chunks:
            return False
        hits = self.bm25_search(query_text, top_k=1)
        if hits:
            # BM25 returned results — check the threshold
            if hits[0].get("bm25_score", 0.0) >= bm25_min:
                return True
        else:
            # BM25 returned nothing (corpus too small for IDF to be positive, or
            # no overlapping tokens). Fall through to cosine without counting this
            # as a "no match" signal — the corpus may still be highly relevant.
            pass
        try:
            dense = self.search(query_embedding, top_k=1)
            if dense and dense[0].get("score", 0.0) >= cosine_min:
                return True
        except Exception:
            pass
        # Both retrievers ran and found nothing above threshold.
        # If BM25 was unavailable (empty hits) and cosine also failed, do one
        # last check: if the corpus is non-empty and tiny (≤ 5 chunks), trust
        # that it was purposefully ingested and always allow the query through.
        if not hits and len(self._chunks) <= 2:
            return True
        return False

    def _is_near_duplicate(self, embedding: List[float]) -> bool:
        """Return True if any of the last 200 stored chunks is cosine-similar enough
        to count as a near-duplicate of *embedding*.

        Bounded to the most-recent 200 chunks so ingestion stays fast.
        Returns False silently when numpy is unavailable.
        """
        check = self._chunks[-200:]
        if not check:
            return False
        try:
            import numpy as np
        except ImportError:
            return False
        q = np.array(embedding, dtype=np.float32)
        norm_q = float(np.linalg.norm(q))
        if norm_q < 1e-8:
            return False
        q_unit = q / norm_q
        for chunk in check:
            emb = chunk.get("embedding")
            if not emb or len(emb) != len(embedding):
                continue
            v = np.array(emb, dtype=np.float32)
            norm_v = float(np.linalg.norm(v))
            if norm_v < 1e-8:
                continue
            if float(q_unit @ (v / norm_v)) >= DEDUP_COSINE_THRESHOLD:
                return True
        return False

    # ── Re-embedding support ──────────────────────────────────────────────────

    async def ensure_embeddings_valid(self, save_path: Optional[str] = None) -> bool:
        """Check stored embeddings match the current model dim; re-embed if not.

        Returns True if re-embedding was triggered, False if everything was fine.
        Called once at startup after the model is loaded.
        """
        from .embedder import get_embedding_dim, embed_text as _embed

        if not self._chunks:
            return False

        current_dim = get_embedding_dim()
        sample_emb = self._chunks[0].get("embedding")
        if sample_emb and len(sample_emb) == current_dim:
            return False  # embeddings are already compatible

        logger.info(
            f"[ChunkStore] Embedding dim mismatch "
            f"(stored={len(sample_emb) if sample_emb else 'none'}, "
            f"model={current_dim}). Re-embedding {len(self._chunks)} chunks…"
        )

        # Strip stale embeddings and re-embed in batches
        texts = [c["text"] for c in self._chunks]
        batch_size = 64
        new_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                embs = await _embed(batch)
                new_embeddings.extend(embs)
            except Exception as e:
                logger.error(f"[ChunkStore] Re-embed batch {i} failed: {e}")
                # Fill with None so indices stay aligned
                new_embeddings.extend([None] * len(batch))

        for chunk, emb in zip(self._chunks, new_embeddings):
            chunk["embedding"] = emb

        self._bm25_dirty = True
        logger.info(f"[ChunkStore] Re-embedding complete.")

        if save_path:
            self.save(save_path)

        return True

    # ── Stats ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._chunks)

    def get_stats(self) -> dict:
        sources: dict = {}
        for c in self._chunks:
            sources[c["source"]] = sources.get(c["source"], 0) + 1
        return {
            "total_chunks": len(self._chunks),
            "unique_sources": len(sources),
            "sources": sources,
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Atomic write to *path*. Embeddings stored inline for instant loads."""
        dir_ = os.path.dirname(path) or "."
        os.makedirs(dir_, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"chunks": self._chunks}, fh, ensure_ascii=False)
            os.replace(tmp, path)
            logger.info(f"[ChunkStore] Saved {len(self._chunks)} chunks → {path}")
        except Exception as e:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            logger.error(f"[ChunkStore] Save failed: {e}")

    def load(self, path: str) -> None:
        """Load from *path*. Safe to call when the file doesn't exist yet."""
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._chunks = data.get("chunks", [])
            self._source_set = {c["source"] for c in self._chunks}
            self._chunk_id_set = {c["id"] for c in self._chunks if "id" in c}
            self._bm25_dirty = True
            logger.info(f"[ChunkStore] Loaded {len(self._chunks)} chunks from {path}")
        except FileNotFoundError:
            logger.info(f"[ChunkStore] No chunk store at {path} — starting fresh")
        except Exception as e:
            logger.warning(f"[ChunkStore] Load failed ({e}) — starting fresh")
