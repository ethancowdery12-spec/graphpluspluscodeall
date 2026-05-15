"""
Contextual Chunk Embedding — Anthropic's Contextual Retrieval approach.

For each stored chunk, generates a 50-100 token LLM context blurb describing
what the chunk is about and where it fits in the source document, then
re-embeds using (context + original text) as the input. This improves
retrieval quality by 35-67% by giving the embedding model richer signal
than the raw chunk text alone.

Reference: Anthropic blog "Introducing Contextual Retrieval"
"""
import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

CONTEXT_SYSTEM = (
    "You are a document analyst. Given a text passage, write 1-2 sentences that "
    "describe what this passage is about and what key information it contains. "
    "Be specific and factual. Do not copy the passage. Do not use phrases like "
    "'This passage discusses' — just state the facts directly."
)

CONTEXT_MAX_TOKENS = 80
CONTEXT_CONCURRENCY = 4  # parallel LLM calls; keep low for local GPU


async def _generate_context(chunk_text: str, source: str) -> str:
    """Generate a short context blurb for a single chunk using the LLM."""
    from .llm import call_llm

    # Give the LLM the first 500 chars of the chunk (enough for context inference)
    snippet = chunk_text[:500]
    prompt = f"Source: {os.path.basename(source)}\n\nPassage:\n{snippet}"

    try:
        result = await call_llm(
            prompt,
            system=CONTEXT_SYSTEM,
            max_tokens=CONTEXT_MAX_TOKENS,
            extra_stop=[".\n\n", "\n\n"],
        )
        ctx = result.get("answer", "").strip()
        # Strip any template bleed-through from local model
        for artifact in ["### Response:", "### Instruction:", "### Input:"]:
            if artifact in ctx:
                ctx = ctx[:ctx.index(artifact)].strip()
        return ctx if ctx else ""
    except Exception as e:
        logger.warning(f"[Contextualizer] Context generation failed for {source}: {e}")
        return ""


async def contextualize_all(
    chunk_store,
    save_path: Optional[str] = None,
    on_progress=None,
    force: bool = False,
) -> dict:
    """Generate context blurbs for all chunks and re-embed with enriched text.

    Args:
        chunk_store: The ChunkStore instance to operate on.
        save_path:   If provided, saves after completion.
        on_progress: Optional async callable(done, total) for progress reporting.
        force:       Re-contextualize chunks that already have a context field.

    Returns dict with counts: {"contextualized": N, "skipped": N, "failed": N}
    """
    from .embedder import embed_text

    chunks = chunk_store._chunks
    total  = len(chunks)

    if total == 0:
        return {"contextualized": 0, "skipped": 0, "failed": 0}

    to_process = [
        (i, c) for i, c in enumerate(chunks)
        if force or not c.get("context")
    ]
    already_done = total - len(to_process)

    logger.info(
        f"[Contextualizer] {len(to_process)} chunks to contextualize "
        f"({already_done} already have context, force={force})"
    )

    sem = asyncio.Semaphore(CONTEXT_CONCURRENCY)
    done_count = already_done
    failed = 0

    async def process_chunk(idx: int, chunk: dict):
        nonlocal done_count, failed
        async with sem:
            ctx = await _generate_context(chunk["text"], chunk.get("source", ""))
            if ctx:
                # Re-embed BEFORE writing context so the chunk stays consistent:
                # either both context and embedding are updated, or neither is.
                enriched = f"{ctx}\n\n{chunk['text']}"
                try:
                    embs = await embed_text([enriched])
                    chunk["context"] = ctx
                    chunk["embedding"] = embs[0]
                except Exception as e:
                    logger.warning(f"[Contextualizer] Re-embed failed idx={idx}: {e}")
                    failed += 1
            else:
                failed += 1

            done_count += 1
            if on_progress:
                try:
                    await on_progress(done_count, total)
                except Exception:
                    pass

    await asyncio.gather(*[process_chunk(i, c) for i, c in to_process])

    # BM25 is built on chunk["text"] which contextualizer never modifies,
    # so no rebuild is needed.

    n_success = len([c for _, c in to_process if c.get("context")])
    logger.info(
        f"[Contextualizer] Done. {n_success} contextualized, "
        f"{failed} failed, {already_done} skipped."
    )

    if save_path:
        chunk_store.save(save_path)

    return {
        "contextualized": n_success,
        "skipped":        already_done,
        "failed":         failed,
        "total":          total,
    }
