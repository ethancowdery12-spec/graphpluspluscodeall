"""
File Router — dispatches any file type to the right extraction pipeline.
Returns (triples, sha256_hex).

Routes:
  Code files  → code_extractor  (AST, tier=EXTRACTED)
  PDF         → pdfplumber text → LLM triples  (tier=INFERRED)
  Images      → llama-mtmd-cli vision → LLM triples  (tier=INFERRED/AMBIGUOUS)
  ZIP         → extract → recurse (depth-limited)
  Text/MD     → LLM triples  (tier=INFERRED)
"""
import hashlib
import logging
import shutil
import tempfile
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional

from .code_extractor import LANG_MAP, extract_code_triples, extract_code_chunks
from .extractor import extract_triples

logger = logging.getLogger(__name__)

# Files that go to the LLM text extractor
TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml", ".toml", ".html", ".ipynb"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MAX_ZIP_DEPTH = 2
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


async def route_file(
    content: bytes,
    filename: str,
    _depth: int = 0,
) -> Tuple[List[dict], str]:
    """
    Dispatch *content* / *filename* to the correct extractor.
    Returns (triples, sha256_hex).
    """
    sha256 = hashlib.sha256(content).hexdigest()
    ext = Path(filename).suffix.lower()

    if len(content) > MAX_FILE_SIZE:
        logger.warning(f"[Router] Skipping {filename} — {len(content)//1024}KB exceeds limit")
        return [], sha256

    try:
        if ext in LANG_MAP:
            source_text = content.decode("utf-8", errors="replace")
            triples = extract_code_triples(source_text, filename)
            # Also add semantic function/class bodies to ChunkStore so
            # hybrid retrieval can find code by natural-language description.
            from .chunk_store import get_chunk_store
            store = get_chunk_store()
            for chunk in extract_code_chunks(source_text, filename):
                await store.add_text(
                    chunk["text"], source=filename, page=chunk.get("page", 0)
                )
        elif ext == ".pdf":
            triples = await _handle_pdf(content, filename)
        elif ext in IMAGE_EXTENSIONS:
            triples = await _handle_image(content, filename)
        elif ext == ".zip" and _depth < MAX_ZIP_DEPTH:
            triples = await _handle_zip(content, filename, _depth)
        elif ext in TEXT_EXTENSIONS or ext == "":
            text = content.decode("utf-8", errors="replace")
            # Add raw text to ChunkStore before LLM extraction
            from .chunk_store import get_chunk_store
            await get_chunk_store().add_text(text, source=filename)
            triples = await extract_triples(text[:6000])
            for t in triples:
                t.setdefault("confidence_tier", "INFERRED")
        else:
            # Unknown binary — create a minimal node so the file is tracked
            triples = [{"subject": filename, "predicate": "is_a", "object": "file",
                        "confidence": 0.99, "confidence_tier": "EXTRACTED"}]

    except Exception as e:
        logger.error(f"[Router] Extraction error for {filename}: {e}")
        triples = []

    return triples, sha256


# ─── PDF ─────────────────────────────────────────────────────────────────────

async def _handle_pdf(content: bytes, filename: str) -> List[dict]:
    try:
        import pdfplumber
        import io

        page_texts: List[str] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                pt = page.extract_text()
                if pt:
                    page_texts.append(pt.strip())

        if not page_texts:
            return [{"subject": filename, "predicate": "is_a", "object": "scanned_pdf",
                     "confidence": 0.8, "confidence_tier": "AMBIGUOUS"}]

        # ── Passage chunking (fast, no LLM) ───────────────────────────────────
        # Add each page as raw chunks to the ChunkStore so passage-retrieval
        # queries can find exact prose.  Done first so it succeeds even if
        # triple extraction later times out.
        from .chunk_store import get_chunk_store
        store = get_chunk_store()
        total_added = 0
        for page_num, page_text in enumerate(page_texts):
            n = await store.add_text(page_text, source=filename, page=page_num + 1)
            total_added += n
            await asyncio.sleep(0)   # yield between pages
        logger.info(f"[PDF] {filename}: added {total_added} passage chunks to ChunkStore")
        # ── End passage chunking ───────────────────────────────────────────────

        full_text = "\n\n".join(page_texts)

        # Chunk into 3000-char windows, extract triples from each, dedupe.
        # For long documents, sample MAX_CHUNKS evenly across the full text
        # so we get proportional coverage (not just the first N pages).
        CHUNK = 3000
        MAX_CHUNKS = 40          # up to 120 000 chars sequentially; more via sampling
        total_len = len(full_text)
        if total_len <= CHUNK * MAX_CHUNKS:
            starts = list(range(0, total_len, CHUNK))
        else:
            step = (total_len - CHUNK) / max(MAX_CHUNKS - 1, 1)
            starts = [int(i * step) for i in range(MAX_CHUNKS)]
        chunks = [full_text[s:s + CHUNK] for s in starts]

        seen = set()
        all_triples: List[dict] = []
        for idx, chunk in enumerate(chunks):
            try:
                chunk_triples = await extract_triples(chunk)
                for t in chunk_triples:
                    t.setdefault("confidence_tier", "INFERRED")
                    key = (t["subject"].lower(), t["predicate"], t["object"].lower())
                    if key not in seen:
                        seen.add(key)
                        all_triples.append(t)
                logger.info(f"[PDF] Chunk {idx+1}/{len(chunks)}: {len(chunk_triples)} triples (total {len(all_triples)})")
            except Exception as chunk_err:
                logger.warning(f"[PDF] Chunk {idx+1}/{len(chunks)} failed: {chunk_err}")
            await asyncio.sleep(0)

        logger.info(f"[PDF] {filename}: extracted {len(all_triples)} unique triples from {len(chunks)} chunks")
        return all_triples

    except ImportError:
        logger.warning("[Router] pdfplumber not installed — cannot extract PDF text")
        return []
    except Exception as e:
        logger.warning(f"[Router] PDF extraction failed for {filename}: {e}")
        return []


# ─── Images ──────────────────────────────────────────────────────────────────

async def _handle_image(content: bytes, filename: str) -> List[dict]:
    """Try llama-mtmd-cli vision, fall back to filename stub."""
    caption = await _call_vision(content, filename)
    if caption:
        triples = await extract_triples(caption[:3000])
        for t in triples:
            t.setdefault("confidence_tier", "INFERRED")
        # Also add a node for the image itself
        triples.insert(0, {"subject": filename, "predicate": "is_a", "object": "image_file",
                           "confidence": 0.99, "confidence_tier": "EXTRACTED"})
        triples.insert(1, {"subject": filename, "predicate": "described_as", "object": caption[:200],
                           "confidence": 0.8, "confidence_tier": "INFERRED"})
        return triples
    return [{"subject": filename, "predicate": "is_a", "object": "image_file",
             "confidence": 0.99, "confidence_tier": "EXTRACTED"}]


async def _call_vision(content: bytes, filename: str) -> Optional[str]:
    """
    Shell out to llama-mtmd-cli.exe for image captioning.
    Set VISION_BIN and VISION_MODEL_PATH in .env to enable.
    """
    import os
    VISION_BIN   = os.getenv("VISION_BIN", "")
    VISION_MODEL = os.getenv("VISION_MODEL_PATH", "")

    if not VISION_BIN or not VISION_MODEL:
        return None
    if not Path(VISION_BIN).exists() or not Path(VISION_MODEL).exists():
        return None

    tmp_dir = tempfile.mkdtemp()
    try:
        img_path = Path(tmp_dir) / Path(filename).name
        img_path.write_bytes(content)
        prompt = ("Describe this image in detail. List all visible entities, "
                  "text, diagrams, relationships, and key concepts as a paragraph.")
        cmd = [VISION_BIN, "-m", VISION_MODEL,
               "--image", str(img_path),
               "-p", prompt,
               "-n", "256", "--temp", "0.1"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=90.0)
            return stdout.decode("utf-8", errors="replace").strip() or None
        except asyncio.TimeoutError:
            proc.kill()
            return None
    except Exception as e:
        logger.warning(f"[Vision] Failed for {filename}: {e}")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── ZIP ─────────────────────────────────────────────────────────────────────

async def _handle_zip(content: bytes, filename: str, depth: int) -> List[dict]:
    """Extract ZIP in memory, recurse into each file."""
    import zipfile
    import io
    all_triples: List[dict] = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            # Safety check: no path traversal
            for member in zf.namelist():
                if ".." in member or member.startswith("/"):
                    continue
                # Skip very large members
                info = zf.getinfo(member)
                if info.file_size > MAX_FILE_SIZE:
                    continue
                if not member.endswith("/"):
                    member_bytes = zf.read(member)
                    sub_triples, _ = await route_file(member_bytes, Path(member).name, _depth=depth + 1)
                    all_triples.extend(sub_triples)
                    await asyncio.sleep(0)  # yield event loop
    except zipfile.BadZipFile:
        pass
    except Exception as e:
        logger.warning(f"[Router] ZIP extraction failed for {filename}: {e}")

    return all_triples
