"""
Directory Walker — walk a local path, hash each file, skip already-ingested
ones, yield (triples, sha256, filepath) for each new file.
"""
import os
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, Set, Tuple, List

from .code_extractor import LANG_MAP
from .file_router import route_file, TEXT_EXTENSIONS, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

# All extensions we can handle
SUPPORTED_EXTENSIONS: Set[str] = (
    set(LANG_MAP.keys())
    | TEXT_EXTENSIONS
    | IMAGE_EXTENSIONS
    | {".pdf", ".zip", ".ipynb"}
)

# Directories to always skip
SKIP_DIRS: Set[str] = {
    "__pycache__", ".git", ".hg", ".svn",
    "node_modules", ".next", "dist", "build", "out",
    ".venv", "venv", "env", ".env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "coverage", ".coverage", "htmlcov",
    ".idea", ".vscode",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_FILES_PER_WALK = 2000         # safety ceiling


async def walk_directory(
    dir_path: str,
    existing_hashes: Set[str],
) -> AsyncGenerator[Tuple[List[dict], str, str], None]:
    """
    Async generator: yields (triples, sha256, filepath) for each file that:
      - Has a supported extension
      - Is under the size limit
      - Has NOT been ingested before (SHA256 not in existing_hashes)

    Skips common build/cache directories automatically.
    """
    root = Path(dir_path).resolve()
    if not root.exists():
        logger.error(f"[DirWalker] Path not found: {dir_path}")
        return
    if not root.is_dir():
        # Single file passed
        content = root.read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
        if sha256 not in existing_hashes:
            triples, sha256 = await route_file(content, root.name)
            yield triples, sha256, str(root)
        return

    count = 0
    for dirpath, dirnames, filenames in os.walk(str(root)):
        # Prune skip dirs in-place (os.walk respects this)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]

        for fname in filenames:
            if count >= MAX_FILES_PER_WALK:
                logger.warning(f"[DirWalker] Reached {MAX_FILES_PER_WALK} file cap — stopping.")
                return

            fpath = Path(dirpath) / fname
            ext   = fpath.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                continue

            try:
                size = fpath.stat().st_size
                if size == 0 or size > MAX_FILE_SIZE:
                    continue

                content = fpath.read_bytes()
                sha256  = hashlib.sha256(content).hexdigest()

                if sha256 in existing_hashes:
                    logger.debug(f"[DirWalker] Skipping (already ingested): {fpath.name}")
                    continue

                rel_path = str(fpath.relative_to(root))
                triples, sha256 = await route_file(content, rel_path)
                count += 1

                yield triples, sha256, str(fpath)
                await asyncio.sleep(0)   # yield event loop between files

            except (PermissionError, OSError) as e:
                logger.debug(f"[DirWalker] Skipping {fpath}: {e}")
                continue
