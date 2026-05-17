"""
Retroactively chunk the Claude Mythos PDF into the ChunkStore.
Runs pdfplumber page-by-page, embeds with sentence-transformers,
and saves chunk_store.json.  No LLM calls — fast.
"""
import sys, os, asyncio, time

# Make sure backend packages are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

PDF_PATH         = r"C:\Users\ethan\Downloads\Claude Mythos Preview System Card (3).pdf"
CHUNK_STORE_PATH = r"C:\Users\ethan\OneDrive\Desktop\GraphRAG\backend\data\chunk_store.json"
SOURCE_NAME      = "Claude Mythos Preview System Card (3).pdf"


async def main():
    from pipeline.chunk_store import get_chunk_store

    store = get_chunk_store()
    store.load(CHUNK_STORE_PATH)
    print(f"Loaded existing store: {len(store)} chunks")

    if store.has_source(SOURCE_NAME):
        print(f"Source '{SOURCE_NAME}' already chunked — nothing to do.")
        return

    import io
    import pdfplumber

    t0 = time.time()
    with open(PDF_PATH, "rb") as f:
        content = f.read()

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        pages = pdf.pages
        total = len(pages)
        print(f"PDF: {total} pages")

        added = 0
        for page_num, page in enumerate(pages, 1):
            pt = page.extract_text()
            if not pt:
                continue
            n = await store.add_text(pt.strip(), source=SOURCE_NAME, page=page_num)
            added += n
            if page_num % 25 == 0 or page_num == total:
                elapsed = time.time() - t0
                print(f"  Page {page_num}/{total} — {added} chunks so far ({elapsed:.1f}s)")

    store.save(CHUNK_STORE_PATH)
    elapsed = time.time() - t0
    print(f"\nDone: {added} chunks added in {elapsed:.1f}s")
    print(f"Total store size: {len(store)} chunks")
    stats = store.get_stats()
    print(f"Sources: {list(stats['sources'].keys())}")


if __name__ == "__main__":
    asyncio.run(main())
