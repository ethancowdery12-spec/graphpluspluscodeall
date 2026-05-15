"""
Multi-hop Query Engine
Intent classification → Semantic node matching → Graph traversal → Context fusion → Generation
"""
import time
import json
import asyncio
from typing import List, Optional
from .llm import call_llm
from .graph_builder import get_graph
from .embedder import embed_query, search_index
from .chunk_store import get_chunk_store
from .reranker import rerank

RAG_FUSION_SYSTEM = (
    "You are a query rewriter. Given a question, output exactly 3 alternative "
    "phrasings that ask for the same information using different words or angles. "
    "Output only the 3 questions, one per line, no numbering, no extra text."
)

_RRF_K_FUSION = 60  # RRF constant for RAG-Fusion cross-query merge

INTENT_SYSTEM = """You are a query intent classifier for a knowledge graph system.
Classify the query into one of: factual, multi-hop, aggregative, comparative.
Extract every named entity / proper noun / capitalized acronym from the query
into the entities array. ALWAYS include at least one entity if any noun is
present. Reply with a single JSON object on one line and nothing else.
Example: {"intent": "factual", "entities": ["RAG"], "confidence": 0.9}"""

# ── Generation system prompts (selected based on available context) ────────────

GENERATION_SYSTEM_HYBRID = (
    "You are a knowledgeable assistant. Read the retrieved passages and graph paths "
    "below, then write a direct, accurate answer in 2-3 paragraphs. "
    "Base your answer on the passage text. Do not echo or copy the passages — "
    "summarize and explain what happened. Do not use headers. Do not restate the question."
)

GENERATION_SYSTEM_WITH_PATHS = (
    "You are a knowledge graph reasoning assistant. "
    "Give a direct, accurate answer in 2-3 short paragraphs. "
    "Use the provided graph paths as evidence. "
    "If the graph is incomplete, fill the gap with your own knowledge naturally — "
    "never mention missing paths, graph limitations, or what the graph lacks. "
    "Do not use markdown headers (#, ##). Do not restate the question."
)

GENERATION_SYSTEM_PASSAGES_ONLY = (
    "You are a knowledgeable assistant. Read the retrieved passages below, "
    "then write a direct, accurate answer in 2-3 paragraphs. "
    "Summarize and explain — do not copy the passage text verbatim. "
    "Do not use markdown headers. Do not restate the question."
)

GENERATION_SYSTEM_NO_PATHS = (
    "You are a knowledgeable assistant. "
    "Give a direct, accurate answer in 2-3 short paragraphs. "
    "Do not use markdown headers. Do not restate the question."
)

# ── Generation prompt templates ────────────────────────────────────────────────

GENERATION_PROMPT_HYBRID = """Question: {question}

Source passages from document:
---
{passages}
---

Graph relationships:
{paths}"""

GENERATION_PROMPT_WITH_PATHS = """Question: {question}

Graph relationships:
{paths}"""

GENERATION_PROMPT_PASSAGES_ONLY = """Question: {question}

Source passages from document:
---
{passages}
---"""

GENERATION_PROMPT_NO_PATHS = "Question: {question}"

# ── Self-RAG critique ──────────────────────────────────────────────────────────

CRITIQUE_SYSTEM = (
    "You are a factual accuracy critic. Given a question, retrieved passages, and a draft answer, "
    "score the answer on groundedness (how much is supported by the passages) and "
    "completeness (how fully it answers the question). "
    "Reply with a single JSON object on one line and nothing else. "
    'Example: {"groundedness": 0.85, "completeness": 0.7, "critique": "Missing X"}'
)


async def query(question: str, on_step=None, on_token=None) -> dict:
    """Full query pipeline: intent → match → traverse → fuse → generate.

    Args:
        question: The user's question string.
        on_step: Optional async callable(step_name, step_data) called immediately
                 after each pipeline step completes. Used for real-time WS events.
        on_token: Optional async callable(delta_text, total_tokens) called for
                 each generated token chunk during the *generation* step.
                 Intent classification is intentionally non-streamed (short).
    """
    graph = get_graph()
    graph.increment_query_count()
    t0 = time.time()
    steps = []

    async def _emit(step_name: str, step_data: dict):
        if on_step:
            try:
                await on_step(step_name, step_data)
            except Exception:
                pass  # never let broadcast errors kill the pipeline

    # ── Step 1: Intent Classification ─────────────────────────────────────────
    # Cap at 80 tokens — the response is a one-line JSON object. A 256-token
    # cap with no JSON-end stop sequence caused the previous 38s blow-up.
    intent_result = await call_llm(
        question, system=INTENT_SYSTEM, max_tokens=80,
        # Stop the moment the JSON closes — prevents the model from rambling
        # past the closing brace and burning the full token budget.
        extra_stop=["}\n", "}\r\n", "\n\n", "###"],
    )
    intent_data = _parse_intent(intent_result["answer"])
    # Always backfill entities from the question itself — the small fine-tuned
    # model often returns entities=[] even when "RAG", "BERT" etc. are in the query.
    if not intent_data.get("entities"):
        intent_data["entities"] = _extract_entities_heuristic(question)
    s1 = {
        "step": "intent_classification",
        "label": "Intent Parser",
        "icon": "🧠",
        "result": intent_data,
        "thinking": intent_result.get("thinking", ""),
        "ms": int((time.time() - t0) * 1000),
    }
    steps.append(s1)
    await _emit("intent_classification", s1)
    t1 = time.time()

    # ── RAG-Fusion (multi-hop / comparative queries only) ─────────────────────
    # Generate 3 query variants and retrieve candidates for each variant, then
    # RRF-merge all candidate lists with the original query's results.
    # Skipped for factual queries to avoid unnecessary LLM + retrieval overhead.
    intent = intent_data.get("intent", "factual")
    fusion_queries: List[str] = [question]  # always include original

    if intent in ("multi-hop", "comparative"):
        try:
            variant_result = await call_llm(
                question, system=RAG_FUSION_SYSTEM,
                max_tokens=120, extra_stop=["\n\n", "###"],
            )
            raw_variants = variant_result.get("answer", "")
            parsed = [
                line.strip().lstrip("0123456789.-) ")
                for line in raw_variants.splitlines()
                if line.strip() and len(line.strip()) > 10
            ]
            fusion_queries.extend(parsed[:3])
        except Exception:
            pass  # if variant generation fails, fall back to original query only

    # ── Step 2: Semantic Node Matching ─────────────────────────────────────────
    # Embed the query ONCE; reuse the same vector for both node search (Step 2)
    # and passage search (Step 3) so we pay the embedding cost only once.
    query_embedding = await embed_query(question)
    semantic_hits = search_index(query_embedding, k=10)

    # Also match by entity names from intent
    start_nodes = [nid for nid, _ in semantic_hits[:5]]
    for entity in intent_data.get("entities", []):
        matched = graph.find_nodes_by_name(entity)
        start_nodes.extend(matched[:2])
    start_nodes = list(dict.fromkeys(start_nodes))[:5]  # unique, max 5

    s2 = {
        "step": "node_matching",
        "label": "Entity Resolver",
        "icon": "🔍",
        "result": {
            "matched_nodes": len(start_nodes),
            "semantic_hits": len(semantic_hits),
            "start_nodes": start_nodes[:4],
        },
        "ms": int((time.time() - t1) * 1000),
    }
    steps.append(s2)
    await _emit("node_matching", s2)
    t2 = time.time()

    # ── Step 3: Passage Retrieval (hybrid BM25+cosine+RRF, optionally fused) ───
    # For multi-hop/comparative queries: embed all fusion variants in parallel,
    # run hybrid_search for each, then RRF-merge all candidate lists.
    # For factual queries: single hybrid_search on the original query only.
    chunk_store = get_chunk_store()

    if len(fusion_queries) > 1:
        # Parallel embed all variant queries
        variant_embeddings = await asyncio.gather(
            *[embed_query(q) for q in fusion_queries]
        )
        # Parallel hybrid search for each variant
        variant_hits_lists = await asyncio.gather(
            *[
                asyncio.to_thread(chunk_store.hybrid_search, q, emb, 20)
                for q, emb in zip(fusion_queries, variant_embeddings)
            ]
        )
        # RRF-merge all candidate lists across queries
        raw_hits = _rag_fusion_merge(variant_hits_lists, top_k=20)
    else:
        raw_hits = chunk_store.hybrid_search(question, query_embedding, top_k=20)

    passage_hits = await rerank(question, raw_hits, top_k=5)

    s3 = {
        "step": "passage_retrieval",
        "label": "Hybrid Retrieval + Rerank",
        "icon": "📄",
        "result": {
            "passages_found":          len(passage_hits),
            "chunk_store_size":        len(chunk_store),
            "candidates_before_rerank": len(raw_hits),
            "fusion_queries":          len(fusion_queries),
            "top_score":               round(passage_hits[0].get("rrf_score", passage_hits[0].get("score", 0)), 3) if passage_hits else 0,
        },
        "ms": int((time.time() - t2) * 1000),
    }
    steps.append(s3)
    await _emit("passage_retrieval", s3)
    t3 = time.time()

    # ── Step 4: Multi-hop Graph Traversal ─────────────────────────────────────
    max_hops = 3 if intent == "multi-hop" else 2
    paths = graph.multi_hop_paths(start_nodes, max_hops=max_hops, top_k=8)

    s4 = {
        "step": "graph_traversal",
        "label": "Multi-hop Traversal",
        "icon": "🕸️",
        "result": {
            "paths_found": len(paths),
            "max_hops": max_hops,
            "intent": intent,
        },
        "paths": paths,
        "ms": int((time.time() - t3) * 1000),
    }
    steps.append(s4)
    await _emit("graph_traversal", s4)
    t4 = time.time()

    # ── Step 5: Context Fusion ─────────────────────────────────────────────────
    path_text    = _paths_to_text(paths)
    passage_text = _passages_to_text(passage_hits)

    s5 = {
        "step": "context_fusion",
        "label": "Context Fusion",
        "icon": "⚡",
        "result": {
            "fused_tokens":   len((path_text + passage_text).split()),
            "paths_used":     len(paths),
            "passages_used":  len(passage_hits),
        },
        "ms": int((time.time() - t4) * 1000),
    }
    steps.append(s5)
    await _emit("context_fusion", s5)
    t5 = time.time()

    # ── Step 6: Generation ─────────────────────────────────────────────────────
    # Select prompt template based on what evidence is available.
    # Passages take priority for specific facts / narrative continuations;
    # graph paths cover structural / multi-hop questions.
    # Build instruction (task directive) + input (evidence) separately so the
    # Alpaca fine-tuned model sees a clean ### Instruction / ### Input split.
    # The question goes in the instruction, context goes in the input.
    # This prevents the model from pattern-matching on passage headers and
    # copying them verbatim as its response.
    if path_text and passage_text:
        gen_instruction = (
            f"Answer this question accurately in 2-3 paragraphs, "
            f"based on the source passages and graph relationships below. "
            f"Question: {question}"
        )
        gen_input = (
            f"Source passages:\n{passage_text[:2000]}\n\n"
            f"Graph relationships:\n{path_text[:800]}"
        )
    elif passage_text:
        gen_instruction = (
            f"Answer this question accurately in 2-3 paragraphs, "
            f"based on the source passages below. "
            f"Question: {question}"
        )
        gen_input = f"Source passages:\n{passage_text[:3000]}"
    elif path_text:
        gen_instruction = (
            f"Answer this question accurately in 2-3 paragraphs, "
            f"using the graph relationships below as evidence. "
            f"Question: {question}"
        )
        gen_input = f"Graph relationships:\n{path_text[:2000]}"
    else:
        gen_instruction = f"Answer this question accurately: {question}"
        gen_input = ""

    # ── Document-retrieval shortcut ────────────────────────────────────────────
    # "Continue this section/passage: [quote]" queries are document-retrieval
    # requests, not synthesis tasks. The LLM would confabulate (paraphrase with
    # wrong details) rather than returning the verbatim source text. Skip LLM
    # entirely and go straight to passage extraction for these.
    import re as _re_q
    _is_continue_query = bool(_re_q.match(
        r'^(?:continue|what comes after|what follows|finish this)[^a-z]',
        question.strip(), flags=_re_q.IGNORECASE,
    ))

    if _is_continue_query:
        # Seed a fake gen_result that will immediately hit the fallback.
        gen_result = {"answer": "", "thinking": "", "source": "shortcut"}
    else:
        gen_result = await call_llm(
            gen_input, instruction=gen_instruction, max_tokens=512,
            on_token=on_token,
            extra_stop=["### Response:", "### Instruction:", "Source passages:"],
        )

    # ── Fallback: detect corrupted generation ──────────────────────────────────
    # The local LLM sometimes loops or echoes the prompt. Detect this and fall
    # back to presenting the top passage directly so the user still gets the
    # right information even when synthesis fails.
    answer_raw = gen_result.get("answer", "")
    # Also treat simulation mode as "no real answer" — if the real LLM is down
    # we'd rather show the actual retrieved passage than a generic fake answer.
    llm_is_fake = gen_result.get("source") == "simulation"
    if llm_is_fake or _answer_is_corrupted(answer_raw, passage_text, path_text, instruction=gen_instruction):
        if passage_hits:
            top        = passage_hits[0]
            top_source = top.get("source", "")
            top_page   = top.get("page", 0)

            # Pull ALL chunks from this source for this page AND the immediately
            # following page, in document order (insertion order = document order
            # since we added them page-by-page, chunk-by-chunk).
            # This ensures we don't stop mid-story at a page break.
            cs = get_chunk_store()
            # Span 3 pages from the best hit — narrative episodes often cross
            # page boundaries and content can run 2+ pages.
            target_pages = {top_page, top_page + 1, top_page + 2}
            story_chunks = [
                c["text"] for c in cs._chunks
                if c.get("source") == top_source
                and c.get("page") in target_pages
            ][:15]  # cap at 15 chunks (~3 pages)

            stitched = _clean_passage_text(
                _stitch_passage_chunks(story_chunks)
            )[:4000]  # hard cap so it stays readable

            # ── Quote-continuation trim ────────────────────────────────────
            # For “Continue this section: [quote]” queries the stitched block
            # typically includes content BEFORE the quoted text as well as
            # the continuation the user wants.
            # Strategy: extract the last 3-5 WORDS from the query (after
            # stripping the “Continue…:” prefix), then find that phrase in
            # the stitched text using whitespace-flexible matching.  Starting
            # from just after the last matched word gives a clean continuation.
            import re as _re2

            def _norm_q(s):
                # Full normalisation: straighten quotes AND collapse whitespace.
                # NOTE: because whitespace is collapsed, character positions in
                # the output do NOT match positions in the input.  Use this for
                # building / matching anchor patterns; use _norm_quotes() when
                # you need m.end() to map back to the original string.
                csq = chr(0x2018)+chr(0x2019)+chr(0x201a)+chr(0x201b)
                cdq = (chr(0x201c)+chr(0x201d)+chr(0x201e)+chr(0x201f)
                       +chr(0x00ab)+chr(0x00bb))
                s = _re2.sub(chr(91)+csq+chr(93), chr(39), s)
                s = _re2.sub(chr(91)+cdq+chr(93), chr(34), s)
                s = _re2.sub(r'\s+', chr(32), s)
                return s.strip()

            def _norm_quotes(s):
                # Quote-only normalisation: straighten curly quotes 1-for-1 so
                # character positions in the result match the original string.
                csq = chr(0x2018)+chr(0x2019)+chr(0x201a)+chr(0x201b)
                cdq = (chr(0x201c)+chr(0x201d)+chr(0x201e)+chr(0x201f)
                       +chr(0x00ab)+chr(0x00bb))
                s = _re2.sub(chr(91)+csq+chr(93), chr(39), s)
                s = _re2.sub(chr(91)+cdq+chr(93), chr(34), s)
                return s

            # Strip the "Continue this section: " prefix + outer quotes
            q_body = _re2.sub(
                r'^(?:continue(?:\s+this\s+\w+)?[:\s]+)',
                chr(39)+chr(39), question.strip(), flags=_re2.IGNORECASE,
            ).strip(chr(34)+chr(39)+chr(0x201c)+chr(0x201d)+chr(0x2018)+chr(0x2019))

            # Build the anchor from text AFTER the last "..." in q_body.
            # The user often types "...text at the very end" where "..." is an
            # abbreviation for omitted text, not a literal PDF sequence.
            # Using only what follows the last ellipsis keeps the anchor
            # contiguous and matchable in the stitched PDF text.
            q_body_norm = _norm_q(q_body)
            ellipsis_variants = (
                chr(0x2026),           # U+2026 HORIZONTAL ELLIPSIS
                '...',                 # three ASCII dots
            )
            for _ell in ellipsis_variants:
                if _ell in q_body_norm:
                    q_body_norm = q_body_norm[q_body_norm.rfind(_ell) + len(_ell):]
                    break
            anchor_words = q_body_norm.split()[-5:]
            if len(anchor_words) >= 2:
                pat = (r'\s+').join(
                    _re2.escape(w) for w in anchor_words)
                # Search in _norm_quotes(stitched): quotes are normalised
                # 1-for-1 so m.end() maps directly back to the original string.
                stitched_nq = _norm_quotes(stitched)
                m = _re2.search(pat, stitched_nq, _re2.IGNORECASE)
                if m:
                    end = m.end()
                    # Advance past any trailing punctuation / quote chars
                    # that still cling to the matched word (e.g. ," or .")
                    while end < len(stitched_nq) and not stitched_nq[end].isspace():
                        end += 1
                    stitched = stitched[end:].lstrip(
                        chr(10)+chr(32)+chr(44)+chr(59)+chr(39)+chr(34)
                    )

            chunk_count = len(story_chunks)
            gen_result["answer"] = (
                f"[Extracted from {top_source}, "
                f"p.{top_page}–{top_page+2}]\n\n{stitched}"
            )
            gen_result["thinking"] = (
                f"LLM synthesis failed (output corrupted). "
                f"Falling back to direct passage extraction: "
                f"{top_source} p.{top_page}–{top_page+2} "
                f"(relevance score {top.get('score', 0):.2f}). "
                f"Stitched {chunk_count} chunk(s) across 3 pages."
            )
        elif path_text:
            gen_result["answer"]  = "Based on the knowledge graph: " + path_text[:600]
            gen_result["thinking"] = "No passage hits; summarised graph paths directly."
        gen_result["source"] = gen_result.get("source", "unknown") + "+fallback"

    s6 = {
        "step": "generation",
        "label": "Hybrid-Fused Generation",
        "icon": "✨",
        "result": {
            "answer_length": len(gen_result["answer"].split()),
            "source": gen_result.get("source", "unknown"),
        },
        "thinking": gen_result.get("thinking", ""),
        "ms": int((time.time() - t5) * 1000),
    }
    steps.append(s6)
    await _emit("generation", s6)
    t6 = time.time()

    # ── Step 7: Self-RAG Critique ──────────────────────────────────────────────
    # Skip when the LLM was never used (simulation, shortcut fallback) — there
    # is nothing to critique.  Also skip when passage_text is empty, since the
    # critique prompt would have no grounding context to evaluate against.
    critique: dict = {}
    if not llm_is_fake and passage_text and not gen_result.get("source", "").endswith("+fallback"):
        critique = await _self_critique(question, gen_result["answer"], passage_text)

        # Refinement: if groundedness is very low, re-generate with a
        # passage-only prompt that forces the model to cite what it read.
        if critique.get("groundedness", 1.0) < 0.4:
            refine_instruction = (
                f"The previous answer was not well grounded in the source passages. "
                f"Rewrite it using only what the passages explicitly state. "
                f"Question: {question}"
            )
            refined = await call_llm(
                f"Source passages:\n{passage_text[:2500]}",
                instruction=refine_instruction,
                max_tokens=512,
                on_token=on_token,
                extra_stop=["### Response:", "### Instruction:", "Source passages:"],
            )
            refined_answer = refined.get("answer", "")
            if not _answer_is_corrupted(
                refined_answer, passage_text, path_text, instruction=refine_instruction
            ):
                gen_result["answer"] = refined_answer
                gen_result["source"] = gen_result.get("source", "unknown") + "+refined"
                critique["refined"] = True

        s7 = {
            "step":    "self_critique",
            "label":   "Self-RAG Critique",
            "icon":    "🔬",
            "result":  critique,
            "ms":      int((time.time() - t6) * 1000),
        }
        steps.append(s7)
        await _emit("self_critique", s7)

    total_ms = int((time.time() - t0) * 1000)

    return {
        "question":  question,
        "answer":    gen_result["answer"],
        "thinking":  gen_result.get("thinking", ""),
        "intent":    intent_data,
        "steps":     steps,
        "paths":     paths,
        "passages":  [
            {"text": p["text"][:300], "source": p["source"],
             "page": p.get("page", 0), "score": round(p.get("score", 0.0), 3)}
            for p in passage_hits
        ],
        "total_ms":  total_ms,
        "source":    gen_result.get("source", "simulation"),
        "provenance": _build_provenance(paths),
        "critique":  critique,
    }


# Question stop-words we never want as "entities" even if capitalized at sentence start
_STOP_HEAD = {"What", "How", "Why", "When", "Where", "Who", "Is", "Are", "Do",
              "Does", "Can", "Should", "Will", "Would", "The", "A", "An",
              "Tell", "Explain", "Describe", "Show", "List", "Compare"}


def _extract_entities_heuristic(question: str) -> List[str]:
    """Heuristic entity extraction used as a backstop when the LLM returns [].

    Pulls out:
      - All-caps acronyms of length 2-8 (RAG, BERT, GPT-4, LLM, …)
      - Title-case words mid-sentence (BERT, Transformer, …)
    Strips question-leading words like "What", "How", etc.
    """
    import re
    found: list[str] = []
    seen: set[str] = set()

    # Acronyms: 2-8 uppercase letters, optionally with digits/hyphens (GPT-4, GPT-3.5)
    for tok in re.findall(r"\b[A-Z][A-Z0-9\-\.]{1,7}\b", question):
        norm = tok.strip(".-")
        if norm and norm.upper() not in seen and len(norm) >= 2:
            found.append(norm)
            seen.add(norm.upper())

    # Title-case words excluding question heads at position 0
    words = question.split()
    for i, w in enumerate(words):
        clean = re.sub(r"[^A-Za-z0-9]+$", "", w)  # strip trailing punctuation
        clean = re.sub(r"^[^A-Za-z0-9]+", "", clean)
        if not clean:
            continue
        # Skip leading question word
        if i == 0 and clean in _STOP_HEAD:
            continue
        # Title case (first letter upper, has at least one lower) and length >= 4
        if (clean[0].isupper() and any(c.islower() for c in clean)
                and len(clean) >= 4 and clean.upper() not in seen):
            found.append(clean)
            seen.add(clean.upper())

    return found[:6]  # cap at 6 entities


def _parse_intent(text: str) -> dict:
    """Parse intent classification from LLM output.

    Tolerates: markdown fences (```json ... ```), trailing prose, partial JSON
    truncated by stop tokens. Falls back to regex extraction.
    """
    import re

    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE)

    # Try to find the first {...} block — model may emit prose around it
    brace_match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    candidate = brace_match.group(0) if brace_match else cleaned.strip()

    try:
        data = json.loads(candidate)
        # Coerce types in case the model returned weird values
        return {
            "intent": str(data.get("intent", "factual")).strip().lower(),
            "entities": [str(e).strip() for e in data.get("entities", []) if e],
            "confidence": float(data.get("confidence", 0.7)),
        }
    except Exception:
        intent_match  = re.search(r'"intent"\s*:\s*"([\w\-]+)"', cleaned)
        intent = intent_match.group(1).lower() if intent_match else "factual"
        # Best-effort: pull any quoted strings from an "entities":[...] array
        ents_match = re.search(r'"entities"\s*:\s*\[(.*?)\]', cleaned, flags=re.DOTALL)
        entities: list[str] = []
        if ents_match:
            entities = [m.group(1) for m in re.finditer(r'"([^"]+)"', ents_match.group(1))]
        return {"intent": intent, "entities": entities, "confidence": 0.7}


def _paths_to_text(paths: List[List[dict]]) -> str:
    """Convert traversal paths to readable text for LLM context."""
    if not paths:
        return ""
    lines = []
    for i, path in enumerate(paths):
        parts = []
        for hop in path:
            src = hop.get("from", {}).get("label", "?")
            pred = hop.get("edge", {}).get("predicate", "→")
            dst = hop.get("to", {}).get("label", "?")
            parts.append(f"{src} --[{pred}]--> {dst}")
        conf = path[0].get("edge", {}).get("confidence", 0) if path else 0
        lines.append(f"Path {i+1} (confidence={conf:.2f}): {' | '.join(parts)}")
    return "\n".join(lines)


def _clean_passage_text(text: str) -> str:
    """Strip common PDF extraction artifacts from a chunk of text.

    Handles:
    - Leading word-fragment lines: lines shorter than 25 chars that consist of
      lowercase letters/punctuation (the tail end of the previous page's last word).
      e.g.  "ts.\n" → stripped (was "results.\n")
           "rval to\n" → stripped (was "interval to\n")
    - Trailing standalone page numbers: a bare integer at the very end.
      e.g.  "...suggested that the\n129" → "...suggested that the"
    - Inline standalone page numbers: bare integers on their own line mid-text.
      e.g.  "...the answer...\n129\nmodel was explicitly..." → removes the "129" line
    """
    import re
    # Strip leading fragment: a line of ≤25 lowercase chars before the real content
    text = re.sub(r'^\s*[a-z][a-z\s\.,;:\'\"]{0,23}\n', '', text)
    # Strip trailing page number (bare integer, possibly with surrounding whitespace)
    text = re.sub(r'\s*\b\d{1,4}\s*$', '', text)
    # Strip inline page numbers (bare 1-4 digit number alone on its own line)
    text = re.sub(r'\n\d{1,4}\n', '\n', text)
    return text.strip()


def _stitch_passage_chunks(texts: List[str], overlap: int = 100) -> str:
    """Merge overlapping chunk texts from the same page into one longer passage.

    Chunks overlap by *overlap* chars (= CHUNK_OVERLAP in chunk_store.py).
    We detect the shared region by checking if the last N chars of the current
    result equal the first N chars of the next chunk, then join without the dupe.
    Falls back to simple newline-join when no overlap is found.
    """
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0].strip()

    result = texts[0]
    for text in texts[1:]:
        stitched = False
        # Try decreasing overlap sizes; the exact overlap is usually CHUNK_OVERLAP
        # but may be ±5 chars due to mid-word boundary splits.
        for n in range(min(overlap + 5, len(result), len(text)), 25, -1):
            if result[-n:] == text[:n]:
                result   = result + text[n:]
                stitched = True
                break
        if not stitched:
            # Check whether text precedes result (reversed order)
            for n in range(min(overlap + 5, len(result), len(text)), 25, -1):
                if text[-n:] == result[:n]:
                    result   = text + result[n:]
                    stitched = True
                    break
        if not stitched:
            result = result.rstrip() + "\n" + text.lstrip()

    return result.strip()


def _answer_is_corrupted(
    answer: str,
    passage_text: str,
    path_text: str = "",
    instruction: str = "",
) -> bool:
    """Return True if the LLM answer looks like it's echoing the prompt.

    Heuristics:
    - Answer is empty or whitespace-only
    - Answer contains '### Response:' (Alpaca template loop)
    - Answer's opening 60 chars appear verbatim in the passage or graph-path
      context that was fed as input (model is copying instead of generating)
    - Answer contains '--[' which is the graph-path edge notation
    - Answer's opening 60 chars appear in the instruction we sent (instruction echo)
    """
    if not answer or not answer.strip():
        return True
    if "### Response:" in answer:
        return True
    if "--[" in answer and "]-->" in answer:
        # Graph-path edge notation (--[predicate]-->) leaked into the answer
        return True
    # Check opening against all supplied context
    lead = answer.strip()[:60].lower()
    if len(lead) > 20:
        if passage_text and lead in passage_text.lower():
            return True
        if path_text and lead in path_text.lower():
            return True
        # Catch instruction echo: local model sometimes repeats the task directive
        if instruction and lead in instruction.lower()[:400]:
            return True
    return False


def _passages_to_text(hits: List[dict]) -> str:
    """Format chunk-store hits as plain text blocks for the generation prompt.

    Uses a simple separator style so Alpaca-tuned models don't pattern-match
    on numbered citation format and echo it back.
    """
    if not hits:
        return ""
    parts = []
    for h in hits:
        src  = h.get("source", "unknown")
        page = h.get("page", 0)
        loc  = f"p.{page}" if page else ""
        header = f"[{src}{(' ' + loc) if loc else ''}]"
        body   = _clean_passage_text(h["text"])[:400]
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def _rag_fusion_merge(candidate_lists: List[List[dict]], top_k: int = 20) -> List[dict]:
    """RRF-merge multiple ranked candidate lists from different query variants.

    Each list is a ranked result from hybrid_search on a different query variant.
    RRF score = Σ 1/(RRF_K + rank_i(doc)) across all lists.
    """
    chunk_ranks: dict = {}  # chunk_id → {list_idx: rank}
    chunk_objs:  dict = {}  # chunk_id → chunk dict

    for list_idx, candidates in enumerate(candidate_lists):
        for rank, chunk in enumerate(candidates):
            cid = chunk["id"]
            chunk_objs.setdefault(cid, chunk)
            chunk_ranks.setdefault(cid, {})[list_idx] = rank + 1  # 1-based

    rrf_scores = []
    for cid, ranks in chunk_ranks.items():
        score = sum(1.0 / (_RRF_K_FUSION + r) for r in ranks.values())
        rrf_scores.append((score, {**chunk_objs[cid], "rrf_score": score}))

    rrf_scores.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in rrf_scores[:top_k]]


async def _self_critique(question: str, answer: str, passage_text: str) -> dict:
    """Ask the LLM to critique its own answer. Returns {} on any failure."""
    try:
        critique_input = (
            f"Question: {question}\n\n"
            f"Passages:\n{passage_text[:1200]}\n\n"
            f"Answer:\n{answer[:500]}"
        )
        result = await call_llm(
            critique_input,
            system=CRITIQUE_SYSTEM,
            max_tokens=80,
            extra_stop=["}\n", "}\r\n", "\n\n", "###"],
        )
        return _parse_critique(result.get("answer", ""))
    except Exception:
        return {}


def _parse_critique(text: str) -> dict:
    import re
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE)
    brace_match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    candidate   = brace_match.group(0) if brace_match else cleaned.strip()
    try:
        data = json.loads(candidate)
        return {
            "groundedness": max(0.0, min(1.0, float(data.get("groundedness", 0.7)))),
            "completeness": max(0.0, min(1.0, float(data.get("completeness", 0.7)))),
            "critique":     str(data.get("critique", "")).strip()[:200],
        }
    except Exception:
        g = re.search(r'"groundedness"\s*:\s*([0-9.]+)', cleaned)
        c = re.search(r'"completeness"\s*:\s*([0-9.]+)', cleaned)
        return {
            "groundedness": float(g.group(1)) if g else 0.7,
            "completeness": float(c.group(1)) if c else 0.7,
            "critique":     "",
        }


def _build_provenance(paths: List[List[dict]]) -> List[dict]:
    """Build citation-style provenance from paths."""
    provenance = []
    for i, path in enumerate(paths):
        entities = []
        for hop in path:
            entities.append(hop.get("from", {}).get("label", ""))
        if path:
            entities.append(path[-1].get("to", {}).get("label", ""))
        conf = sum(h.get("edge", {}).get("confidence", 0.5) for h in path) / max(len(path), 1)
        provenance.append({
            "path_id": i + 1,
            "entities": [e for e in entities if e],
            "hops": len(path),
            "confidence": round(conf, 3),
        })
    return provenance
