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

INTENT_SYSTEM = """You are a query intent classifier for a knowledge graph system.
Classify the query into one of: factual, multi-hop, aggregative, comparative.
Extract every named entity / proper noun / capitalized acronym from the query
into the entities array. ALWAYS include at least one entity if any noun is
present. Reply with a single JSON object on one line and nothing else.
Example: {"intent": "factual", "entities": ["RAG"], "confidence": 0.9}"""

GENERATION_SYSTEM_WITH_PATHS = """You are GraphRAG++, a knowledge graph reasoning engine.
Answer questions using ONLY the provided graph paths as evidence.
Cite paths by their number (e.g., "Path 1") when supporting a claim.
Do not mention the existence or absence of evidence — just answer."""

GENERATION_SYSTEM_NO_PATHS = """You are an expert assistant.
Answer the user's question concisely and accurately from your own knowledge.
Do NOT mention graphs, paths, evidence, or that you have no context.
Just answer the question directly."""

GENERATION_PROMPT_WITH_PATHS = """Question: {question}

Knowledge Graph Paths:
{paths}

Answer the question using these paths. Cite path numbers when relevant."""

GENERATION_PROMPT_NO_PATHS = """Question: {question}

Provide a direct, concise answer."""


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

    # ── Step 2: Semantic Node Matching ─────────────────────────────────────────
    query_embedding = await embed_query(question)
    semantic_hits = search_index(query_embedding, k=10)

    # Also match by entity names from intent
    start_nodes = [nid for nid, _ in semantic_hits[:5]]
    for entity in intent_data.get("entities", []):
        matched = graph.find_nodes_by_name(entity)
        start_nodes.extend(matched[:2])
    start_nodes = list(dict.fromkeys(start_nodes))[:8]  # unique, max 8

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

    # ── Step 3: Multi-hop Traversal ────────────────────────────────────────────
    intent = intent_data.get("intent", "factual")
    max_hops = 4 if intent == "multi-hop" else 2
    paths = graph.multi_hop_paths(start_nodes, max_hops=max_hops, top_k=5)

    s3 = {
        "step": "graph_traversal",
        "label": "Multi-hop Traversal",
        "icon": "🕸️",
        "result": {
            "paths_found": len(paths),
            "max_hops": max_hops,
            "intent": intent,
        },
        "paths": paths,
        "ms": int((time.time() - t2) * 1000),
    }
    steps.append(s3)
    await _emit("graph_traversal", s3)
    t3 = time.time()

    # ── Step 4: Context Fusion ─────────────────────────────────────────────────
    path_text = _paths_to_text(paths)
    stats_text = json.dumps(graph.get_stats(), indent=2)

    s4 = {
        "step": "context_fusion",
        "label": "Context Fusion",
        "icon": "⚡",
        "result": {
            "fused_tokens": len(path_text.split()),
            "paths_used": len(paths),
        },
        "ms": int((time.time() - t3) * 1000),
    }
    steps.append(s4)
    await _emit("context_fusion", s4)
    t4 = time.time()

    # ── Step 5: Generation ─────────────────────────────────────────────────────
    # Branch the prompt on path availability so the model never sees a
    # placeholder like "No graph paths found", which it tends to meta-comment.
    if path_text:
        gen_prompt = GENERATION_PROMPT_WITH_PATHS.format(
            question=question, paths=path_text[:4000],
        )
        gen_system = GENERATION_SYSTEM_WITH_PATHS
    else:
        gen_prompt = GENERATION_PROMPT_NO_PATHS.format(question=question)
        gen_system = GENERATION_SYSTEM_NO_PATHS
    gen_result = await call_llm(
        gen_prompt, system=gen_system, max_tokens=1024, on_token=on_token,
    )

    s5 = {
        "step": "generation",
        "label": "Graph-Fused Generation",
        "icon": "✨",
        "result": {
            "answer_length": len(gen_result["answer"].split()),
            "source": gen_result.get("source", "unknown"),
        },
        "thinking": gen_result.get("thinking", ""),
        "ms": int((time.time() - t4) * 1000),
    }
    steps.append(s5)
    await _emit("generation", s5)

    total_ms = int((time.time() - t0) * 1000)

    return {
        "question": question,
        "answer": gen_result["answer"],
        "thinking": gen_result.get("thinking", ""),
        "intent": intent_data,
        "steps": steps,
        "paths": paths,
        "total_ms": total_ms,
        "source": gen_result.get("source", "simulation"),
        "provenance": _build_provenance(paths),
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
