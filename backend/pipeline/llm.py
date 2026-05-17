"""
GraphRAG++ LLM Adapter

Priority chain:
  1. Gemini API  — google-genai SDK (needs GEMINI_API_KEY)
  2. Llama.cpp   — local AMD/Vulkan GPU server (needs LOCAL_LLAMA_CPP_URL)
  3. Cloud Run   — Modal / GCP endpoint (needs CLOUD_RUN_URL)
  4. HF API      — HuggingFace Inference API (needs HF_TOKEN)
  5. Simulation  — deterministic mock (always works)
"""
import os
import re
import json
import asyncio
import random
import logging
from typing import Optional, Callable, Awaitable

TokenCallback = Callable[[str, int], Awaitable[None]]

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
HF_TOKEN            = os.getenv("HF_TOKEN", "")
BASE_MODEL_ID       = "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2"
HF_MODEL_ID         = os.getenv("HF_MODEL_ID", BASE_MODEL_ID)
CLOUD_RUN_URL       = os.getenv("CLOUD_RUN_URL", "").rstrip("/")
LOCAL_LLAMA_CPP_URL = os.getenv("LOCAL_LLAMA_CPP_URL", "").rstrip("/")
USE_SIMULATION      = os.getenv("USE_SIMULATION", "false").lower() == "true"

# Alpaca template — used only for llama.cpp / HF fine-tuned model calls
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

# ── Comprehensive system prompt for general-purpose models ────────────────────
# This replaces the fine-tuning signal that the old Qwen adapter relied on.
# It explicitly documents every output format the pipeline expects.
GRAPHRAG_SYSTEM_PROMPT = """You are the reasoning core of GraphRAG++, a hybrid knowledge graph and retrieval-augmented generation system. You have four distinct tasks depending on the instruction you receive. Read the instruction carefully and produce exactly the output format specified below for that task.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 1 — TRIPLE EXTRACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Triggered when: instruction mentions "extract", "triples", "entity-relation", "knowledge graph"

Goal: Extract factual (subject, predicate, object) triples from the provided text. Every triple must represent a real relationship that can be directly verified from the text.

Rules:
- subject and object must be proper nouns, named entities, or short noun phrases (2–6 words max). Never use pronouns, partial sentences, or sentence fragments.
- predicate must be a short verb phrase in snake_case (e.g. "uses", "based_on", "introduced_by", "enables", "requires", "is_a", "part_of", "trained_on", "evaluates_on", "outperforms", "implements", "defined_in", "calls", "inherits_from").
- confidence: float 0.0–1.0. Use 0.95–1.0 for directly stated facts, 0.75–0.94 for strongly implied, 0.5–0.74 for inferred.
- confidence_tier: "EXTRACTED" for directly stated verifiable facts; "INFERRED" for LLM-derived high-confidence relationships; "AMBIGUOUS" for uncertain or speculative ones.
- Extract 5–20 triples per chunk. Prefer precision over volume — only include triples you are confident about.
- Do NOT extract trivial triples like ("text", "is_a", "text") or ("document", "contains", "words").

Output format — a raw JSON array, no markdown fences, no preamble:
[
  {"subject": "BERT", "predicate": "based_on", "object": "Transformer", "confidence": 0.98, "confidence_tier": "EXTRACTED"},
  {"subject": "Transformer", "predicate": "introduced_by", "object": "Vaswani et al.", "confidence": 0.99, "confidence_tier": "EXTRACTED"},
  {"subject": "BERT", "predicate": "enables", "object": "Bidirectional Context", "confidence": 0.95, "confidence_tier": "EXTRACTED"}
]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 2 — INTENT CLASSIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Triggered when: instruction mentions "intent", "classify", "query type"

Goal: Classify the user's query into one of four intent types and extract the key named entities.

Intent types:
- "factual"      — single-hop, direct lookup (e.g. "What is BERT?", "Who made GPT-4?")
- "multi-hop"    — requires traversing ≥2 relationships (e.g. "How does LoRA relate to PEFT and training efficiency?")
- "aggregative"  — summarise or list across many entities (e.g. "What fine-tuning techniques exist?")
- "comparative"  — explicitly compares two or more entities (e.g. "How does RAG differ from GraphRAG?")

entities: list of the key named entities, concepts, or proper nouns in the query (up to 5).
confidence: your confidence in the intent classification, 0.0–1.0.

Output format — a raw JSON object, no markdown fences, no preamble:
{"intent": "multi-hop", "entities": ["LoRA", "PEFT", "fine-tuning"], "confidence": 0.93}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 3 — QUERY VARIANT GENERATION (RAG-Fusion)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Triggered when: instruction mentions "variants", "rephrase", "alternative queries", "RAG-Fusion"

Goal: Generate 3 semantically distinct paraphrases of the user's query. Each variant should approach the same information need from a different angle to maximise retrieval coverage across the vector index.

Rules:
- Keep the same core information need.
- Vary vocabulary, specificity, and framing (e.g. more technical, more conceptual, more concrete).
- Do NOT add new constraints or change the question's intent.
- Each variant should be a complete, standalone question.

Output format — a raw JSON array of 3 strings, no markdown fences, no preamble:
["variant one as a complete question?", "variant two with different phrasing?", "variant three from another angle?"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 4 — ANSWER SYNTHESIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Triggered when: instruction mentions "answer", "synthesize", "generate", "explain", or gives you graph paths and passages to work with.

Goal: Synthesise a clear, grounded, accurate answer to the user's question using the provided context (graph traversal paths + retrieved passages).

Rules:
- Ground every claim in the provided context. Do not hallucinate facts not present in the paths or passages.
- If the context is insufficient to fully answer the question, say so clearly and answer what you can.
- Prioritise graph paths for structured relationship claims; prioritise passages for verbatim facts, numbers, and prose.
- Write in clear, concise prose. Use bullet points only if the question explicitly asks for a list.
- Do not repeat the question. Do not use filler phrases like "Based on the provided context...".
- Length: 2–6 paragraphs depending on complexity. Short factual questions get short answers.
- You may think step-by-step internally before writing the answer, but only output the final answer — no meta-commentary.

Output format — plain prose, no JSON, no markdown headers:
[Your synthesised answer here.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL RULES (all tasks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Output ONLY what the task requires. No preamble, no explanation, no "Here is the output:", no markdown code fences around JSON.
- For Tasks 1–3, your entire response must be valid JSON parseable by Python's json.loads().
- Never truncate JSON arrays mid-output. If context is too long, reduce the number of triples rather than cutting off mid-array.
- If you are unsure which task applies, default to Task 4 (answer synthesis) and respond in prose.
"""


# ── Lazy state ────────────────────────────────────────────────────────────────
_local_model     = None
_local_tokenizer = None
_local_loaded    = False
_hf_client       = None
_gemini_client   = None


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_response(raw: str) -> tuple[str, str]:
    """Extract <think> block and clean answer from any model output."""
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    if "<think>" in answer and "</think>" not in answer:
        answer = re.sub(r"<think>.*$", "", answer, flags=re.DOTALL).strip()

    if not answer and thinking:
        return "", thinking

    if thinking and thinking == answer:
        thinking = ""

    return thinking, answer


def build_alpaca_prompt(instruction: str, input_text: str) -> str:
    return ALPACA_TEMPLATE.format(instruction=instruction, input=input_text)


# ── Tier 1: Gemini ────────────────────────────────────────────────────────────

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None and GEMINI_API_KEY:
        try:
            from google import genai
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"[LLM] Gemini client ready (model={GEMINI_MODEL})")
        except Exception as e:
            logger.warning(f"[LLM] Gemini client init failed: {e}")
            _gemini_client = None
    return _gemini_client


async def _call_gemini(
    instruction: str,
    input_text: str,
    max_tokens: int,
    on_token: Optional[TokenCallback] = None,
) -> dict:
    client = _get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini client not available")

    from google import genai
    from google.genai import types

    # Build the full user message from instruction + input
    user_message = f"{instruction}\n\n{input_text}" if input_text.strip() else instruction

    # gemini-2.5-flash counts thinking tokens toward max_output_tokens.
    # For short structured tasks (JSON extraction / intent / variants) we want
    # thinking ON so accuracy is high; for long prose synthesis we disable it
    # so the entire budget goes to the actual answer.
    is_synthesis = any(w in instruction.lower() for w in ("answer", "synthesize", "synthesise", "explain", "paragraphs"))
    thinking_budget = 0 if is_synthesis else 1024

    try:
        thinking_cfg = types.ThinkingConfig(thinking_budget=thinking_budget)
    except Exception:
        thinking_cfg = None  # older SDK version without ThinkingConfig

    config_kwargs: dict = dict(
        system_instruction=GRAPHRAG_SYSTEM_PROMPT,
        max_output_tokens=max(max_tokens, 4096),  # floor at 4096 so answers aren't cut
        temperature=0.1,
    )
    if thinking_cfg is not None:
        config_kwargs["thinking_config"] = thinking_cfg

    config = types.GenerateContentConfig(**config_kwargs)

    if on_token is not None:
        # Streaming path
        raw_parts: list[str] = []
        total_tokens = 0

        def _stream():
            return client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=user_message,
                config=config,
            )

        stream = await asyncio.to_thread(_stream)
        for chunk in stream:
            delta = chunk.text or ""
            if delta:
                raw_parts.append(delta)
                total_tokens += len(delta.split())
                try:
                    await on_token(delta, total_tokens)
                except Exception as cb_err:
                    logger.debug(f"[LLM] token callback raised: {cb_err}")
        raw = "".join(raw_parts)
    else:
        def _generate():
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_message,
                config=config,
            )

        response = await asyncio.to_thread(_generate)
        raw = response.text or ""

    # Strip markdown fences Gemini sometimes wraps JSON in
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())

    thinking, answer = _parse_response(raw)
    return {"thinking": thinking, "answer": answer, "source": "gemini"}


# ── Tier 2: Llama.cpp (local AMD/Vulkan) ─────────────────────────────────────

async def _call_llama_cpp(
    instruction: str,
    input_text: str,
    max_tokens: int,
    on_token: Optional[TokenCallback] = None,
    extra_stop: Optional[list[str]] = None,
) -> dict:
    import httpx
    prompt = build_alpaca_prompt(instruction, input_text)
    stops = ["### Instruction:", "### Input:", "</s>"]
    if extra_stop:
        stops.extend(extra_stop)
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stop": stops,
    }

    if on_token is None:
        async with httpx.AsyncClient(timeout=120.0) as c:
            resp = await c.post(f"{LOCAL_LLAMA_CPP_URL}/completion", json=payload)
            resp.raise_for_status()
            raw = resp.json().get("content", "")
    else:
        payload["stream"] = True
        raw_parts: list[str] = []
        total_tokens = 0
        async with httpx.AsyncClient(timeout=600.0) as c:
            async with c.stream("POST", f"{LOCAL_LLAMA_CPP_URL}/completion", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("content", "")
                    if not delta:
                        continue
                    raw_parts.append(delta)
                    total_tokens += 1
                    try:
                        await on_token(delta, total_tokens)
                    except Exception as cb_err:
                        logger.debug(f"[LLM] token callback raised: {cb_err}")
        raw = "".join(raw_parts)

    thinking, answer = _parse_response(raw)
    return {"thinking": thinking, "answer": answer, "source": "llama_cpp_local"}


# ── Tier 3: Cloud Run / Modal ─────────────────────────────────────────────────

async def _call_cloud_run(instruction: str, input_text: str, max_tokens: int) -> dict:
    import httpx
    payload = {"instruction": instruction, "input": input_text, "max_tokens": max_tokens}
    async with httpx.AsyncClient(timeout=600.0) as c:
        resp = await c.post(f"{CLOUD_RUN_URL}/predict", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return {"thinking": data.get("thinking", ""), "answer": data.get("answer", ""), "source": "modal_cloud"}


# ── Tier 4: HF Inference API ──────────────────────────────────────────────────

def _get_hf_client():
    global _hf_client
    if _hf_client is None and HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            _hf_client = InferenceClient(model=HF_MODEL_ID, token=HF_TOKEN)
        except Exception:
            _hf_client = None
    return _hf_client


async def _call_hf_api(instruction: str, input_text: str, max_tokens: int) -> dict:
    client = _get_hf_client()
    prompt = build_alpaca_prompt(instruction, input_text)
    raw = await asyncio.to_thread(
        lambda: client.text_generation(prompt, max_new_tokens=max_tokens, temperature=0.1, do_sample=True)
    )
    thinking, answer = _parse_response(raw)
    return {"thinking": thinking, "answer": answer, "source": "hf_api"}


# ── Local LoRA (legacy, CUDA only) ────────────────────────────────────────────

def _try_load_local_model() -> bool:
    global _local_model, _local_tokenizer, _local_loaded
    if _local_loaded:
        return _local_model is not None
    _local_loaded = True
    local_lora = os.getenv("LOCAL_LORA_PATH", "")
    if not local_lora or not os.path.isdir(local_lora):
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_cfg,
                                                    device_map="auto", token=HF_TOKEN or None,
                                                    trust_remote_code=True)
        _local_model = PeftModel.from_pretrained(base, local_lora)
        _local_model.eval()
        raw_tok = AutoTokenizer.from_pretrained(local_lora, trust_remote_code=True, token=HF_TOKEN or None)
        _local_tokenizer = raw_tok.tokenizer if hasattr(raw_tok, "tokenizer") else raw_tok
        if _local_tokenizer.pad_token is None:
            _local_tokenizer.pad_token = _local_tokenizer.eos_token
        return True
    except Exception as e:
        logger.warning(f"[LLM] Local LoRA load failed: {e}")
        _local_model = None
        return False


async def _call_local(instruction: str, input_text: str, max_tokens: int) -> dict:
    import torch
    prompt = build_alpaca_prompt(instruction, input_text)
    inputs = _local_tokenizer([prompt], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = await asyncio.to_thread(
            lambda: _local_model.generate(**inputs, max_new_tokens=max_tokens,
                                          temperature=0.1, do_sample=True,
                                          pad_token_id=_local_tokenizer.eos_token_id)
        )
    raw = _local_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    thinking, answer = _parse_response(raw)
    return {"thinking": thinking, "answer": answer, "source": "local_lora"}


# ── Public API ────────────────────────────────────────────────────────────────

async def call_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = 1024,
    instruction: Optional[str] = None,
    on_token: Optional[TokenCallback] = None,
    extra_stop: Optional[list[str]] = None,
) -> dict:
    """
    Unified LLM call with 4-tier fallback.

    For Gemini (tier 1): uses the comprehensive GRAPHRAG_SYSTEM_PROMPT as the
    system instruction; instruction + prompt form the user turn.

    For llama.cpp / HF (tiers 2+): falls back to Alpaca template since those
    endpoints use the fine-tuned model.

    Returns: {thinking, answer, source}
    """
    if USE_SIMULATION:
        return await _simulate(instruction or prompt, prompt)

    # Resolve instruction text (system folds in for non-Gemini backends)
    if instruction and system:
        instr = f"{system}\n\n{instruction}"
    elif system:
        instr = system
    elif instruction:
        instr = instruction
    else:
        instr = "Complete the following request."

    input_text = prompt

    # 1. Gemini
    if GEMINI_API_KEY and _get_gemini_client() is not None:
        try:
            return await _call_gemini(instr, input_text, max_tokens, on_token=on_token)
        except Exception as e:
            logger.warning(f"[LLM] Gemini failed: {e}. Trying next...")

    # 2. Local llama.cpp
    if LOCAL_LLAMA_CPP_URL:
        try:
            return await _call_llama_cpp(instr, input_text, max_tokens,
                                         on_token=on_token, extra_stop=extra_stop)
        except Exception as e:
            logger.warning(f"[LLM] Llama.cpp failed: {e}. Trying next...")

    # 3. Cloud Run / Modal
    if CLOUD_RUN_URL:
        try:
            return await _call_cloud_run(instr, input_text, max_tokens)
        except Exception as e:
            logger.warning(f"[LLM] Modal failed: {e}. Trying next...")

    # 4. Local LoRA (CUDA)
    if _try_load_local_model():
        try:
            return await _call_local(instr, input_text, max_tokens)
        except Exception as e:
            logger.warning(f"[LLM] Local LoRA failed: {e}. Trying next...")

    # 5. HF Inference API
    if _get_hf_client():
        try:
            return await _call_hf_api(instr, input_text, max_tokens)
        except Exception as e:
            logger.warning(f"[LLM] HF API failed: {e}. Trying next...")

    # 6. Simulation
    return await _simulate(instr, input_text)


# ── Simulation ────────────────────────────────────────────────────────────────

async def _simulate(instruction: str, input_text: str) -> dict:
    await asyncio.sleep(0.3)
    keywords = ["transformer", "attention", "bert", "gpt", "llm", "neural",
                "graph", "embedding", "fine-tuning", "rag", "vector"]
    found = [k for k in keywords if k in input_text.lower()]

    if "extract" in instruction.lower() or "triple" in instruction.lower():
        triples = _sim_extract_triples(input_text)
        answer  = json.dumps(triples)
        thinking = f"Identified {len(triples)} entity-relation triples."
    elif "intent" in instruction.lower() or "classify" in instruction.lower():
        intent  = random.choice(["multi-hop", "factual", "aggregative", "comparative"])
        answer  = json.dumps({"intent": intent, "entities": found[:3],
                               "confidence": round(random.uniform(0.82, 0.98), 3)})
        thinking = f"Classified as {intent} query pattern."
    else:
        answer  = f"Based on graph traversal: {', '.join(found[:2] or ['related concepts'])} are connected via multi-hop paths."
        thinking = "Processing via structured chain-of-thought."

    return {"thinking": thinking, "answer": answer, "source": "simulation"}


def _sim_extract_triples(text: str) -> list:
    pool = [
        {"subject": "Transformer", "predicate": "introduced_by", "object": "Vaswani et al.", "confidence": 0.97, "confidence_tier": "EXTRACTED"},
        {"subject": "Transformer", "predicate": "uses", "object": "Self-Attention", "confidence": 0.99, "confidence_tier": "EXTRACTED"},
        {"subject": "BERT", "predicate": "based_on", "object": "Transformer", "confidence": 0.98, "confidence_tier": "EXTRACTED"},
        {"subject": "GPT-4", "predicate": "developed_by", "object": "OpenAI", "confidence": 0.97, "confidence_tier": "EXTRACTED"},
        {"subject": "RAG", "predicate": "combines", "object": "Retrieval", "confidence": 0.91, "confidence_tier": "INFERRED"},
        {"subject": "GraphRAG", "predicate": "extends", "object": "RAG", "confidence": 0.95, "confidence_tier": "INFERRED"},
        {"subject": "Knowledge Graph", "predicate": "enables", "object": "Multi-hop Reasoning", "confidence": 0.93, "confidence_tier": "INFERRED"},
    ]
    return random.sample(pool, min(random.randint(3, 6), len(pool)))
