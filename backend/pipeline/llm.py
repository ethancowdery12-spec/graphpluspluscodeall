"""
GraphRAG++ LLM Adapter  (v2 — fine-tuned LoRA integration)

Priority chain:
  1. Local LoRA  — load base model + fine-tuned adapter (needs CUDA GPU + LOCAL_LORA_PATH)
  2. HF API      — call base model via Inference API with Alpaca prompts (needs HF_TOKEN)
  3. Gradio      — HuggingFace Space fallback
  4. Simulation  — deterministic mock (always works)

Alpaca template is used for ALL real-model calls because that is the exact
format the LoRA adapter was fine-tuned on.
"""
import os
import re
import json
import asyncio
import random
import logging
from typing import Optional, Callable, Awaitable

# Type alias: async callback invoked once per streamed token chunk.
# Args: (delta_text, total_tokens_so_far)
TokenCallback = Callable[[str, int], Awaitable[None]]

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN           = os.getenv("HF_TOKEN", "")
BASE_MODEL_ID      = "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2"
HF_MODEL_ID        = os.getenv("HF_MODEL_ID", BASE_MODEL_ID)
LOCAL_LORA_PATH     = os.getenv("LOCAL_LORA_PATH", "")
CLOUD_RUN_URL       = os.getenv("CLOUD_RUN_URL", "").rstrip("/")
LOCAL_LLAMA_CPP_URL = os.getenv("LOCAL_LLAMA_CPP_URL", "").rstrip("/")
USE_SIMULATION      = os.getenv("USE_SIMULATION", "false").lower() == "true"

# Alpaca prompt template — exactly matches fine-tuning format
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

# ── Lazy-loaded local model state ─────────────────────────────────────────────
_local_model     = None
_local_tokenizer = None
_local_loaded    = False   # True = we already tried loading (prevents repeated attempts)
_hf_client       = None


# ── Prompt helpers ────────────────────────────────────────────────────────────

def build_alpaca_prompt(instruction: str, input_text: str) -> str:
    """Wrap instruction + input in the Alpaca format used during fine-tuning."""
    return ALPACA_TEMPLATE.format(instruction=instruction, input=input_text)


def _parse_response(raw: str) -> tuple[str, str]:
    """Extract <think> block and final answer from model output.

    Edge cases handled:
      - Model wraps the *entire* response in <think>...</think> (Qwen3.5 sometimes
        does this). We unwrap and treat the contents as the answer.
      - Unclosed <think> tag (truncation): strip everything up to and including
        the lone opening tag.
      - No tags: pass-through.
    """
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Unclosed <think> tag — drop the partial reasoning block.
    if "<think>" in answer and "</think>" not in answer:
        answer = re.sub(r"<think>.*$", "", answer, flags=re.DOTALL).strip()

    # Whole response was wrapped in <think>. Promote the contents to the answer.
    if not answer and thinking:
        return "", thinking

    # If the parser produced thinking == answer (rare model bug), keep only one.
    if thinking and thinking == answer:
        thinking = ""

    return thinking, answer


# ── Local LoRA model loading ──────────────────────────────────────────────────

def _try_load_local_model() -> bool:
    """
    Attempt to load the base model + fine-tuned LoRA adapter.
    Returns True if successful.
    Requirements: CUDA GPU + LOCAL_LORA_PATH env var set + peft/transformers installed.
    """
    global _local_model, _local_tokenizer, _local_loaded
    if _local_loaded:
        return _local_model is not None

    _local_loaded = True  # mark so we only try once

    if not LOCAL_LORA_PATH or not os.path.isdir(LOCAL_LORA_PATH):
        logger.info("[LLM] LOCAL_LORA_PATH not set or not found — skipping local model")
        return False

    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("[LLM] No CUDA GPU detected — skipping local LoRA mode")
            return False

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"[LLM] GPU detected: {torch.cuda.get_device_name(0)}  ({vram_gb:.0f} GB)")

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        logger.info(f"[LLM] Loading base model {BASE_MODEL_ID} in 4-bit...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            token=HF_TOKEN or None,
            trust_remote_code=True,
        )

        logger.info(f"[LLM] Applying LoRA adapter from {LOCAL_LORA_PATH} ...")
        _local_model = PeftModel.from_pretrained(base, LOCAL_LORA_PATH)
        _local_model.eval()

        # Load the plain text tokenizer (Qwen VL returns a processor)
        raw_tok = AutoTokenizer.from_pretrained(
            LOCAL_LORA_PATH,
            trust_remote_code=True,
            token=HF_TOKEN or None,
        )
        _local_tokenizer = raw_tok.tokenizer if hasattr(raw_tok, "tokenizer") else raw_tok
        if _local_tokenizer.pad_token is None:
            _local_tokenizer.pad_token = _local_tokenizer.eos_token

        used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"[LLM] Local LoRA model ready  ({used:.1f} GB VRAM used)")
        return True

    except Exception as e:
        logger.warning(f"[LLM] Local model load failed: {e}")
        _local_model = None
        return False


async def _call_local(instruction: str, input_text: str, max_tokens: int) -> dict:
    """Run inference with the locally loaded LoRA model."""
    import torch

    prompt = build_alpaca_prompt(instruction, input_text)
    inputs = _local_tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output = await asyncio.to_thread(
            lambda: _local_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=_local_tokenizer.eos_token_id,
            )
        )

    raw = _local_tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    thinking, answer = _parse_response(raw)
    return {"thinking": thinking, "answer": answer, "source": "local_lora"}


# ── HF Inference API ──────────────────────────────────────────────────────────

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
        lambda: client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
        )
    )
    thinking, answer = _parse_response(raw)
    return {"thinking": thinking, "answer": answer, "source": "hf_api"}


async def _call_cloud_run(instruction: str, input_text: str, max_tokens: int) -> dict:
    """Call the deployed Modal / Cloud Run GPU inference service."""
    import httpx
    logger.info(f"[LLM] Calling Modal endpoint: {CLOUD_RUN_URL}")
    payload = {"instruction": instruction, "input": input_text, "max_tokens": max_tokens}
    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(f"{CLOUD_RUN_URL}/predict", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return {"thinking": data.get("thinking", ""),
            "answer":  data.get("answer", ""),
            "source":  "modal_cloud"}


async def _call_llama_cpp(
    instruction: str,
    input_text: str,
    max_tokens: int,
    on_token: Optional[TokenCallback] = None,
    extra_stop: Optional[list[str]] = None,
) -> dict:
    """Call local llama.cpp server.

    If `on_token` is provided, uses SSE streaming and invokes the callback
    after each chunk with (delta_text, total_tokens_so_far). The full content
    is still accumulated and returned the same way as the non-streaming path.

    `extra_stop` lets callers add task-specific stop sequences (e.g. ["\\n}"]
    for a one-line JSON classifier) on top of the default Alpaca stops.
    """
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
        # Fast path — single non-streaming POST.
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{LOCAL_LLAMA_CPP_URL}/completion", json=payload)
            resp.raise_for_status()
            raw = resp.json().get("content", "")
    else:
        # Streaming path — llama-server emits SSE lines like `data: {...}\n\n`.
        payload["stream"] = True
        raw_parts: list[str] = []
        total_tokens = 0
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream(
                "POST", f"{LOCAL_LLAMA_CPP_URL}/completion", json=payload
            ) as resp:
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
                    total_tokens += 1  # approximate: 1 chunk ≈ 1 token for llama-server
                    try:
                        await on_token(delta, total_tokens)
                    except Exception as cb_err:
                        logger.debug(f"[LLM] token callback raised: {cb_err}")
        raw = "".join(raw_parts)

    thinking, answer = _parse_response(raw)
    return {"thinking": thinking, "answer": answer, "source": "llama_cpp_local"}


async def _call_gradio(instruction: str, input_text: str, max_tokens: int) -> dict:
    from gradio_client import Client
    prompt  = build_alpaca_prompt(instruction, input_text)
    client  = Client(BASE_MODEL_ID, hf_token=HF_TOKEN or None)
    result  = await asyncio.to_thread(
        lambda: client.predict(message=prompt, api_name="/chat")
    )
    raw = result if isinstance(result, str) else str(result)
    thinking, answer = _parse_response(raw)
    return {"thinking": thinking, "answer": answer, "source": "gradio"}


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
    Unified LLM call.  Pass either:
      - instruction + prompt  →  uses Alpaca template (best with fine-tuned model)
      - prompt only           →  prompt used as-is (legacy path)

    If `on_token` is supplied AND the active backend supports streaming
    (currently only llama_cpp_local), the callback fires after each token
    chunk. Other backends ignore it.

    Returns: {thinking, answer, source}
    """
    if USE_SIMULATION:
        return await _simulate(instruction or prompt, prompt)

    # Resolve instruction / input
    instr      = instruction or "Complete the following request."
    input_text = prompt

    # 1. Local Llama.cpp server (fastest for local AMD/Vulkan)
    if LOCAL_LLAMA_CPP_URL:
        try:
            return await _call_llama_cpp(
                instr, input_text, max_tokens,
                on_token=on_token, extra_stop=extra_stop,
            )
        except Exception as e:
            logger.warning(f"[LLM] Local Llama.cpp failed: {e}. Trying next...")

    # 2. Cloud Run / Modal endpoint (remote A100 GPU)
    if CLOUD_RUN_URL:
        try:
            return await _call_cloud_run(instr, input_text, max_tokens)
        except Exception as e:
            logger.warning(f"[LLM] Modal cloud failed: {e}. Trying next...")

    # 3. Local LoRA (needs CUDA)
    if _try_load_local_model():
        try:
            return await _call_local(instr, input_text, max_tokens)
        except Exception as e:
            logger.warning(f"[LLM] Local LoRA failed: {e}. Trying next...")

    # 4. HF Inference API
    if _get_hf_client():
        try:
            return await _call_hf_api(instr, input_text, max_tokens)
        except Exception as e:
            logger.warning(f"[LLM] HF API failed: {e}. Trying next...")

    # 5. Gradio fallback
    try:
        return await _call_gradio(instr, input_text, max_tokens)
    except Exception:
        pass

    # 6. Simulation
    return await _simulate(instr, input_text)


# ── Simulation (unchanged, kept as final fallback) ────────────────────────────

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
    elif "path" in input_text.lower() or "graph" in input_text.lower():
        answer  = _sim_generate_answer(input_text)
        thinking = "Traversing knowledge graph paths. Fusing top-3 context segments."
    else:
        answer  = f"Based on graph traversal: {', '.join(found[:2] or ['related concepts'])} are connected via 3 multi-hop paths (avg confidence 0.91)."
        thinking = "Processing via structured chain-of-thought."

    return {"thinking": thinking, "answer": answer, "source": "simulation"}


def _sim_extract_triples(text: str) -> list:
    pool = [
        {"subject": "Transformer", "predicate": "introduced_by", "object": "Vaswani et al.", "confidence": 0.97},
        {"subject": "Transformer", "predicate": "uses", "object": "Self-Attention", "confidence": 0.99},
        {"subject": "BERT", "predicate": "based_on", "object": "Transformer", "confidence": 0.98},
        {"subject": "GPT-4", "predicate": "developed_by", "object": "OpenAI", "confidence": 0.97},
        {"subject": "RAG", "predicate": "combines", "object": "Retrieval", "confidence": 0.91},
        {"subject": "GraphRAG", "predicate": "extends", "object": "RAG", "confidence": 0.95},
        {"subject": "Knowledge Graph", "predicate": "enables", "object": "Multi-hop Reasoning", "confidence": 0.93},
        {"subject": "LLM", "predicate": "requires", "object": "Large-scale Training Data", "confidence": 0.89},
    ]
    return random.sample(pool, min(random.randint(3, 6), len(pool)))


def _sim_generate_answer(prompt: str) -> str:
    return random.choice([
        "The Transformer (Vaswani et al., 2017) uses self-attention that enables parallel processing. BERT extends this with bidirectional training. GraphRAG++ links these via: Transformer → enables → BERT → extends → GPT → powers → Modern LLMs.",
        "Multi-hop traversal (depth=3): Self-Attention → enables → Long-range Dependencies → critical_for → Semantic Understanding → foundation_of → Modern NLP. Confidence: 0.94, validated against 847 triples.",
        "4 convergent paths found. Highest-confidence (0.97): Fine-tuning → adapts → Pre-trained Models → power → Production AI Systems. Counterfactual check: passed.",
    ])
