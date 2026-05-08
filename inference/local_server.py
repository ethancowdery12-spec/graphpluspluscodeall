import os
import re
import logging
import asyncio
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
GITHUB_LORA_URL = "https://github.com/ethancowdery12-spec/GraphRAG-plus-plus.git"

CACHE_DIR = os.path.expanduser("~/.graphrag-inference")
BASE_MODEL_DIR = os.path.join(CACHE_DIR, "base-model")
ADAPTER_DIR = os.path.join(CACHE_DIR, "lora-adapter")

HF_TOKEN = os.getenv("HF_TOKEN", "")

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

# ── Global model state ────────────────────────────────────────────────────────
model     = None
tokenizer = None


def _download_and_prepare():
    logger.info("Preparing models locally...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. Base Model
    marker = os.path.join(BASE_MODEL_DIR, ".download_complete")
    if not os.path.exists(marker):
        logger.info(f"Downloading base model {BASE_MODEL_ID} to {BASE_MODEL_DIR}...")
        snapshot_download(
            BASE_MODEL_ID,
            local_dir=BASE_MODEL_DIR,
            token=HF_TOKEN or None,
        )
        with open(marker, "w") as f:
            f.write("ok")
        logger.info("Base model download complete.")
    else:
        logger.info("Base model already cached.")

    # 2. LoRA Adapter
    if not os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
        logger.info(f"Cloning LoRA adapter from {GITHUB_LORA_URL} to {ADAPTER_DIR}...")
        os.makedirs(ADAPTER_DIR, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", GITHUB_LORA_URL, ADAPTER_DIR],
            check=True,
        )
        logger.info("LoRA adapter cloned.")
    else:
        logger.info("LoRA adapter already present.")


def _load_model():
    """Load model + tokenizer into GPU memory."""
    global model, tokenizer
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_DIR, use_fast=False, token=HF_TOKEN or None
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading Base Model in 4-bit...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        quantization_config=bnb_cfg,
        device_map="auto",
        token=HF_TOKEN or None,
    )

    logger.info("Applying LoRA Adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    logger.info("Model fully loaded and ready for inference!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download + load model on startup."""
    logger.info("=== Local GraphRAG++ Inference Server starting ===")
    await asyncio.to_thread(_download_and_prepare)
    await asyncio.to_thread(_load_model)
    logger.info("=== Server ready ===")
    yield
    logger.info("=== Server shutting down ===")


app = FastAPI(title="Local GraphRAG++ Inference API", lifespan=lifespan)

# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    instruction: str
    input: str
    max_tokens: int = 512
    temperature: float = 0.1

class PredictResponse(BaseModel):
    thinking: str
    answer: str
    source: str = "local"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "platform": "local"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    prompt = ALPACA_TEMPLATE.format(instruction=req.instruction, input=req.input)

    import torch
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output = await asyncio.to_thread(
            lambda: model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=req.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        )

    raw = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    thinking    = think_match.group(1).strip() if think_match else ""
    answer      = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    return PredictResponse(thinking=thinking, answer=answer)

if __name__ == "__main__":
    import uvicorn
    # Optional: run with python local_server.py
    uvicorn.run("local_server:app", host="0.0.0.0", port=8001, reload=False)
