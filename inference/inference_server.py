"""
GraphRAG++ Inference Server — Cloud Run GPU
Loads the merged LoRA model from GCS and serves POST /predict
"""
import os
import re
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GCS_MODEL_PATH = os.getenv("GCS_MODEL_PATH", "")   # gs://bucket/graphrag-model
HF_TOKEN       = os.getenv("HF_TOKEN", "")
LOCAL_MODEL_DIR = "/tmp/graphrag-model"

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


def _download_from_gcs():
    """Download model from GCS to local /tmp using Python SDK (no gsutil needed)."""
    if not GCS_MODEL_PATH:
        raise RuntimeError("GCS_MODEL_PATH env var not set")

    from google.cloud import storage as gcs

    # Parse gs://bucket/prefix
    path        = GCS_MODEL_PATH.replace("gs://", "")
    bucket_name = path.split("/")[0]
    prefix      = "/".join(path.split("/")[1:]).rstrip("/") + "/"

    logger.info(f"Downloading model from gs://{bucket_name}/{prefix} ...")
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blobs  = list(client.list_blobs(bucket_name, prefix=prefix))

    for blob in blobs:
        rel_path   = blob.name[len(prefix):]
        if not rel_path:
            continue
        local_path = os.path.join(LOCAL_MODEL_DIR, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        logger.info(f"  {blob.name} ({blob.size / 1e6:.1f} MB)")
        blob.download_to_filename(local_path)

    logger.info(f"Model download complete → {LOCAL_MODEL_DIR}")



def _load_model():
    """Load model + tokenizer into GPU memory."""
    global model, tokenizer
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_dir = LOCAL_MODEL_DIR

    logger.info(f"Loading tokenizer from {model_dir} ...")
    raw_tok = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, token=HF_TOKEN or None
    )
    tokenizer = raw_tok.tokenizer if hasattr(raw_tok, "tokenizer") else raw_tok
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model in 4-bit ...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )
    model.eval()

    used  = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"Model ready — {used:.1f}/{total:.0f} GB VRAM")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download + load model on startup."""
    logger.info("=== GraphRAG++ Inference Server starting ===")
    await asyncio.to_thread(_download_from_gcs)
    await asyncio.to_thread(_load_model)
    logger.info("=== Server ready ===")
    yield
    logger.info("=== Server shutting down ===")


app = FastAPI(title="GraphRAG++ Inference API", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    instruction: str
    input: str
    max_tokens: int = 512
    temperature: float = 0.1


class PredictResponse(BaseModel):
    thinking: str
    answer: str
    source: str = "cloud_run"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


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
