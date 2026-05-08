import os
import re
import runpod
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import snapshot_download
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
GITHUB_LORA_URL = "https://github.com/ethancowdery12-spec/GraphRAG-plus-plus.git"
CACHE_DIR = "/model-cache"
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

# ── Global Model Setup ────────────────────────────────────────────────────────
model = None
tokenizer = None

def setup_models():
    """Download and load models on container start."""
    global model, tokenizer
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. Base Model
    marker = os.path.join(BASE_MODEL_DIR, ".download_complete")
    if not os.path.exists(marker):
        print(f"[INFO] Downloading base model {BASE_MODEL_ID} to {BASE_MODEL_DIR}...")
        snapshot_download(
            BASE_MODEL_ID,
            local_dir=BASE_MODEL_DIR,
            token=HF_TOKEN or None,
        )
        with open(marker, "w") as f:
            f.write("ok")
    else:
        print("[INFO] Base model already cached.")

    # 2. LoRA Adapter
    if not os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
        print(f"[INFO] Cloning LoRA adapter from {GITHUB_LORA_URL} to {ADAPTER_DIR}...")
        os.makedirs(ADAPTER_DIR, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", GITHUB_LORA_URL, ADAPTER_DIR],
            check=True,
        )
    else:
        print("[INFO] LoRA adapter already present.")

    # 3. Load Model
    print("[INFO] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_DIR, use_fast=False, token=HF_TOKEN or None
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading Base Model in 4-bit...")
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

    print("[INFO] Applying LoRA Adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    print("[INFO] Model fully loaded and ready for RunPod inference!")


# Execute setup during initialization
setup_models()


# ── RunPod Handler ────────────────────────────────────────────────────────────

def handler(job):
    """
    RunPod endpoint handler logic.
    Accepts job["input"] with the payload.
    """
    job_input = job.get("input", {})
    
    instruction = job_input.get("instruction", "")
    input_text = job_input.get("input", "")
    max_tokens = job_input.get("max_tokens", 512)
    temperature = job_input.get("temperature", 0.1)

    if not instruction:
        return {"error": "Missing 'instruction' field in input"}

    prompt = ALPACA_TEMPLATE.format(instruction=instruction, input=input_text)

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    thinking    = think_match.group(1).strip() if think_match else ""
    answer      = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    return {
        "thinking": thinking,
        "answer": answer,
        "source": "runpod_serverless"
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
