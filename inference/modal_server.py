import os
import re
import modal

# ── Modal App Definition ──────────────────────────────────────────────────────
app = modal.App("graphrag-inference")

BASE_MODEL_ID = "lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
GITHUB_LORA_URL = "https://github.com/ethancowdery12-spec/GraphRAG-plus-plus.git"
MODEL_CACHE_DIR = "/model-cache"
ADAPTER_DIR = "/tmp/graphrag-lora"

# Persistent volume — survives across deploys, only downloads model once
model_volume = modal.Volume.from_name("graphrag-model-cache", create_if_missing=True)

# Container image — just Python deps, NO model download during build
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers>=4.46.0",
        "peft>=0.13.2",
        "bitsandbytes>=0.44.1",
        "accelerate>=1.1.1",
        "huggingface-hub",
        "fastapi[standard]",
        "sentencepiece",
        "tokenizers",
        "tiktoken",
    )
    .apt_install("git")
)


# ── GPU Inference Class ───────────────────────────────────────────────────────
@app.cls(
    gpu="L4",
    image=image,
    scaledown_window=300,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_CACHE_DIR: model_volume},
)
class InferenceGPU:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from huggingface_hub import snapshot_download
        import subprocess

        hf_token = os.environ.get("HF_TOKEN")

        # ── 1. Download base model to volume (only first time) ────────────
        base_path = os.path.join(MODEL_CACHE_DIR, "base-model-qwen36")
        marker = os.path.join(base_path, ".download_complete")
        if not os.path.exists(marker):
            print("[INFO] First run - downloading base model to persistent volume...")
            snapshot_download(
                BASE_MODEL_ID,
                local_dir=base_path,
                token=hf_token,
            )
            # Write marker so we skip next time
            with open(marker, "w") as f:
                f.write("ok")
            model_volume.commit()
            print("[OK] Base model cached to volume.")
        else:
            print("[OK] Base model already cached in volume.")

        # ── 2. Clone LoRA adapter from GitHub (tiny, ~4MB, always fresh) ──
        if not os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
            print("[INFO] Cloning LoRA adapter from GitHub...")
            subprocess.run(
                ["git", "clone", "--depth", "1", GITHUB_LORA_URL, ADAPTER_DIR],
                check=True,
            )
            print("[OK] LoRA adapter cloned.")
        else:
            print("[OK] LoRA adapter already present.")

        # ── 3. Load tokenizer from base model ────────────────────────────
        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path, use_fast=False, token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── 4. Load base model in 4-bit ──────────────────────────────────
        print("Loading Base Model in 4-bit...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=bnb_cfg,
            device_map="auto",
            token=hf_token,
        )

        # ── 5. Apply LoRA adapter ────────────────────────────────────────
        print("Applying LoRA Adapter...")
        self.model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        self.model.eval()
        print("[OK] Model fully loaded and ready for inference!")

    @modal.method()
    def predict(self, instruction: str, input_text: str, max_tokens: int, temperature: float):
        import torch

        ALPACA_TEMPLATE = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n"
        )
        prompt = ALPACA_TEMPLATE.format(instruction=instruction, input=input_text)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return raw


# ── Web Endpoint ──────────────────────────────────────────────────────────────
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from pydantic import BaseModel
    import re

    web_app = FastAPI(title="GraphRAG++ Modal API")

    class PredictRequest(BaseModel):
        instruction: str
        input: str
        max_tokens: int = 512
        temperature: float = 0.1

    @web_app.post("/predict")
    def predict_endpoint(req: PredictRequest):
        api = InferenceGPU()
        raw = api.predict.remote(req.instruction, req.input, req.max_tokens, req.temperature)

        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        thinking    = think_match.group(1).strip() if think_match else ""
        answer      = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        return {"thinking": thinking, "answer": answer, "source": "modal_l4"}

    @web_app.get("/health")
    def health():
        return {"status": "ok", "platform": "modal", "model": BASE_MODEL_ID}

    return web_app
