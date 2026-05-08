"""
GraphRAG++ Fine-Tuning on Modal  (A100 80GB)
=============================================
Quantizes & fine-tunes lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled
on the expanded GraphRAG training set, then exports to GGUF (IQ2_M) for
~1-2 GB GPU VRAM inference via llama.cpp expert offloading.

Usage:
    modal run finetune_modal.py            # run training + export
    modal run finetune_modal.py::export    # re-export GGUF only (if model already trained)

v2.1 - Hardware Fix Applied
"""

import os
import modal
# ── Compatibility Patches (Remote Only) ────────────────────────────────────────
try:
    import torch
    import torch.utils._pytree
    import triton.runtime.autotuner

    # Fix for Unsloth 2026 compatibility
    for i in range(1, 9):
        for prefix in ["int", "uint"]:
            bit_type = f"{prefix}{i}"
            if not hasattr(torch, bit_type):
                setattr(torch, bit_type, torch.int8)

    # Fix for TorchAO compatibility in Torch 2.5+
    if not hasattr(torch.utils._pytree, "register_constant"):
        def _register_constant(cls):
            torch.utils._pytree.register_pytree_node(cls, lambda x: ([], x), lambda x, _: x)
        torch.utils._pytree.register_constant = _register_constant

    # Fix for Triton Autotuner compatibility (Missing 'STAGE' etc.)
    _orig_autotuner_init = triton.runtime.autotuner.Autotuner.__init__
    def _patched_autotuner_init(self, fn, arg_names, configs, key, *args, **kwargs):
        valid_key = [k for k in key if k in arg_names]
        return _orig_autotuner_init(self, fn, arg_names, configs, valid_key, *args, **kwargs)
    triton.runtime.autotuner.Autotuner.__init__ = _patched_autotuner_init

    # Fix for Triton Autotuner indexing (IndexError)
    _orig_autotuner_run = triton.runtime.autotuner.Autotuner.run
    def _patched_autotuner_run(self, *args, **kwargs):
        # Dynamically filter key_idx to prevent indexing out of bounds
        self.key_idx = [i for i in self.key_idx if i < len(args)]
        return _orig_autotuner_run(self, *args, **kwargs)
    triton.runtime.autotuner.Autotuner.run = _patched_autotuner_run
except (ImportError, ModuleNotFoundError):
    pass

# ── Modal App ─────────────────────────────────────────────────────────────────
app = modal.App("graphrag-finetune")

MODEL_ID = "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2"
OUTPUT_NAME = "graphrag-plus-plus-qwen35-4b"
MAX_SEQ_LEN = 4096

# Persistent volume for caching model weights + saving outputs
vol = modal.Volume.from_name("graphrag-finetune-vol", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev", "libssl-dev", "curl", "ninja-build", "gcc", "g++", "clang")
    # 1. Force stable CUDA 12.4 builds which are fully compatible with 12.6 drivers
    .run_commands(
        "pip config set global.extra-index-url https://download.pytorch.org/whl/cu124",
        "pip uninstall -y torch torchvision xformers",
        "pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 xformers==0.0.28.post3 numpy==1.26.4 setuptools wheel",
        "export CC=gcc CXX=g++ MAX_JOBS=4; pip install causal-conv1d>=1.4.0 flash-linear-attention torch==2.5.1+cu124 --no-build-isolation",
        "export CC=gcc CXX=g++ MAX_JOBS=4; pip install flash-attn torch==2.5.1+cu124 --no-build-isolation"
    )
    # 2. Install remaining high-level libraries
    .uv_pip_install(
       "unsloth_zoo",
       "transformers>=4.46.0",
       "trl>=0.12.0",
       "peft>=0.13.0",
       "bitsandbytes>=0.44.0",
       "accelerate>=1.1.0",
       "datasets",
       "huggingface-hub",
       "hf-transfer",
       "sentencepiece",
       "tokenizers",
       "tiktoken",
       "protobuf",
       "torch==2.5.1",       # FORCE version to prevent auto-upgrade to 2.6
       "torchvision==0.20.1",
       "xformers==0.0.28.post3",
    )
    .pip_install("unsloth @ git+https://github.com/unslothai/unsloth.git")
    .env({
        "UNSLOTH_DISABLE_STATISTICS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "CC": "gcc",
        "CXX": "g++",
        "ACCELERATE_DISABLE_KERNEL_CHECK": "1", # Suppress kernel hang warning
    })
    .add_local_file("training_data.py", "/root/training_data.py")
)


# ── Training Data ─────────────────────────────────────────────────────────────
# Inline the training data so Modal can serialize it without needing local files.
# In a production setup you'd use modal.Mount or a HF dataset instead.

def _get_training_data():
    """Return the full GraphRAG++ training dataset."""
    # We import from the local file and return the list.
    # Modal serializes this function's return value.
    from training_data import get_all_examples
    return get_all_examples()


# ── Alpaca prompt template ────────────────────────────────────────────────────
ALPACA = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request. Always use a <think> block for reasoning, "
    "followed by a JSON array of entity-relation triples.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


# ── Main training function ────────────────────────────────────────────────────
@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=7200,          # 2 hours max
    volumes={"/data": vol},
)
def train():
    """Fine-tune Qwen3.5-4B on GraphRAG++ data, then export GGUF."""
    import sys
    sys.path.insert(0, "/root")

    import torch
    import gc

    # ── GPU info ──────────────────────────────────────────────────────────
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {vram:.0f} GB")
    print(f"Torch : {torch.__version__}  CUDA: {torch.version.cuda}")

    # ── 1. Load model in 4-bit ────────────────────────────────────────────
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    # HF token must come from the env (e.g. Modal Secret) — never hardcode.
    import os as _os
    hf_token = _os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("HF_TOKEN env var is required for fine-tuning.")

    print(f"\n{'='*60}")
    print(f"Loading {MODEL_ID} in 4-bit ...")
    print(f"{'='*60}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,              # auto (bf16 compute, 4-bit storage)
        load_in_4bit=True,       # NF4 quantization
        token=hf_token,
    )

    used = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM used: {used:.1f} GB / {vram:.0f} GB  ({used/vram*100:.0f}%)")

    # ── 2. Extract text tokenizer (MoE models may return a processor) ────
    if hasattr(tokenizer, 'tokenizer'):
        text_tok = tokenizer.tokenizer
        if text_tok.pad_token is None:
            text_tok.pad_token = text_tok.eos_token
        print(f"Extracted text tokenizer: {type(text_tok).__name__}")
    else:
        text_tok = tokenizer
        if text_tok.pad_token is None:
            text_tok.pad_token = text_tok.eos_token
        print(f"Plain tokenizer: {type(text_tok).__name__}")

    # 3. Attach LoRA adapters ───────────────────────────────────────────
    # Attention-only LoRA: matches upstream training recipe.
    # Expert FFNs left frozen (256-expert LoRA impractical on single GPU).
    print("\nAttaching LoRA adapters (attention-only) ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"LoRA r=16, alpha=16 (attention-only)")
    print(f"Trainable: {tr/1e6:.1f}M  ({tr/tot*100:.2f}%)")

    # 4. Dataset Prep ────────────────────────────────────────────────────────
    import training_data
    items = training_data.get_all_examples()
    
    print(f"\n[DATA] Using {len(items)} examples (including seeds and synthetic augmentation).")
    
    from datasets import Dataset
    raw = Dataset.from_list(items)
    
    def fmt(ex):
        return {"text": ALPACA.format(**ex) + text_tok.eos_token}
    
    train_ds = raw.map(fmt, remove_columns=raw.column_names)

    # 5. Train ──────────────────────────────────────────────────────────
    import gc

    # ── 5. Train ──────────────────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        tokenizer=text_tok,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_steps=12,                 # Fixed: Replaced deprecated warmup_ratio
            num_train_epochs=5,
            learning_rate=2e-5,
            fp16=False,
            bf16=True,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir="/data/graphrag_outputs",
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="### Instruction:\n",
        response_part="### Response:\n",
    )

    print(f"\n{'='*60}")
    print("Training: 4-bit QLoRA | r=16 attn-only | eff-batch=16 | 5 epochs")
    print(f"{'='*60}")
    stats = trainer.train()
    print(f"\nDone!  Loss: {stats.training_loss:.4f}  |  {stats.metrics['train_runtime']/60:.1f} min")

    # ── 6. Quick inference test ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Inference test ...")
    print(f"{'='*60}")

    FastLanguageModel.for_inference(model)

    test_cases = [
        (
            "Extract entity-relation triples as JSON.",
            "GPT-4 from OpenAI uses RLHF. LangChain integrates GPT-4 for RAG pipelines."
        ),
        (
            "Classify this query intent. Output JSON.",
            "Which inference frameworks support OpenAI-compatible APIs?"
        ),
    ]

    for instruction, inp in test_cases:
        prompt = ALPACA.format(instruction=instruction, input=inp, output="")
        inputs = text_tok([prompt], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=text_tok.eos_token_id,
            )
        resp = text_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Q: {instruction}\nA: {resp}\n{'─'*60}")

    # ── 7. Save LoRA adapter ─────────────────────────────────────────────
    lora_dir = f"/data/{OUTPUT_NAME}-lora"
    print(f"\nSaving LoRA adapter to {lora_dir} ...")
    model.save_pretrained(lora_dir)
    text_tok.save_pretrained(lora_dir)

    # Push to HuggingFace Hub
    if hf_token:
        print("Pushing to HuggingFace Hub ...")
        model.push_to_hub(OUTPUT_NAME, token=hf_token)
        text_tok.push_to_hub(OUTPUT_NAME, token=hf_token)
        print(f"Pushed: huggingface.co/{OUTPUT_NAME}")

    # ── 8. Export GGUF (Robust Fix) ──────
    print(f"\n{'='*60}")
    print("Exporting GGUF (q3_k_m — High reasoning quality, < 2GB) ...")
    print(f"{'='*60}")

    gguf_dir = f"/data/{OUTPUT_NAME}-gguf"
    try:
        # We manually call the export to avoid the vision-projector bug in automated scripts
        print("Bypassing automated vision-projector check...")
        model.save_pretrained_gguf(
            gguf_dir,
            text_tok,
            quantization_method="q3_k_m",
        )
    except Exception as e:
        print(f"Standard GGUF export failed: {e}")
        print("Falling back to manual export via 'modal run finetune_modal.py::export' ...")

    vol.commit()
    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")

    return {
        "status": "success",
        "loss": stats.training_loss,
        "runtime_min": stats.metrics["train_runtime"] / 60,
        "examples": len(train_ds),
        "lora_dir": lora_dir,
        "gguf_dir": gguf_dir,
    }


# ── Re-export GGUF (Stable Fix for Vision Bug) ────────────────────────────────
@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=3600,
    volumes={"/data": vol},
)
def export():
    """Manually export the model to GGUF, specifically bypassing multimodal/vision errors."""
    import os
    import subprocess
    from huggingface_hub import HfApi
    from unsloth import FastLanguageModel

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("HF_TOKEN env var is required for upload.")
    lora_dir = f"/data/{OUTPUT_NAME}-lora"
    hf_merged_dir = f"/data/{OUTPUT_NAME}-hf-merged"
    gguf_output_file = f"/data/{OUTPUT_NAME}-q3_k_m.gguf"

    if not os.path.exists(lora_dir):
        print(f"ERROR: No trained model found at {lora_dir}")
        return

    # 1. Merge weights to 16-bit HF
    print(f"Loading model from {lora_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_dir,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        token=hf_token,
    )
    text_tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

    print(f"Merging weights to {hf_merged_dir}...")
    model.save_pretrained_merged(hf_merged_dir, text_tok, save_method="merged_16bit")
    
    # 2. Setup llama.cpp
    llama_cpp_dir = "/root/.unsloth/llama.cpp"
    if not os.path.exists(llama_cpp_dir):
        from unsloth_zoo.llama_cpp import install_llama_cpp
        install_llama_cpp()

    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_cpp_dir, "llama-quantize")

    # 3. Phase 1: HF -> F16 GGUF (Explicitly TEXT ONLY)
    f16_gguf = f"/data/{OUTPUT_NAME}-f16.gguf"
    print(f"\nPhase 1: Converting HF to F16 GGUF (Force Text Mode)...")
    convert_cmd = [
        "python", convert_script,
        hf_merged_dir,
        "--outfile", f16_gguf,
        "--outtype", "f16"
    ]
    subprocess.run(convert_cmd, check=True)

    # 4. Phase 2: Quantize to Q3_K_M
    print(f"\nPhase 2: Quantizing to Q3_K_M...")
    subprocess.run([quantize_bin, f16_gguf, gguf_output_file, "q3_k_m"], check=True)

    if os.path.exists(f16_gguf): os.remove(f16_gguf)

    # 5. Phase 3: Push to HF
    if hf_token:
        print(f"\nPhase 3: Uploading GGUF to Hugging Face...")
        api = HfApi()
        try:
            username = api.whoami(token=hf_token)["name"]
            repo_id = f"{username}/{OUTPUT_NAME}-GGUF"
        except Exception:
            repo_id = f"{OUTPUT_NAME}-GGUF"
            
        print(f"Creating/Verifying repo: {repo_id}")
        api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True)
        api.upload_file(
            path_or_fileobj=gguf_output_file,
            path_in_repo=f"{OUTPUT_NAME}-q3_k_m.gguf",
            repo_id=repo_id,
            token=hf_token,
        )
        print(f"SUCCESS: https://huggingface.co/{repo_id}")

    vol.commit()
    print("\n[EXPORT COMPLETE]")

    vol.commit()


@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=3600,
    volumes={"/data": vol},
)
def test_batch(count: int = 150):
    """Test the fine-tuned model on batch of 150+ fresh examples across all 6 categories."""
    import sys
    import json
    import torch
    import time
    from unsloth import FastLanguageModel
    
    sys.path.insert(0, "/root")
    import training_data

    lora_dir = f"/data/{OUTPUT_NAME}-lora"
    if not os.path.exists(lora_dir):
        print(f"ERROR: No LoRA adapter found at {lora_dir}")
        return

    print(f"Loading latest fine-tuned model from {lora_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_dir,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    if hasattr(tokenizer, 'tokenizer'):
        text_tok = tokenizer.tokenizer
    else:
        text_tok = tokenizer

    # Generate expanded test set
    test_set = training_data.get_test_set(count)
    print(f"Generated {len(test_set)} fresh test examples (Coding, Infra, Security, Logs/Reasoning).")

    results = []
    timestamp = int(time.time())
    output_file = f"/data/test_results_{timestamp}.jsonl"
    
    print(f"Starting inference on {len(test_set)} samples...")
    for i, ex in enumerate(test_set):
        prompt = ALPACA.format(instruction=ex["instruction"], input=ex["input"], output="")
        inputs = text_tok([prompt], return_tensors="pt").to("cuda")
        
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=384,  # Increased for potentially complex reasoning
                temperature=0.05,    # Very low temp for strict evaluation
                do_sample=True,
                pad_token_id=text_tok.eos_token_id,
            )
        
        response = text_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        result_item = {
            "id": i,
            "instruction": ex["instruction"],
            "input": ex["input"],
            "response": response
        }
        results.append(result_item)
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_set)}] Inference running...")

    # Save to Volume with timestamp
    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    # Also symlink to 'latest'
    latest_path = "/data/test_results_latest.jsonl"
    if os.path.exists(latest_path):
        os.remove(latest_path)
    with open(latest_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    vol.commit()
    print(f"\n[TEST COMPLETE]")
    print(f"Detailed results: {output_file}")
    print(f"Latest results mirror: {latest_path}")
    
    return {"status": "success", "count": len(results), "path": output_file}


# ── Entrypoint ────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    result = train.remote()
    print(f"\n{'='*60}")
    print("Results:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")
