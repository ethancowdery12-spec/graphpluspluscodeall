"""
Upload fine-tuned GraphRAG++ model to HuggingFace Hub
=====================================================
Uploads both the LoRA adapter and GGUF quantized model.

Usage:
    python upload_to_hf.py --token YOUR_HF_TOKEN
    python upload_to_hf.py --token YOUR_HF_TOKEN --lora-only
    python upload_to_hf.py --token YOUR_HF_TOKEN --gguf-only
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def upload_lora(api: HfApi, lora_dir: str, repo_id: str, token: str):
    """Upload LoRA adapter to HuggingFace Hub."""
    print(f"Uploading LoRA adapter from {lora_dir} → {repo_id}")

    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")

    # Create a model card
    card = f"""---
base_model: lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled
tags:
  - qwen3.6
  - mixture-of-experts
  - lora
  - graphrag
  - entity-extraction
  - reasoning
license: apache-2.0
---

# GraphRAG++ Fine-Tuned LoRA Adapter

LoRA adapter fine-tuned on the GraphRAG++ pipeline tasks:
- Entity-Relation Extraction (25 examples)
- Query Intent Classification (15 examples)
- Graph-Grounded QA (15 examples)

## Base Model
[lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled)

## Training Config
- LoRA: r=16, alpha=16, dropout=0.0, targets=[q_proj, k_proj, v_proj, o_proj]
- 4-bit QLoRA (NF4)
- LR: 2e-5, cosine scheduler, warmup 3%
- Optimizer: adamw_8bit
- 5 epochs, effective batch size 16
- Trained on A100 80GB via Modal
"""
    card_path = os.path.join(lora_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card)

    api.upload_folder(
        repo_id=repo_id,
        folder_path=lora_dir,
        token=token,
        commit_message="Upload GraphRAG++ LoRA adapter",
    )
    print(f"✓ LoRA uploaded: https://huggingface.co/{repo_id}")


def upload_gguf(api: HfApi, gguf_dir: str, repo_id: str, token: str):
    """Upload GGUF quantized model to HuggingFace Hub."""
    print(f"Uploading GGUF from {gguf_dir} → {repo_id}")

    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")

    card = f"""---
base_model: lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled
tags:
  - qwen3.6
  - mixture-of-experts
  - gguf
  - quantized
  - graphrag
  - llama-cpp
license: apache-2.0
---

# GraphRAG++ GGUF (Q2_K Quantized)

GGUF quantized version of the GraphRAG++ fine-tuned model.

## Quantization
- Method: Q2_K (aggressive — designed for expert offloading)
- GPU VRAM with expert offloading: ~1-2 GB
- Full RAM (all experts loaded): ~8-10 GB

## Usage with llama.cpp
```bash
# Expert offloading: only shared experts + attention on GPU
./llama-server -m graphrag-qwen36-q2k.gguf \\
    --n-gpu-layers 999 \\
    --override-kv llama.expert_used_count.i32=2 \\
    -c 4096
```

## Usage with LM Studio
Search for this model in LM Studio's model browser.

## Base Model
[lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled)
"""
    card_path = os.path.join(gguf_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card)

    api.upload_folder(
        repo_id=repo_id,
        folder_path=gguf_dir,
        token=token,
        commit_message="Upload GraphRAG++ GGUF (Q2_K)",
    )
    print(f"✓ GGUF uploaded: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload GraphRAG++ model to HuggingFace")
    parser.add_argument("--token", required=True, help="HuggingFace token")
    parser.add_argument("--lora-dir", default="graphrag-plus-plus-qwen36-35b-lora",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--gguf-dir", default="graphrag-plus-plus-qwen36-35b-gguf",
                        help="Path to GGUF directory")
    parser.add_argument("--lora-repo", default="graphrag-plus-plus-qwen36-35b",
                        help="HuggingFace repo for LoRA")
    parser.add_argument("--gguf-repo", default="graphrag-plus-plus-qwen36-35b-GGUF",
                        help="HuggingFace repo for GGUF")
    parser.add_argument("--lora-only", action="store_true", help="Only upload LoRA")
    parser.add_argument("--gguf-only", action="store_true", help="Only upload GGUF")

    args = parser.parse_args()
    api = HfApi()

    if not args.gguf_only:
        if os.path.exists(args.lora_dir):
            upload_lora(api, args.lora_dir, args.lora_repo, args.token)
        else:
            print(f"⚠ LoRA dir not found: {args.lora_dir}")

    if not args.lora_only:
        if os.path.exists(args.gguf_dir):
            upload_gguf(api, args.gguf_dir, args.gguf_repo, args.token)
        else:
            print(f"⚠ GGUF dir not found: {args.gguf_dir}")


if __name__ == "__main__":
    main()
