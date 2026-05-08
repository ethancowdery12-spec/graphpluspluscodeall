import os
from huggingface_hub import hf_hub_download

REPO_ID = "EthanHCowdery5678234/graphrag-plus-plus-qwen35-4b-GGUF"
FILENAME = "graphrag-plus-plus-qwen35-4b-q3_k_m.gguf"
TOKEN = os.getenv("HF_TOKEN", "")
if not TOKEN:
    raise SystemExit("HF_TOKEN env var is required; export it before running.")

print(f"Downloading {FILENAME} from {REPO_ID}...")
path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    token=TOKEN,
    local_dir=".",
    local_dir_use_symlinks=False
)
print(f"Done! Model saved to: {path}")
