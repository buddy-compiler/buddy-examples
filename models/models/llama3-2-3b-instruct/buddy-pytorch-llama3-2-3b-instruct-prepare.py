from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "meta-llama/Llama-3.2-3B-Instruct",
    local_dir=Path(__file__).resolve().parent / "output/llama3-2-3b-instruct",
)
