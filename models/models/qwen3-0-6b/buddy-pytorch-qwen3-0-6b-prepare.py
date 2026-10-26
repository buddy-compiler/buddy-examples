from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen3-0.6B",
    local_dir=Path(__file__).resolve().parent / "output/qwen3-0-6b",
)
