from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    local_dir=Path(__file__).resolve().parent / "output/deepseek-r1-0528-qwen3-8b",
)
