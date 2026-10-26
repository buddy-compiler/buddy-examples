from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "MiniMaxAI/MiniMax-H3",
    local_dir=Path(__file__).resolve().parent / "output/minimax-h3-fl2va",
)
