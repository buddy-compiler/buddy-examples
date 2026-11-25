from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "lerobot/pi0fast-base",
    local_dir=Path(__file__).resolve().parent / "output/pi0-fast-base",
)
