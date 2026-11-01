from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "lerobot/pi05_base",
    local_dir=Path(__file__).resolve().parent / "output/pi05-base",
)
