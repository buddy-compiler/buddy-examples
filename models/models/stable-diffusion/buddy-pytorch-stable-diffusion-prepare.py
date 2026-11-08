from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "borno1/stable_diffusion_2_base",
    local_dir=Path(__file__).resolve().parent / "output/stable-diffusion",
)
