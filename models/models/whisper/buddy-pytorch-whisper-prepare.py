from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "openai/whisper-tiny",
    local_dir=Path(__file__).resolve().parent / "output/whisper",
)
