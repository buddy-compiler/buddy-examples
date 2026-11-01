from pathlib import Path

from huggingface_hub import snapshot_download

output = Path(__file__).resolve().parent / "output"
snapshot_download("lerobot/smolvla_base", local_dir=output / "smolvla-base")
# The policy builds its VLM from the config and processor of SmolVLM2; the
# weights of the VLM layers come with the policy checkpoint.
snapshot_download(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    local_dir=output / "smolvlm2-500m-video-instruct",
    allow_patterns=["*.json", "*.txt", "*.jinja"],
)
