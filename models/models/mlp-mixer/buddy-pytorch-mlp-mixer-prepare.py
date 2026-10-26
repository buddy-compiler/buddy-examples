from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "timm/mixer_b16_224.goog_in21k_ft_in1k",
    revision="0006300d604941f96f16cb7d229137363e1b2ac6",
    local_dir=Path(__file__).resolve().parent / "output/mlp-mixer",
)

snapshot_download(
    "evanarlian/imagenet_1k_resized_256",
    repo_type="dataset",
    allow_patterns=["data/train-*.parquet"],
    local_dir=Path(__file__).resolve().parents[2] / "datasets/imagenet-256",
)
