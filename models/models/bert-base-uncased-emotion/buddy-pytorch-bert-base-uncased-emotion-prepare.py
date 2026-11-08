from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "bhadresh-savani/bert-base-uncased-emotion",
    local_dir=Path(__file__).resolve().parent / "output/bert-base-uncased-emotion",
)

snapshot_download(
    "dair-ai/emotion",
    repo_type="dataset",
    allow_patterns=["split/train-*.parquet", "split/validation-*.parquet"],
    local_dir=Path(__file__).resolve().parents[2] / "datasets/emotion",
)
