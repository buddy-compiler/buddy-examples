from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "convaiinnovations/laya",
    revision="55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    local_dir=Path(__file__).resolve().parent / "output/laya",
)

snapshot_download(
    "LocalLLaMA/typed-decisions",
    repo_type="dataset",
    revision="80cb93ae477fdfc91e3b6114f7a281c444dda843",
    allow_patterns=["all/train-*.parquet"],
    local_dir=Path(__file__).resolve().parents[2] / "datasets/typed-decisions",
)
