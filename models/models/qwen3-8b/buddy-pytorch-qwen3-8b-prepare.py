from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen3-8B",
    local_dir=Path(__file__).resolve().parent / "output/qwen3-8b",
)

snapshot_download(
    "Salesforce/wikitext",
    repo_type="dataset",
    revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    allow_patterns=["wikitext-2-raw-v1/*.parquet"],
    local_dir=Path(__file__).resolve().parents[2] / "datasets/wikitext",
)
