from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    local_dir=Path(__file__).resolve().parent / "output/deepseek-r1-distill-qwen-1.5b",
)

snapshot_download(
    "Salesforce/wikitext",
    repo_type="dataset",
    revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    allow_patterns=["wikitext-2-raw-v1/*.parquet"],
    local_dir=Path(__file__).resolve().parents[2] / "datasets/wikitext",
)
