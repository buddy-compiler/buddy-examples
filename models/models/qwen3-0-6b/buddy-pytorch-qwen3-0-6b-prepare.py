from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen3-0.6B",
    local_dir=Path(__file__).resolve().parent / "output/qwen3-0-6b",
)

# LAMBADA (OpenAI's English test set, 5153 passages): the zero-shot accuracy
# of the last word.
snapshot_download(
    "EleutherAI/lambada_openai",
    repo_type="dataset",
    revision="900124bf3b8235c6daf21033af9948b3f07346c4",
    allow_patterns=["en/test/en.parquet"],
    local_dir=Path(__file__).resolve().parents[2] / "datasets/lambada",
)
