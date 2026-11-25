from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    "meta-llama/Llama-3.2-3B-Instruct",
    local_dir=Path(__file__).resolve().parent / "output/llama3-2-3b-instruct",
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
