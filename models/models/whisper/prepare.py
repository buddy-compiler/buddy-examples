import argparse
from pathlib import Path
import tomllib

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download Whisper weights and processor"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with Path(__file__).with_name("configs").joinpath("model.toml").open("rb") as file:
        config = tomllib.load(file)["parameters"]
    snapshot_download(
        repo_id=config["checkpoint"],
        local_dir=args.output,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "normalizer.json"],
    )


if __name__ == "__main__":
    main()
