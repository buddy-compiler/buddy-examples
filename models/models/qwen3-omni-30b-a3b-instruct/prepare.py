import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = snapshot_download(MODEL, revision=REVISION,
        allow_patterns=["*.safetensors", "*.json", "*.txt"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint.json").write_text(json.dumps({
        "model": MODEL, "revision": REVISION, "path": checkpoint
    }, indent=2) + "\n")
    print(checkpoint)


if __name__ == "__main__":
    main()
