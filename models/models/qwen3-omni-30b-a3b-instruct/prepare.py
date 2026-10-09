from importlib import import_module
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    import_module(".configs.importer-param", __package__).add_arguments(parser)
    args = parser.parse_args()
    checkpoint = snapshot_download(args.checkpoint, revision=args.revision,
        allow_patterns=["*.safetensors", "*.json", "*.txt"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint.json").write_text(json.dumps({
        "model": args.checkpoint, "revision": args.revision, "path": checkpoint
    }, indent=2) + "\n")
    print(checkpoint)


if __name__ == "__main__":
    main()
