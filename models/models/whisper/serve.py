import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Serve Whisper through vLLM")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            str(args.model.resolve(strict=True)),
            "--served-model-name",
            "whisper",
            "--dtype",
            "float32",
            "--max-model-len",
            "448",
            "--max-num-seqs",
            "4",
            "--kv-cache-memory-bytes",
            "536870912",
            "--enforce-eager",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
    )


if __name__ == "__main__":
    main()
