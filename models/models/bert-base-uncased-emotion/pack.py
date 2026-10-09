import json
from pathlib import Path

from stack.compiler.package import pack


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
parser.add_argument("--entrypoint", required=True)
parser.add_argument("--backend", required=True)
parser.add_argument("--task-runtime", choices=("linux", "tile"), required=True)
args = parser.parse_args()
directory = args.model_dir.resolve()
metadata = "bert.payload/model.json"
model = json.loads((directory / metadata).read_text())
validation = model["validation_inputs"]
execution = {
    "kind": "python",
    "entrypoint": args.entrypoint,
    "p2e": {
        "kind": "native",
        "task_runtime": args.task_runtime,
        "input": "resource",
        "arguments": [".", "--input"],
        "reference_settings": {"inputs": [item["text"] for item in validation]},
        "prepared_inputs": [item["resource"] for item in validation],
    },
}
pack(
    directory,
    model["chip"],
    "bert",
    model["model"],
    "bert-run",
    metadata,
    [
        f"bert.payload/{name}"
        for name in (
            "model.json",
            "params.f32",
            "weights.bin",
            "scales.bin",
            "quant-index.json",
        )
    ]
    + [item["resource"] for item in validation],
    args.rax_pack,
    execution,
    args.backend,
)
