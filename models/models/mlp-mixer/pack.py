import argparse
import json
from pathlib import Path

from stack.compiler.package import pack

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
parser.add_argument("--backend", required=True)
args = parser.parse_args()
output = args.output_dir.resolve()
resources = ["model.json"] + [
    str(p.relative_to(output))
    for p in sorted((output / "mixer.payload").iterdir())
    if p.name != "quant.rhal.mlir"
]
metadata = json.loads((output / "model.json").read_text())
pack(
    output,
    metadata["chip"],
    "mlp-mixer",
    "mixer",
    "mixer-run",
    "model.json",
    resources,
    args.rax_pack,
    {
        "kind": "native",
        "input": "file",
        "arguments": ["--tensor"],
        "task_runtime": "tile",
    },
    args.backend,
)
