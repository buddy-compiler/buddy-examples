import json
import shutil
from pathlib import Path
from stack.compiler.package import pack

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
parser.add_argument("--arguments", nargs="*", required=True)
parser.add_argument("--backend", required=True)
args = parser.parse_args()
output = args.output_dir.resolve()
resources = ["model.json"] + [
    str(p.relative_to(output))
    for p in sorted((output / "resnet18.payload").iterdir())
    if p.name != "quant.rhal.mlir"
]
shutil.copy2(Path(__file__).parent / "Labels.txt", output / "Labels.txt")
resources.append("Labels.txt")
metadata = json.loads((output / "model.json").read_text())
pack(
    output,
    metadata["chip"],
    "resnet18",
    "resnet18",
    "resnet18-run",
    "model.json",
    resources,
    args.rax_pack,
    {"kind": "native", "input": "file", "arguments": args.arguments},
    args.backend,
)
