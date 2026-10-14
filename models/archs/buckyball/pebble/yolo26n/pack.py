import sys
from pathlib import Path

repo_root = Path(sys.argv[sys.argv.index("--repo-root") + 1]).resolve()
sys.path.insert(0, str(repo_root))

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
parser.add_argument("--repo-root", type=Path, required=True)
args = parser.parse_args()
output = args.output_dir.resolve()
resources = ["model.json"] + [
    str(p.relative_to(output))
    for p in sorted((output / "yolo26.payload").iterdir())
    if p.name != "quant.rhal.mlir"
]
shutil.copy2(Path(__file__).parent / "labels.txt", output / "labels.txt")
resources.append("labels.txt")
metadata = json.loads((output / "model.json").read_text())
pack(
    output,
    metadata["chip"],
    "yolo26",
    "yolo26",
    "yolo26-run",
    "model.json",
    resources,
    args.rax_pack,
    {"kind": "native", "input": "file", "arguments": args.arguments},
    args.backend,
)
