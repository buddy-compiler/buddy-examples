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
    for p in sorted((output / "mobilenetv3.payload").iterdir())
    if p.name != "quant.rhal.mlir"
]
shutil.copy2(Path(__file__).parent / "Labels.txt", output / "Labels.txt")
resources.append("Labels.txt")
metadata = json.loads((output / "model.json").read_text())
pack(
    output,
    metadata["chip"],
    "mobilenetv3",
    "mobilenetv3",
    "mobilenetv3-run",
    "model.json",
    resources,
    args.rax_pack,
    {"kind": "native", "input": "file", "arguments": args.arguments},
    args.backend,
)
