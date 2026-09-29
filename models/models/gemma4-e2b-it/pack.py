import argparse
import json
import shutil
from pathlib import Path

from stack.compiler.package import pack

parser = argparse.ArgumentParser()
parser.add_argument("--generated-dir", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--chip", required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
args = parser.parse_args()
source, output = args.generated_dir.resolve(), args.model_dir.resolve()
for name in ("arg0.data", "vocab.txt"):
    shutil.copy2(source / name, output / name)
(output / "model.json").write_text(json.dumps({
    "chip": args.chip, "model": "google/gemma-4-E2B-it", "precision": "f32"
}))
pack(output, args.chip, "gemma4-e2b-it", "google/gemma-4-E2B-it",
     "gemma4-e2b-it-run", "model.json", ["model.json", "arg0.data", "vocab.txt"],
     args.rax_pack, {"kind": "native", "input": "text", "arguments": []}, "buckyball")
