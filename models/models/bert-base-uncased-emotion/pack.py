import json
from pathlib import Path

from stack.compiler.package import pack


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
parser.add_argument("--entrypoint", required=True)
parser.add_argument("--backend", required=True)
args = parser.parse_args()
directory = args.model_dir.resolve()
metadata = "bert.payload/model.json"
model = json.loads((directory / metadata).read_text())
pack(directory, model["chip"], "bert", model["model"], "bert-run", metadata,
     [f"bert.payload/{name}" for name in
      ("model.json", "params.f32", "weights.bin", "scales.bin", "quant-index.json")], args.rax_pack, {"kind": "python", "entrypoint": args.entrypoint}, args.backend)
