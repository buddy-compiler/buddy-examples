import argparse
import json
from pathlib import Path
import shutil

from stack.compiler.package import pack


parser = argparse.ArgumentParser()
parser.add_argument("--generated-dir", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
parser.add_argument("--backend", required=True)
args = parser.parse_args()
source, output = args.generated_dir.resolve(), args.model_dir.resolve()
shutil.copy2(source / "model.json", output / "model.json")
shutil.copytree(source / "mixer.payload", output / "mixer.payload", dirs_exist_ok=True)
resources = ["model.json"] + [str(p.relative_to(output)) for p in sorted((output / "mixer.payload").iterdir()) if p.name != "quant.rhal.mlir"]
metadata = json.loads((output / "model.json").read_text())
pack(output, metadata["chip"], "mlp-mixer", "mixer", "mixer-run", "model.json",
     resources, args.rax_pack, {"kind": "native", "input": "file", "arguments": ["--tensor"], "task_runtime": "tile"}, args.backend)
