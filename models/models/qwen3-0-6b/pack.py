import json
from pathlib import Path
import shutil

from stack.compiler.package import pack


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--generated-dir", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
parser.add_argument("--entrypoint", required=True)
parser.add_argument("--backend", required=True)
args = parser.parse_args()
source, output = args.generated_dir.resolve(), args.model_dir.resolve()
metadata = json.loads((source / "model.json").read_text())
shutil.copy2(source / "model.json", output / "model.json")
resources = ["model.json"]
shutil.copytree(source / "tokenizer", output / "tokenizer", dirs_exist_ok=True)
for path in sorted((source / "tokenizer").rglob("*")):
    if path.is_file():
        name = path.relative_to(source)
        resources.append(str(name))
for name in dict.fromkeys(stage["parameters"] for stage in metadata["stages"]):
    destination = output / name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for filename in ("params.f32", "weights.bin"):
        shutil.copy2(source / name / filename, destination / filename)
        resources.append(f"{name}/{filename}")
    for filename in ("quant-index.json", "scales.bin"):
        payload = source / name / f"{name}.payload" / filename
        if filename == "scales.bin" and payload.stat().st_size == 0:
            continue
        shutil.copy2(payload, destination / filename)
        resources.append(f"{name}/{filename}")
pack(output, metadata["chip"], "qwen", metadata["model"], "qwen-run", "model.json", resources, args.rax_pack, {"kind": "python", "entrypoint": args.entrypoint, "p2e": {"kind": "host-worker", "workers": metadata["execution_tiles"], "protocol": "bbmux1", "task_runtime": "tile"}}, args.backend)
