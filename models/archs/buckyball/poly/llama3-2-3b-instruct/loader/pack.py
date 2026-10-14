import json
from pathlib import Path
import shutil

from stack.compiler.package import pack
from .inputs import prepare_inputs


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
parser.add_argument("--run-config", type=Path, required=True)
parser.add_argument("--backend", required=True)
args = parser.parse_args()
output = args.output_dir.resolve()
metadata = json.loads((output / "model.json").read_text())
resources = ["model.json"]
for path in sorted((output / "tokenizer").rglob("*")):
    if path.is_file():
        name = path.relative_to(output)
        resources.append(str(name))
for name in dict.fromkeys(stage["parameters"] for stage in metadata["stages"]):
    destination = output / name
    for filename in ("params.f32", "weights.bin"):

        resources.append(f"{name}/{filename}")
    for filename in ("quant-index.json", "scales.bin"):
        shutil.copy2(
            output / name / f"{name}.payload" / filename, destination / filename
        )
        resources.append(f"{name}/{filename}")
prepared, reference = prepare_inputs(output, args.run_config, metadata)
resources.extend(prepared)
execution = {
    "kind": "native",
    "input": "resource",
    "arguments": ["/root"],
    "prepared_inputs": prepared,
    "reference_settings": reference,
    "output": "json",
    "p2e": {"kind": "native", "task_runtime": "ant"},
}
pack(
    output,
    metadata["chip"],
    "llama",
    metadata["model"],
    "llama-run",
    "model.json",
    resources,
    args.rax_pack,
    execution,
    args.backend,
)
