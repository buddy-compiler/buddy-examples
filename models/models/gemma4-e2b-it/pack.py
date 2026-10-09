import argparse
import json
import shutil
from pathlib import Path

from stack.compiler.package import pack
from .inputs import prepare_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--generated-dir", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--chip", required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
parser.add_argument("--run-config", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path, required=True)
args = parser.parse_args()
source, output = args.generated_dir.resolve(), args.model_dir.resolve()
for name in ("model.json", "vocab.txt"):
    shutil.copy2(source / name, output / name)
metadata = json.loads((output / "model.json").read_text())
resources = ["model.json", "vocab.txt"]
(output / "tasks").mkdir(exist_ok=True)
shutil.copy2(source / "tasks/manifest.json", output / "tasks/manifest.json")
resources.append("tasks/manifest.json")
for phase in ("prefill", "decode"):
    destination = output / phase
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    for name in ("params.f32", "weights.bin"):
        shutil.copy2(source / phase / name, destination / name)
        resources.append(f"{phase}/{name}")
    for name in ("quant-index.json", "scales.bin"):
        payload = source / phase / f"{phase}.payload" / name
        if name == "scales.bin" and payload.stat().st_size == 0:
            continue
        shutil.copy2(payload, destination / name)
        resources.append(f"{phase}/{name}")
prepared, reference = prepare_inputs(output, args.run_config, args.tokenizer)
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
    args.chip,
    "gemma4-e2b-it",
    "google/gemma-4-E2B-it",
    "gemma4-e2b-it-run",
    "model.json",
    [*resources, *prepared],
    args.rax_pack,
    execution,
    "buckyball",
)
