import sys
from pathlib import Path

repo_root = Path(sys.argv[sys.argv.index("--repo-root") + 1]).resolve()
sys.path.insert(0, str(repo_root))

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from stack.compiler.package import pack
from inputs import prepare_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--chip", required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
parser.add_argument("--run-config", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path, required=True)
parser.add_argument("--repo-root", type=Path, required=True)
args = parser.parse_args()
output = args.output_dir.resolve()
weight_digests = []
for phase in ("prefill", "decode"):
    with (output / phase / "weights.bin").open("rb") as stream:
        weight_digests.append(hashlib.file_digest(stream, "sha256").hexdigest())
if weight_digests[0] != weight_digests[1]:
    raise ValueError("Gemma prefill and decode require identical canonical weights")
metadata = json.loads((output / "model.json").read_text())
if (
    metadata["parameters"]["prefill"]["weight_bytes"]
    != metadata["parameters"]["decode"]["weight_bytes"]
):
    raise ValueError("Gemma phases disagree on canonical weight bytes")
if metadata["mxfp8_parameter_bytes"] != (output / "prefill/weights.bin").stat().st_size:
    raise ValueError("Gemma physical weight count differs from its canonical payload")
shutil.copy2(output / "prefill/weights.bin", output / "weights.bin")
resources = ["model.json", "vocab.txt", "weights.bin"]
(output / "tasks").mkdir(exist_ok=True)
resources.append("tasks/manifest.json")
for phase in ("prefill", "decode"):
    destination = output / phase
    resources.append(f"{phase}/params.f32")
    for name in ("quant-index.json", "scales.bin"):
        payload = output / phase / f"{phase}.payload" / name
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
