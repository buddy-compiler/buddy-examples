import sys
from pathlib import Path

repo_root = Path(sys.argv[sys.argv.index("--repo-root") + 1]).resolve()
sys.path.insert(0, str(repo_root))

import argparse
import json
from pathlib import Path
from stack.compiler.package import pack
from stack.models.models.laya.inputs import prepare_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
parser.add_argument("--run-config", type=Path, required=True)
parser.add_argument("--repo-root", type=Path, required=True)
args = parser.parse_args()
output = args.output_dir.resolve()
metadata = json.loads((output / "model.json").read_text())
resources = ["model.json"]
for entry in metadata["stages"]:
    for filename in ("params.f32", "weights.bin"):
        path = f'{entry["name"]}/{filename}'
        if (output / path).stat().st_size:
            resources.append(path)
for path in (output / "tokenizer").iterdir():
    if path.is_file():
        resources.append(str(path.relative_to(output)))
# Empty weight files are still required by the executable; RAX resources can be empty.
for entry in metadata["stages"]:
    name = f'{entry["name"]}/weights.bin'
    if name not in resources:
        resources.append(name)
        (output / name).write_bytes(b"")
prepared, reference, files = prepare_inputs(output, args.run_config, metadata)
resources.extend([*prepared, "inputs/labels.json"])
pack(
    output,
    metadata["chip"],
    "laya",
    metadata["model"],
    "laya-run",
    "model.json",
    resources,
    args.rax_pack,
    {
        "kind": "native",
        "input": "resource",
        "arguments": ["/root"],
        "prepared_inputs": prepared,
        "reference_settings": reference,
        "reference_files": files,
        "output": "json",
        "p2e": {"kind": "native", "task_runtime": "ant"},
    },
    "buckyball",
)
