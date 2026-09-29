import argparse
import json
from pathlib import Path
import shutil
from stack.compiler.package import pack

parser = argparse.ArgumentParser()
parser.add_argument("--generated-dir", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--rax-pack", type=Path, required=True)
args = parser.parse_args()
source, output = args.generated_dir.resolve(), args.model_dir.resolve()
metadata = json.loads((source / "model.json").read_text())
resources = ["model.json"]
for entry in metadata["stages"]:
    for filename in ("params.f32", "weights.bin"):
        path = f'{entry["name"]}/{filename}'
        if (source / path).stat().st_size:
            resources.append(path)
for path in (source / "tokenizer").iterdir():
    if path.is_file():
        resources.append(str(path.relative_to(source)))
for name in resources:
    (output / name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / name, output / name)
# Empty weight files are still required by the executable; RAX resources can be empty.
for entry in metadata["stages"]:
    name = f'{entry["name"]}/weights.bin'
    if name not in resources:
        resources.append(name)
        (output / name).write_bytes(b"")
pack(
    output,
    metadata["chip"],
    "laya",
    metadata["model"],
    "laya-run",
    "model.json",
    resources,
    args.rax_pack,
    {"kind": "python", "entrypoint": "stack.models.models.laya.run:run"},
    "buckyball",
)
