import json
import subprocess
from pathlib import Path


def pack(directory: Path, chip: str, model_type: str, model_name: str, program: str,
         metadata: str, resources: list[str], packer: Path, execution: dict, backend: str, *, embed_payload: bool = True):
    layout = {"version": 4, "execution": execution, "chip": chip, "model_type": model_type, "metadata": metadata,
              "resources": {f"resource_{index}": path for index, path in enumerate(resources)}}
    layout_path = directory / "layout.json"
    layout_path.write_text(json.dumps(layout, sort_keys=True))
    entries = dict(layout["resources"], layout="layout.json")
    manifest = [f'rhal.module @model attributes {{version = "0.1.0", model_name = {json.dumps(model_name)}}} {{']
    for index, (name, path) in enumerate(entries.items(), 1):
        size = (directory / path).stat().st_size
        manifest.append(f'  rhal.constant @{name} {{id = {index} : i32, storage = "external", type = tensor<{size}xi8>, uri = "file:{path}"}}')
    manifest.extend([
        f'  rhal.codeobj @program {{id = 1 : i32, kind = "device_elf", backend = "{backend}", uri = "file:{program}"}}',
        '}',
    ])
    path = directory / f"{model_type}.rhal.mlir"
    path.write_text("\n".join(manifest))
    command = [str(packer), str(path), "-o", str(directory / f"{model_type}.rax")]
    if embed_payload:
        command.append("--embed-payload")
    subprocess.run(command, check=True)
