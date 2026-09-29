import json
import subprocess
import tempfile
from pathlib import Path


class Package:
    def __init__(self, path: Path, loader: Path, log_root: Path):
        manifest = json.loads(subprocess.check_output([str(loader), str(path)], text=True))
        if len(manifest["programs"]) != 1:
            raise ValueError("a tile package must contain exactly one program")
        program = manifest["programs"][0]
        if program["kind"] != "DeviceELF":
            raise ValueError("a tile package requires a device ELF")
        self.program = Path(program["path"])
        resources = manifest["resources"]
        layout = json.loads(Path(resources.pop("layout")).read_text())
        if layout["version"] != 4:
            raise ValueError("unsupported tile package layout")
        if set(resources) != set(layout["resources"]):
            raise ValueError("package resources differ from its layout")
        self.model_type = layout["model_type"]
        self.execution = layout["execution"]
        self.chip = layout["chip"]
        log_root.mkdir(parents=True, exist_ok=True)
        self.workspace = tempfile.TemporaryDirectory(prefix="model-", dir=log_root)
        self.directory = Path(self.workspace.name)
        try:
            for name, relative in layout["resources"].items():
                relative = Path(relative)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"invalid package resource path: {relative}")
                target = self.directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(resources[name])
            self.metadata_path = self.directory / layout["metadata"]
            if layout["metadata"] not in layout["resources"].values():
                raise ValueError("metadata is not a package resource")
            self.metadata = json.loads(self.metadata_path.read_text())
            if self.metadata["chip"] != self.chip:
                raise ValueError("package chip differs from model metadata")
        except BaseException:
            self.close()
            raise

    def close(self):
        self.workspace.cleanup()
