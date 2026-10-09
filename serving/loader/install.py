import argparse
import json
import hashlib
import shlex
import shutil
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("binary")
    parser.add_argument("destination", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dataset", default="")
    parser.add_argument(
        "--model-storage", choices=("initramfs", "ddr"), default="initramfs"
    )
    parser.add_argument("--worker-cpus", type=int, nargs="+")
    parser.add_argument("--resource-index", type=Path)
    args = parser.parse_args()
    source, binary, destination, config, dataset = (
        args.source,
        args.binary,
        args.destination,
        args.config,
        args.dataset,
    )
    layout = json.loads((source / "layout.json").read_text())
    execution = layout["execution"]
    has_p2e = "p2e" in execution
    if has_p2e:
        execution = {**execution, **execution["p2e"]}
    if execution["kind"] == "host-worker":
        workers = execution["workers"]
        if (
            type(workers) is not int
            or workers < 1
            or execution.get("protocol") != "bbmux1"
        ):
            raise ValueError(
                "host-worker requires positive workers and protocol=bbmux1"
            )
        if (
            args.worker_cpus is None
            or len(args.worker_cpus) != execution["workers"]
            or len(set(args.worker_cpus)) != len(args.worker_cpus)
            or min(args.worker_cpus) < 0
        ):
            raise ValueError(
                "host-worker requires one distinct explicit CPU ID per worker"
            )
        if dataset:
            raise ValueError(
                "host-worker inputs are supplied by the host, not a dataset launcher"
            )
    elif execution["kind"] != "native":
        raise ValueError("guest launcher requires native or host-worker execution")
    (source / binary).resolve(strict=True)
    settings = (
        tomllib.loads(config.read_text()) if execution["kind"] == "native" else None
    )
    if execution["kind"] == "native" and execution["input"] == "resource":
        if execution["reference_settings"].keys() - settings.keys() or any(
            settings[key] != value
            for key, value in execution["reference_settings"].items()
        ):
            raise ValueError(
                "running settings changed; rebuild prepared model resources"
            )
    if "reference_files" in execution:
        for relative, expected in execution["reference_files"].items():
            with (config.parent / relative).open("rb") as source_file:
                actual = hashlib.file_digest(source_file, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(
                    "input files changed; rebuild prepared model resources"
                )
    if has_p2e:
        destination.mkdir(parents=True, exist_ok=True)
        installed = [binary, "layout.json"]
        if args.model_storage == "initramfs":
            installed.extend(layout["resources"].values())
        for value in installed:
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid native artifact path")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
    else:
        shutil.copytree(source, destination, dirs_exist_ok=True)
    if execution["kind"] == "host-worker":
        model_dir = "/root"
        resource_arguments = []
        prepare = ""
        if args.model_storage == "ddr":
            if args.resource_index is None:
                raise ValueError("DDR storage requires an explicit resource index")
            shutil.copy2(args.resource_index, destination / "model.index")
            resource_arguments = ["--resource-index", "/root/model.index"]
            small = min(
                (
                    name
                    for name in layout["resources"].values()
                    if (source / name).stat().st_size >= 16
                ),
                key=lambda name: (source / name).stat().st_size,
            )
            with (source / small).open("rb") as data:
                prefix = data.read(16).hex()
            prepare = (
                shlex.join(
                    [
                        "/root/guest/resource-check",
                        "/root/model.index",
                        model_dir,
                        small,
                        prefix,
                    ]
                )
                + "\n"
            )
        elif args.resource_index is not None:
            raise ValueError("Resource index is only valid for DDR storage")
        launcher = destination / "run-model"
        launcher.write_text(
            "#!/bin/sh\nset -eu\n"
            + prepare
            + "exec /root/guest/worker-mux "
            + shlex.join(
                [
                    "/root/" + binary,
                    model_dir,
                    *resource_arguments,
                    *map(str, args.worker_cpus),
                ]
            )
            + "\n"
        )
        launcher.chmod(0o755)
        return
    commands = []
    if args.model_storage == "ddr":
        if args.resource_index is None:
            raise ValueError("DDR storage requires an explicit resource index")
        shutil.copy2(args.resource_index, destination / "model.index")
        small = min(
            (
                name
                for name in layout["resources"].values()
                if (source / name).stat().st_size >= 16
            ),
            key=lambda name: (source / name).stat().st_size,
        )
        with (source / small).open("rb") as data:
            prefix = data.read(16).hex()
        commands.append(
            ["/root/guest/resource-check", "/root/model.index", "/root", small, prefix]
        )
    elif args.resource_index is not None:
        raise ValueError("Resource index is only valid for DDR storage")
    if dataset:
        arguments = [arg for arg in execution["arguments"] if arg != "--image"]
        commands.append(["./" + binary, *arguments, "--dataset", dataset])
    else:
        inputs = settings["inputs"]
        if execution["input"] == "resource":
            inputs = execution["prepared_inputs"]
        for index, value in enumerate(inputs):
            if execution["input"] == "file":
                path = (config.parent / value).resolve(strict=True)
                target = Path("inputs") / f"{index}{path.suffix}"
                (destination / target).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination / target)
                value = str(target)
            elif execution["input"] == "resource":
                path = Path(value)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("invalid prepared input resource")
                (source / path).resolve(strict=True)
            elif execution["input"] != "text":
                raise ValueError(f"unknown input kind: {execution['input']}")
            arguments = ["./" + binary, *execution["arguments"], value]
            if args.model_storage == "ddr":
                arguments.append("/root/model.index")
            commands.append(arguments)
    launcher = destination / "run-model"
    launcher.write_text(
        "#!/bin/sh\nset -eu\ncd /root\n"
        + "\n".join(shlex.join(command) for command in commands)
        + "\n"
    )
    launcher.chmod(0o755)


if __name__ == "__main__":
    main()
