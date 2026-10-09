import argparse
import hashlib
import tomllib
from importlib import import_module
from pathlib import Path
from contextlib import closing, ExitStack

from .guest import LinuxGuest, firmware_profile, validate_firmware

from .package import Package
from .executor import run_program


def main():
    parser = argparse.ArgumentParser(description="Run a compiled model package")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--firmware", type=Path)
    parser.add_argument("--guest-memory-mib", type=int)
    parser.add_argument("--load-manifest", type=Path)
    parser.add_argument("--itrace", action="store_true")
    parser.add_argument("--mtrace", action="store_true")
    paths = parser.parse_args()
    config_path = paths.run_config.resolve(strict=True)
    with config_path.open("rb") as source:
        settings = tomllib.load(source)
    fields = {
        "inputs",
        "core_index",
        "timeout",
        "memory_mib",
        "tile_indices",
        "max_tokens",
        "max_audio_tokens",
        "threads",
        "record_io",
        "temperature",
        "p2e",
    }
    unknown = settings.keys() - fields
    if unknown:
        raise ValueError(f"Unknown running parameters: {sorted(unknown)}")
    if (
        not isinstance(settings["inputs"], list)
        or not settings["inputs"]
        or not all(isinstance(value, (str, dict)) for value in settings["inputs"])
    ):
        raise ValueError(
            "running inputs must be a nonempty list of strings or request tables"
        )
    if "p2e" in settings:
        p2e = settings["p2e"]
        if not isinstance(p2e, dict) or set(p2e) != {"console_socket", "timeout"}:
            raise ValueError("p2e requires console_socket and timeout")
        p2e["console_socket"] = str(
            (config_path.parent / p2e["console_socket"]).resolve(strict=True)
        )
    args = argparse.Namespace(**vars(paths), **settings)
    with closing(
        Package(args.model.resolve(), args.loader.resolve(), args.log_dir.resolve())
    ) as package, ExitStack() as session:
        execution = package.execution
        if (
            execution.get("p2e", {}).get("task_runtime") == "ant"
            and execution["kind"] != "native"
        ):
            raise ValueError("Ant model execution must run inside the Linux guest")
        if (
            execution["kind"] == "native"
            and execution["input"] == "resource"
            and paths.firmware is None
        ):
            raise ValueError("prepared model resources require Linux guest firmware")
        if execution["kind"] == "native" and execution["input"] == "resource":
            if execution["reference_settings"].keys() - settings.keys() or any(
                settings[key] != value
                for key, value in execution["reference_settings"].items()
            ):
                raise ValueError(
                    "running settings changed; rebuild prepared model resources"
                )
            if "reference_files" in execution:
                for name, digest in execution["reference_files"].items():
                    with (config_path.parent / name).open("rb") as source:
                        if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                            raise ValueError(
                                "running input files changed; rebuild prepared model resources"
                            )
        if "p2e" in settings and execution["kind"] != "python":
            raise ValueError("native guest execution does not accept a host console")
        if paths.guest_memory_mib is not None and paths.guest_memory_mib < 1:
            raise ValueError("guest-memory-mib must be positive")
        if paths.firmware is not None:
            if "p2e" in settings:
                raise ValueError(
                    "--firmware cannot also connect to an external p2e console"
                )
            endpoint = execution.get("p2e", {})
            if execution["kind"] != "native" or endpoint.get("kind") != "native":
                raise ValueError(
                    "Linux BEMU firmware requires packaged native guest execution"
                )
            repo = Path(__file__).resolve().parents[2]
            firmware, memory, profile = firmware_profile(
                repo, package.chip, None, paths.firmware, paths.guest_memory_mib
            )
            manifest = validate_firmware(
                firmware, profile, package, paths.load_manifest
            )
            native = execution["kind"] == "native"
            endpoint = session.enter_context(
                LinuxGuest(
                    args.simulator.resolve(),
                    firmware,
                    memory,
                    args.log_dir.resolve(),
                    args.timeout,
                    args.itrace,
                    args.mtrace,
                    manifest,
                    native=native,
                    output=execution.get("output"),
                    result_count=len(execution["prepared_inputs"]),
                )
            )
            if not native:
                args.p2e = endpoint
        elif paths.guest_memory_mib is not None or paths.load_manifest is not None:
            raise ValueError("guest memory/load manifest options require --firmware")
        elif (
            execution.get("p2e", {}).get("task_runtime") == "ant"
            and "p2e" not in settings
        ):
            raise ValueError("Ant models require --firmware or an explicit p2e console")
        if execution["kind"] == "native":
            if paths.firmware is None:
                for value in args.inputs:
                    if execution["input"] == "file":
                        value = str((config_path.parent / value).resolve(strict=True))
                    elif execution["input"] != "text":
                        raise ValueError(
                            f"unknown native input kind: {execution['input']}"
                        )
                    run_program(package, args, [*execution["arguments"], value])
        elif execution["kind"] == "python":
            module, symbol = execution["entrypoint"].split(":")
            run = getattr(import_module(module), symbol)
            run(package, args)
        else:
            raise ValueError(f"unknown execution kind: {execution['kind']}")


if __name__ == "__main__":
    main()
