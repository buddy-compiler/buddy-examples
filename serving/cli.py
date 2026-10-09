import argparse
import tomllib
from importlib import import_module
from pathlib import Path
from contextlib import closing

from .package import Package
from .executor import run_program


def main():
    parser = argparse.ArgumentParser(description="Run a compiled model package")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
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
        "memory_utilization",
        "tile_indices",
        "max_num_seqs",
        "max_tokens",
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
        p2e["console_socket"] = str((config_path.parent / p2e["console_socket"]).resolve(strict=True))
    args = argparse.Namespace(**vars(paths), **settings)
    with closing(
        Package(args.model.resolve(), args.loader.resolve(), args.log_dir.resolve())
    ) as package:
        execution = package.execution
        if (args.itrace or args.mtrace) and execution["kind"] != "native":
            raise ValueError("itrace and mtrace require native model execution")
        if "p2e" in settings:
            endpoint = execution.get("p2e", {})
            if (execution["kind"] != "python" or endpoint.get("kind") != "host-worker"
                    or endpoint.get("protocol") != "bbmux1"):
                raise ValueError("P2E transport requires a Python model with a bbmux1 host-worker endpoint")
            if args.tile_indices != list(range(endpoint["workers"])):
                raise ValueError("selected controllers differ from the packaged worker topology")
        if execution["kind"] == "native":
            for value in args.inputs:
                if execution["input"] == "file":
                    value = str((config_path.parent / value).resolve(strict=True))
                elif execution["input"] != "text":
                    raise ValueError(f"unknown native input kind: {execution['input']}")
                run_program(package, args, [*execution["arguments"], value])
        elif execution["kind"] == "python":
            module, symbol = execution["entrypoint"].split(":")
            run = getattr(import_module(module), symbol)
            run(package, args)
        else:
            raise ValueError(f"unknown execution kind: {execution['kind']}")


if __name__ == "__main__":
    main()
