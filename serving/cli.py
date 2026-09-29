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
    paths = parser.parse_args()
    config_path = paths.run_config.resolve(strict=True)
    with config_path.open("rb") as source:
        settings = tomllib.load(source)
    fields = {"inputs", "core_index", "timeout", "memory_mib",
              "memory_utilization", "tile_indices", "max_num_seqs", "max_tokens",
              "threads", "record_io", "temperature"}
    unknown = settings.keys() - fields
    if unknown:
        raise ValueError(f"Unknown running parameters: {sorted(unknown)}")
    if not isinstance(settings["inputs"], list) or not settings["inputs"] or not all(isinstance(value, str) for value in settings["inputs"]):
        raise ValueError("running inputs must be a nonempty list of strings")
    args = argparse.Namespace(**vars(paths), **settings)
    with closing(Package(args.model.resolve(), args.loader.resolve(), args.log_dir.resolve())) as package:
        execution = package.execution
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
