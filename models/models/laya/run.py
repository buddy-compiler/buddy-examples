import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from stack.serving.executor import Pool
from .inputs import encode, decode


def run(package, args):
    config = package.metadata
    tokenizer = AutoTokenizer.from_pretrained(package.directory / "tokenizer")
    requests = []
    for value in args.inputs:
        data = json.loads((args.run_config.resolve().parent / value).read_text())
        requests.extend(data if isinstance(data, list) else [data])
    tiles = args.tile_indices
    traces = [args.log_dir.resolve() / f"trace-tile-{tile}" for tile in tiles]
    for path in traces:
        path.mkdir(parents=True, exist_ok=True)
    pool = Pool(
        args.simulator.resolve(),
        {"program": str(package.program), "directory": str(package.directory)},
        args.log_dir.resolve(),
        tiles,
        args.record_io,
        [(str(path),) for path in traces],
        args.memory_mib,
    )
    try:
        pending = []
        for request in requests:
            inputs, labels = encode(
                tokenizer, request, config["sequence_length"], config["options"]
            )
            payload = b"".join(
                inputs[name].tobytes()
                for name in ("tokens", "mask", "positions", "valid", "qtype")
            )
            pending.append(
                (
                    request,
                    labels,
                    pool.submit(payload, 4 * (config["options"] + config["actions"])),
                )
            )
        for request, labels, future in pending:
            result = np.frombuffer(
                future.result(timeout=args.timeout), dtype=np.float32
            )
            if not np.isfinite(result).all():
                raise ValueError("non-finite Laya output")
            print(
                json.dumps(
                    decode(
                        result[: config["options"]],
                        result[config["options"] :],
                        request,
                        labels,
                        config["settings"],
                    )
                ),
                flush=True,
            )
    finally:
        pool.close()
