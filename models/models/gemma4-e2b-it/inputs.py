import struct
import json
import subprocess
import tomllib
from pathlib import Path


def prepare_inputs(directory: Path, config: Path, tokenizer: Path):
    metadata = json.loads((directory / "model.json").read_text())
    settings = tomllib.loads(config.read_text())
    tiles = settings["tile_indices"]
    max_tokens = settings["max_tokens"]
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 1
    ):
        raise ValueError("Gemma max_tokens must be a positive integer")
    if (
        not tiles
        or len(tiles) & (len(tiles) - 1)
        or len(set(tiles)) != len(tiles)
        or any(type(tile) is not int or tile <= 0 for tile in tiles)
        or tiles != metadata["execution_tiles"]
        or settings["temperature"] != 0
    ):
        raise ValueError(
            "Gemma prepared compute tiles must match the captured power-of-two task plan"
        )
    resources = []
    (directory / "inputs").mkdir(exist_ok=True)
    for index, prompt in enumerate(settings["inputs"]):
        data = subprocess.check_output(
            [str(tokenizer), str(directory / "vocab.txt"), prompt]
        )
        count = struct.unpack_from("<Q", data)[0]
        if len(data) != 8 + count * 8 or not 1 <= count <= metadata["prefill_length"]:
            raise ValueError("invalid prepared Gemma token sequence")
        if max_tokens > metadata["cache_length"] + 1 - count:
            raise ValueError("Gemma requested generation exceeds the 512-token cache")
        tokens = data[8:]
        packet = struct.pack("<QQQQd", count, max_tokens, 2, len(tiles), 0.0)
        packet += (
            tokens
            + struct.pack("<qq", 1, 106)
            + struct.pack("<" + "Q" * len(tiles), *tiles)
        )
        relative = f"inputs/request-{index}.bin"
        (directory / relative).write_bytes(packet)
        resources.append(relative)
    return resources, {
        key: settings[key]
        for key in ("inputs", "temperature", "tile_indices", "max_tokens")
    }
