import json
import math
import struct
import tomllib
from pathlib import Path


def prepare_inputs(directory: Path, config: Path, metadata: dict):
    from transformers import AutoTokenizer

    settings = tomllib.loads(config.read_text())
    prompts = settings["inputs"]
    tiles = settings["tile_indices"]
    max_tokens = settings["max_tokens"]
    temperature = settings["temperature"]
    if (
        not prompts
        or not all(isinstance(text, str) for text in prompts)
        or max_tokens < 1
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("native generation requires text inputs and valid parameters")
    if len(tiles) != metadata["parts"] or len(set(tiles)) != len(tiles):
        raise ValueError("prepared tiles differ from the compiled model")
    tokenizer = AutoTokenizer.from_pretrained(
        directory / "tokenizer", local_files_only=True
    )
    generation = json.loads(
        (directory / "tokenizer/generation_config.json").read_text()
    )
    stop = generation["eos_token_id"]
    stop = stop if isinstance(stop, list) else [stop]
    (directory / "inputs").mkdir(exist_ok=True)
    resources = []
    for index, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        if not tokens or len(tokens) > metadata["prefill_length"]:
            raise ValueError("prompt does not fit the compiled prefill length")
        if len(tokens) + max_tokens - 1 > metadata["cache_length"]:
            raise ValueError("generation exceeds the compiled context capacity")
        data = struct.pack(
            "<QQQQd", len(tokens), max_tokens, len(stop), len(tiles), temperature
        )
        data += struct.pack(f"<{len(tokens)}q", *tokens)
        data += struct.pack(f"<{len(stop)}q", *stop)
        data += struct.pack(f"<{len(tiles)}Q", *tiles)
        relative = f"inputs/request-{index}.bin"
        (directory / relative).write_bytes(data)
        resources.append(relative)
    reference = {
        key: settings[key]
        for key in ("inputs", "max_tokens", "temperature", "tile_indices")
    }
    return resources, reference
