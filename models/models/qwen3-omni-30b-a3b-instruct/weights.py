import json
from pathlib import Path

from safetensors import safe_open


class Checkpoint:
    def __init__(self, directory: Path):
        self.directory = directory
        self.files = json.loads(
            (directory / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.metadata = json.loads((directory / "config.json").read_text())

    def tensor(self, name, *, axis=None, rank=0, parts=1):
        with safe_open(
            self.directory / self.files[name], framework="pt", device="cpu"
        ) as source:
            tensor = source.get_slice(name)
            if axis is None:
                return tensor[:].float()
            shape = tensor.get_shape()
            if parts < 1 or not 0 <= rank < parts or shape[axis] % parts:
                raise ValueError(
                    f"invalid tensor partition: {name}, rank={rank}, parts={parts}"
                )
            width = shape[axis] // parts
            indices = [slice(None)] * len(shape)
            indices[axis] = slice(rank * width, (rank + 1) * width)
            return tensor[tuple(indices)].float().contiguous()
