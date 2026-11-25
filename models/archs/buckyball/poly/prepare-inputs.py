"""Prepare decoder workload inputs without running the reference model."""

from importlib import import_module
from pathlib import Path
import math
import sys

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
name, output = sys.argv[1:]
output = Path(output)
directory = output / "prompt"
directory.mkdir(parents=True, exist_ok=True)
model = import_module(f"models.models.{name}.model")
tokenizer = AutoTokenizer.from_pretrained(model.checkpoint)
prompt = (model.checkpoint.parent.parent / "input.txt").read_text().strip()
tokens = tokenizer(prompt, return_tensors="pt").input_ids
# The prompt token ids alone, the input of the Linux check (runner.cpp).
tokens.numpy().astype("int64").tofile(directory / "ids.data")
capacity = math.ceil((tokens.shape[1] + 7) / 32) * 32
_, start, keys, values = model.inputs(tokens.shape[1])
keys = [cache[:, :, :capacity, :].contiguous() for cache in keys]
values = [cache[:, :, :capacity, :].contiguous() for cache in values]
arguments = [tokens, start, *keys, *values]
axes = [(1,), (), *[(2,)] * (len(keys) + len(values))]
order = [int(i) for i in (output / "input-order.txt").read_text().split()]
sizes = []
with (directory / "inputs.data").open("wb") as stream:
    for i in order:
        tensor = arguments[i]
        data = tensor.numpy().tobytes()
        stream.write(data + bytes(-len(data) % 64))
        dimensions = "".join(f" {tensor.shape[axis]}" for axis in axes[i])
        sizes.append(f"input {len(data)}{dimensions}\n")
_, count = map(int, (output / "candidates.txt").read_text().split())
sizes.extend((f"output {count * 4}\n", f"output {count * 8}\n"))
for cache in [*keys, *values]:
    sizes.append(f"output {cache.numel() * cache.element_size()} {cache.shape[2]}\n")
(directory / "sizes.txt").write_text("".join(sizes))
