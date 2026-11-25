"""Prepare the explicit V/K-head/K-tail state of the Qwen3-8B workload."""

from importlib import import_module
from pathlib import Path
import math
import sys

from transformers import AutoTokenizer
from frontend.state import packed_keys

output = Path(sys.argv[1])
directory = output / "prompt"
directory.mkdir(parents=True, exist_ok=True)
model = import_module("models.models.qwen3-8b.model")
tokenizer = AutoTokenizer.from_pretrained(model.checkpoint)
prompt = (model.checkpoint.parent.parent / "input.txt").read_text().strip()
tokens = tokenizer(prompt, return_tensors="pt").input_ids
# The prompt token ids alone, the input of the Linux check (runner.cpp).
tokens.numpy().astype("int64").tofile(directory / "ids.data")
capacity = math.ceil((tokens.shape[1] + 7) / 32) * 32
_, start, keys, values = model.inputs(tokens.shape[1])
keys = [cache[:, :, :capacity, :].contiguous() for cache in keys]
values = [cache[:, :, :capacity, :].contiguous() for cache in values]
states = packed_keys(keys, values)
layout = [
    tuple(map(int, line.split()))
    for line in (output / "state-layout.txt").read_text().splitlines()
]
for state, (heads, row_bytes, code_bytes, exponent) in zip(states, layout, strict=True):
    if (
        state.shape[:3] != (1, heads, capacity)
        or state.shape[-1] * state.element_size() != row_bytes
    ):
        raise ValueError("State layout differs from the program's declared planes")
arguments = [tokens, start, *states]
axes = [(1,), (), *[(2,)] * len(states)]
order = [int(i) for i in (output / "input-order.txt").read_text().split()]
sizes = []
with (directory / "inputs.data").open("wb") as stream:
    for i in order:
        value = arguments[i]
        data = value.contiguous().numpy().tobytes()
        stream.write(data + bytes(-len(data) % 64))
        dimensions = "".join(f" {value.shape[axis]}" for axis in axes[i])
        sizes.append(f"input {len(data)}{dimensions}\n")
_, count = map(int, (output / "candidates.txt").read_text().split())
sizes.extend((f"output {count * 4}\n", f"output {count * 8}\n"))
for state in states:
    sizes.append(f"output {state.numel() * state.element_size()} {capacity}\n")
(directory / "sizes.txt").write_text("".join(sizes))
