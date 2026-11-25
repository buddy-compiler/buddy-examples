# ===- buddy-pytorch-golden.py --------------------------------------------
#
# Qwen3-0.6B on poly: the golden data of baremetal.cpp.
# For a short prompt written from position 0 into an empty KV cache, the
# inputs of the device program in its order and its outputs in PyTorch
# (frontend/reference.py: the candidates of the next token, the largest
# logits of the last position on each rank's piece of the vocabulary with
# their indices, then the keys and the values) for the prompt and every
# decode step after it, each on the token torch.argmax picks; each
# file the values one after the other at 64-byte boundaries, and their sizes
# in bytes with their dynamic sizes (output/prompt/: files of the model image).
#
#   buddy-pytorch-golden.py <design.pb>   reads output/input-order.txt of
#                                         buddy-pytorch-importer.py
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import math
import sys

import torch

torch.set_num_threads(16)
from transformers import AutoTokenizer

from scripts.design import load
from frontend import reference
from frontend.partition.graph import candidates

model_dir = Path(__file__).resolve().parent
design_pb = sys.argv[1]
output = Path(sys.argv[2])
directory = output / "prompt"
directory.mkdir(parents=True, exist_ok=True)

model = import_module("models.models.qwen3-0-6b.model")
decoder = model.load()
tokenizer = AutoTokenizer.from_pretrained(model.checkpoint)
# The device program splits the decoder across a rank on each poly hart of
# design.pb, as frontend/partition/graph.py cuts it.
ranks = sum(hart.component == "poly" for hart in load(design_pb).device.harts)
# The program reads the inputs of the decoder in this order.
order = [int(index) for index in (output / "input-order.txt").read_text().split()]

# baremetal.cpp runs eight decoder steps over one KV cache: the prompt, then
# each generated token but the last, fed back one at a time. The cache holds
# every position they write, in whole MXINT8 blocks: the cache positions are
# the K of the probability-value product, which the device reads in blocks of
# 32 values (runner.cpp grows its cache in the same blocks).
steps = 8
block = 32
prompt = (model.checkpoint.parent.parent / "input.txt").read_text().strip()
tokens = tokenizer(prompt, return_tensors="pt").input_ids
positions = tokens.shape[1] + steps - 1
cache_capacity = math.ceil(positions / block) * block
_, start, keys, values = model.inputs(tokens.shape[1])
keys = [cache[:, :, :cache_capacity, :].contiguous() for cache in keys]
values = [cache[:, :, :cache_capacity, :].contiguous() for cache in values]
arguments = [tokens, start, *keys, *values]
# The dynamic sizes of the decoder, as the importer marks them: the token
# count (axis 1 of the tokens) and the cache capacity (axis 2 of every key
# and value cache).
argument_axes = [(1,), (), *[(2,)] * (len(keys) + len(values))]
output_axes = [(), (), *[(2,)] * (len(keys) + len(values))]


def top(logits):
    """The candidates of the next token as the program returns them: on each
    rank's piece of the vocabulary, its `candidates` largest logits of the
    last position, the first index first among equal values, and their
    indices in the vocabulary, the ranks in order."""
    last = logits[0, -1]
    values, indices = [], []
    for low, high in reference.pieces(last.shape[0], ranks):
        order = torch.sort(last[low:high], descending=True, stable=True).indices
        values.append(last[low:high][order[:candidates]])
        indices.append(order[:candidates] + low)
    return torch.cat(values)[None], torch.cat(indices)[None]


with torch.no_grad():
    plain, _, _ = decoder(tokens, start, keys, values)
# Fixed weights are quantized once for all reference decode steps.
for module in decoder.modules():
    if isinstance(module, torch.nn.Linear):
        module.weight = torch.nn.Parameter(
            reference.mxint8(module.weight.detach(), 1), requires_grad=False
        )
logits, new_keys, new_values = reference.run(
    decoder, tokens, start, keys, values, ranks
)
# The final outputs baremetal.cpp checks: the candidate logits of every step
# (candidates.data) and the token each step picks (tokens.data).
step_candidates = [top(logits)[0]]
step_tokens = [logits[0, -1].argmax()]
print(
    f"{tokens.shape[1]} tokens: device program in PyTorch vs FP32 model: "
    f"max |logit difference| {(logits - plain).abs().max().item()}, "
    f"next token {tokenizer.decode(logits[0, -1].argmax())!r} "
    f"(FP32 {tokenizer.decode(plain[0, -1].argmax())!r})"
)
sizes = []
for kind, tensors, axes in (
    (
        "input",
        [arguments[position] for position in order],
        [argument_axes[position] for position in order],
    ),
    ("output", [*top(logits), *new_keys, *new_values], output_axes),
):
    packed = bytearray()
    for tensor, dynamic in zip(tensors, axes, strict=True):
        data = tensor.contiguous().numpy().tobytes()
        packed.extend(data + bytes(-len(data) % 64))
        dimensions = "".join(f" {tensor.shape[axis]}" for axis in dynamic)
        sizes.append(f"{kind} {len(data)}{dimensions}\n")
    if kind == "input":
        assert (directory / "inputs.data").read_bytes() == packed
    else:
        (directory / "outputs.data").write_bytes(packed)
assert (directory / "sizes.txt").read_text() == "".join(sizes)

# The decode steps after the prompt, each on the token the previous one
# chose; their outputs follow those of the prompt.
position = tokens.shape[1]
with (directory / "outputs.data").open("ab") as file:
    for step in range(1, steps):
        tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        start = torch.tensor([position], dtype=start.dtype).reshape(start.shape)
        logits, new_keys, new_values = reference.run(
            decoder, tokens, start, new_keys, new_values, ranks
        )
        for tensor in [*top(logits), *new_keys, *new_values]:
            data = tensor.contiguous().numpy().tobytes()
            file.write(data + bytes(-len(data) % 64))
        step_candidates.append(top(logits)[0])
        step_tokens.append(logits[0, -1].argmax())
        position += 1
torch.cat(step_candidates).numpy().tofile(directory / "candidates.data")
torch.stack(step_tokens).numpy().astype("int64").tofile(directory / "tokens.data")
