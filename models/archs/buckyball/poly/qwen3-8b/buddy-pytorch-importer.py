# ===- buddy-pytorch-importer.py ------------------------------------------
#
# Qwen3-8B on poly: one decoder step of any number of tokens, quantized to
# the MXINT8 products of poly and partitioned across the reduce core and a
# rank on each poly core of design.pb as a device program (program.mlir) and
# the graphs of each class of core (graph-<class>.mlir). The token count is a
# dynamic size: one compiled program runs any count.
#
#   buddy-pytorch-importer.py <design.pb>   into output/, with the packed
#                                           parameters, the inputs and the
#                                           outputs of the program in PyTorch
#                                           for a short prompt (output/prompt/)
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import sys

import torch
from torch._inductor.decomposition import decompositions
from transformers import AutoTokenizer

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from scripts.design import load
from frontend import reference
from models.archs.buckyball.poly.layout import ball
from frontend.partition.graph import cut
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

torch.set_num_threads(4)

model_dir = Path(__file__).resolve().parent
design_pb = sys.argv[1]
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

# ===----------------------------------------------------------------------===
# Step 1: Import one graph of the decoder, the token count dynamic; 8 tokens
# only give the trace an example.
# ===----------------------------------------------------------------------===
model = import_module("models.models.qwen3-8b.model")
decoder = model.load()
tokens, start, keys, values = model.inputs(8)
torch._dynamo.mark_dynamic(tokens, 1, min=1, max=model.capacity)
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=decompositions,
    trace=None,
)
with torch.no_grad():
    (graph,) = compiler.importer(decoder, tokens, start, keys, values)
params = compiler.imported_params[graph]
# Dynamo orders the inputs of the graph as the forward first reads them.
example = [tokens, start, *keys, *values]
order = [
    next(i for i, e in enumerate(example) if e is v) for v in graph._runtime_inputs_ref
]

# ===----------------------------------------------------------------------===
# Step 2.1: Export the original graph.
# ===----------------------------------------------------------------------===
graph.dump(output / "01-import.txt")

# ===----------------------------------------------------------------------===
# Step 2.2: Absorb the transposes PyTorch puts on the Linear weights.
# ===----------------------------------------------------------------------===
permute(graph)
graph.dump(output / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize every matrix product to MXINT8.
# ===----------------------------------------------------------------------===
params = quantize(graph, params)
graph.dump(output / "03-quanted.txt")

# ===----------------------------------------------------------------------===
# Step 2.4: Partition the graph across the cores as a device program:
# launches of the semantic graphs of the model, each graph defined once per
# class (graph-<class of core>.mlir), on the reduce core and the poly ranks.
# ===----------------------------------------------------------------------===
design = load(design_pb)
device = cut(graph, params, design)
# The weights of the poly cores in the layout their Ball reads in place.
ball(
    device,
    min(
        p.thread.mem.bank.entries for p in design.profiles if p.core.component == "poly"
    ),
)
(output / "04-partitioned.txt").write_text(
    "".join(
        f"{s.name}: graph {s.graph} (class {s.graph_class}, on {s.core_class}), {len(s.ops)} operations\n"
        f"  inputs: {', '.join(s.inputs)}\n  outputs: {', '.join(s.outputs)}\n"
        for s in device.segments
    )
)
(output / "program.mlir").write_text(device.program())
for core_class in ("reduce", "poly"):
    (output / f"graph-{core_class}.mlir").write_text(device.module(core_class))

# ===----------------------------------------------------------------------===
# Step 3: the packed parameters; for a short prompt written from
# position 0 into an empty KV cache, the inputs of the program in its order
# and its outputs in PyTorch, each file the values one after the other at
# 64-byte boundaries, and their sizes in bytes (output/prompt/: the files of
# the model image).
# ===----------------------------------------------------------------------===
directory = output / "prompt"
directory.mkdir(exist_ok=True)
for element, values_ in device.packed().items():
    values_.tofile(directory / f"pack-{element}.data")
# The runner passes the token count after the first input: the tokens.
if order[0] != 0:
    raise ValueError("the tokens are not the first input of the program")
tokenizer = AutoTokenizer.from_pretrained(model.checkpoint)
with (output / "vocab.txt").open("w") as stream:
    for token, index in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        stream.write(token.replace("\n", "\\n") + "\n")
(output / "input-order.txt").write_text(" ".join(map(str, order)) + "\n")
prompt = (model.checkpoint.parent.parent / "input.txt").read_text().strip()
tokens = tokenizer(prompt, return_tensors="pt").input_ids
_, start, keys, values = model.inputs(tokens.shape[1])
arguments = [tokens, start, *keys, *values]
logits, new_keys, new_values = reference.run(
    decoder, tokens, start, keys, values, device.ranks
)
with torch.no_grad():
    plain, _, _ = decoder(tokens, start, keys, values)
print(
    f"{tokens.shape[1]} tokens: device program in PyTorch vs FP32 model: "
    f"max |logit difference| {(logits - plain).abs().max().item()}, "
    f"next token {tokenizer.decode(logits[0, -1].argmax())!r} "
    f"(FP32 {tokenizer.decode(plain[0, -1].argmax())!r})"
)
sizes = []
for kind, tensors in (
    ("input", [arguments[position] for position in order]),
    ("output", [logits, *new_keys, *new_values]),
):
    with open(directory / f"{kind}s.data", "wb") as file:
        for tensor in tensors:
            data = tensor.contiguous().numpy().tobytes()
            file.write(data + bytes(-len(data) % 64))
            sizes.append(f"{kind} {len(data)}\n")
(directory / "sizes.txt").write_text("".join(sizes))
