# ===- buddy-pytorch-importer.py ------------------------------------------
#
# Qwen3-0.6B on poly: one decoder step of any number of tokens, quantized to
# the MXINT8 products of poly and partitioned across the reduce hart and a
# rank on each poly hart of design.pb as a device program (program.mlir) and
# the graphs of each class of hart (graph-<class>.mlir). The token count is a
# dynamic size: one compiled program runs any count.
#
#   buddy-pytorch-importer.py <design.pb>   into output/, with the packed
#                                           parameters (output/prompt/), the
#                                           vocabulary and the order of the
#                                           inputs of the program
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import sys

import torch

torch.set_num_threads(16)
from torch._inductor.decomposition import decompositions
from transformers import AutoTokenizer

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from models.archs.buckyball.poly.topk import topk
from models.archs.buckyball.poly.cache import update
from scripts.design import load
from frontend.partition.layout import ball
from frontend.attention import (
    group_queries,
    quantize_kv_in_cache_layout,
    split_attention_products,
)
from frontend.partition.graph import candidates, cut
from frontend.permute.graph import permute
from frontend.quant.graph import quantize
from frontend.integer_activations import integer_activations

model_dir = Path(__file__).resolve().parent
design_pb = sys.argv[1]
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)

# ===----------------------------------------------------------------------===
# Step 1: Import one graph of the decoder, the token count dynamic; 8 tokens
# only give the trace an example.
# ===----------------------------------------------------------------------===
model = import_module("models.models.qwen3-0-6b.model")
decoder = model.load()
tokens, start, keys, values = model.inputs(8)
torch._dynamo.mark_dynamic(tokens, 1, min=1, max=model.capacity)
for cache in [*keys, *values]:
    torch._dynamo.mark_dynamic(cache, 2, min=32, max=model.capacity)
compiler = DynamoCompiler(
    primary_registry={
        **tosa.ops_registry,
        **quant.ops_registry,
        "TopkOp": topk,
        "IndexPutOp": update,
    },
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
group_queries(graph)
graph.dump(output / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize every matrix product to MXINT8.
# ===----------------------------------------------------------------------===
params = quantize(graph, params)
params = integer_activations(graph, params)
quantize_kv_in_cache_layout(graph)
split_attention_products(graph)
graph.dump(output / "03-quanted.txt")

# ===----------------------------------------------------------------------===
# Step 2.4: Partition the graph across the harts as a device program:
# launches of the semantic graphs of the model, each graph defined once per
# class (graph-<class of hart>.mlir), on the reduce hart and the poly ranks.
# ===----------------------------------------------------------------------===
design = load(design_pb)
device = cut(graph, params, design)
# The weights of the poly harts in the layout their Ball reads in place.
ball(
    device,
    min(
        p.thread.mem.bank.entries for p in design.profiles if p.hart.component == "poly"
    ),
)
(output / "04-partitioned.txt").write_text(
    "".join(
        f"{s.name}: graph {s.graph} (class {s.graph_class}, on {s.hart_class}), {len(s.ops)} operations\n"
        f"  inputs: {', '.join(s.inputs)}\n  outputs: {', '.join(s.outputs)}\n"
        for s in device.segments
    )
)
(output / "program.mlir").write_text(
    device.program(
        "program", list(device.packs), [node.name for node in device.graph.inputs]
    )
)
(reducer,) = [
    hart.component
    for hart in design.device.harts
    if hart.hart_id == design.workload.control_hart
]
for name, component in (("reduce", reducer), ("poly", "poly")):
    (output / f"graph-{name}.mlir").write_text(device.module(component))

# ===----------------------------------------------------------------------===
# Step 3: the packed parameters (output/prompt/: files of the model image),
# the vocabulary and the order in which the program reads the inputs of the
# decoder (input-order.txt), for the runners and buddy-pytorch-golden.py.
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
# The candidates of the next token of a rank and of all of them, which the
# program returns (frontend/partition/graph.py).
(output / "candidates.txt").write_text(f"{candidates} {candidates * device.ranks}\n")
(directory / "input-order.txt").write_text(" ".join(map(str, order)) + "\n")
