# ===- buddy-pytorch-importer.py ------------------------------------------
#
# One decoder step of any number of tokens, into output/: the device
# program and the graphs of each class of core (graph-<class>.mlir). The
# token count is a dynamic size: one compiled program runs any count.
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import sys

import torch
from torch._inductor.decomposition import decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from scripts.design import load
from frontend.partition.graph import cut
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

model_dir = Path(__file__).resolve().parent
design_pb = sys.argv[1]
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
llama3 = import_module("models.models.llama3-2-3b-instruct.model")

# ===----------------------------------------------------------------------===
# Step 1: Import one graph of the checkpoint for any number of tokens (8
# only gives the trace an example).
# ===----------------------------------------------------------------------===
tokens, start, keys, values = llama3.inputs(8)
torch._dynamo.mark_dynamic(tokens, 1, min=1, max=llama3.capacity)
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=decompositions,
    trace=None,
)
with torch.no_grad():
    (graph,) = compiler.importer(llama3.load(), tokens, start, keys, values)
params = compiler.imported_params[graph]

# ===----------------------------------------------------------------------===
# Step 2.1: Export the original graph and FP32 parameters.
# ===----------------------------------------------------------------------===
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")

# ===----------------------------------------------------------------------===
# Step 2.2: Absorb the transposes PyTorch puts on the Linear weights.
# ===----------------------------------------------------------------------===
permute(graph)
graph.dump(output / "02-permuted.txt")
# graph.dump_parameters(output / "02-permuted-parameters")

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
device = cut(graph, params, load(design_pb))
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
