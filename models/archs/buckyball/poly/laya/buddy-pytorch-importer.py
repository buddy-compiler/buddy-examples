from importlib import import_module
from pathlib import Path
import sys

import torch

torch.set_num_threads(16)
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch._inductor.decomposition import decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from scripts.design import load
from frontend.partition.layout import ball
from frontend import attention
from frontend import norm
from frontend import integer_softmax
from frontend.integer_activations import integer_activations
from frontend.attention import (
    fuse_rotary,
    quantize_kv_in_cache_layout,
    split_attention_products,
)
from frontend.partition.graph import cut, unfuse
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

model_dir = Path(__file__).resolve().parent
design_pb = sys.argv[1]
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)

# ===----------------------------------------------------------------------===
# Step 1: Import the complete checkpoint model.
# ===----------------------------------------------------------------------===
laya = import_module("models.models.laya.model")
# The recipe runs the checkpoint fine-tuned on typed decisions.
laya.checkpoint = laya.checkpoint / "typed-decisions"
model = laya.load()
compiler = DynamoCompiler(
    primary_registry={
        **tosa.ops_registry,
        **quant.ops_registry,
        **attention.ops_registry,
        **norm.ops_registry,
        **integer_softmax.ops_registry,
    },
    aot_autograd_decomposition=decompositions,
    trace=None,
)
# The attention of the head runs as its math: its two products are matrix
# products of the graph, like those of the encoder.
with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
    # The token count is a dynamic size of the tokens and their mask: one
    # compiled program runs any count (128 only gives the trace an example).
    tokens, mask, positions, valid, qtype = laya.inputs()
    torch._dynamo.mark_dynamic(tokens, 1, min=2, max=128)
    torch._dynamo.mark_dynamic(mask, 1, min=2, max=128)
    (graph,) = compiler.importer(model, tokens, mask, positions, valid, qtype)
graph.order_inputs((tokens, mask, positions, valid, qtype))
params = compiler.imported_params[graph]

# ===----------------------------------------------------------------------===
# Step 2.1: Export the original graph.
# ===----------------------------------------------------------------------===
graph.dump(output / "01-import.txt")

# ===----------------------------------------------------------------------===
# Step 2.2: Absorb the Linear weight transposes without changing parameters.
# ===----------------------------------------------------------------------===
permute(graph)
graph.dump(output / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize every matrix product to MXINT8.
# ===----------------------------------------------------------------------===
params = quantize(graph, params)
params = norm.integer_norm(graph, params)
params = integer_activations(graph, params)
attention.reuse_softmax_maximum(graph)
params = integer_softmax.integer_softmax(graph, params)
attention.reuse_rotary_views(graph)
quantize_kv_in_cache_layout(graph)
split_attention_products(graph)
graph.dump(output / "03-quanted.txt")

# ===----------------------------------------------------------------------===
# Step 2.4: Partition the graph across the harts as a device program:
# launches of the semantic graphs of the model, each graph defined once per
# class (graph-<class of hart>.mlir), on the reduce hart and the poly ranks.
# ===----------------------------------------------------------------------===
design = load(design_pb)
# Each rank writes its q, k and v each whole: the fused Linear as three.
params = unfuse(graph, params)
device = cut(graph, params, design)
# The RoPE of q and k in one pass over the Linear's output, written head
# after head: no transpose or concatenation of them on the CPU.
fuse_rotary(device)
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

sample_dir = output / "sample"
sample_dir.mkdir(exist_ok=True)
for element, values in device.packed().items():
    values.tofile(sample_dir / f"pack-{element}.data")
