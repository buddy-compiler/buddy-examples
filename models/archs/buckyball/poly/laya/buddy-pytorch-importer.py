from importlib import import_module
from pathlib import Path
import sys

import torch
import numpy as np
import json
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch._inductor.decomposition import decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from scripts.design import load
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
# Step 1: Import the complete checkpoint model.
# ===----------------------------------------------------------------------===
laya = import_module("models.models.laya.model")
decisions = import_module("models.models.laya.data")
# The recipe runs the checkpoint fine-tuned on typed decisions.
laya.checkpoint = laya.checkpoint / "typed-decisions"
model = laya.load()
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
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
# graph.dump_parameters(output / "01-import-parameters")

# ===----------------------------------------------------------------------===
# Step 2.2: Absorb the Linear weight transposes without changing parameters.
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

sample_dir = output / "sample"
sample_dir.mkdir(exist_ok=True)
for element, values in device.packed().items():
    values.tofile(sample_dir / f"pack-{element}.data")
sample, labels = decisions.load(64, 1)
sample = tuple(value[:1] for value in sample)
(output / "input.data").write_bytes(
    b"".join(value.numpy().tobytes() for value in sample)
)
from frontend.reference import evaluate

with sdpa_kernel(SDPBackend.MATH):
    baseline, expected = evaluate(model, sample)
np.concatenate([value.numpy().reshape(-1) for value in expected]).tofile(
    output / "expected.data"
)
(output / "reference.json").write_text(
    json.dumps(
        {
            "fp32_choice": baseline[0].argmax(-1).item(),
            "choice": expected[0].argmax(-1).item(),
            "action": expected[1].argmax(-1).item(),
            "label": labels[0].item(),
        }
    )
    + "\n"
)
