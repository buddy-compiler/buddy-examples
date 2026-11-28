from importlib import import_module
from pathlib import Path
import ctypes
import sys

import numpy as np
import torch
from torch.overrides import TorchFunctionMode
from torch._inductor.decomposition import decompositions
from transformers import AutoTokenizer

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from cache import update
from frontend.weights import weights

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)


# ===----------------------------------------------------------------------===
# Step 1: Import one FP32 graph of the decoder, the token count dynamic; 8
# tokens only give the trace an example. The matrix products go to the mesh
# in the midend (convert-linalg-to-magia), which computes them in FP16; the
# weights of the Linears are kept the way the mesh reads them.
# ===----------------------------------------------------------------------===
model = import_module("models.models.qwen3-0-6b.model")
decoder = model.load()
tokens, start, keys, values = model.inputs(8)
torch._dynamo.mark_dynamic(tokens, 1, min=1, max=model.capacity)
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, "IndexPutOp": update},
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
graph.dump(output / "01-import.txt")
params = weights(graph, params)
sort_graph(graph)
graph.dump(output / "02-weights.txt")
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir()
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
# forward takes the parameters grouped by type: the FP16 weights, the rest.
for index, dtype in enumerate((torch.float16, torch.float32)):
    np.concatenate(
        [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")
(output / "input-order.txt").write_text(" ".join(map(str, order)) + "\n")


# ===----------------------------------------------------------------------===
# Step 2: The check of the compiled model (check.cpp): the prompt and the
# tokens the model writes after it, each the largest logit, with the matrix
# products the graph hands to the mesh computed as the mesh computes them:
# the operands rounded to FP16, the hart's reference magia_gemm (argv[1], a
# shared library of stack/runtime/reference/gemm.c), the result back in FP32.
# ===----------------------------------------------------------------------===
reference = ctypes.CDLL(sys.argv[1])
reference.magia_gemm.restype = None
reference.magia_gemm.argtypes = [ctypes.c_uint64] * 7


sliced = []


def gemm(x, w):
    """X [M, N] times W [N, K] on the mesh, from zero."""
    xs = np.ascontiguousarray(x.to(torch.float16).numpy())
    ws = np.ascontiguousarray(w.to(torch.float16).numpy())
    zs = np.zeros((xs.shape[0], ws.shape[1]), dtype=np.float16)
    reference.magia_gemm(
        xs.ctypes.data, ws.ctypes.data, zs.ctypes.data, 1, *xs.shape, ws.shape[1]
    )
    # The slice of the model an RTL simulation runs (slice.cpp): the first
    # product of the first decode step, the query projection of layer 0 for
    # one token and its first 64 output features (all 1024 terms), its
    # operands and result as the mesh holds them.
    if xs.shape[0] == 1 and not sliced:
        sliced.append(True)
        xs.tofile(output / "slice-x.data")
        np.ascontiguousarray(ws[:, :64]).tofile(output / "slice-w.data")
        np.ascontiguousarray(zs[:, :64]).tofile(output / "slice-z.data")
    return torch.from_numpy(zs).float()


class Mesh(TorchFunctionMode):
    """The Linears, which have no bias, and the two products of an attention
    (four dimensions: a product a head) go to the mesh."""

    def __torch_function__(self, function, types, args=(), kwargs=None):
        if function is torch.nn.functional.linear:
            x, weight, bias = args
            if bias is not None:
                raise ValueError("a Linear of the decoder has a bias")
            return gemm(x.reshape(-1, x.shape[-1]), weight.t()).reshape(
                *x.shape[:-1], weight.shape[0]
            )
        if function is torch.matmul and args[0].ndim == 4:
            a, b = args
            return torch.stack(
                [gemm(x, w) for x, w in zip(a.flatten(0, 1), b.flatten(0, 1))]
            ).reshape(*a.shape[:-1], b.shape[-1])
        return function(*args, **(kwargs or {}))


steps = 8
tokenizer = AutoTokenizer.from_pretrained(model.checkpoint)
prompt = (model.checkpoint.parent.parent / "input.txt").read_text().strip()
tokens = tokenizer(prompt, return_tensors="pt").input_ids
_, start, keys, values = model.inputs(tokens.shape[1])
tokens.numpy().astype("int64").tofile(output / "check-prompt.data")
written = []
last = []
position = 0
with torch.no_grad(), Mesh():
    for _ in range(steps):
        logits, keys, values = decoder(tokens, start, keys, values)
        position += tokens.shape[1]
        last.append(logits[0, -1, :].numpy().astype(np.float32))
        tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        start = torch.tensor([position], dtype=start.dtype)
        written.append(tokens.item())
np.array(written, dtype=np.int64).tofile(output / "check-tokens.data")
# The logits of the last position of every step, the output the check
# compares.
np.stack(last).tofile(output / "check-logits.data")
print(f"{prompt!r} -> {tokenizer.decode(written)!r}")
