from importlib import import_module
from pathlib import Path
import ctypes
import sys

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions

from frontend.weights import weights

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)


# ===----------------------------------------------------------------------===
# Step 1: Import the model as one FP32 graph. The matrix products go to the
# mesh in the midend (convert-linalg-to-magia), which computes them in FP16;
# the weights of the Linears are kept the way the mesh reads them.
# ===----------------------------------------------------------------------===
bert = import_module("models.models.bert-base-uncased-emotion.model")
emotion = import_module("models.models.bert-base-uncased-emotion.data")
model = bert.load()
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    trace=None,
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *bert.inputs())
params = compiler.imported_params[graph]
graph.dump(output / "01-import.txt")
params = weights(graph, params)
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


# ===----------------------------------------------------------------------===
# Step 2: The check of the compiled model (check.cpp): the first validation
# sentences and the logits of the model, with the matrix products the graph
# hands to the mesh computed as the mesh computes them: the operands rounded
# to FP16, the hart's reference magia_gemm (argv[1], a shared library of
# stack/runtime/reference/gemm.c), the result back in FP32.
# ===----------------------------------------------------------------------===
reference = ctypes.CDLL(sys.argv[1])
reference.magia_gemm.restype = None
reference.magia_gemm.argtypes = [ctypes.c_uint64] * 6


sliced = []


def gemm(x, w):
    """X [M, N] times W [N, K] on the mesh, from zero."""
    xs = np.ascontiguousarray(x.to(torch.float16).numpy())
    ws = np.ascontiguousarray(w.to(torch.float16).numpy())
    zs = np.zeros((xs.shape[0], ws.shape[1]), dtype=np.float16)
    reference.magia_gemm(
        xs.ctypes.data, ws.ctypes.data, zs.ctypes.data, *xs.shape, ws.shape[1]
    )
    # The slice of the model an RTL simulation runs (slice.cpp): the first
    # product of the model, the query projection of layer 0, for the first
    # four tokens of the first sentence and its first 64 output features (all
    # 768 terms), its operands and result as the mesh holds them.
    if not sliced:
        sliced.append(True)
        xs[:4].tofile(output / "slice-x.data")
        np.ascontiguousarray(ws[:, :64]).tofile(output / "slice-w.data")
        np.ascontiguousarray(zs[:4, :64]).tofile(output / "slice-z.data")
    return torch.from_numpy(zs).float()


class Mesh(TorchFunctionMode):
    """The Linears, their bias added in FP32, and the two products of an
    attention (four dimensions: a product a head) go to the mesh."""

    def __torch_function__(self, function, types, args=(), kwargs=None):
        if function is torch.nn.functional.linear:
            x, weight, bias = args
            return (
                gemm(x.reshape(-1, x.shape[-1]), weight.t()).reshape(
                    *x.shape[:-1], weight.shape[0]
                )
                + bias
            )
        if function is torch.matmul and args[0].ndim == 4:
            a, b = args
            return torch.stack(
                [gemm(x, w) for x, w in zip(a.flatten(0, 1), b.flatten(0, 1))]
            ).reshape(*a.shape[:-1], b.shape[-1])
        return function(*args, **(kwargs or {}))


tests, _ = emotion.load("validation", 64)
with torch.no_grad(), Mesh():
    expected = model(*tests)
for name, value in zip(("ids", "types", "mask", "positions"), tests):
    value.numpy().astype("int64").tofile(output / f"check-{name}.data")
expected.numpy().astype("float32").tofile(output / "check-logits.data")

# ===----------------------------------------------------------------------===
# The dataset of the accuracy runs (dataset.cpp, from the model region): the
# 2000 validation sentences, one record per file (32 int64s each of ids,
# positions, types and mask), the logits of the model with the mesh's
# products as above on each, and their labels.
# ===----------------------------------------------------------------------===
tests, labels = emotion.load("validation", 2000)
(output / "dataset").mkdir(exist_ok=True)
ids, types, mask, positions = tests
for name, value in (
    ("ids", ids),
    ("positions", positions),
    ("types", types),
    ("masks", mask),
):
    value.numpy().astype("int64").tofile(output / f"dataset/{name}.data")
with torch.no_grad(), Mesh():
    logits = torch.cat(
        [model(*batch) for batch in zip(*(value.split(100) for value in tests))]
    )
logits.numpy().astype("float32").tofile(output / "dataset/reference.data")
labels.numpy().astype("int64").tofile(output / "dataset/labels.data")
