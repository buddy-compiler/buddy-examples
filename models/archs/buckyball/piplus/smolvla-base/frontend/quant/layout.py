"""The weights of the quantized graphs in the layout the mxint8mm Ball reads.

A weight packed along K ([N, K + K / 32]: the K codes, then the K / 32
exponent bytes) is stored by K blocks of kc values, kc the largest multiple of
32 dividing K up to 128: [blocks][N][S], a row of a block the kc codes, the
kc / 32 exponent bytes and zero padding to S = kc + 16 bytes, the packed row
of kc values on 128-bit bank rows. The rows of a block are contiguous, so the
thread moves a tile of them into a bank as they are. 128 keeps a 32-row tile
of packed rows (9 bank rows each) in the 512-row banks of the piplus threads.

The graph reads the logical weight back with ordinary operations: a slice
drops the padding, mxint8_to_fp32 decodes along axis 2 ([blocks, N, kc]), a
permute [0, 2, 1] and a view give [K, N], the operand of the product the
transpose of the weight was. On the CPU the graph computes exactly what it
did; convert-linalg-to-acc of piplus recognizes the chain and reads the
parameter in place.

Not imported by buddy-pytorch-calibrate.py: the layout changes no value."""

import torch

from buddy.compiler.graph.operation import PermuteOp, SliceOp, ViewOp
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import MXINT8ToFP32DequantOp
from frontend.quant.graph import operation, replace_uses


def block(k):
    """The K block: the largest multiple of 32 dividing K up to 128."""
    kc = 128
    while k % kc:
        kc -= 32
    return kc


def lay_out(graph, params):
    """Every weight read by a product through mxint8_to_fp32 and a transpose,
    stored [blocks][N][S] and read back through slice, mxint8_to_fp32 along
    axis 2, permute and view."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for weight in list(graph.params):
        if weight.tensor_meta["dtype"] != TensorDType.Int8:
            continue
        (decode,) = (graph.node_table[name] for name in weight.children)
        assert isinstance(decode, MXINT8ToFP32DequantOp)
        (transpose,) = (graph.node_table[name] for name in decode.children)
        assert isinstance(transpose, PermuteOp) and list(transpose.args[1]) == [1, 0]
        packed = values[weight.name]
        n = packed.shape[0]
        k = packed.shape[1] * 32 // 33
        kc = block(k)
        blocks = k // kc
        codes = packed[:, :k].reshape(n, blocks, kc)
        exponents = packed[:, k:].reshape(n, blocks, kc // 32)
        padding = torch.zeros(n, blocks, 16 - kc // 32, dtype=packed.dtype)
        values[weight.name] = (
            torch.cat([codes, exponents, padding], 2).permute(1, 0, 2).contiguous()
        )
        weight.tensor_meta["shape"] = [blocks, n, kc + 16]
        i8, f32 = TensorDType.Int8, TensorDType.Float32
        # The slice reads the parameter; the decode reads the slice.
        weight.children.remove(decode.name)
        decode.parents.remove(weight.name)
        decode.args[0] = weight.name + "_blocks"
        sliced = operation(
            graph,
            SliceOp,
            weight.name + "_blocks",
            [weight],
            [2, 0, kc + kc // 32],
            [blocks, n, kc + kc // 32],
            i8,
        )
        sliced.add_children(decode.name)
        decode.add_parent(sliced.name)
        decode.args[1] = 2
        decode.tensor_meta["shape"] = [blocks, n, kc]
        transpose.args[1] = [0, 2, 1]
        transpose.tensor_meta["shape"] = [blocks, kc, n]
        view = operation(
            graph, ViewOp, transpose.name + "_k", [transpose], [[k, n]], [k, n], f32
        )
        replace_uses(graph, transpose, view)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]
