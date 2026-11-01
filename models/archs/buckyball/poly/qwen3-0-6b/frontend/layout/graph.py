import math

import torch
from buddy.compiler.graph.operation import SliceOp
from compiler.frontend.partition import tensor_type
from compiler.frontend.quant import MXINT8ToFP32DequantOp


def ball(device):
    """Every MXINT8 weight a poly core decodes along K, stored as
    Mxint8matmulBall reads its rows in place, one bank row group per row:
    the K codes, the K / 32 exponents, zeros to 16 * S(K) bytes
    (S(K) = K / 16 + ceil(K / 512) bank rows); the rows padded with zeros
    to a multiple of four, the least N of the Ball. The graph slices the
    padding off before the decoding, so software decoding computes the same
    values. The packs of the parameters are laid out again."""
    done = set()
    for segment in device.segments:
        if segment.core_class != "poly":
            continue
        ops = []
        for key, node in segment.ops:
            name = node.args[0] if isinstance(node, MXINT8ToFP32DequantOp) else None
            if name not in segment.params or node.args[1] != 1:
                ops.append((key, node))
                continue
            if name in done:
                raise ValueError(f"{name}: a weight decoded twice")
            done.add(name)
            placeholder, value = device.parameters[name]
            rows, packed = value.shape
            k = packed * 32 // 33
            width = 16 * (k // 16 + -(-k // 512))
            height = -(-rows // 4) * 4
            if (height, width) == (rows, packed):
                ops.append((key, node))
                continue
            padded = torch.zeros(height, width, dtype=value.dtype)
            padded[:rows, :packed] = value
            placeholder.tensor_meta["shape"] = [height, width]
            device.parameters[name] = (placeholder, padded)
            source, shape = name, [height, width]
            for axis, size in ((0, rows), (1, packed)):
                if shape[axis] == size:
                    continue
                piece = SliceOp()
                piece.name = f"{name}_ball{axis}"
                for argument in (source, axis, 0, size):
                    piece.add_argument(argument)
                piece.add_parent(source)
                shape[axis] = size
                piece.tensor_meta = {
                    "shape": list(shape),
                    "dtype": placeholder.tensor_meta["dtype"],
                }
                ops.append((key, piece))
                source = piece.name
            node._arguments[0] = source
            node._parents = [
                source if parent == name else parent for parent in node._parents
            ]
            ops.append((key, node))
        segment.ops = ops
    device.packs, device.offsets = {}, {}
    for segment in device.segments:
        for name in segment.params:
            if name in device.offsets:
                continue
            node, _ = device.parameters[name]
            element = tensor_type({"shape": [], "dtype": node.tensor_meta["dtype"]})[
                7:-1
            ]
            count = math.prod(int(d) for d in node.tensor_meta["shape"])
            offset = device.packs.get(element, 0)
            device.offsets[name] = (element, offset, count)
            device.packs[element] = offset + count
