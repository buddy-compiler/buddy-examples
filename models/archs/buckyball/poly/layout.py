import math

import torch
from buddy.compiler.graph.operation import SliceOp, PermuteOp, ViewOp
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.partition import tensor_type
from compiler.frontend.quant import MXINT8ToFP32DequantOp


def ball(device, depth):
    """Every MXINT8 weight a poly core decodes along K, stored as
    Mxint8matmulBall reads [K block, output row, packed values] in place.
    Each block holds codes, exponents and padding to a bank row. Four
    output rows must fit in a bank. The graph restores the logical [N, K]
    tensor, so the layout changes no quantized value."""
    host_parameters = {
        name
        for segment in device.segments
        if segment.core_class != "poly"
        for name in segment.params
    }
    done = set()
    for segment in device.segments:
        if segment.core_class != "poly":
            continue
        ops = []
        for key, node in segment.ops:
            name = node.args[0] if isinstance(node, MXINT8ToFP32DequantOp) else None
            if (
                name not in segment.params
                or name in host_parameters
                or node.args[1] != 1
            ):
                ops.append((key, node))
                continue
            if name in done:
                raise ValueError(f"{name}: a weight decoded twice")
            done.add(name)
            placeholder, value = device.parameters[name]
            rows, packed = value.shape
            k = packed * 32 // 33
            limit = min(k, (depth // 4 * 16 // 33) * 32)
            chunk = next(c for c in range(limit // 32 * 32, 0, -32) if k % c == 0)
            blocks = k // chunk
            width = 16 * (chunk // 16 + -(-chunk // 512))
            height = -(-rows // 4) * 4
            padded = torch.zeros(blocks, height, width, dtype=value.dtype)
            padded[:, :rows, :chunk] = (
                value[:, :k].reshape(rows, blocks, chunk).permute(1, 0, 2)
            )
            padded[:, :rows, chunk : chunk + chunk // 32] = (
                value[:, k:].reshape(rows, blocks, chunk // 32).permute(1, 0, 2)
            )
            placeholder.tensor_meta["shape"] = [blocks, height, width]
            device.parameters[name] = (placeholder, padded)
            source, shape = name, [blocks, height, width]
            for axis, size in ((1, rows), (2, chunk + chunk // 32)):
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
            decoded = MXINT8ToFP32DequantOp()
            decoded.name = node.name + "_blocks"
            decoded.add_argument(source)
            decoded.add_argument(2)
            decoded.add_parent(source)
            decoded.tensor_meta = {
                "shape": [blocks, rows, chunk],
                "dtype": TensorDType.Float32,
            }
            ordered = PermuteOp()
            ordered.name = node.name + "_rows"
            ordered.add_argument(decoded.name)
            ordered.add_argument([1, 0, 2])
            ordered.add_parent(decoded.name)
            ordered.tensor_meta = {
                "shape": [rows, blocks, chunk],
                "dtype": TensorDType.Float32,
            }
            restored = ViewOp()
            restored.name = node.name
            restored.add_argument(ordered.name)
            restored.add_argument([rows, k])
            restored.add_parent(ordered.name)
            restored.tensor_meta = dict(node.tensor_meta)
            ops.extend((key, item) for item in (decoded, ordered, restored))
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
