import math

from buddy.compiler.graph.operation import PermuteOp, SliceOp, ViewOp
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.partition import tensor_type
from compiler.frontend.quant import MXINT8ToFP32DequantOp
from frontend.quant.layout import stored


def ball(device):
    """Every weight a product of the partitioned program reads through
    mxint8_to_fp32 and a transpose, stored as the mxint8mm Ball of piplus
    reads it in place ([blocks][N][S], frontend/quant/layout.py) and read
    back through slice, mxint8_to_fp32 along axis 2, permute and view. A
    rank's piece of a weight is stored on its own, in the blocks of its K."""
    for segment in device.segments:
        readers = {}
        for _, node in segment.ops:
            for parent in node.parents:
                readers.setdefault(parent, []).append(node)
        ops = []
        replaced = {}
        for key, node in segment.ops:
            if node.name in replaced:
                ops.extend((key, item) for item in replaced.pop(node.name))
                continue
            name = node.args[0] if isinstance(node, MXINT8ToFP32DequantOp) else None
            if name not in segment.params or node.args[1] != 1:
                ops.append((key, node))
                continue
            (transpose,) = readers[node.name]
            if not isinstance(transpose, PermuteOp) or list(transpose.args[1]) != [
                1,
                0,
            ]:
                raise ValueError(f"{node.name}: a weight read without its transpose")
            placeholder, value = device.parameters[name]
            n, k = node.tensor_meta["shape"]
            if list(placeholder.tensor_meta["shape"]) == [n, k + k // 32]:
                value = stored(value)
                placeholder.tensor_meta["shape"] = list(value.shape)
                device.parameters[name] = (placeholder, value)
            blocks, _, _ = placeholder.tensor_meta["shape"]
            kc = k // blocks
            sliced = SliceOp()
            sliced.name = f"{name}_blocks"
            for argument in (name, 2, 0, kc + kc // 32):
                sliced.add_argument(argument)
            sliced.add_parent(name)
            sliced.tensor_meta = {
                "shape": [blocks, n, kc + kc // 32],
                "dtype": TensorDType.Int8,
            }
            decoded = MXINT8ToFP32DequantOp()
            decoded.name = node.name
            decoded.add_argument(sliced.name)
            decoded.add_argument(2)
            decoded.add_parent(sliced.name)
            decoded.tensor_meta = {
                "shape": [blocks, n, kc],
                "dtype": TensorDType.Float32,
            }
            ordered = PermuteOp()
            ordered.name = f"{transpose.name}_blocks"
            ordered.add_argument(decoded.name)
            ordered.add_argument([0, 2, 1])
            ordered.add_parent(decoded.name)
            ordered.tensor_meta = {
                "shape": [blocks, kc, n],
                "dtype": TensorDType.Float32,
            }
            operand = ViewOp()
            operand.name = transpose.name
            operand.add_argument(ordered.name)
            operand.add_argument([k, n])
            operand.add_parent(ordered.name)
            operand.tensor_meta = dict(transpose.tensor_meta)
            ops.extend((key, item) for item in (sliced, decoded))
            replaced[transpose.name] = [ordered, operand]
        if replaced:
            raise ValueError(f"{segment.name}: transposes of weights not read")
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
