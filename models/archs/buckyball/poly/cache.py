"""Insert the contiguous token interval used by the decoder's KV-cache updates."""

import buddy_mlir.ir as ir
from buddy_mlir.dialects import arith, tensor


def update(node, symbols):
    cache = symbols[(str(node.args[0]), 0)]
    positions = symbols[(str(node.args[1][2]), 0)]
    values = symbols[(str(node.args[2]), 0)]
    shape = list(ir.RankedTensorType(values.type).shape)
    index = ir.IndexType.get()
    zero = arith.ConstantOp(index, ir.IntegerAttr.get(index, 0)).result
    start = arith.IndexCastOp(index, tensor.ExtractOp(positions, [zero])).result
    offsets = [0, 0, ir.ShapedType.get_dynamic_size(), 0]
    dynamic_sizes = []
    for axis, size in enumerate(shape):
        if size < 0:
            dim = arith.ConstantOp(index, ir.IntegerAttr.get(index, axis)).result
            dynamic_sizes.append(tensor.DimOp(values, dim).result)
    return tensor.InsertSliceOp(
        values,
        cache,
        [start],
        dynamic_sizes,
        [],
        ir.DenseI64ArrayAttr.get(offsets),
        ir.DenseI64ArrayAttr.get(shape),
        ir.DenseI64ArrayAttr.get([1, 1, 1, 1]),
    ).result
