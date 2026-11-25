import buddy_mlir.ir as ir
from buddy_mlir.dialects import arith, bufferization, memref, scf, tensor
from buddy.compiler.ops import linalg


def topk_op(node, symbol_table):
    """Expose the packed hardware result and retain the standard reference body.

    YOLO scores are nonnegative sigmoid outputs; U32 encodings preserve their
    order. The public packed tensor is shared by value and index unpacking.
    """
    source = symbol_table[(str(node.args[0]), 0)]
    source_type = ir.RankedTensorType(source.type)
    if (
        len(node.args) != 2
        or node.kwargs
        or len(source_type.shape) != 2
        or source_type.shape[0] != 1
    ):
        raise ValueError(
            "YOLO Top-K requires one row and the largest/sorted last-axis contract"
        )
    element = source_type.element_type
    floating = element == ir.F32Type.get()
    if not floating and element != ir.IntegerType.get_signless(32):
        raise ValueError("YOLO Top-K scores must be FP32 or nonnegative INT32")
    k = node.args[1]
    i32, i64 = ir.IntegerType.get_signless(32), ir.IntegerType.get_signless(64)
    packed_type = ir.MemRefType.get([(k + 1) // 2, 4], i32)
    region = scf.ExecuteRegionOp([packed_type], no_inline=True)
    region.operation.attributes["pebble.topk_u32"] = ir.IntegerAttr.get(i64, k)
    block = region.operation.regions[0].blocks.append()
    with ir.InsertionPoint(block):
        values, indices = linalg.topk_op(node, symbol_table)
        value_buffer, index_buffer = (
            values.operation.operands[0],
            indices.operation.operands[0],
        )
        values.operation.erase()
        indices.operation.erase()
        packed = memref.AllocOp(packed_type, [], []).result
        zero, one, two, limit = [
            arith.ConstantOp(ir.IndexType.get(), x).result for x in (0, 1, 2, k)
        ]
        loop = scf.ForOp(zero, limit, one)
        with ir.InsertionPoint(loop.body):
            rank = loop.induction_variable
            row = arith.DivUIOp(rank, two).result
            column = arith.MulIOp(arith.RemUIOp(rank, two).result, two).result
            key = memref.LoadOp(value_buffer, [zero, rank]).result
            if floating:
                key = arith.BitcastOp(i32, key).result
            at = arith.TruncIOp(
                i32, memref.LoadOp(index_buffer, [zero, rank]).result
            ).result
            memref.StoreOp(key, packed, [row, column])
            memref.StoreOp(at, packed, [row, arith.AddIOp(column, one).result])
            scf.YieldOp([])
        if k % 2:
            tail = arith.ConstantOp(ir.IndexType.get(), k // 2).result
            three = arith.ConstantOp(ir.IndexType.get(), 3).result
            value = arith.ConstantOp(i32, 0).result
            memref.StoreOp(value, packed, [tail, two])
            memref.StoreOp(value, packed, [tail, three])
        scf.YieldOp([packed])
    packed_tensor = bufferization.ToTensorOp(
        ir.RankedTensorType.get([(k + 1) // 2, 4], i32),
        region.results[0],
        restrict=True,
    )
    values = memref.AllocOp(ir.MemRefType.get([1, k], element), [], []).result
    indices = memref.AllocOp(ir.MemRefType.get([1, k], i64), [], []).result
    zero, one, two, limit = [
        arith.ConstantOp(ir.IndexType.get(), x).result for x in (0, 1, 2, k)
    ]
    loop = scf.ForOp(zero, limit, one)
    with ir.InsertionPoint(loop.body):
        rank = loop.induction_variable
        row = arith.DivUIOp(rank, two).result
        column = arith.MulIOp(arith.RemUIOp(rank, two).result, two).result
        key = tensor.ExtractOp(packed_tensor.result, [row, column]).result
        if floating:
            key = arith.BitcastOp(element, key).result
        at = arith.ExtUIOp(
            i64,
            tensor.ExtractOp(
                packed_tensor.result, [row, arith.AddIOp(column, one).result]
            ).result,
        ).result
        memref.StoreOp(key, values, [zero, rank])
        memref.StoreOp(at, indices, [zero, rank])
        scf.YieldOp([])
    return (
        bufferization.ToTensorOp(
            ir.RankedTensorType.get([1, k], element), values, restrict=True
        ),
        bufferization.ToTensorOp(
            ir.RankedTensorType.get([1, k], i64), indices, restrict=True
        ),
    )
