"""Keep the rank's small sorted candidate list while scanning logits once."""

import buddy_mlir.ir as ir
from buddy_mlir.dialects import arith, bufferization, memref, scf


def topk(node, symbols):
    source = symbols[(str(node.args[0]), 0)]
    shape = list(ir.RankedTensorType(source.type).shape)
    dtype = ir.RankedTensorType(source.type).element_type
    k, axis = node.args[1:3]
    if len(shape) != 2 or axis not in (-1, 1) or dtype != ir.F32Type.get():
        raise ValueError("Candidate selection requires a rank-two FP32 tensor")
    if not 0 < k <= shape[1] or (len(node.args) > 3 and not node.args[3]):
        raise ValueError("Candidate selection requires a positive largest top-k")
    index = ir.IndexType.get()
    i64 = ir.IntegerType.get_signless(64)
    constants = [
        arith.ConstantOp(index, ir.IntegerAttr.get(index, n)).result
        for n in (0, 1, shape[0], shape[1], k, k - 1)
    ]
    zero, one, rows, columns, count, last = constants
    negative_inf = arith.ConstantOp(
        dtype, ir.FloatAttr.get(dtype, float("-inf"))
    ).result
    zero_i64 = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, 0)).result
    values = memref.AllocOp(ir.MemRefType.get([shape[0], k], dtype), [], []).result
    indices = memref.AllocOp(ir.MemRefType.get([shape[0], k], i64), [], []).result
    memory = bufferization.ToBufferOp(ir.MemRefType.get(shape, dtype), source).result
    row_loop = scf.ForOp(zero, rows, one)
    with ir.InsertionPoint(row_loop.body):
        row = row_loop.induction_variable
        initialize = scf.ForOp(zero, count, one)
        with ir.InsertionPoint(initialize.body):
            at = [row, initialize.induction_variable]
            memref.StoreOp(negative_inf, values, at)
            memref.StoreOp(zero_i64, indices, at)
            scf.YieldOp([])
        scan = scf.ForOp(zero, columns, one)
        with ir.InsertionPoint(scan.body):
            column = scan.induction_variable
            value = memref.LoadOp(memory, [row, column]).result
            lowest = memref.LoadOp(values, [row, last]).result
            better = arith.CmpFOp(arith.CmpFPredicate.OGT, value, lowest).result
            insert = scf.IfOp(better, has_else=False)
            with ir.InsertionPoint(insert.then_block):
                position = arith.IndexCastOp(i64, column).result
                shift = scf.ForOp(zero, count, one, [value, position])
                with ir.InsertionPoint(shift.body):
                    at = [row, shift.induction_variable]
                    candidate, candidate_index = shift.inner_iter_args
                    old = memref.LoadOp(values, at).result
                    old_index = memref.LoadOp(indices, at).result
                    precedes = arith.CmpFOp(
                        arith.CmpFPredicate.OGT, candidate, old
                    ).result
                    equal = arith.CmpFOp(arith.CmpFPredicate.OEQ, candidate, old).result
                    earlier = arith.CmpIOp(
                        arith.CmpIPredicate.slt, candidate_index, old_index
                    ).result
                    precedes = arith.OrIOp(
                        precedes, arith.AndIOp(equal, earlier)
                    ).result
                    memref.StoreOp(arith.SelectOp(precedes, candidate, old), values, at)
                    memref.StoreOp(
                        arith.SelectOp(precedes, candidate_index, old_index),
                        indices,
                        at,
                    )
                    scf.YieldOp(
                        [
                            arith.SelectOp(precedes, old, candidate).result,
                            arith.SelectOp(precedes, old_index, candidate_index).result,
                        ]
                    )
                scf.YieldOp([])
            scf.YieldOp([])
        scf.YieldOp([])
    return (
        bufferization.ToTensorOp(
            ir.RankedTensorType.get([shape[0], k], dtype), values, restrict=True
        ).result,
        bufferization.ToTensorOp(
            ir.RankedTensorType.get([shape[0], k], i64), indices, restrict=True
        ).result,
    )
