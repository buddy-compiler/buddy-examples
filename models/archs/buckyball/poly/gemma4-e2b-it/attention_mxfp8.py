import math
import struct

from buddy_mlir import ir
from buddy_mlir.dialects import arith, bufferization, func, memref


def lower_prefill_attention(node, symbols):
    inputs = [symbols[(name, 0)] for name in node.args[:6]]
    types = [ir.RankedTensorType(value.type) for value in inputs]
    q, kc, ks, vc, vs, mask = [list(t.shape) for t in types]
    f32, i8 = ir.F32Type.get(), ir.IntegerType.get_signless(8)
    if (
        any(t.rank != 4 or any(d <= 0 for d in t.shape) for t in types)
        or [t.element_type for t in types] != [f32, i8, i8, i8, i8, f32]
        or q[-1] % 32
        or kc != vc
        or ks != vs
        or ks != [*kc[:-1], kc[-1] // 32]
        or q[0] != kc[0]
        or q[1] % kc[1]
        or q[-1] != kc[-1]
        or mask != [q[0], 1, q[2], kc[2]]
    ):
        raise ValueError("Gemma MXFP8 attention requires static compressed KV shapes")
    storage = memref.AllocOp(
        ir.MemRefType.get([q[0], q[2], q[1], q[3]], f32), [], []
    ).result
    output = memref.TransposeOp(
        ir.MemRefType.get(
            q,
            f32,
            ir.StridedLayoutAttr.get(0, [q[1] * q[2] * q[3], q[3], q[1] * q[3], 1]),
        ),
        storage,
        ir.AffineMap.get_permutation([0, 2, 1, 3]),
    ).result
    scores = memref.AllocOp(ir.MemRefType.get(q[:-1], f32), [], []).result
    unranked = ir.UnrankedMemRefType.get(f32, None)
    arguments = [memref.CastOp(unranked, v).result for v in (output, scores)]
    for value, tensor in zip(inputs, types):
        layout = ir.StridedLayoutAttr.get(
            ir.ShapedType.get_dynamic_size(),
            [ir.ShapedType.get_dynamic_size()] * tensor.rank,
        )
        buffer = bufferization.ToBufferOp(
            ir.MemRefType.get(tensor.shape, tensor.element_type, layout),
            value,
            read_only=True,
        ).result
        arguments.append(
            memref.CastOp(
                ir.UnrankedMemRefType.get(tensor.element_type, None), buffer
            ).result
        )
    scale = float(node.args[6])
    if not math.isfinite(scale):
        raise ValueError("Gemma MXFP8 attention requires a finite scale")
    bits = struct.unpack("<I", struct.pack("<f", scale))[0]
    arguments.append(arith.ConstantOp(ir.IntegerType.get_signless(32), bits).result)
    module = ir.InsertionPoint.current.block.owner
    while module.operation.name != "builtin.module":
        module = module.operation.parent
    name = "rvv_flash_attention_mxfp8"
    table = ir.SymbolTable(module)
    signature = ir.FunctionType.get([v.type for v in arguments], [])
    if name not in table:
        with ir.InsertionPoint.at_block_begin(module.regions[0].blocks[0]):
            function = func.FuncOp(name, signature, visibility="private")
            function.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    elif ir.TypeAttr(table[name].attributes["function_type"]).value != signature:
        raise ValueError("Gemma MXFP8 attention callee type mismatch")
    func.CallOp([], name, arguments)
    shape = [q[0], q[2], q[1] * q[3]]
    flat = memref.CollapseShapeOp(
        ir.MemRefType.get(shape, f32), storage, [[0], [1], [2, 3]]
    ).result
    return bufferization.ToTensorOp(
        ir.RankedTensorType.get(shape, f32), flat, restrict=True, writable=True
    ).result
