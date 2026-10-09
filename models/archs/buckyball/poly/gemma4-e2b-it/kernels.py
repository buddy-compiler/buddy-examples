import math
import struct

from buddy.compiler.graph.operation import FlashAttentionForCpuPrefillOp, Op
from buddy_mlir import ir
from buddy_mlir.dialects import arith, bufferization, func, memref


class BankFlashAttentionOp(Op):
    pass


def map_flash_attention(graph):
    for node in list(graph.body):
        if not isinstance(node, FlashAttentionForCpuPrefillOp):
            continue
        mask_name = node.kwargs.get("attn_mask")
        if mask_name is None or len(node.args) != 3:
            raise ValueError("Gemma bank attention requires Q/K/V and an explicit mask")
        mask_name = str(mask_name)
        if mask_name not in graph.node_table:
            raise ValueError("Gemma bank attention mask is absent from the graph")
        replacement = BankFlashAttentionOp()
        replacement._name = node.name
        replacement._arguments = [*node.args, mask_name]
        replacement._keyword_arguments = {
            key: value for key, value in node.kwargs.items() if key != "attn_mask"
        }
        replacement._tensor_meta = dict(node.tensor_meta)
        replacement._op_type = node._op_type
        replacement._parents = list(dict.fromkeys([*node._parents, mask_name]))
        replacement._children = list(node._children)
        replacement.trace_meta = node.trace_meta
        graph._body[graph._body.index(node)] = replacement
        graph.node_table[node.name] = replacement
        if node.name not in graph.node_table[mask_name]._children:
            graph.node_table[mask_name]._children.append(node.name)
    for group in graph.op_groups:
        graph.op_groups[group] = [
            graph.node_table[node.name] for node in graph.op_groups[group]
        ]


def lower_flash_attention(node, symbols):
    q, k, v, mask = [symbols[(name, 0)] for name in node.args]
    inputs = [q, k, v, mask]
    types = [ir.RankedTensorType(value.type) for value in inputs]
    shape = list(types[0].shape)
    key_shape, value_shape, mask_shape = [list(t.shape) for t in types[1:]]
    f32 = ir.F32Type.get()
    if (
        any(t.rank != 4 or t.element_type != f32 for t in types)
        or any(size <= 0 for t in types for size in t.shape)
        or shape[-1] % 16
        or key_shape[:2] != shape[:2]
        or key_shape[-1] != shape[-1]
        or value_shape != key_shape
        or mask_shape != [shape[0], 1, shape[2], key_shape[2]]
        or [list(s) for s in node.tensor_meta["shape"]] != [shape, shape[:-1]]
    ):
        raise ValueError(
            "Gemma bank attention requires static FP32 Q/K/V and mask shapes"
        )
    output_types = [
        ir.RankedTensorType.get(shape, f32),
        ir.RankedTensorType.get(shape[:-1], f32),
    ]
    outputs = [
        memref.AllocOp(ir.MemRefType.get(t.shape, f32), [], []).result
        for t in output_types
    ]
    unranked = ir.UnrankedMemRefType.get(f32, None)
    arguments = [memref.CastOp(unranked, value).result for value in outputs]
    for value, t in zip(inputs, types):
        layout = ir.StridedLayoutAttr.get(
            ir.ShapedType.get_dynamic_size(),
            [ir.ShapedType.get_dynamic_size()] * t.rank,
        )
        buffer = bufferization.ToBufferOp(
            ir.MemRefType.get(t.shape, f32, layout), value, read_only=True
        ).result
        arguments.append(memref.CastOp(unranked, buffer).result)
    scale = node.kwargs.get("scale")
    scale = 1 / math.sqrt(shape[-1]) if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("Gemma bank attention requires a finite scale")
    scale_bits = struct.unpack("<I", struct.pack("<f", scale))[0]
    arguments.append(
        arith.ConstantOp(ir.IntegerType.get_signless(32), scale_bits).result
    )
    module = ir.InsertionPoint.current.block.owner
    while module.operation.name != "builtin.module":
        module = module.operation.parent
    name = "rvv_flash_attention"
    table = ir.SymbolTable(module)
    function_type = ir.FunctionType.get([arg.type for arg in arguments], [])
    if name not in table:
        with ir.InsertionPoint.at_block_begin(module.regions[0].blocks[0]):
            function = func.FuncOp(name, function_type, visibility="private")
            function.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    elif ir.TypeAttr(table[name].attributes["function_type"]).value != function_type:
        raise ValueError("Gemma bank attention callee type mismatch")
    func.CallOp([], name, arguments)
    return tuple(
        bufferization.ToTensorOp(t, value, restrict=True, writable=True).result
        for t, value in zip(output_types, outputs)
    )
