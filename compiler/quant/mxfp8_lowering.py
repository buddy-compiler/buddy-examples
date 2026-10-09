def lower_codec(inputs, *, encoding):
    from buddy_mlir import ir
    from buddy_mlir.dialects import bufferization, func, memref

    input_type = ir.RankedTensorType(inputs[0].type)
    shape = list(input_type.shape)
    if not shape or any(size <= 0 for size in shape) or shape[-1] % 32:
        raise ValueError(
            "MXFP8 requires positive static dimensions and a last dimension divisible by 32"
        )
    scale_shape = [*shape[:-1], shape[-1] // 32]
    i8, f32 = ir.IntegerType.get_signless(8), ir.F32Type.get()
    if encoding:
        if input_type.element_type != f32:
            raise ValueError("MXFP8 encoder requires FP32 input")
        output_types = [
            ir.RankedTensorType.get(shape, i8),
            ir.RankedTensorType.get(scale_shape, i8),
        ]
    else:
        if input_type.element_type != i8 or ir.RankedTensorType(
            inputs[1].type
        ) != ir.RankedTensorType.get(scale_shape, i8):
            raise ValueError("MXFP8 decoder requires matching INT8 code/scale tensors")
        output_types = [ir.RankedTensorType.get(shape, f32)]
    outputs = [
        memref.AllocOp(ir.MemRefType.get(t.shape, t.element_type), [], []).result
        for t in output_types
    ]
    arguments = []
    for output in outputs:
        t = ir.MemRefType(output.type)
        arguments.append(
            memref.CastOp(
                ir.UnrankedMemRefType.get(t.element_type, None), output
            ).result
        )
    for value in inputs:
        t = ir.RankedTensorType(value.type)
        layout = ir.StridedLayoutAttr.get(
            ir.ShapedType.get_dynamic_size(),
            [ir.ShapedType.get_dynamic_size()] * t.rank,
        )
        buffer = bufferization.ToBufferOp(
            ir.MemRefType.get(t.shape, t.element_type, layout), value, read_only=True
        ).result
        arguments.append(
            memref.CastOp(
                ir.UnrankedMemRefType.get(t.element_type, None), buffer
            ).result
        )
    name = "rvv_mxfp8_encode" if encoding else "rvv_mxfp8_decode"
    module = ir.InsertionPoint.current.block.owner
    while module.operation.name != "builtin.module":
        module = module.operation.parent
    table = ir.SymbolTable(module)
    if name not in table:
        with ir.InsertionPoint.at_block_begin(module.regions[0].blocks[0]):
            function = func.FuncOp(
                name, ([arg.type for arg in arguments], []), visibility="private"
            )
            function.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    func.CallOp([], name, arguments)
    tensors = [
        bufferization.ToTensorOp(t, value, restrict=True, writable=True).result
        for t, value in zip(output_types, outputs)
    ]
    return tuple(tensors) if encoding else tensors[0]
