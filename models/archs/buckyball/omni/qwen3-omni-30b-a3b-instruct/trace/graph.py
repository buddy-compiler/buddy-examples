import math

import torch
from torch._inductor.decomposition import decompositions
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy_mlir import ir
from stack.compiler.quant.mxfp8_graph import quantize_graph

from ..permute.layout import reorder, TILE_K, TILE_ROWS


def emit(name, stage, inputs, target, quantized, output, compiler_build):
    compiler = DynamoCompiler(primary_registry=tosa.ops_registry,
                              aot_autograd_decomposition=decompositions,
                              func_name=f"forward_{name}")
    with torch.no_grad():
        graphs = compiler.importer(stage, **inputs)
    if len(graphs) != 1:
        raise ValueError(f"{name} did not capture as one graph")
    graph = graphs[0]
    params = compiler.imported_params[graph]
    positions = {value.data_ptr(): index for value, index in zip(graph._runtime_inputs_ref, graph._inputs)}
    graph._inputs = [positions[value.data_ptr()] for value in inputs.values()]
    graph._runtime_inputs_ref = list(inputs.values())
    names = {value.data_ptr(): key for key, value in
             list(stage.named_parameters()) + list(stage.named_buffers())}
    ordered_names = [names[value.data_ptr()] for value in params]
    parameter_shapes = [list(value.shape) for value in params]
    graph.fuse_ops([simply_fuse])
    parameters = output / name
    parameters.mkdir(parents=True, exist_ok=True)
    if quantized:
        quantize_graph(graph, params, ordered_names, parameters, name,
                       compiler_build / "bin/rax-pack", weights=quantized,
                       reorder=reorder, tile_rows=TILE_ROWS, tile_k=TILE_K)
    else:
        (parameters / "params.f32").write_bytes(b"".join(value.detach().numpy().tobytes() for value in params))
    symbol = f"subgraph_{name}"
    graph.op_groups[symbol] = graph.op_groups.pop("subgraph0")
    graph.group_map_device[symbol] = graph.device
    from examples.balls.smatmul.compiler.python.fp32 import apply
    apply(graph)
    driver = GraphDriver(graph)
    subgraph = driver.subgraphs[0]
    subgraph.lower_to_top_level_ir()
    module = subgraph._imported_module
    with module.context:
        function = next(op for op in module.body.operations if op.operation.name == "func.func")
        attrs = []
        for argument in function.regions[0].blocks[0].arguments:
            shape = list(ir.RankedTensorType(argument.type).shape)
            strides = [math.prod(shape[i + 1:]) for i in range(len(shape))]
            attrs.append(ir.DictAttr.get({
                "bufferization.buffer_layout": ir.Attribute.parse(f"strided<{strides}, offset: ?>"),
                "bufferization.writable": ir.BoolAttr.get(False),
            }))
        function.attributes["arg_attrs"] = ir.ArrayAttr.get(attrs)
    (output / f"{name}.mlir").write_text(str(module))
    forward = driver.construct_main_graph(True)
    with forward.context:
        for op in forward.body.operations:
            if op.operation.name == "func.func" and ir.StringAttr(op.attributes["sym_name"]).value == symbol:
                op.attributes["buckyball.target"] = ir.StringAttr.get(target)
    (output / f"{name}-forward.mlir").write_text(str(forward))
    return {"floats": sum(p.numel() for p in params if p.dtype == torch.float32),
            "bytes": sum(p.numel() for p in params if p.dtype == torch.int8),
            "parameters": ordered_names, "shapes": parameter_shapes, "quantized": sorted(quantized)}
