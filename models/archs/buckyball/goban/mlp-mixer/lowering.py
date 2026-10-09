from copy import deepcopy
from math import prod

from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.graph import Graph, NodeType
from buddy.compiler.graph.operation import MegaKernelOp, OutputOp, PlaceholderOp
from buddy.compiler.graph.type import TensorDType
from buddy_mlir import ir


def emit(graph, output):
    kernels = [node for node in graph.body if isinstance(node, MegaKernelOp)]
    if len(kernels) != 50 or any(len(node._stages) != 1 for node in kernels):
        raise ValueError("Mixer-B/16 requires fifty independent Linear kernels")
    shapes = []
    for index, original in enumerate(kernels):
        kernel = deepcopy(original)
        source = graph.node_table[str(kernel.args[0])]
        rows, reduction = source.tensor_meta["shape"]
        columns = kernel.tensor_meta["shape"][1]
        chunk = 192 if rows == 768 else (1 if rows == 1 else 64)
        shapes.append((rows, reduction, columns, chunk))
        isolated = Graph(graph._ops_registry, f"linear{index}", graph.device)
        for parameter in graph.params:
            isolated.add_node(deepcopy(parameter), NodeType.FakeNode)
        activation = PlaceholderOp()
        activation.name = str(kernel.args[0])
        activation._tensor_meta = {"shape": [chunk, reduction], "dtype": TensorDType.Float32}
        isolated.add_node(activation, NodeType.InputNode)
        kernel._tensor_meta["shape"] = [chunk, columns]
        kernel._children = ["output"]
        kernel._stages[0]._tensor_meta["shape"] = [chunk, columns]
        isolated.add_node(kernel)
        result = OutputOp()
        result.name = "output"
        result.add_argument(kernel.name)
        isolated.add_node(result)
        isolated.op_groups = {f"subgraph{index}": [kernel]}
        isolated.group_map_device = {f"subgraph{index}": graph.device}
        driver = GraphDriver(isolated)
        subgraph = driver.subgraphs[0]
        subgraph.lower_to_top_level_ir()
        module = subgraph._imported_module
        with module.context:
            function = next(op for op in module.body.operations if op.operation.name == "func.func")
            attributes = []
            for argument in function.regions[0].blocks[0].arguments:
                shape = list(ir.RankedTensorType(argument.type).shape)
                strides = [prod(shape[i + 1:]) for i in range(len(shape))]
                attributes.append(ir.DictAttr.get({"bufferization.buffer_layout": ir.Attribute.parse(f"strided<{strides}, offset: ?>")}))
            function.attributes["arg_attrs"] = ir.ArrayAttr.get(attributes)
        (output / f"subgraph{index}.mlir").write_text(str(module))
        (output / f"linear{index}.mlir").write_text(str(driver.construct_main_graph(True)))
    return shapes
