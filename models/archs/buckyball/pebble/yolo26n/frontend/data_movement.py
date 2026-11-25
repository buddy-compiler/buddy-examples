import math
import torch
from buddy.compiler.graph.operation import (
    AddOp,
    ArangeOp,
    CatOp,
    MulOp,
    TensorConstantOp,
    ToCopyOp,
    UnsafeIndexOp,
    UnsqueezeOp,
    ViewOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import ConcatColumnsOp, GatherRowsOp
from frontend.quant.graph import operation, replace_result


def data_movement(graph):
    for node in list(graph.body):
        if node.tensor_meta.get("dtype") != TensorDType.Int8:
            continue
        shape = list(node.tensor_meta["shape"])
        if isinstance(node, UnsafeIndexOp):
            source = graph.node_table[node.args[0]]
            batch, height, width, channels = source.tensor_meta["shape"]
            assert batch == 1 and len(shape) == 4 and shape[3] == channels
            names = node.args[1]
            assert names[0] is None and names[3] is None
            y = graph.node_table[names[1]]
            assert isinstance(y, UnsqueezeOp) and y.args[1] == -1
            coordinates = []
            for name, size in ((y.args[0], shape[1]), (names[2], shape[2])):
                cast = graph.node_table[name]
                assert (
                    isinstance(cast, ToCopyOp)
                    and cast.kwargs["dtype"] == TensorDType.Int64
                )
                multiply = graph.node_table[cast.args[0]]
                assert isinstance(multiply, MulOp) and isinstance(
                    multiply.args[1], (int, float)
                )
                add = graph.node_table[multiply.args[0]]
                assert isinstance(add, AddOp) and isinstance(add.args[1], (int, float))
                sequence = graph.node_table[add.args[0]]
                assert isinstance(sequence, ArangeOp) and sequence.args == [size]
                coordinates.append(
                    (
                        (torch.arange(size, dtype=torch.float32) + add.args[1])
                        * multiply.args[1]
                    ).long()
                )
            indices = TensorConstantOp()
            indices.name = node.name + "_rows"
            indices.add_argument(
                (coordinates[0][:, None] * width + coordinates[1][None, :])
                .reshape(-1)
                .numpy()
            )
            indices.tensor_meta = {
                "shape": [shape[1] * shape[2]],
                "dtype": TensorDType.Int64,
            }
            graph.add_node(indices)
            matrix = operation(
                graph,
                ViewOp,
                node.name + "_source",
                [source],
                [height * width, channels],
                TensorDType.Int8,
            )
            matrix.add_argument([height * width, channels])
            moved = operation(
                graph,
                GatherRowsOp,
                node.name + "_dma",
                [indices, matrix],
                [shape[1] * shape[2], channels],
                TensorDType.Int8,
            )
        elif isinstance(node, CatOp) and len(shape) == 4 and node.args[1] % 4 == 3:
            rows = math.prod(shape[:-1])
            matrices = []
            for name in node.args[0]:
                source = graph.node_table[name]
                assert list(source.tensor_meta["shape"][:-1]) == shape[:-1]
                columns = source.tensor_meta["shape"][-1]
                matrix = operation(
                    graph,
                    ViewOp,
                    name + "_concat_" + node.name,
                    [source],
                    [rows, columns],
                    TensorDType.Int8,
                )
                matrix.add_argument([rows, columns])
                matrices.append(matrix)
            moved = operation(
                graph,
                ConcatColumnsOp,
                node.name + "_dma",
                matrices,
                [rows, shape[-1]],
                TensorDType.Int8,
            )
        else:
            continue
        restored = operation(
            graph, ViewOp, node.name + "_shape", [moved], shape, TensorDType.Int8
        )
        restored.add_argument(shape)
        replace_result(graph, node, restored)
    sort_graph(graph)
