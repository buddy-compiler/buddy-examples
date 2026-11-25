import math
import torch
from buddy.compiler.graph.operation import (
    MulOp,
    OutputOp,
    SigmoidOp,
    SliceOp,
    SoftmaxOp,
    ViewOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import (
    FP32ToInt8QuantOp,
    FP32ToInt32QuantOp,
    FixedRequantOp,
    Int32ToFP32DequantOp,
    LookupI32Op,
    SoftmaxI8Op,
)
from models.archs.buckyball.pebble.fixed import operation, parameter
from frontend.quant.graph import replace_result


def nonlinear(graph, params):
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    exponent = parameter(
        graph,
        values,
        "softmax_q30_table",
        torch.round(
            torch.exp(-torch.arange(1024, dtype=torch.float64) / 64) * 2**30
        ).to(torch.int32),
    )
    indices = torch.arange(1024)
    signed = torch.where(indices < 512, indices, indices - 1024).double()
    sigmoid_table = parameter(
        graph,
        values,
        "sigmoid_q24_table",
        torch.round(torch.sigmoid(signed / 64) * 2**24).to(torch.int32),
    )
    for node in list(graph.body):
        if not isinstance(node, SoftmaxOp):
            continue
        assert node.args[1:] == [-1, False]
        source = graph.node_table[node.args[0]]
        assert isinstance(source, MulOp) and isinstance(source.args[1], (int, float))
        view = graph.node_table[source.args[0]]
        assert isinstance(view, ViewOp)
        decoded = graph.node_table[view.args[0]]
        assert isinstance(decoded, Int32ToFP32DequantOp)
        shape = list(node.tensor_meta["shape"])
        matrix_shape = [math.prod(shape[:-1]), shape[-1]]
        raw = graph.node_table[decoded.args[0]]
        matrix = operation(
            graph, ViewOp, node.name + "_matrix", [raw], matrix_shape, TensorDType.Int32
        )
        matrix.add_argument(matrix_shape)
        scale = values[decoded.args[1]] * source.args[1]
        assert scale.numel() == 1 and scale.item() > 0
        mantissa, power = torch.frexp(scale.double().reshape(()) * 64)
        shift = 30 - power.to(torch.int64)
        assert 0 <= shift.item() <= 61
        multiplier = torch.round(mantissa * 2**30).to(torch.int32)
        m = parameter(
            graph, values, node.name + "_m", multiplier.expand(16).contiguous()
        )
        s = parameter(
            graph,
            values,
            node.name + "_s",
            shift.to(torch.int32).expand(16).contiguous(),
        )
        codes = operation(
            graph,
            FixedRequantOp,
            node.name + "_q6",
            [matrix, m, s],
            matrix_shape,
            TensorDType.Int32,
        )
        codes.add_argument(6)
        (consumer,) = [graph.node_table[name] for name in node.children]
        assert isinstance(consumer, FP32ToInt8QuantOp)
        factor = graph.node_table[consumer.args[1]]
        if isinstance(factor, SliceOp):
            factor = graph.node_table[factor.args[0]]
        factor_values = values[factor.name].reshape(-1)
        assert torch.all(factor_values == factor_values[0])
        output_multiplier = int(torch.round(factor_values[0].double() * 65536).item())
        assert 0 < output_multiplier < 2**30
        normalized = operation(
            graph,
            SoftmaxI8Op,
            node.name + "_integer",
            [codes, exponent],
            matrix_shape,
            TensorDType.Int8,
        )
        normalized.add_argument(output_multiplier)
        restored = operation(
            graph, ViewOp, node.name + "_shape", [normalized], shape, TensorDType.Int8
        )
        restored.add_argument(shape)
        replace_result(graph, consumer, restored)

    for node in list(graph.body):
        if not isinstance(node, SigmoidOp):
            continue
        shape = list(node.tensor_meta["shape"])
        source = graph.node_table[node.args[0]]
        scale = parameter(
            graph,
            values,
            node.name + "_index_scale",
            torch.full([1] * len(shape), 64.0),
        )
        codes = operation(
            graph,
            FP32ToInt32QuantOp,
            node.name + "_indices",
            [source, scale],
            shape,
            TensorDType.Int32,
        )
        matrix_shape = [math.prod(shape[:-1]), shape[-1]]
        matrix = operation(
            graph,
            ViewOp,
            node.name + "_matrix",
            [codes],
            matrix_shape,
            TensorDType.Int32,
        )
        matrix.add_argument(matrix_shape)
        looked = operation(
            graph,
            LookupI32Op,
            node.name + "_lookup",
            [matrix, sigmoid_table],
            matrix_shape,
            TensorDType.Int32,
        )
        restored = operation(
            graph, ViewOp, node.name + "_shape", [looked], shape, TensorDType.Int32
        )
        restored.add_argument(shape)
        step = parameter(
            graph, values, node.name + "_step", torch.full([1] * len(shape), 2.0**-24)
        )
        decoded = operation(
            graph,
            Int32ToFP32DequantOp,
            node.name + "_values",
            [restored, step],
            shape,
            TensorDType.Float32,
        )
        replace_result(graph, node, decoded)
    while True:
        dead = [
            node
            for node in graph.body
            if not node.children and not isinstance(node, OutputOp)
        ]
        if not dead:
            break
        for node in reversed(dead):
            graph.delete_node(
                node, [graph.node_table[parent] for parent in node.parents]
            )
    sort_graph(graph)
    return [values[node.name] for node in graph.params]
