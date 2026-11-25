import torch

from buddy.compiler.graph.operation import (
    AddMMOp,
    Conv2dOp,
    MatmulOp,
    MeanOp,
    OutputOp,
    PermuteOp,
    PlaceholderOp,
    SliceOp,
    TOp,
)
from buddy.compiler.graph.transform.fuse_ops import decompose_addmm_to_mm_add
from buddy.compiler.graph.transform.quantization.quantize import sort_graph


def nhwc(shape):
    batch, channels, height, width = shape
    return [batch, height, width, channels]


def permute(graph, params):
    """Align the layouts with the pebble Balls; returns the new parameters.

    The whole feature extractor runs in NHWC: the runner passes the image in
    NHWC, and the weights keep the [F, C, H, W] checkpoint layout. The global
    average pools reduce H and W. The flatten before the classifier reads a
    1x1 image, whose NHWC bytes already are NCHW.
    """
    image = graph.inputs[0]
    image.tensor_meta["shape"] = nhwc(image.tensor_meta["shape"])
    for node in graph.body:
        if (
            isinstance(node, (PlaceholderOp, OutputOp))
            or len(node.tensor_meta["shape"]) != 4
        ):
            continue
        node.tensor_meta["shape"] = nhwc(node.tensor_meta["shape"])
        if isinstance(node, Conv2dOp):
            node._layout = "NHWC_FCHW"
        if isinstance(node, MeanOp):
            node.args[1] = [1, 2]
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for transpose in [node for node in graph.body if isinstance(node, TOp)]:
        view = PermuteOp()
        view.name = transpose.name
        graph.displace_node(transpose, view)
        view.add_argument([1, 0])
    traces = {
        node.name: node.trace_meta for node in graph.body if isinstance(node, AddMMOp)
    }
    decompose_addmm_to_mm_add(graph)
    for name, trace in traces.items():
        graph.node_table[name + "_decomposed_mm"].trace_meta = trace
    for matmul in [node for node in graph.body if isinstance(node, MatmulOp)]:
        transpose = graph.node_table[matmul.args[1]]
        weight = graph.node_table[transpose.args[0]]
        values[weight.name] = values[weight.name].T.contiguous()
        inputs, outputs = values[weight.name].shape
        padded = (outputs + 15) // 16 * 16
        values[weight.name] = torch.nn.functional.pad(
            values[weight.name], (0, padded - outputs)
        )
        weight.tensor_meta["shape"] = [inputs, padded]
        graph.replace_as_parent(transpose, [matmul.name], weight)
        weight.add_children(matmul.name)
        graph.delete_node(transpose, [weight])
        if padded == outputs:
            continue
        add = graph.node_table[matmul.children[0]]
        bias = graph.node_table[add.args[0]]
        values[bias.name] = torch.nn.functional.pad(
            values[bias.name], (0, padded - outputs)
        )
        bias.tensor_meta["shape"] = [padded]
        matmul.tensor_meta["shape"] = [1, padded]
        add.tensor_meta["shape"] = [1, padded]
        unpadded = SliceOp()
        unpadded.name = add.name + "_unpadded"
        for argument in (add.name, 1, 0, outputs):
            unpadded.add_argument(argument)
        unpadded.tensor_meta = {
            "shape": [1, outputs],
            "dtype": add.tensor_meta["dtype"],
        }
        users = list(add.children)
        graph.replace_as_parent(add, users, unpadded)
        for user in users:
            unpadded.add_children(user)
        unpadded.add_parent(add.name)
        add.children.clear()
        add.add_children(unpadded.name)
        graph.add_node(unpadded)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]
