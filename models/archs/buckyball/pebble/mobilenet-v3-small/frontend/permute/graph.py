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


def nhwc(shape):
    batch, channels, height, width = shape
    return [batch, height, width, channels]


def padded(size):
    return (size + 15) // 16 * 16


def insert_after(graph, node, new):
    """`new` reads `node` and takes over its users."""
    users = [user for user in node.children]
    graph.replace_as_parent(node, users, new)
    for user in users:
        new.add_children(user)
    new.add_parent(node.name)
    node.children.clear()
    node.add_children(new.name)
    graph.add_node(new)
    graph.body.remove(new)
    graph.body.insert(graph.body.index(node) + 1, new)


def permute(graph, params):
    """Align the layouts with the pebble Balls; returns the new parameters."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    output = next(node for node in graph.body if isinstance(node, OutputOp))
    logits_name = output.args[0]
    classes = graph.node_table[logits_name].tensor_meta["shape"][1]

    # The whole feature extractor runs in NHWC: the runner passes the image in
    # NHWC, and the weights keep the [F, C, H, W] checkpoint layout. The
    # global average pools reduce H and W. The flatten before the classifier
    # reads a 1x1 image, whose NHWC bytes already are NCHW.
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

    # x @ W.T + b of a Linear becomes x @ W' + b': smatmul reads its weights
    # as [K][N] with N in rows of 16, so W' is W.T stored [K, N] and both
    # padded with zeros to multiples of 16. The padded outputs are zero, and
    # so are the padded inputs of the next Linear.
    # t(W) is the 2-D permute(W, [1, 0]) the Buddy decomposition matches.
    for transpose in [node for node in graph.body if isinstance(node, TOp)]:
        permute = PermuteOp()
        permute.name = transpose.name
        graph.displace_node(transpose, permute)
        permute.add_argument([1, 0])
    # The decomposition drops the trace of an addmm; its matmul keeps it.
    traces = {
        node.name: node.trace_meta for node in graph.body if isinstance(node, AddMMOp)
    }
    decompose_addmm_to_mm_add(graph)
    for name, trace in traces.items():
        graph.node_table[name + "_decomposed_mm"].trace_meta = trace
    for matmul in [node for node in graph.body if isinstance(node, MatmulOp)]:
        transpose = graph.node_table[matmul.args[1]]
        weight = graph.node_table[transpose.args[0]]
        outputs, inputs = values[weight.name].shape
        stored = torch.zeros(padded(inputs), padded(outputs))
        stored[:inputs, :outputs] = values[weight.name].T
        values[weight.name] = stored
        weight.tensor_meta["shape"] = list(stored.shape)
        graph.replace_as_parent(transpose, [matmul.name], weight)
        weight.add_children(matmul.name)
        graph.delete_node(transpose, [weight])

        add = graph.node_table[matmul.children[0]]
        bias = graph.node_table[add.args[0]]
        stored_bias = torch.zeros(padded(outputs))
        stored_bias[:outputs] = values[bias.name]
        values[bias.name] = stored_bias
        bias.tensor_meta["shape"] = [padded(outputs)]
        for node in (matmul, add, *(graph.node_table[name] for name in add.children)):
            if not isinstance(node, OutputOp):
                node.tensor_meta["shape"] = [1, padded(outputs)]

    # The model returns the logits without their padding. The decomposition
    # replaced the node of the logits by one of the same name.
    logits = graph.node_table[logits_name]
    unpadded = SliceOp()
    unpadded.name = logits.name + "_unpadded"
    unpadded.add_argument(logits.name)
    unpadded.add_argument(1)
    unpadded.add_argument(0)
    unpadded.add_argument(classes)
    unpadded.tensor_meta = {
        "shape": [1, classes],
        "dtype": logits.tensor_meta["dtype"],
    }
    insert_after(graph, logits, unpadded)
    return [values[node.name] for node in graph.params]
