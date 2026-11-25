import torch

from buddy.compiler.graph.operation import (
    AddMMOp,
    AddOp,
    Conv2dOp,
    MatmulOp,
    MaxPool2dOp,
    MeanOp,
    PermuteOp,
    ReluOp,
    SliceOp,
    TOp,
)
from buddy.compiler.graph.transform.fuse_ops import decompose_addmm_to_mm_add


def nhwc(shape):
    batch, channels, height, width = shape
    return [batch, height, width, channels]


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
    index = graph.body.index(node) + 1
    graph.body.insert(index, new)
    # The inputs and parameters at or after `index` move one place down.
    graph._inputs = [i + (i >= index) for i in graph._inputs]
    graph._fake_params = [i + (i >= index) for i in graph._fake_params]


def permute(graph, params):
    """Align the layouts with the pebble Balls; returns the new parameters."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))

    # The convolutions, residual adds and pooling run in NHWC; the weights
    # keep the [F, C, H, W] checkpoint layout. The runner passes NHWC images.
    image = graph.inputs[0]
    image.tensor_meta["shape"] = nhwc(image.tensor_meta["shape"])
    for node in graph.body:
        if isinstance(node, (Conv2dOp, MaxPool2dOp, ReluOp, AddOp, MeanOp)):
            node.tensor_meta["shape"] = nhwc(node.tensor_meta["shape"])
        if isinstance(node, Conv2dOp):
            node._layout = "NHWC_FCHW"
        if isinstance(node, MaxPool2dOp):
            node._layout = "NHWC"

    # The global average pooling reduces H and W, axes 1 and 2 in NHWC. Its
    # 1x1 result has the same bytes in NHWC and NCHW, so the flatten stays.
    mean = next(node for node in graph.body if isinstance(node, MeanOp))
    mean.args[1] = [1, 2]

    # x @ W.T + b of the Linear becomes x @ W' + b: smatmul reads its weights
    # as [K][N], so W' is W.T stored [K, N].
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
        values[weight.name] = values[weight.name].T.contiguous()
        weight.tensor_meta["shape"] = list(values[weight.name].shape)
        graph.replace_as_parent(transpose, [matmul.name], weight)
        weight.add_children(matmul.name)
        graph.delete_node(transpose, [weight])

        # The pebble compiler reads the weight in place, sixteen output
        # columns a line: N is padded with zeros to a multiple of 16, and so
        # is the bias. A slice after the bias drops the padded outputs.
        inputs, outputs = values[weight.name].shape
        padded = (outputs + 15) // 16 * 16
        if padded == outputs:
            continue
        values[weight.name] = torch.nn.functional.pad(
            values[weight.name], (0, padded - outputs)
        )
        weight.tensor_meta["shape"] = [inputs, padded]
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
        unpadded.add_argument(add.name)
        unpadded.add_argument(1)
        unpadded.add_argument(0)
        unpadded.add_argument(outputs)
        unpadded.tensor_meta = {
            "shape": [1, outputs],
            "dtype": add.tensor_meta["dtype"],
        }
        insert_after(graph, add, unpadded)
    return [values[node.name] for node in graph.params]
