from buddy.compiler.graph.operation import (
    AddMMOp,
    AddOp,
    Conv2dOp,
    MatmulOp,
    MaxPool2dOp,
    MeanOp,
    PermuteOp,
    ReluOp,
    TOp,
)
from buddy.compiler.graph.transform.fuse_ops import decompose_addmm_to_mm_add


def nhwc(shape):
    batch, channels, height, width = shape
    return [batch, height, width, channels]


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
    return [values[node.name] for node in graph.params]
