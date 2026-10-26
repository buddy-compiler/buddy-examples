from buddy.compiler.graph.operation import (
    AddMMOp,
    Conv2dOp,
    MatmulOp,
    MaxPool2dOp,
    PermuteOp,
    ReluOp,
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
    graph.body.insert(graph.body.index(node) + 1, new)


def permute(graph, params):
    """Align the layouts with the pebble Balls; returns the new parameters."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))

    # The convolutions run in NHWC; the weights keep the [F, C, H, W]
    # checkpoint layout. The runner passes NHWC images.
    image = graph.inputs[0]
    image.tensor_meta["shape"] = nhwc(image.tensor_meta["shape"])
    for node in graph.body:
        if (
            isinstance(node, (Conv2dOp, MaxPool2dOp, ReluOp))
            and len(node.tensor_meta["shape"]) == 4
        ):
            node.tensor_meta["shape"] = nhwc(node.tensor_meta["shape"])
        if isinstance(node, Conv2dOp):
            node._layout = "NHWC_FCHW"
        if isinstance(node, MaxPool2dOp):
            node._layout = "NHWC"

    # The adaptive average pooling and the flatten before classifier.1 read
    # channels first: one transpose back to NCHW.
    pool = [node for node in graph.body if isinstance(node, MaxPool2dOp)][-1]
    batch, height, width, channels = pool.tensor_meta["shape"]
    nchw = PermuteOp()
    nchw.name = pool.name + "_nchw"
    nchw.add_argument(pool.name)
    nchw.add_argument([0, 3, 1, 2])
    nchw.tensor_meta = {
        "shape": [batch, channels, height, width],
        "dtype": pool.tensor_meta["dtype"],
    }
    insert_after(graph, pool, nchw)

    # x @ W.T + b of a Linear becomes x @ W' + b: smatmul reads its weights
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
