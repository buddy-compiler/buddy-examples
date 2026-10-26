from buddy.compiler.graph.operation import Conv2dOp, PermuteOp


def nhwc(shape):
    batch, channels, height, width = shape
    return [batch, height, width, channels]


def renamed(argument, old, new):
    """`argument` reading `new` wherever it read `old`, in lists too."""
    if isinstance(argument, list):
        return [renamed(item, old, new) for item in argument]
    return new if argument == old else argument


def permute(graph, params):
    """Align the layouts with the pebble Balls; returns the new parameters.

    The convolutions run in NHWC; everything else of YOLO26 (SiLU, split,
    concat, attention, upsampling and the Detect head) keeps NCHW. Each
    convolution reads its input through an NCHW-to-NHWC permute and writes
    its output through the inverse one. The weights keep the [F, C, H, W]
    checkpoint layout.
    """
    for conv in [node for node in graph.body if isinstance(node, Conv2dOp)]:
        source = graph.node_table[conv.args[0]]
        shape = source.tensor_meta["shape"]
        to_nhwc = PermuteOp()
        to_nhwc.name = conv.name + "_input_nhwc"
        to_nhwc.add_argument(source.name)
        to_nhwc.add_argument([0, 2, 3, 1])
        to_nhwc.add_parent(source.name)
        to_nhwc.add_children(conv.name)
        to_nhwc.tensor_meta = {
            "shape": nhwc(shape),
            "dtype": source.tensor_meta["dtype"],
        }
        source.children[source.children.index(conv.name)] = to_nhwc.name
        conv.parents[conv.parents.index(source.name)] = to_nhwc.name
        conv.args[0] = to_nhwc.name
        graph.add_node(to_nhwc)
        graph.body.remove(to_nhwc)
        graph.body.insert(graph.body.index(conv), to_nhwc)

        batch, channels, height, width = conv.tensor_meta["shape"]
        conv.tensor_meta["shape"] = [batch, height, width, channels]
        conv._layout = "NHWC_FCHW"
        to_nchw = PermuteOp()
        to_nchw.name = conv.name + "_output_nchw"
        to_nchw.add_argument(conv.name)
        to_nchw.add_argument([0, 3, 1, 2])
        to_nchw.tensor_meta = {
            "shape": [batch, channels, height, width],
            "dtype": conv.tensor_meta["dtype"],
        }
        # A user can read the output several times, also inside a list.
        for user in [graph.node_table[name] for name in conv.children]:
            user._arguments = renamed(user.args, conv.name, to_nchw.name)
            user._parents = renamed(user.parents, conv.name, to_nchw.name)
            to_nchw.add_children(user.name)
        to_nchw.add_parent(conv.name)
        conv.children.clear()
        conv.add_children(to_nchw.name)
        graph.add_node(to_nchw)
        graph.body.remove(to_nchw)
        graph.body.insert(graph.body.index(conv) + 1, to_nchw)
    return params
