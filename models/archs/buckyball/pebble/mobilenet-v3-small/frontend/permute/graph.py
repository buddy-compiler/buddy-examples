from buddy.compiler.graph.operation import Conv2dOp, MeanOp, OutputOp, PlaceholderOp


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
    return params
