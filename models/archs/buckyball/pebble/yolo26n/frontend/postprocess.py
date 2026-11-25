import torch
from buddy.compiler.graph.operation import (
    AddOp,
    CatOp,
    GatherOp,
    GetItemOp,
    MaxDimOp,
    MulOp,
    OutputOp,
    PermuteOp,
    RepeatOp,
    SigmoidOp,
    SplitWithSizesOp,
    SubOp,
    TopkOp,
    UnsqueezeOp,
    ViewOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import (
    ConcatColumnsOp,
    FixedRequantOp,
    GatherRowsOp,
    Int32ToFP32DequantOp,
    LookupI32Op,
    RowMaxU32Op,
)
from models.archs.buckyball.pebble.fixed import coefficients, parameter
from frontend.quant.graph import operation, replace_argument, replace_result


def postprocess(graph, params):
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    (sigmoid,) = [node for node in graph.body if isinstance(node, SigmoidOp)]
    (cat,) = [graph.node_table[name] for name in sigmoid.children]
    assert isinstance(cat, CatOp) and cat.args[0][1] == sigmoid.name
    (permuted,) = [graph.node_table[name] for name in cat.children]
    assert isinstance(permuted, PermuteOp) and permuted.args[1] == [0, 2, 1]
    (split,) = [graph.node_table[name] for name in permuted.children]
    assert isinstance(split, SplitWithSizesOp) and split.args[1:] == [[4, 80], -1]
    parts = {
        graph.node_table[name].args[1]: graph.node_table[name]
        for name in split.children
    }
    boxes, scores = parts[0], parts[1]
    users = [graph.node_table[name] for name in scores.children]
    (maximum,) = [node for node in users if isinstance(node, MaxDimOp)]
    (gather,) = [node for node in users if isinstance(node, GatherOp)]
    assert len(users) == 2
    (max_value,) = [graph.node_table[name] for name in maximum.children]
    assert isinstance(max_value, GetItemOp) and max_value.args[1] == 0
    box_source = graph.node_table[cat.args[0][0]]
    box_view = operation(
        graph,
        PermuteOp,
        boxes.name + "_direct",
        [box_source],
        boxes.tensor_meta["shape"],
        TensorDType.Float32,
    )
    box_view.add_argument([0, 2, 1])
    replace_result(graph, boxes, box_view)

    # Keep the classification heads in Q6 logits and Q24 probabilities.
    # Only maxima and the selected anchors are decoded to FP32.
    indices = torch.arange(1024)
    signed = torch.where(indices < 512, indices, indices - 1024).double()
    table = parameter(
        graph,
        values,
        "head_sigmoid_q24",
        torch.round(torch.sigmoid(signed / 64) * 2**24).to(torch.int32),
    )
    logits = graph.node_table[sigmoid.args[0]]
    assert isinstance(logits, CatOp) and logits.args[1] % 3 == 2
    heads, maxima = [], []
    for name in logits.args[0]:
        flattened = graph.node_table[name]
        assert isinstance(flattened, ViewOp)
        layout = graph.node_table[flattened.args[0]]
        assert isinstance(layout, PermuteOp) and layout.args[1] == [0, 3, 1, 2]
        decoded = graph.node_table[layout.args[0]]
        assert isinstance(decoded, Int32ToFP32DequantOp)
        raw = graph.node_table[decoded.args[0]]
        batch, height, width, channels = raw.tensor_meta["shape"]
        assert batch == 1 and channels == 80 and height * width % 16 == 0
        m, s = coefficients(values[decoded.args[1]], 6)
        multiplier = parameter(graph, values, name + "_q6_multiplier", m)
        shift = parameter(graph, values, name + "_q6_shift", s)
        codes = operation(
            graph,
            FixedRequantOp,
            name + "_q6",
            [raw, multiplier, shift],
            raw.tensor_meta["shape"],
            TensorDType.Int32,
        )
        codes.add_argument(6)
        lookup_shape = [height * width // 16, channels * 16]
        matrix = operation(
            graph,
            ViewOp,
            name + "_lookup_matrix",
            [codes],
            lookup_shape,
            TensorDType.Int32,
        )
        matrix.add_argument(lookup_shape)
        looked = operation(
            graph,
            LookupI32Op,
            name + "_probabilities",
            [matrix, table],
            lookup_shape,
            TensorDType.Int32,
        )
        shape = [height * width, channels]
        matrix = operation(
            graph, ViewOp, name + "_class_matrix", [looked], shape, TensorDType.Int32
        )
        matrix.add_argument(shape)
        reduced = operation(
            graph,
            RowMaxU32Op,
            name + "_class_max",
            [matrix],
            [height * width],
            TensorDType.Int32,
        )
        for source, tail, output in (
            (matrix, [channels], heads),
            (reduced, [], maxima),
        ):
            shape = [1, height * width, *tail]
            view = operation(
                graph,
                ViewOp,
                source.name + "_batch",
                [source],
                shape,
                TensorDType.Int32,
            )
            view.add_argument(shape)
            output.append(view)
    for original, inputs in ((scores, heads), (max_value, maxima)):
        combined = operation(
            graph,
            CatOp,
            original.name + "_integer",
            inputs,
            original.tensor_meta["shape"],
            TensorDType.Int32,
        )
        combined.args[:] = [[node.name for node in inputs], 1]
        if original is max_value:
            (topk,) = [graph.node_table[name] for name in original.children]
            assert isinstance(topk, TopkOp)
            topk.tensor_meta["dtype"] = (TensorDType.Int32, TensorDType.Int64)
            assert all(graph.node_table[name].args[1] == 1 for name in topk.children)
        replace_result(graph, original, combined)
    repeated = graph.node_table[gather.args[2]]
    assert isinstance(repeated, RepeatOp) and repeated.args[1] == [1, 1, 80]
    expanded_indices = graph.node_table[repeated.args[0]]
    assert isinstance(expanded_indices, UnsqueezeOp)
    index_source = graph.node_table[expanded_indices.args[0]]
    count = gather.tensor_meta["shape"][1]
    indices = operation(
        graph,
        ViewOp,
        gather.name + "_indices",
        [index_source],
        [count],
        TensorDType.Int64,
    )
    indices.add_argument([count])
    matrices = [graph.node_table[head.args[0]] for head in heads]
    selected = operation(
        graph,
        GatherRowsOp,
        gather.name + "_rows",
        [indices, *matrices],
        [count, 80],
        TensorDType.Int32,
    )
    restored = operation(
        graph,
        ViewOp,
        gather.name + "_selected",
        [selected],
        [1, count, 80],
        TensorDType.Int32,
    )
    restored.add_argument([1, count, 80])
    replace_result(graph, gather, restored)
    gather = restored
    (flat_scores,) = [graph.node_table[name] for name in gather.children]
    assert isinstance(flat_scores, ViewOp)
    flat_scores.tensor_meta["dtype"] = TensorDType.Int32
    (topk,) = [graph.node_table[name] for name in flat_scores.children]
    assert isinstance(topk, TopkOp)
    topk.tensor_meta["dtype"] = (TensorDType.Int32, TensorDType.Int64)
    (selected_values,) = [
        graph.node_table[name]
        for name in topk.children
        if graph.node_table[name].args[1] == 0
    ]
    selected_values.tensor_meta["dtype"] = TensorDType.Int32
    users = list(selected_values.children)
    step = parameter(graph, values, "selected_score_step", torch.full([1, 1], 2.0**-24))
    decoded = operation(
        graph,
        Int32ToFP32DequantOp,
        "selected_score_values",
        [selected_values, step],
        selected_values.tensor_meta["shape"],
        TensorDType.Float32,
    )
    for name in users:
        user = graph.node_table[name]
        replace_argument(graph, user, user.args.index(selected_values.name), decoded)

    # Gather raw boxes and their scales before decoding. Anchor and stride
    # parameters are selected only for the final detections as well.
    (box_gather,) = [graph.node_table[name] for name in box_view.children]
    assert isinstance(box_gather, GatherOp) and isinstance(box_source, MulOp)
    repeated = graph.node_table[box_gather.args[2]]
    assert isinstance(repeated, RepeatOp) and repeated.args[1] == [1, 1, 4]
    index_source = graph.node_table[repeated.args[0]]
    count = box_gather.tensor_meta["shape"][1]
    indices = operation(
        graph, ViewOp, "box_indices", [index_source], [count], TensorDType.Int64
    )
    indices.add_argument([count])
    decoded_cat = graph.node_table[box_source.args[0]]
    low = graph.node_table[decoded_cat.args[0][0]]
    assert isinstance(decoded_cat, CatOp) and isinstance(low, SubOp)
    anchor_view = graph.node_table[low.args[0]]
    anchors = graph.node_table[anchor_view.args[0]]
    part = graph.node_table[low.args[1]]
    split = graph.node_table[part.args[0]]
    raw_cat = graph.node_table[split.args[0]]
    raw_heads, scales = [], []
    for name in raw_cat.args[0]:
        flattened = graph.node_table[name]
        layout = graph.node_table[flattened.args[0]]
        decoded = graph.node_table[layout.args[0]]
        assert isinstance(decoded, Int32ToFP32DequantOp)
        raw = graph.node_table[decoded.args[0]]
        batch, height, width, channels = raw.tensor_meta["shape"]
        assert batch == 1 and channels == 4
        matrix = operation(
            graph,
            ViewOp,
            name + "_box_matrix",
            [raw],
            [height * width, 4],
            TensorDType.Int32,
        )
        matrix.add_argument([height * width, 4])
        raw_heads.append(matrix)
        scales.append(values[decoded.args[1]].reshape(1, 4).expand(height * width, 4))
    scale_table = parameter(
        graph, values, "box_scale_rows", torch.cat(scales).contiguous()
    )
    selected = operation(
        graph,
        GatherRowsOp,
        "box_raw_rows",
        [indices, *raw_heads],
        [count, 4],
        TensorDType.Int32,
    )
    selected_scales = operation(
        graph,
        GatherRowsOp,
        "box_selected_scales",
        [indices, scale_table],
        [count, 4],
        TensorDType.Float32,
    )
    decoded = operation(
        graph,
        Int32ToFP32DequantOp,
        "box_decoded_rows",
        [selected, selected_scales],
        [count, 4],
        TensorDType.Float32,
    )
    anchor_table = parameter(
        graph,
        values,
        "box_anchor_rows",
        values[anchors.name].T.repeat(1, 2).contiguous(),
    )
    strides = graph.node_table[box_source.args[1]]
    stride_table = parameter(
        graph,
        values,
        "box_stride_rows",
        values[strides.name].reshape(-1, 1).expand(-1, 4).contiguous(),
    )
    selected_anchors = operation(
        graph,
        GatherRowsOp,
        "box_selected_anchors",
        [indices, anchor_table],
        [count, 4],
        TensorDType.Float32,
    )
    selected_strides = operation(
        graph,
        GatherRowsOp,
        "box_selected_strides",
        [indices, stride_table],
        [count, 4],
        TensorDType.Float32,
    )
    signs = parameter(
        graph, values, "box_corner_signs", torch.tensor([[-1.0, -1.0, 1.0, 1.0]])
    )
    delta = operation(
        graph,
        MulOp,
        "box_signed_distances",
        [decoded, signs],
        [count, 4],
        TensorDType.Float32,
    )
    corners = operation(
        graph,
        AddOp,
        "box_corners",
        [selected_anchors, delta],
        [count, 4],
        TensorDType.Float32,
    )
    scaled = operation(
        graph,
        MulOp,
        "box_scaled",
        [corners, selected_strides],
        [count, 4],
        TensorDType.Float32,
    )
    result = operation(
        graph, ViewOp, "selected_boxes", [scaled], [1, count, 4], TensorDType.Float32
    )
    result.add_argument([1, count, 4])
    replace_result(graph, box_gather, result)
    (output,) = [node for node in graph.body if isinstance(node, OutputOp)]
    packed = graph.node_table[output.args[0]]
    assert isinstance(packed, CatOp) and packed.args[1] % 3 == 2
    matrices = []
    for number, name in enumerate(packed.args[0]):
        source = graph.node_table[name]
        assert source.tensor_meta["dtype"] == TensorDType.Float32
        columns = source.tensor_meta["shape"][-1]
        matrix = operation(
            graph,
            ViewOp,
            f"detection_part{number}",
            [source],
            [count, columns],
            TensorDType.Float32,
        )
        matrix.add_argument([count, columns])
        matrices.append(matrix)
    joined = operation(
        graph,
        ConcatColumnsOp,
        "detection_rows",
        matrices,
        [count, 6],
        TensorDType.Float32,
    )
    detections = operation(
        graph, ViewOp, "detections", [joined], [1, count, 6], TensorDType.Float32
    )
    detections.add_argument([1, count, 6])
    replace_result(graph, packed, detections)
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
