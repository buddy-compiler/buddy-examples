from buddy.compiler.graph.operation import (
    BatchMatmulOp,
    CloneOp,
    CosOp,
    EmbeddingOp,
    GetItemOp,
    OutputOp,
    PermuteOp,
    PlaceholderOp,
    RsqrtOp,
    SliceOp,
    SinOp,
    TopkOp,
    TransposeMatmulFusedOp,
    ViewOp,
    WhereOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.partition import partition
from compiler.frontend.quant import (
    FP32ToMXINT8QuantOp,
    MXINT8ToFP32DequantOp,
    MXINT5ToFP32DequantOp,
    MXINT8ActivationOp,
)
from frontend.integer_activations import discard
from frontend.quant.graph import mxint8, replace_argument

# The candidates of the next token each rank sends: the top-k of the sampler
# runner.cpp runs by default, the generation config of the checkpoint
# (temperature 0.6, top-k 20, top-p 0.95; run.args). The union of the ranks'
# candidates holds the 20 largest logits of the vocabulary, all that top-k
# sampling reads; greedy decoding reads the largest.
candidates = 20


def cut(graph, params, design):
    """The decoder across poly: the reduce hart and a rank on each of its
    poly harts (as many as design.pb has), with the attention heads, the FFN
    channels and the vocabulary split across the ranks. Each rank owns the
    KV cache of its heads; every hart computes the positions, the mask and
    the rotary tables it uses."""
    # The program returns the candidates of the next token, not the logits:
    # the LM head computes the last position only, and each rank sends the
    # `candidates` largest logits of its piece of the vocabulary and their
    # indices, not the piece.
    (output,) = [node for node in graph.body if isinstance(node, OutputOp)]
    logits = graph.node_table[output.args[0]]
    head = graph.node_table[logits.args[0]]
    restored = graph.node_table[head.args[0]]
    codes = graph.node_table[restored.args[0]]
    hidden = graph.node_table[codes.args[0]]
    tokens, width = hidden.tensor_meta["shape"]
    last = SliceOp()
    last.name = "last"
    for argument in (hidden.name, 0, tokens - 1, tokens):
        last.add_argument(argument)
    last.add_parent(hidden.name)
    last.tensor_meta = {"shape": [1, width], "dtype": hidden.tensor_meta["dtype"]}
    graph.add_node(last)
    hidden.children[hidden.children.index(codes.name)] = last.name
    last.add_children(codes.name)
    codes.args[0] = last.name
    codes.parents[codes.parents.index(hidden.name)] = last.name
    codes.tensor_meta["shape"] = [1, width + width // 32]
    restored.tensor_meta["shape"] = [1, width]
    vocabulary = head.tensor_meta["shape"][-1]
    head.tensor_meta["shape"] = [1, vocabulary]
    graph.delete_node(logits, [graph.node_table[name] for name in logits.parents])
    top = TopkOp()
    top.name = "top"
    for argument in (head.name, candidates, -1):
        top.add_argument(argument)
    top.add_parent(head.name)
    head.add_children(top.name)
    top.tensor_meta = {
        "shape": [[1, candidates], [1, candidates]],
        "dtype": [TensorDType.Float32, TensorDType.Int64],
    }
    graph.add_node(top)
    items = []
    for index, name in enumerate(("top_values", "top_indices")):
        item = GetItemOp()
        item.name = name
        item.add_argument(top.name)
        item.add_argument(index)
        item.add_parent(top.name)
        top.add_children(name)
        item.tensor_meta = {
            "shape": [1, candidates],
            "dtype": top.tensor_meta["dtype"][index],
        }
        graph.add_node(item)
        item.add_children(output.name)
        items.append(name)
    output.args[0:1] = items
    sort_graph(graph)

    (embedding,) = [node for node in graph.body if isinstance(node, EmbeddingOp)]
    hidden, frontier = {embedding.name}, [embedding]
    while frontier:
        for child in frontier.pop().children:
            if child not in hidden:
                hidden.add(child)
                frontier.append(graph.node_table[child])
    replicated = {
        node.name
        for node in graph.body
        if node.name not in hidden and not isinstance(node, PlaceholderOp)
    }
    # The inputs are the tokens, the start position, then the keys and the
    # values of every layer: [1, heads, capacity, head_dim].
    states = {node.name: 1 for node in graph.inputs[2:]}
    # The Linears that read a normalized activation (q, k, v, gate, up and
    # the LM head) split along their output channels; the Linears after them
    # read their pieces and split along K.
    columns = set()
    for linear in graph.body:
        if not isinstance(linear, TransposeMatmulFusedOp) or linear.name in replicated:
            continue
        node = graph.node_table[linear.args[0]]
        while isinstance(
            node,
            (
                ViewOp,
                SliceOp,
                CloneOp,
                PermuteOp,
                FP32ToMXINT8QuantOp,
                MXINT8ToFP32DequantOp,
            ),
        ):
            node = graph.node_table[node.args[0]]
        # A normalization ends in its scale (and shift) after the reciprocal
        # square root.
        frontier = [node]
        for _ in range(3):
            frontier = [graph.node_table[name] for n in frontier for name in n.parents]
            if any(isinstance(n, RsqrtOp) for n in frontier):
                columns.add(linear.name)
                break
    ranks = sum(hart.component == "poly" for hart in design.device.harts)
    compressed = {
        node.name
        for node in graph.body
        if isinstance(node, TransposeMatmulFusedOp)
        and isinstance(graph.node_table[node.args[1]], MXINT5ToFP32DequantOp)
    }
    for name in columns & compressed:
        linear = graph.node_table[name]
        children = list(linear.children)
        restored = mxint8(
            graph, linear, name + "_output", len(linear.tensor_meta["shape"]) - 1
        )
        for child in children:
            user = graph.node_table[child]
            for index, argument in enumerate(list(user.args)):
                if argument == name:
                    replace_argument(graph, user, index, restored)
    if compressed:
        values = dict(zip((node.name for node in graph.params), params, strict=True))
        for activation in list(graph.body):
            if not isinstance(activation, MXINT8ActivationOp):
                continue
            if values[activation.args[2]][2048].item() != 0:
                raise ValueError("Direct quantized SiLU requires table[2048] == 0")
            for index in range(2 if activation.args[4] else 1):
                quant = graph.node_table[activation.args[index]]
                if not isinstance(quant, FP32ToMXINT8QuantOp):
                    continue
                source = graph.node_table[quant.args[0]]
                while isinstance(source, (ViewOp, CloneOp)):
                    parent = graph.node_table[source.args[0]]
                    if (
                        source.tensor_meta["shape"][-1]
                        != parent.tensor_meta["shape"][-1]
                    ):
                        break
                    source = parent
                if not isinstance(source, MXINT8ToFP32DequantOp):
                    continue
                packed = graph.node_table[source.args[0]]
                if not isinstance(packed, FP32ToMXINT8QuantOp):
                    continue
                if (
                    quant.tensor_meta["shape"] != packed.tensor_meta["shape"]
                    or quant.args[1] != len(source.tensor_meta["shape"]) - 1
                    or source.args[1] != quant.args[1]
                    or packed.args[1] != quant.args[1]
                ):
                    continue
                # Zero underflow blocks may change exponent on requantization;
                # this SiLU table and the optional gate map them to zero.
                replace_argument(graph, activation, index, packed)
                discard(graph, [quant])
        sort_graph(graph)
    preparation = {
        node.name
        for node in graph.body
        if node.name in replicated and isinstance(node, (CosOp, SinOp, WhereOp))
    }
    (reducer,) = [
        hart.component for hart in design.device.harts if hart.component != "poly"
    ]
    return partition(
        graph,
        params,
        ranks,
        states,
        replicated,
        columns,
        {},
        preparation,
        set(),
        set(),
        classify,
        compressed - columns,
        {},
        [reducer] + ["poly"] * ranks,
        False,
    )


def classify(device, segment):
    """The semantic graphs of the decoder. On each rank: `attention`, the
    attention of a layer over its heads (q, k, v, their norms and RoPE, its
    slice of the KV cache, scores, softmax, context, its K piece of o_proj);
    `mlp`, the MLP of a layer over its channels (gate, up, SiLU·up, its K
    piece of down); `lm_head`, its piece of the logits of the last position
    and its `candidates` largest with their indices. On the reduce
    hart: `embed`, the embedding and the start of the first attention (its
    norm, the quantized inputs of q, k, v); `attention_out`, the end of an
    attention (the partial sums of the ranks in rank order, the residual)
    and the start of the MLP (its norm, the quantized inputs of gate and
    up); `mlp_out`, the end of an MLP and the start of the next attention;
    `final`, the end of the last MLP, the final norm and the input of the
    LM head; `cache`, the KV cache of a layer joined from the ranks'
    slices; `lm_head_out`, the candidates of the next token: those of the
    ranks in rank order, their indices offset to the vocabulary."""

    def head(segment):
        return any(
            name.rsplit("_r", 1)[0] == device.output.args[0] for name in segment.outputs
        )

    if segment.hart_class == "poly" and any(
        "_local_r" in node.name for _, node in segment.ops
    ):
        return "prepare"
    ops = [node for _, node in segment.ops]
    if segment.hart_class == "poly":
        if any(isinstance(node, BatchMatmulOp) for node in ops):
            return "attention"
        return "lm_head" if head(segment) else "mlp"
    if segment.join:
        return "cache"
    if any(isinstance(node, EmbeddingOp) for node in ops):
        return "embed"
    if any(head(reader) for reader in segment.readers):
        return "final"
    (block,) = {
        device.owner[name].graph_class
        for name in segment.inputs
        if name in device.owner and device.owner[name].hart_class == "poly"
    }
    return f"{block}_out"
