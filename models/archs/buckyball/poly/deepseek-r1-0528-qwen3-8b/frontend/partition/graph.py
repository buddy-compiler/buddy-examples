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
from compiler.frontend.quant import FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp

# The candidates of the next token each rank sends: the top-k of the sampler
# runner.cpp runs by default, the generation config of the checkpoint
# (none: the README's temperature 0.6 and top-p 0.95; run.args) with the
# top-k transformers samples with when it names none, 50. The union of the
# ranks' candidates holds the 50 largest logits of the vocabulary, all that
# top-k sampling reads; greedy decoding reads the largest.
candidates = 50


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
    # Prepare each normalized activation once using the compute hart's
    # quantizer; all ranks consume the packed result.
    shared = set()
    for name in columns:
        restored = graph.node_table[graph.node_table[name].args[0]]
        assert isinstance(restored, MXINT8ToFP32DequantOp)
        packed = graph.node_table[restored.args[0]]
        assert isinstance(packed, FP32ToMXINT8QuantOp)
        shared.add(packed.name)
        replicated.add(packed.name)
        source = graph.node_table[packed.args[0]]
        while isinstance(source, (ViewOp, CloneOp, PermuteOp)):
            replicated.add(source.name)
            source = graph.node_table[source.args[0]]
    preparation = {
        node.name
        for node in graph.body
        if node.name in replicated and isinstance(node, (CosOp, SinOp, WhereOp))
    }
    (reducer,) = [
        hart.component
        for hart in design.device.harts
        if hart.hart_id == design.workload.control_hart
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
        shared,
        set(),
        classify,
        set(),
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
    and its `candidates` largest with their indices; `prepare`,
    normalized activations quantized once for all ranks. On the reduce
    hart: `embed`, the embedding and the start of the first attention (its
    norm); `attention_out`, the end of an
    attention (the partial sums of the ranks in rank order, the residual)
    and the start of the MLP (its norm); `mlp_out`, the end of an MLP and
    the start of the next attention;
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
        if segment.join:
            return "prepare"
        if any(isinstance(node, BatchMatmulOp) for node in ops):
            return "attention"
        return "lm_head" if head(segment) else "mlp"
    if segment.join:
        return "cache"
    if any(isinstance(node, EmbeddingOp) for node in ops):
        return "embed"
    readers = [
        consumer
        for reader in segment.readers
        for consumer in (
            reader.readers if reader.join and reader.hart_class == "poly" else [reader]
        )
    ]
    if any(head(reader) for reader in readers):
        return "final"
    (block,) = {
        device.owner[name].graph_class
        for name in segment.inputs
        if name in device.owner and device.owner[name].hart_class == "poly"
    }
    return f"{block}_out"
