from math import gcd

from buddy.compiler.graph.operation import (
    BatchMatmulOp,
    CloneOp,
    CosOp,
    EmbeddingOp,
    GetItemOp,
    OutputOp,
    ExpandOp,
    IndexPutOp,
    PermuteOp,
    PlaceholderOp,
    PowOp,
    RsqrtOp,
    SliceOp,
    SinOp,
    TopkOp,
    TransposeMatmulFusedOp,
    UnsqueezeOp,
    ViewOp,
    WhereOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.partition import partition
from compiler.frontend.quant import FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp

# The candidates of the next token each rank sends: the top-k of the sampler
# runner.cpp runs by default, the generation config of the checkpoint, which
# samples none: greedy decoding, which reads the largest logit only.
candidates = 1


def cut(graph, params, design):
    """The decoder across poly: the reduce hart and a rank on each of its
    poly harts (as many as design.pb has), with the query heads, the FFN
    channels and the vocabulary split across the ranks. Gemma 4 E2B has one
    KV head: the first poly hart produces the keys and values and prepares
    the packed keys and values, which all query ranks share. Each consumer
    keeps its decoding/layout markers and reads the quantized operands.
    The unsharded per-layer projection also runs there;
    the reduce hart normalizes and distributes its results.
    The first poly hart prepares masks and rotary tables once per forward;
    their consumers share them through device SHM."""
    # The program returns the candidates of the next token, not the logits:
    # the LM head and the softcapping of its logits compute the last
    # position only, and each rank sends the `candidates` largest softcapped
    # logits of its piece of the vocabulary and their indices, not the piece.
    (output,) = [node for node in graph.body if isinstance(node, OutputOp)]
    capped = [graph.node_table[output.args[0]]]
    while not isinstance(capped[-1], ViewOp):
        capped.append(graph.node_table[capped[-1].args[0]])
    logits = capped.pop()
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
    workers = [hart for hart in design.device.harts if hart.component == "poly"]
    ranks = len(workers)
    capacity = min(
        design.threads[index].mem.bank.entries
        * design.threads[index].mem.bank.width
        // 32
        for hart in workers
        for index in hart.thread_indices
    )
    row_width = gcd(vocabulary // ranks, capacity)
    while vocabulary // ranks // row_width < max(
        len(hart.thread_indices) for hart in workers
    ):
        row_width //= 2
    head.tensor_meta["shape"] = [1, vocabulary]
    first = capped[-1]
    first.args[0] = head.name
    first.parents[first.parents.index(logits.name)] = head.name
    head.add_children(first.name)
    logits.children.clear()
    graph.delete_node(logits, [graph.node_table[name] for name in logits.parents])
    for node in capped:
        node.tensor_meta["shape"] = [1, vocabulary // row_width, row_width]
    # Softcap rows fit the banks and expose independent work to every thread.
    rows = ViewOp()
    rows.name = "softcap_rows"
    rows.add_argument(head.name)
    rows.add_argument([1, vocabulary // row_width, row_width])
    rows.add_parent(head.name)
    rows.tensor_meta = {
        "shape": [1, vocabulary // row_width, row_width],
        "dtype": head.tensor_meta["dtype"],
    }
    head.children[head.children.index(first.name)] = rows.name
    rows.add_children(first.name)
    first.args[0] = rows.name
    first.parents[first.parents.index(head.name)] = rows.name
    graph.add_node(rows)
    flat = ViewOp()
    flat.name = "softcap_flat"
    flat.add_argument(capped[0].name)
    flat.add_argument([1, vocabulary])
    flat.add_parent(capped[0].name)
    flat.tensor_meta = {"shape": [1, vocabulary], "dtype": head.tensor_meta["dtype"]}
    capped[0].children[capped[0].children.index(output.name)] = flat.name
    flat.add_children(output.name)
    graph.add_node(flat)
    top = TopkOp()
    top.name = "top"
    for argument in (flat.name, candidates, -1):
        top.add_argument(argument)
    top.add_parent(flat.name)
    flat.children[flat.children.index(output.name)] = top.name
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

    hidden = set()
    frontier = [node for node in graph.body if isinstance(node, EmbeddingOp)]
    while frontier:
        node = frontier.pop()
        for child in node.children:
            if child not in hidden:
                hidden.add(child)
                frontier.append(graph.node_table[child])
    # The key and value projections: the Linears the cache writes come from.
    projections, frontier = set(), [
        graph.node_table[node.args[2]]
        for node in graph.body
        if isinstance(node, IndexPutOp)
    ]
    while frontier:
        node = frontier.pop()
        if isinstance(node, TransposeMatmulFusedOp):
            projections.add(node)
            continue
        frontier.extend(graph.node_table[name] for name in node.parents)
    # Their inputs and weights, and all that follows from them up to the
    # attention products.
    keys = {name for node in projections for name in node.args[:2]}
    frontier = list(projections)
    while frontier:
        node = frontier.pop()
        keys.add(node.name)
        for child in node.children:
            user = graph.node_table[child]
            if (
                child not in keys
                and not isinstance(user, BatchMatmulOp)
                and child != "output"
            ):
                frontier.append(user)
    replicated = keys | {
        node.name
        for node in graph.body
        if node.name not in hidden and not isinstance(node, PlaceholderOp)
    }
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
        # square root, which Gemma 4 computes as a power of -0.5.
        frontier = [node]
        for _ in range(3):
            frontier = [graph.node_table[name] for n in frontier for name in n.parents]
            if any(
                isinstance(n, RsqrtOp) or (isinstance(n, PowOp) and n.args[1] == -0.5)
                for n in frontier
            ):
                columns.add(linear.name)
                break
    shared = set()
    for product in graph.body:
        if not isinstance(product, BatchMatmulOp):
            continue
        value = graph.node_table[product.args[1]]
        while isinstance(value, (ViewOp, PermuteOp, ExpandOp, UnsqueezeOp, CloneOp)):
            value = graph.node_table[value.args[0]]
        if value.name in keys:
            shared.add(
                value.args[0]
                if isinstance(value, MXINT8ToFP32DequantOp)
                else value.name
            )
    shared |= {
        node.name
        for node in graph.body
        if node.name in replicated
        and node.name not in hidden
        and isinstance(node, (CosOp, SinOp, WhereOp))
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
        {},
        replicated,
        columns,
        {},
        set(),
        shared,
        set(),
        classify,
        set(),
        {},
        [reducer] + ["poly"] * ranks,
        False,
    )


def classify(device, segment):
    """The semantic graphs of Gemma 4. On each rank: `attention`, the
    attention of a layer over its query heads, reading the shared keys and
    values (sliding or full, including layers sharing an earlier cache);
    `projection`, independent dense projections before attention;
    `cache`, the shared KV producer on the first rank; `mlp`, the MLP of a layer
    over its channels (the layers that share KV have a wider one);
    `per_layer`, the gate of the per-layer input of a layer over its
    channels, which reads its piece of that input; `lm_head`, its piece of
    the softcapped logits of the last position and its `candidates` largest
    with their indices. On the reduce hart: `embed`,
    the embedding and the start of the first attention; after each block of a layer the partial sums of
    the ranks in rank order, the norms and the residual of the block and the
    start of the next block: `attention_out`, `mlp_out` and `layer_out`
    (with the layer scale; the next attention reads its own keys and values
    or not); `final`, the end of the last layer, the final norm and the
    input of the LM head; `lm_head_out`, the candidates of the
    next token: those of the ranks in rank order, their indices offset to
    the vocabulary."""

    def head(segment):
        return any(
            name.rsplit("_r", 1)[0] == device.output.args[0] for name in segment.outputs
        )

    ops = [node for _, node in segment.ops]
    if segment.hart_class == "poly":
        if segment.join:
            return "cache"
        if any(isinstance(node, BatchMatmulOp) for node in ops):
            return "attention"
        if head(segment):
            return "lm_head"
        if any(isinstance(device.nodes.get(name), SliceOp) for name in segment.inputs):
            return "per_layer"
        if any(
            device.owner[name].graph_class == "embed"
            for name in segment.inputs
            if name in device.owner
        ):
            return "projection"
        return "mlp"
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
    return f"{block}_out" if block != "per_layer" else "layer_out"
