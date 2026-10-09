from buddy.compiler.graph.operation import OutputOp
from buddy.compiler.graph.type import TensorDType


def verify_config(config, cache):
    count = config.num_hidden_layers - config.num_kv_shared_layers
    pattern = ["sliding_attention"] * 4 + ["full_attention"]
    if (
        count != 15
        or len(cache.layers) != 15
        or config.layer_types[:count] != pattern * 3
        or config.num_key_value_heads != 1
        or config.head_dim != 256
        or config.global_head_dim != 512
        or config.sliding_window != 512
        or config.vocab_size != 262144
    ):
        raise ValueError("Gemma checkpoint does not match the native hybrid-cache ABI")
    for index, layer in enumerate(cache.layers):
        shape = [1, 1, 512, 512 if index % 5 == 4 else 256]
        if (
            list(layer.key_codes.shape) != shape
            or list(layer.value_codes.shape) != shape
        ):
            raise ValueError(
                f"Gemma cache layer {index} does not match the native KV shape"
            )


def verify_graph(graph, phase, length):
    f32, i64, i8 = TensorDType.Float32, TensorDType.Int64, TensorDType.Int8
    inputs = [([1, length], i64)]
    inputs.append(([1, length], i64))
    outputs = []
    for group in range(3):
        for layer in range(5):
            width = 512 if layer == 4 else 256
            cache = ([1, 1, 512, width], i8)
            scale = ([1, 1, 512, width // 32], i8)
            counter = ([1], i64)
            inputs.extend([counter, cache, scale, cache, scale])
            outputs.extend([counter, cache, scale, cache, scale])
    outputs.append(([1, length, 262144], f32))
    output = next(node for node in graph.body if isinstance(node, OutputOp))
    nodes = [graph.node_table[name] for name in output.args]
    for kind, actual, expected in (
        ("input", graph.inputs, inputs),
        ("output", nodes, outputs),
    ):
        if len(actual) != len(expected):
            raise ValueError(
                f"Gemma {phase} {kind} count does not match the native ABI"
            )
        for position, (node, (shape, dtype)) in enumerate(zip(actual, expected)):
            if (
                list(node.tensor_meta["shape"]) != shape
                or node.tensor_meta["dtype"] != dtype
            ):
                raise ValueError(
                    f"Gemma {phase} {kind} {position} ({node.name}) has {node.tensor_meta}; expected {shape}/{dtype}"
                )
    for position, node in enumerate(nodes[:-1]):
        if not node.args or node.args[0] != graph.inputs[position + 2].name:
            raise ValueError(
                f"Gemma {phase} cache output {position} does not update its declared input"
            )
    position = graph.inputs[1].name
    memo = {}

    def depends(name):
        if name == position:
            return True
        if name not in memo:
            memo[name] = any(
                depends(parent) for parent in graph.node_table[name]._parents
            )
        return memo[name]

    masks = [
        node
        for node in graph.body
        if type(node).__name__ == "WhereOp"
        and list(node.tensor_meta["shape"]) == [1, 1, length, 512]
    ]
    if len(masks) != 2 or not all(depends(node.name) for node in masks):
        raise ValueError(
            f"Gemma {phase} full/sliding masks must depend on runtime position IDs"
        )
    for operation in ("SinOp", "CosOp"):
        if not any(
            type(node).__name__ == operation and depends(node.name)
            for node in graph.body
        ):
            raise ValueError(
                f"Gemma {phase} rotary {operation} must depend on runtime position IDs"
            )
