import torch
from torch import nn
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

from stack.compiler.quant import mxfp8_cache as codec
from .attention import AttentionHeads


class Prepare(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.norm = layer.input_layernorm

    def forward(self, hidden):
        return self.norm(hidden)


class KVProducer(nn.Module):
    def __init__(self, layer, key_projection, value_projection):
        super().__init__()
        if layer.self_attn.is_kv_shared_layer:
            raise ValueError("A shared Gemma layer cannot produce KV")
        self.key_projection, self.value_projection = key_projection, value_projection
        self.key_norm, self.value_norm = layer.self_attn.k_norm, layer.self_attn.v_norm
        self.width = layer.self_attn.head_dim

    def forward(self, hidden, cosine, sine):
        shape = (*hidden.shape[:-1], 1, self.width)
        key = self.key_norm(self.key_projection(hidden).reshape(shape))
        key = apply_rotary_pos_emb(key, cosine, sine, unsqueeze_dim=2).transpose(1, 2)
        value = self.value_norm(self.value_projection(hidden).reshape(shape)).transpose(
            1, 2
        )
        key_codes, key_scales = codec.encode(key)
        value_codes, value_scales = codec.encode(value)
        return key_codes, key_scales, value_codes, value_scales


class PostAttention(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.post = layer.post_attention_layernorm
        self.pre = layer.pre_feedforward_layernorm

    def forward(self, projected, residual):
        hidden = residual + self.post(projected)
        return hidden, self.pre(hidden)


class GateUp(nn.Module):
    def __init__(self, layer, gate_projection, up_projection):
        super().__init__()
        self.gate, self.up = gate_projection, up_projection
        self.activation = layer.mlp.act_fn

    def forward(self, hidden):
        return self.activation(self.gate(hidden)) * self.up(hidden)


class PostLayer(nn.Module):
    def __init__(self, layer, input_gate, input_projection):
        super().__init__()
        if layer.enable_moe_block or not layer.hidden_size_per_layer_input:
            raise ValueError(
                "Gemma task layer requires the E2B dense per-layer-input contract"
            )
        self.post = layer.post_feedforward_layernorm
        self.gate, self.projection = input_gate, input_projection
        self.activation = layer.act_fn
        self.input_norm = layer.post_per_layer_input_norm
        self.register_buffer("scalar", layer.layer_scalar)

    def forward(self, projected, residual, per_layer_input):
        hidden = residual + self.post(projected)
        gated = self.activation(self.gate(hidden)) * per_layer_input
        hidden = hidden + self.input_norm(self.projection(gated))
        return hidden * self.scalar


def layer_tasks(layer, weights, layouts, tiles, *, prefill):
    from .linear import PackedLinear
    from .partition import head_ranges, kv_sources, panel_ranges, participants

    tiles = participants(tiles)
    projections = {
        "q": layer.self_attn.q_proj,
        "o": layer.self_attn.o_proj,
        "gate": layer.mlp.gate_proj,
        "up": layer.mlp.up_proj,
        "down": layer.mlp.down_proj,
        "input_gate": layer.per_layer_input_gate,
        "input_projection": layer.per_layer_projection,
    }
    if not layer.self_attn.is_kv_shared_layer:
        projections.update(k=layer.self_attn.k_proj, v=layer.self_attn.v_proj)
    ranges = {
        name: panel_ranges(
            projection.out_features, projection.in_features, layouts[name], tiles
        )
        for name, projection in projections.items()
    }
    if ranges["gate"] != ranges["up"] or layouts["gate"] != layouts["up"]:
        raise ValueError(
            "Gemma gate/up tasks require matching original panel boundaries"
        )

    def matrix(name, span):
        return PackedLinear(
            weights[name], projections[name].in_features, layouts[name], span
        )

    def full(name):
        projection = projections[name]
        span = panel_ranges(
            projection.out_features, projection.in_features, layouts[name], [tiles[0]]
        )[0]
        return matrix(name, span)

    heads = head_ranges(layer.self_attn.config.num_attention_heads, tiles)
    ranks = []
    for rank, tile in enumerate(tiles):
        tasks = {
            "tile": tile,
            "ranges": {name: spans[rank] for name, spans in ranges.items()},
        }
        for name in ("q", "o", "down"):
            if ranges[name][rank]["columns"]:
                tasks[name] = matrix(name, ranges[name][rank])
        if ranges["gate"][rank]["columns"]:
            tasks["gateup"] = GateUp(
                layer,
                matrix("gate", ranges["gate"][rank]),
                matrix("up", ranges["up"][rank]),
            )
        first, end = heads[rank]
        if first < end:
            tasks["attention"] = AttentionHeads(layer, first, end, prefill=prefill)
        ranks.append(tasks)
    result = {
        "ranks": ranks,
        "prepare": Prepare(layer),
        "post_attention": PostAttention(layer),
        "post_layer": PostLayer(layer, full("input_gate"), full("input_projection")),
        "kv": kv_sources(layer.config, tiles)[layer.layer_idx],
    }
    if not layer.self_attn.is_kv_shared_layer:
        result["kv_producer"] = KVProducer(layer, full("k"), full("v"))
    return result


def capture_task(compiler, task, inputs, *, symbol, prefill):
    from buddy.compiler.graph import GraphDriver
    from buddy.compiler.graph.transform import (
        apply_classic_fusion,
        flash_attention_prefill,
        gqa_attention_fusion,
        simply_fuse,
    )
    from buddy_mlir import ir
    from .linear import register
    from .attention import register as register_attention
    from .kernels import map_flash_attention

    register(compiler)
    codec.register(compiler)
    register_attention(compiler)
    with torch.no_grad():
        graphs = compiler.importer(task, *inputs)
    if len(graphs) != 1:
        raise ValueError("A Gemma compute task must capture exactly one graph")
    graph = graphs[0]
    graph.fuse_ops(
        [
            simply_fuse,
            apply_classic_fusion,
            flash_attention_prefill if prefill else gqa_attention_fusion,
        ]
    )
    map_flash_attention(graph)
    if len(graph.op_groups) != 1 or not symbol:
        raise ValueError(
            "A Gemma task requires one group and an explicit unique symbol"
        )
    old_name = next(iter(graph.op_groups))
    graph.op_groups[symbol] = graph.op_groups.pop(old_name)
    graph.group_map_device[symbol] = graph.group_map_device.pop(old_name)
    driver = GraphDriver(graph)
    graph.task_arguments = driver._subgraphs_inputs[symbol]
    if len(driver.subgraphs) != 1:
        raise ValueError("A Gemma task must lower as one explicit compute subgraph")
    subgraph = driver.subgraphs[0]
    subgraph.lower_to_top_level_ir()
    module = subgraph._imported_module
    with module.context:
        for function in module.body.operations:
            if function.operation.name != "func.func" or not function.regions[0].blocks:
                continue
            attributes = []
            for argument in function.regions[0].blocks[0].arguments:
                shape = list(ir.RankedTensorType(argument.type).shape)
                strides, product = [], 1
                for size in reversed(shape):
                    strides.insert(0, product)
                    product *= size
                if (
                    str(ir.RankedTensorType(argument.type).element_type) != "i8"
                    or len(shape) != 1
                ):
                    strides = [ir.ShapedType.get_dynamic_size()] * len(shape)
                attributes.append(
                    ir.DictAttr.get(
                        {
                            "bufferization.buffer_layout": ir.StridedLayoutAttr.get(
                                ir.ShapedType.get_dynamic_size(), strides
                            ),
                            "bufferization.writable": ir.BoolAttr.get(False),
                        }
                    )
                )
            function.attributes["arg_attrs"] = ir.ArrayAttr.get(attributes)
    return graph, module
