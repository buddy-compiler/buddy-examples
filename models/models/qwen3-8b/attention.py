from torch import nn

from .stages import Attention, shard_linear


class AttentionBody(Attention):
    def __init__(self, layer, config, cache_length, frequencies, rank, parts, attention_scaling, cache_quantizer):
        super().__init__(layer, config, cache_length, frequencies, rank, 2 * parts, attention_scaling, cache_quantizer)
        self.o_proj = nn.Identity()


class AttentionProjection(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.o_proj = shard_linear(attention.o_proj, 0, 0, 2)

    def forward(self, hidden):
        return self.o_proj(hidden)


def parameters(layer, config, frequencies, rank, parts):
    group, half = rank % parts, rank // parts
    query = config.num_attention_heads // parts * config.head_dim
    kv = config.num_key_value_heads // parts * config.head_dim
    hidden = config.hidden_size
    qbegin, kbegin = group * query + half * query // 2, group * kv + half * kv // 2
    return {
        "frequencies": frequencies,
        "k_norm.weight": layer.self_attn.k_norm.weight,
        "norm.weight": layer.input_layernorm.weight,
        "q_norm.weight": layer.self_attn.q_norm.weight,
        "q_proj.weight": layer.self_attn.q_proj.weight[qbegin:qbegin + query // 2],
        "k_proj.weight": layer.self_attn.k_proj.weight[kbegin:kbegin + kv // 2],
        "v_proj.weight": layer.self_attn.v_proj.weight[kbegin:kbegin + kv // 2],
        "o_proj.weight": layer.self_attn.o_proj.weight[
            half * hidden // 2:(half + 1) * hidden // 2, group * query:(group + 1) * query],
    }
