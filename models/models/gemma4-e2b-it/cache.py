import torch
from transformers.cache_utils import Cache as TransformersCache, StaticLayer


class Layer(StaticLayer):
    def __init__(self, width, length, codec, sliding):
        super().__init__(length)
        shape = (1, 1, length, width)
        scale_shape = (1, 1, length, width // 32)
        self.key_codes = torch.zeros(shape, dtype=torch.int8)
        self.key_scales = torch.full(scale_shape, 127, dtype=torch.int8)
        self.value_codes = torch.zeros(shape, dtype=torch.int8)
        self.value_scales = torch.full(scale_shape, 127, dtype=torch.int8)
        self.is_sliding = sliding
        self.codec = codec
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        count = key_states.shape[-2]
        positions = (
            torch.arange(count, device=key_states.device) + self.cumulative_length
        )
        self.cumulative_length.add_(count)
        key_codes, key_scales = self.codec.encode(key_states)
        value_codes, value_scales = self.codec.encode(value_states)
        self.key_codes.index_copy_(2, positions, key_codes)
        self.key_scales.index_copy_(2, positions, key_scales)
        self.value_codes.index_copy_(2, positions, value_codes)
        self.value_scales.index_copy_(2, positions, value_scales)
        return (
            self.codec.decode(self.key_codes, self.key_scales),
            self.codec.decode(self.value_codes, self.value_scales),
        )

    def reset(self):
        self.cumulative_length.zero_()
        self.key_codes.zero_()
        self.value_codes.zero_()
        self.key_scales.fill_(127)
        self.value_scales.fill_(127)


class Cache(TransformersCache):
    def __init__(self, config, length, codec):
        count = config.num_hidden_layers - config.num_kv_shared_layers
        super().__init__(
            layers=[
                Layer(
                    (
                        config.global_head_dim
                        if kind == "full_attention"
                        else config.head_dim
                    ),
                    length,
                    codec,
                    kind == "sliding_attention",
                )
                for kind in config.layer_types[:count]
            ]
        )
