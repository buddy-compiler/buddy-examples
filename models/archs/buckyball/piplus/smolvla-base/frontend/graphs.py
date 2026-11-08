import copy

import torch
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


class Vision(torch.nn.Module):
    """One camera image to its image tokens: the vision encoder and the
    connector into the language model."""

    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, image):
        return self.flow.vlm_with_expert.embed_image(image)


class Prefix(torch.nn.Module):
    """The prompt, the state and the image tokens of every camera to the keys
    and values of every language model layer, which the denoising steps read,
    and the padding mask of the prefix. The tokens of each camera are an
    input of their own: the cameras are encoded apart, on any core."""

    def __init__(self, flow):
        super().__init__()
        # The prefix embeds the image tokens Vision computed: a copy of the
        # flow whose embed_image passes them through. The copies get their
        # own submodule tables, which copy.copy would share with the flow.
        self.flow = copy.copy(flow)
        self.flow._modules = dict(flow._modules)
        self.flow.vlm_with_expert = copy.copy(flow.vlm_with_expert)
        self.flow.vlm_with_expert._modules = dict(flow.vlm_with_expert._modules)
        self.flow.vlm_with_expert.embed_image = lambda tokens: tokens

    def forward(self, lang_tokens, lang_masks, state, *tokens):
        masks = [torch.ones(1, dtype=torch.bool)] * len(tokens)
        embeddings, pad_masks, att_masks = self.flow.embed_prefix(
            list(tokens), masks, lang_tokens, lang_masks.bool(), state
        )
        _, cache = self.flow.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad_masks, att_masks),
            position_ids=torch.cumsum(pad_masks, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        layers = sorted(cache)
        return (
            pad_masks,
            *[cache[layer]["key_states"] for layer in layers],
            *[cache[layer]["value_states"] for layer in layers],
        )


class Step(torch.nn.Module):
    """One denoising step: the actions x_t at time t and the prefix to the
    velocity of the actions."""

    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, x_t, time, pad_masks, *cache):
        half = len(cache) // 2
        past = {
            layer: {"key_states": cache[layer], "value_states": cache[half + layer]}
            for layer in range(half)
        }
        return self.flow.denoise_step(pad_masks, past, x_t, time)
