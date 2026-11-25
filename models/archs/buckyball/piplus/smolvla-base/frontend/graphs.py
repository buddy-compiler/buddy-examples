import copy
import types

import torch
from lerobot.policies.smolvla.modeling_smolvla import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)

from frontend import rope


def rotate(flow):
    """RoPE over whole rows (rope.py): every attention gains the rotated q and
    k projections, and flow's vlm_with_expert attends with them."""
    for module in flow.modules():
        if all(hasattr(module, name) for name in ("q_proj", "k_proj", "head_dim")):
            module.q_rot = rope.rotated(module.q_proj, module.head_dim)
            module.k_rot = rope.rotated(module.k_proj, module.head_dim)
    vlm = flow.vlm_with_expert
    vlm.forward_attn_layer = types.MethodType(rope.forward_attn_layer, vlm)
    vlm.forward_cross_attn_layer = types.MethodType(rope.forward_cross_attn_layer, vlm)


class Vision(torch.nn.Module):
    """One camera image to its image tokens: the vision encoder and the
    connector into the language model."""

    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, image):
        # embed_image, without the attention mask: every patch of a camera
        # image is valid, so the encoder's additive mask is all zeros. The
        # graph then holds no mask to build and add to every head's scores.
        # The embeddings as a projection of the patches in token order: the
        # patch convolution in channel order, flattened and transposed back
        # (Idefics3VisionEmbeddings), is a transpose the hart CPU makes
        # element by element. With every patch valid, the position ids are
        # the patch indices, so the whole position table is added.
        model = self.flow.vlm_with_expert.get_vlm_model()
        vision = model.vision_model
        embeddings = vision.embeddings
        size = embeddings.patch_size
        pixels = image.to(dtype=vision.dtype)
        batch, channels, height, width = pixels.shape
        patches = pixels.reshape(
            batch, channels, height // size, size, width // size, size
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            batch, (height // size) * (width // size), channels * size * size
        )
        convolution = embeddings.patch_embedding
        hidden = torch.nn.functional.linear(
            patches,
            convolution.weight.reshape(convolution.out_channels, -1),
            convolution.bias,
        )
        hidden = hidden + embeddings.position_embedding.weight
        hidden = vision.encoder(
            inputs_embeds=hidden, attention_mask=None
        ).last_hidden_state
        return model.connector(vision.post_layernorm(hidden))


class Prefix(torch.nn.Module):
    """The prompt, the state and the image tokens of every camera to the keys
    and values of every language model layer, which the denoising steps read,
    and the padding mask of the prefix. The tokens of each camera are an
    input of their own: the cameras are encoded apart, on any hart."""

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
        rotate(self.flow)
        # The layers attend at the angles of the positions (rope.angles).
        head_dim = (
            self.flow.vlm_with_expert.get_vlm_model()
            .text_model.layers[0]
            .self_attn.head_dim
        )
        self.encode = lambda positions: rope.angles(positions, head_dim)

    def forward(self, lang_tokens, lang_masks, state, *tokens):
        # The keys and values of the cross layers projected by the expert,
        # once for every step (rope.forward_cross_attn_layer reads them).
        # Dynamo unpacks `*cache` as a tuple, which it cannot assign into:
        # the projections would run outside the graph.
        masks, angles, *cache = self.raw(lang_tokens, lang_masks, state, *tokens)
        cache = list(cache)
        vlm = self.flow.vlm_with_expert
        experts = vlm.get_model_layers([vlm.get_vlm_model().text_model, vlm.lm_expert])[
            1
        ]
        half = len(cache) // 2
        for layer, expert in enumerate(experts):
            if (
                expert is None
                or "cross" not in vlm.attention_mode
                or (
                    vlm.self_attn_every_n_layers > 0
                    and layer % vlm.self_attn_every_n_layers == 0
                )
            ):
                continue
            for index, projection in (
                (layer, expert.self_attn.k_proj),
                (half + layer, expert.self_attn.v_proj),
            ):
                states = cache[index]
                cache[index] = projection(states.reshape(*states.shape[:2], -1)).view(
                    *states.shape[:2], -1, expert.self_attn.head_dim
                )
        return (masks, angles, *cache)

    def raw(self, lang_tokens, lang_masks, state, *tokens):
        """The masks, the angles of the positions and the keys and values of
        every layer as the language model makes them."""
        masks = [torch.ones(1, dtype=torch.bool)] * len(tokens)
        embeddings, pad_masks, att_masks = self.flow.embed_prefix(
            list(tokens), masks, lang_tokens, lang_masks.bool(), state
        )
        # The prefix gains masked tokens up to a multiple of 32: its
        # attention takes no padding of its scores and values in any layer.
        tail = -embeddings.shape[1] % 32
        embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, tail))
        pad_masks = torch.nn.functional.pad(pad_masks, (0, tail), value=False)
        att_masks = torch.nn.functional.pad(att_masks, (0, tail))
        _, cache = self.flow.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad_masks, att_masks),
            position_ids=self.encode(torch.cumsum(pad_masks, dim=1) - 1),
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        layers = sorted(cache)
        # What every denoising step attends with, once: its attention mask
        # (the prefix's tokens, then its causal chunk) and the angles of the
        # positions of its chunk, which depend on the prefix only.
        chunk = -(-self.flow.config.chunk_size // 32) * 32
        suffix = torch.ones(1, chunk, dtype=torch.bool)
        masks = torch.cat(
            [
                pad_masks[:, None, :].expand(1, chunk, pad_masks.shape[1]),
                make_att_2d_masks(suffix, torch.ones(1, chunk)),
            ],
            dim=2,
        )
        positions = pad_masks.sum(-1)[:, None] + torch.cumsum(suffix, dim=1) - 1
        return (
            masks,
            self.encode(positions),
            *[cache[layer]["key_states"] for layer in layers],
            *[cache[layer]["value_states"] for layer in layers],
        )


class Step(torch.nn.Module):
    """One denoising step: the actions x_t at time t, the prefix's masks and
    angles for the steps and its keys and values to the velocity of the
    actions."""

    def __init__(self, flow):
        super().__init__()
        # The chunk of actions padded to a multiple of 32 tokens: with the
        # prefix's, the keys of the cross attention (the prefix) and of the
        # self attention (the prefix and the chunk) need no padding. The
        # actions attend causally, so the padding tokens at the end change no
        # action; their velocities are dropped.
        #
        # The hidden of the action expert padded to a multiple of 32 too:
        # zero rows and columns of the weights carry zero columns along the
        # residual stream, so its products take no padding of their K. An
        # RMSNorm over the wider row is the same with eps and its weight
        # scaled by 720 / 736 and its square root.
        self.flow = copy.deepcopy(flow)
        self.actions = flow.config.chunk_size
        self.flow.config.chunk_size = -(-self.actions // 32) * 32
        hidden = self.flow.vlm_with_expert.expert_hidden_size
        wide = -(-hidden // 32) * 32

        # The action embedding stays as wide as the time embedding it is
        # concatenated with; the products after it make the wide rows.
        for module in self.flow.modules():
            if (
                isinstance(module, torch.nn.Linear)
                and hidden in (module.in_features, module.out_features)
                and module is not self.flow.action_in_proj
            ):
                rows = wide if module.out_features == hidden else module.out_features
                columns = wide if module.in_features == hidden else module.in_features
                weight = torch.zeros(rows, columns, dtype=module.weight.dtype)
                weight[: module.out_features, : module.in_features] = module.weight
                module.weight = torch.nn.Parameter(weight)
                if module.bias is not None:
                    bias = torch.zeros(rows, dtype=module.bias.dtype)
                    bias[: module.out_features] = module.bias
                    module.bias = torch.nn.Parameter(bias)
                module.in_features, module.out_features = columns, rows
            elif (
                type(module).__name__ == "LlamaRMSNorm"
                and module.weight.shape[0] == hidden
            ):
                weight = torch.zeros(wide, dtype=module.weight.dtype)
                weight[:hidden] = module.weight * (hidden / wide) ** 0.5
                module.weight = torch.nn.Parameter(weight)
                module.variance_epsilon *= hidden / wide
        # The time of a step enters only through action_time_mlp_in of
        # [action embedding, time embedding]: W [a; e(t)] + b = W_a a +
        # (W_e e(t) + b). The step takes the time term as its input (the
        # program has it as a constant per step, time_bias), so no sine,
        # cosine or concatenation is left in the graph: W_a is its own Linear.
        mlp_in = self.flow.action_time_mlp_in
        width = self.flow.action_in_proj.out_features
        self.action_part = torch.nn.Linear(width, mlp_in.out_features, bias=False)
        self.action_part.weight = torch.nn.Parameter(
            mlp_in.weight[:, :width].detach().clone()
        )
        self.register_buffer("time_weight", mlp_in.weight[:, width:].detach().clone())
        self.register_buffer("time_offset", mlp_in.bias.detach().clone())
        rotate(self.flow)
        # The expert of a cross layer attends at its tokens' indices.
        vlm = self.flow.vlm_with_expert
        vlm.register_buffer(
            "cross_angles",
            rope.angles(
                torch.arange(self.flow.config.chunk_size)[None],
                vlm.lm_expert.layers[0].self_attn.head_dim,
            ),
        )

    def time_bias(self, time):
        """The time term of action_time_mlp_in at `time` ([1]): [1, hidden]."""
        embedding = create_sinusoidal_pos_embedding(
            time,
            self.flow.vlm_with_expert.expert_hidden_size,
            self.flow.config.min_period,
            self.flow.config.max_period,
            device=time.device,
        ).to(torch.float32)
        return torch.nn.functional.linear(embedding, self.time_weight, self.time_offset)

    def forward(self, x_t, time_bias, masks, angles, *cache):
        # The policy's denoise_step, with the masks and angles the prefix
        # made for every step and the time term of the step (time_bias).
        half = len(cache) // 2
        past = {
            layer: {"key_states": cache[layer], "value_states": cache[half + layer]}
            for layer in range(half)
        }
        x_t = torch.nn.functional.pad(
            x_t, (0, 0, 0, self.flow.config.chunk_size - self.actions)
        )
        # embed_suffix with the time term given.
        hidden = self.action_part(self.flow.action_in_proj(x_t)) + time_bias[:, None, :]
        suffix = self.flow.action_time_mlp_out(torch.nn.functional.silu(hidden))
        outputs, _ = self.flow.vlm_with_expert.forward(
            attention_mask=masks,
            position_ids=angles,
            past_key_values=past,
            inputs_embeds=[None, suffix],
            use_cache=self.flow.config.use_cache,
            fill_kv_cache=False,
        )
        velocity = self.flow.action_out_proj(
            outputs[1][:, -self.flow.config.chunk_size :].to(dtype=torch.float32)
        )
        return velocity[:, : self.actions]
