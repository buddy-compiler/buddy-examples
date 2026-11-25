"""The attention layers of smolvlm_with_expert (lerobot 0.6.0) with RoPE as
x cos + rotate_half(x) sin over whole rows: rotate_half of a projection is
the projection by the rotated weight (`rotated`), so no half of a row is
split off or concatenated. The rotated weight's rows are the weight's,
permuted and negated, so its MXINT8 codes and exponents are the weight's.
The cache of a cross layer holds the expert's keys and values, which the
prefix projects once (graphs.Prefix), not every denoising step.

The layers take the angles of their positions (`angles`) for position_ids:
the prefix makes the steps' once, and the expert's in a cross layer, whose
positions are its tokens' indices, are a constant of the step
(vlm_with_expert.cross_angles)."""

import torch


def rotated(linear, head_dim):
    """The projection rotate_half(linear(x)) per head: output row j of a head
    is -row j + d/2 for j < d/2, row j - d/2 otherwise."""
    half = head_dim // 2
    weight = linear.weight.view(-1, 2, half, linear.in_features)
    weight = torch.stack((-weight[:, 1], weight[:, 0]), dim=1).reshape_as(linear.weight)
    result = torch.nn.Linear(linear.in_features, linear.out_features, bias=False)
    result.weight = torch.nn.Parameter(weight.detach().clone())
    return result


def angles(positions, head_dim, max_wavelength=10_000):
    """cos and sin [2, B, L, D] of RoPE positions [B, L] for heads of
    head_dim, the angles of each half repeated over the row."""
    d_half = head_dim // 2
    freq_exponents = (2.0 / head_dim) * torch.arange(d_half, dtype=torch.float32)
    timescale = (max_wavelength**freq_exponents).repeat(2)
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    return torch.stack((torch.cos(radians), torch.sin(radians)))


def rope(x, x_rot, angles):
    """x cos + rotate_half(x) sin of x [B, L, H, D] at angles [2, B, L, D]."""
    return (
        x.to(torch.float32) * angles[0][:, :, None, :]
        + x_rot.to(torch.float32) * angles[1][:, :, None, :]
    )


def forward_attn_layer(
    self,
    model_layers,
    inputs_embeds,
    layer_idx,
    position_ids,
    attention_mask,
    batch_size,
    head_dim,
    use_cache: bool = True,
    fill_kv_cache: bool = True,
    past_key_values=None,
) -> list[torch.Tensor]:
    query_states = []
    key_states = []
    value_states = []
    query_rots = []
    key_rots = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = model_layers[i][layer_idx]
        if hidden_states is None or layer is None:
            continue
        hidden_states = layer.input_layernorm(hidden_states)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

        hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
        query_rots.append(layer.self_attn.q_rot(hidden_states).view(hidden_shape))
        key_rots.append(layer.self_attn.k_rot(hidden_states).view(hidden_shape))

    # B,L,H,D with L sequence length, H number of heads, D head dim
    # concatenate on the number of embeddings/tokens
    query_states = torch.cat(query_states, dim=1)
    key_states = torch.cat(key_states, dim=1)
    value_states = torch.cat(value_states, dim=1)
    query_rots = torch.cat(query_rots, dim=1)
    key_rots = torch.cat(key_rots, dim=1)
    seq_len = query_states.shape[1]
    if seq_len < position_ids.shape[2]:
        _position_ids = position_ids[:, :, :seq_len]
        _attention_mask = attention_mask[:, :seq_len, :seq_len]
    else:
        _position_ids = position_ids
        _attention_mask = attention_mask

    attention_mask_ = _attention_mask
    position_ids_ = _position_ids

    query_states = rope(query_states, query_rots, position_ids_)
    key_states = rope(key_states, key_rots, position_ids_)

    if use_cache and past_key_values is None:
        past_key_values = {}

    if use_cache:
        if fill_kv_cache:
            past_key_values[layer_idx] = {
                "key_states": key_states,
                "value_states": value_states,
            }
        else:
            # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
            # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
            # the max len, then we (for instance) double the cache size. This implementation already exists
            # in `transformers`. (molbap)
            key_states = torch.cat(
                [past_key_values[layer_idx]["key_states"], key_states], dim=1
            )
            value_states = torch.cat(
                [past_key_values[layer_idx]["value_states"], value_states], dim=1
            )

    attention_interface = self.get_attention_interface()

    att_output = attention_interface(
        attention_mask_, batch_size, head_dim, query_states, key_states, value_states
    )
    return [att_output], past_key_values


def forward_cross_attn_layer(
    self,
    model_layers,
    inputs_embeds,
    layer_idx,
    position_ids,
    attention_mask,
    batch_size,
    head_dim,
    use_cache: bool = True,
    fill_kv_cache: bool = True,
    past_key_values=None,
) -> list[torch.Tensor]:
    attention_interface = self.get_attention_interface()

    att_outputs = []
    assert len(inputs_embeds) == 2 or (
        use_cache and past_key_values is not None and not fill_kv_cache
    ), f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"

    if len(inputs_embeds) == 2 and not past_key_values:
        # Prefix attention
        seq_len = inputs_embeds[0].shape[1]
        position_id = position_ids[:, :, :seq_len]
        prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

        layer = model_layers[0][layer_idx]

        hidden_states = layer.input_layernorm(inputs_embeds[0])

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

        hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
        value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        query_states = rope(
            query_state,
            layer.self_attn.q_rot(hidden_states).view(hidden_shape),
            position_id,
        )
        key_states = rope(
            key_state,
            layer.self_attn.k_rot(hidden_states).view(hidden_shape),
            position_id,
        )

        att_output = attention_interface(
            prefix_attention_mask,
            batch_size,
            head_dim,
            query_states,
            key_states,
            value_states,
        )
        att_outputs.append(att_output)

    if use_cache and past_key_values is None:
        past_key_values = {}

    if use_cache:
        if fill_kv_cache:
            past_key_values[layer_idx] = {
                "key_states": key_states,
                "value_states": value_states,
            }
        else:
            # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
            # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
            # the max len, then we (for instance) double the cache size. This implementation already exists
            # in `transformers`. (molbap)
            key_states = past_key_values[layer_idx]["key_states"]
            value_states = past_key_values[layer_idx]["value_states"]

    # Expert
    expert_layer = model_layers[1][layer_idx]
    if expert_layer is not None:
        expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

        expert_input_shape = expert_hidden_states.shape[:-1]
        expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

        expert_hidden_states = expert_hidden_states.to(
            dtype=expert_layer.self_attn.q_proj.weight.dtype
        )
        expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(
            expert_hidden_shape
        )

        # The cache of a cross layer holds the expert's keys and values: the
        # prefix projects them once for every step (graphs.Prefix).
        expert_key_states = key_states
        expert_value_states = value_states

        expert_attention_mask = attention_mask[
            :, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :
        ]  # take into account kv

        expert_query_states = rope(
            expert_query_state,
            expert_layer.self_attn.q_rot(expert_hidden_states).view(
                expert_hidden_shape
            ),
            self.cross_angles,
        )

        att_output = attention_interface(
            expert_attention_mask,
            batch_size,
            head_dim,
            expert_query_states,
            expert_key_states,
            expert_value_states,
        )
        att_outputs.append(att_output)
    else:
        att_outputs.append(None)

    # att_output = att_output.to(dtype=models[i].dtype)
    return att_outputs, past_key_values
