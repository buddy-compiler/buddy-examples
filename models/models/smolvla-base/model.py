# ===- model.py ----------------------------------------------------------------
#
# SmolVLA as one graph: the VLM runs once over the images, the prompt and the
# robot state; the action expert then integrates the flow from the noise to an
# action chunk in ten Euler steps.
#
# ===---------------------------------------------------------------------------

from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla import smolvlm_with_expert
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

checkpoint = Path(__file__).resolve().parent / "output"
config = PreTrainedConfig.from_pretrained(checkpoint / "smolvla-base")
config.vlm_model_name = str(checkpoint / "smolvlm2-500m-video-instruct")
# The VLM layers are loaded from the policy checkpoint, not from SmolVLM2.
config.load_vlm_weights = False
config.device = "cpu"


# RoPE writes the two rotated halves next to each other.
def apply_rope(x, positions, max_wavelength=10_000):
    d_half = x.shape[-1] // 2
    dtype = x.dtype
    x = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    radians = radians[..., None, :]
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


# Grouped-query attention: the query heads of a key/value head attend to it
# together, so the keys and values are not repeated per query head.
def eager_attention_forward(
    self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
):
    length = query_states.shape[1]
    source = key_states.shape[1]
    kv_heads = self.num_key_value_heads
    groups = self.num_attention_heads // kv_heads
    query = query_states.to(torch.float32).view(
        batch_size, length, kv_heads, groups, head_dim
    )
    query = query.permute(0, 2, 3, 1, 4).reshape(
        batch_size, kv_heads, groups * length, head_dim
    )
    key = key_states.to(torch.float32).transpose(1, 2)
    value = value_states.transpose(1, 2)
    scores = torch.matmul(query, key.transpose(2, 3)) * head_dim**-0.5
    scores = scores.view(batch_size, kv_heads, groups, length, source)
    scores = torch.where(
        attention_mask[:, None, None, :, :], scores, torch.finfo(torch.float32).min
    )
    probs = torch.softmax(scores, dim=-1).to(value.dtype)
    output = torch.matmul(
        probs.view(batch_size, kv_heads, groups * length, source), value
    )
    output = output.view(batch_size, kv_heads, groups, length, head_dim).permute(
        0, 3, 1, 2, 4
    )
    return output.reshape(batch_size, length, kv_heads * groups * head_dim)


class Policy(torch.nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, images, lang_tokens, lang_masks, state, noise):
        img_masks = [torch.ones(1, dtype=torch.bool)] * images.shape[0]
        return self.flow.sample_actions(
            list(images.unbind(0)),
            img_masks,
            lang_tokens,
            lang_masks.bool(),
            state,
            noise=noise,
        )


def load():
    """The policy in FP32, whatever the checkpoint stores, with the rewrites
    above in place."""
    policy = SmolVLAPolicy.from_pretrained(checkpoint / "smolvla-base", config=config)
    return rewrite(policy)


def rewrite(policy):
    """The policy in FP32 with the rewrites above in place."""
    flow = policy.float().eval().model

    # SmolVLM interpolates patch positions for images of any size, which
    # scatters into the position ids. The policy feeds whole square images of
    # the encoder size, where the positions are just the patch order.
    embeddings = flow.vlm_with_expert.get_vlm_model().vision_model.embeddings

    def whole_image_embeddings(pixel_values, patch_attention_mask):
        patches = embeddings.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        positions = torch.arange(embeddings.num_patches)[None, :]
        return patches + embeddings.position_embedding(positions)

    embeddings.forward = whole_image_embeddings
    smolvlm_with_expert.apply_rope = apply_rope
    smolvlm_with_expert.SmolVLMWithExpertModel.eager_attention_forward = (
        eager_attention_forward
    )
    return Policy(flow).eval()


def inputs():
    """Example inputs: the camera images in [-1, 1], the prompt tokens and
    their mask, the robot state and the noise of the action chunk."""
    cameras = len(config.image_features)
    height, width = config.resize_imgs_with_padding
    return (
        torch.zeros((cameras, 1, 3, height, width)),
        torch.zeros((1, config.tokenizer_max_length), dtype=torch.int64),
        torch.ones((1, config.tokenizer_max_length), dtype=torch.int64),
        torch.zeros((1, config.max_state_dim)),
        torch.zeros((1, config.chunk_size, config.max_action_dim)),
    )
