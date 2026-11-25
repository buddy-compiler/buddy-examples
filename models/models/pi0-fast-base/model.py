# ===- model.py ----------------------------------------------------------------
#
# pi0-FAST as one graph: PaliGemma runs once over the images and the prompt
# (which carries the discretized robot state), then decodes the FAST tokens of
# an action chunk one at a time, each the most likely after the ones before
# it, with the keys and values of the tokens before it cached, up to an EOS.
# The tokens are the policy's output: the FAST tokenizer turns them into
# actions afterwards.
#
# ===---------------------------------------------------------------------------

from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

checkpoint = Path(__file__).resolve().parent / "output/pi0-fast-base"
config = PreTrainedConfig.from_pretrained(checkpoint)
config.device = "cpu"
# The Gemma vocabulary of PaliGemma, kept with the checkpoint (its hub
# repository is gated).
config.text_tokenizer_name = str(checkpoint.parent / "gemma-tokenizer")


class Start:
    """The token the decoding routine appends to the padded prompt, where
    lerobot writes a BOS: the space of "Action: ". The base checkpoint is
    openpi's, whose sequence is the prompt, "Action: ", the FAST tokens and
    an EOS with no BOS between (after a BOS it only repeats the BOS); the
    prompt of observation.py ends with "Action:"."""

    def __init__(self, tokenizer):
        self.bos_token_id = tokenizer.encode("Action: ", add_special_tokens=False)[-1]


class Policy(torch.nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, images, tokens, masks):
        img_masks = [torch.ones(1, dtype=torch.bool)] * images.shape[0]
        return self.flow.sample_actions_fast_kv_cache(
            list(images.unbind(0)),
            img_masks,
            tokens,
            masks.bool(),
            max_decoding_steps=self.flow.config.max_decoding_steps,
            temperature=0.0,
        )


def load():
    """The policy in FP32, whatever the checkpoint stores."""
    policy = PI0FastPolicy.from_pretrained(checkpoint, config=config)
    flow = policy.float().eval().model
    # Gemma attends through the eager implementation, explicit products.
    flow.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
        "eager"
    )
    flow._paligemma_tokenizer = Start(flow._paligemma_tokenizer)
    return Policy(flow).eval()


def inputs():
    """Example inputs: the camera images in [-1, 1], the prompt tokens (with
    the discretized state) and their mask."""
    cameras = len(config.image_features)
    height, width = config.image_resolution
    return (
        torch.zeros((cameras, 1, 3, height, width)),
        torch.zeros((1, config.tokenizer_max_length), dtype=torch.int64),
        torch.ones((1, config.tokenizer_max_length), dtype=torch.int64),
    )
