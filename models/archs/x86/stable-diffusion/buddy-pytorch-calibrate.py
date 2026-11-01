# ===- buddy-pytorch-calibrate.py -----------------------------------------
#
# Stable Diffusion quantization check: the text encoder, the UNet and the VAE
# decoder as the quantized graphs compute them (Linears and attention
# products in MXINT8, convolutions in INT8), rejected when the image is below
# 30 dB PSNR against FP32. Every activation scale is computed online, so
# nothing is calibrated: calibration.json records the accepted PSNR.
#
# ===---------------------------------------------------------------------------

from importlib import import_module
from pathlib import Path
import json

import torch
from diffusers import PNDMScheduler
from diffusers.models.attention_processor import Attention, AttnProcessor
from torch.overrides import TorchFunctionMode
from transformers import CLIPTokenizer

from compiler.frontend.quant import dequantize_mxint8, quantize, quantize_mxint8

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
text_encoder = import_module("models.models.stable-diffusion.model-text_encoder")
unet = import_module("models.models.stable-diffusion.model-unet")
vae = import_module("models.models.stable-diffusion.model-vae")


def attention_scores(self, query, key, attention_mask=None):
    """The attention probabilities from one batched product, scaled: the score
    product is then a BatchMatmul the quantization reads like every other
    product, not the baddbmm diffusers fuses the scale into."""
    scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale
    if attention_mask is not None:
        scores = scores + attention_mask
    return scores.softmax(dim=-1)


Attention.get_attention_scores = attention_scores
models = {
    "text_encoder": text_encoder.load(),
    "unet": unet.load(),
    "vae": vae.load(),
}
# The attention runs through explicit products, as the importer.
models["text_encoder"].text_encoder.config._attn_implementation = "eager"
models["unet"].unet.set_attn_processor(AttnProcessor())
models["vae"].vae.set_attn_processor(AttnProcessor())
tokenizer = CLIPTokenizer.from_pretrained(text_encoder.checkpoint / "tokenizer")
scheduler = PNDMScheduler.from_pretrained(text_encoder.checkpoint / "scheduler")
# The runner: 30 PLMS steps, guidance 7.5.
steps = 30


class Products(TorchFunctionMode):
    """Calls `hook(operands)` on every product of two activations and
    computes the product of what the hook returns."""

    def __init__(self, hook):
        super().__init__()
        self.hook = hook

    def __torch_function__(self, function, types, args=(), kwargs=None):
        name = getattr(function, "__name__", None)
        # `a @ b` arrives as torch._C.TensorBase.matmul.
        if name in ("matmul", "bmm"):
            args = self.hook(args)
        return function(*args, **(kwargs or {}))


def generate(prompt, seed, product):
    """The image of the runner for `prompt`, in [0, 1]. Every product goes
    through `product(operands)`."""

    def run(graph, *inputs):
        with Products(product):
            return models[graph](*inputs)[0]

    with torch.no_grad():
        states = [
            run(
                "text_encoder",
                tokenizer(
                    text, padding="max_length", max_length=77, return_tensors="pt"
                ).input_ids,
            )
            for text in ("", prompt)
        ]
        states = torch.cat(states)
        latents = torch.randn(
            (1, 4, 64, 64), generator=torch.Generator().manual_seed(seed)
        )
        scheduler.set_timesteps(steps)
        for timestep in scheduler.timesteps:
            noise = run(
                "unet",
                torch.cat([latents, latents]),
                timestep.reshape(1).float(),
                states,
            )
            noise = noise[:1] + 7.5 * (noise[1:] - noise[:1])
            latents = scheduler.step(noise, timestep, latents).prev_sample
        image = run("vae", latents / 0.18215)
    return ((image + 1) / 2).clamp(0, 1)


def tokenwise(x, axis):
    """x as INT8 values with one scale per token: the largest magnitude
    along `axis` over 127."""
    step = x.abs().amax(axis, keepdim=True).clamp(min=1e-30) / 127
    return quantize(x, 1 / step) * step


def split(x, axis):
    """x as the two INT8 terms the graph reads it with: its INT8 value plus
    the INT8 value of what that leaves out, one scale per pixel each."""
    head = tokenwise(x, axis)
    return head + tokenwise(x - head, axis)


def mxint8(values, axis):
    """Values through MXINT8 along `axis`, as the graph computes them."""
    return dequantize_mxint8(quantize_mxint8(values, axis), axis)


def quantized_product(operands):
    # The products whose K is not a multiple of 32 stay FP32.
    if operands[0].shape[-1] % 32 != 0:
        return operands
    return mxint8(operands[0], -1), mxint8(operands[1], -2)


prompt = "a photo of a flower in a garden"
fp32 = generate(prompt, 42, lambda operands: operands)

# ===----------------------------------------------------------------------===
# Generate the image again with the quantized graphs: the weights of the
# Linears packed in MXINT8 along K and their inputs quantized online, the
# convolutions with INT8 weights (one scale per output channel) and inputs
# as two INT8 terms (one online scale per pixel each), biases FP32.
# Rejected below 30 dB PSNR.
# ===----------------------------------------------------------------------===
for model in models.values():
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = mxint8(module.weight.detach(), 1)
            module.register_forward_pre_hook(lambda module, args: mxint8(args[0], -1))
        elif isinstance(module, torch.nn.Conv2d):
            weight = module.weight.detach()
            step = weight.abs().amax([1, 2, 3], keepdim=True) / 127
            module.weight.data = quantize(weight, 1 / step) * step
            module.register_forward_pre_hook(lambda module, args: split(args[0], 1))
quantized = generate(prompt, 42, quantized_product)
psnr = (10 * torch.log10(1 / (quantized - fp32).square().mean())).item()
print(f"Quantized image: {psnr:.2f} dB PSNR against FP32")
if psnr < 30:
    raise RuntimeError(
        f"Quantized image is {psnr:.2f} dB PSNR against FP32, below 30 dB"
    )
(output / "calibration.json").write_text(json.dumps({"psnr": psnr}, indent=2) + "\n")
