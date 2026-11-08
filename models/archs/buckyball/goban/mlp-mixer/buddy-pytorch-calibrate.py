from importlib import import_module
from pathlib import Path
import json

import torch

from compiler.frontend.quant import quantize, quantize_bias
from frontend.quant.graph import smooth_norms

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
imagenet = import_module("models.models.mlp-mixer.data")


def load(pattern, every, count):
    """The images of `imagenet.load`, cut into the [196, 768] patches the
    graph reads."""
    images, labels = imagenet.load(pattern, every, count)
    patches = images.reshape(-1, 3, 14, 16, 14, 16).permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(-1, 196, 768), labels


# ===----------------------------------------------------------------------===
# Record the largest magnitude of every input channel of every Linear over
# about 1024 training images; for the token-mixing Linears, of every row of
# the [C, K] input instead, a channel of the image. The graph classifies one
# image at a time; the software runs the images in batches of 64, and the
# maxima reduce over the batch axis.
# ===----------------------------------------------------------------------===
mixer = import_module("models.models.mlp-mixer.model")
model = mixer.load()
images, _ = load("train-*.parquet", 12, 1)
channel_maxima = {}


def record(name, x):
    maximum = x.abs().amax((0, 2) if "mlp_tokens" in name else (0, 1))
    channel_maxima[name] = torch.maximum(channel_maxima.get(name, maximum), maximum)


hooks = [
    module.register_forward_pre_hook(
        lambda module, args, name=name: record(name, args[0])
    )
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
]
with torch.no_grad():
    for batch in images.split(64):
        model(batch)
for hook in hooks:
    hook.remove()

# ===----------------------------------------------------------------------===
# Smooth the inputs of the channel-mixing fc1s and of the head through the
# LayerNorm that produces them: the norm divides every channel by s, the
# Linear weight takes s. s = sqrt(a / w) from the largest input magnitude a
# and weight magnitude w of the channel. A token-mixing Linear converts its
# input with one multiplier per row, from the largest magnitude of the row;
# every other Linear converts its whole input with one multiplier, from its
# largest (smoothed) magnitude.
# ===----------------------------------------------------------------------===
smooth = {}
for name, module in model.named_modules():
    if name == "head" or name.endswith("mlp_channels.fc1"):
        a = channel_maxima[name].clamp(min=1e-5)
        w = module.weight.detach().abs().amax(0).clamp(min=1e-5)
        smooth[name] = (a / w).sqrt()
maxima = {
    name: (
        maximum.tolist()
        if "mlp_tokens" in name
        else (maximum / smooth[name] if name in smooth else maximum).max().item()
    )
    for name, maximum in channel_maxima.items()
}

# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on 2000 validation images: the scales
# are rejected when the accuracy drops more than 1 point below FP32. Every
# Linear reads its weight with one scale per output channel and its input
# with the scales above; a token-mixing Linear keeps its bias in FP32.
# ===----------------------------------------------------------------------===
tests, labels = load("val-*.parquet", 1, 4)


def accuracy(model):
    with torch.no_grad():
        correct = sum(
            (model(batch)[:, 0].argmax(1) == label).sum().item()
            for batch, label in zip(tests.split(64), labels.split(64))
        )
    return correct / len(labels) * 100


fp32 = accuracy(model)
quantized = mixer.load()
smooth_norms(quantized, {name: factor.tolist() for name, factor in smooth.items()})
for name, module in quantized.named_modules():
    if not isinstance(module, torch.nn.Linear):
        continue
    step = module.weight.detach().abs().amax(1, keepdim=True) / 127
    module.weight.data = quantize(module.weight.detach(), 1 / step) * step
    if "mlp_tokens" in name:
        scale = torch.tensor(maxima[name]).reshape(-1, 1) / 127
    else:
        scale = torch.full([1, 1], maxima[name] / 127)
        bias_step = step.reshape(-1) * scale.reshape(-1)
        module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, args, scale=scale: quantize(args[0], 1 / scale) * scale
    )
int8 = accuracy(quantized)
print(f"FP32 accuracy {fp32:.2f}%, INT8 accuracy {int8:.2f}% on {len(labels)} images")
if fp32 - int8 > 1:
    raise RuntimeError(
        f"INT8 accuracy {int8:.2f}% is more than 1 below FP32 {fp32:.2f}%"
    )
calibration = {
    **maxima,
    "smooth": {name: factor.tolist() for name, factor in smooth.items()},
}
(output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
