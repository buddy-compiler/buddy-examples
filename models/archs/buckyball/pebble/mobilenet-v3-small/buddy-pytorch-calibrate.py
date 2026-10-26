from pathlib import Path
import json

import torch
from datasets import load_dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torch.nn.utils.fusion import fuse_conv_bn_eval

from stack.compiler.frontend.quant import quantize, quantize_bias
from frontend.quant.graph import FP32

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
model_source = model_dir.parents[3] / "models/mobilenet-v3-small"
imagenet = model_dir.parents[3] / "datasets/imagenet-256/data"

# ===----------------------------------------------------------------------===
# Load the checkpoint and fold every batch normalization into the
# convolution before it, as the importer does.
# ===----------------------------------------------------------------------===
model = mobilenet_v3_small(weights=None).eval()
model.load_state_dict(
    torch.load(model_source / "output/mobilenet-v3-small/model.pth", weights_only=True)
)
for block in model.modules():
    if (
        isinstance(block, torch.nn.Sequential)
        and isinstance(block[0], torch.nn.Conv2d)
        and isinstance(block[1], torch.nn.BatchNorm2d)
    ):
        block[0] = fuse_conv_bn_eval(block[0], block[1])
        block[1] = torch.nn.Identity()

# The runner resizes the image to 224x224 and normalizes it.
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def images(split, count):
    files = sorted(str(path) for path in imagenet.glob(f"{split}-*.parquet"))
    samples = load_dataset("parquet", data_files=files, split="train")
    samples = samples.shuffle(seed=0).select(range(count))
    for start in range(0, count, 64):
        batch = samples[start : start + 64]
        yield (
            torch.stack([preprocess(image.convert("RGB")) for image in batch["image"]]),
            torch.tensor(batch["label"]),
        )


# ===----------------------------------------------------------------------===
# Sample the inputs of every convolution and Linear over 1024 training images
# and clip each at the magnitude whose INT8 encoding has the least squared
# error: a few outliers would otherwise take most of the 256 codes.
# ===----------------------------------------------------------------------===
torch.manual_seed(0)
samples = {}
hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: samples.setdefault(name, []).append(
            inputs[0].flatten()[torch.randperm(inputs[0].numel())[:200000]]
        )
    )
    for name, module in model.named_modules()
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
]
with torch.no_grad():
    for batch, _ in images("train", 1024):
        model(batch)
for hook in hooks:
    hook.remove()


def error(values, clip):
    step = clip / 127
    return ((quantize(values, 1 / step) * step - values) ** 2).mean().item()


maxima = {}
for name, values in samples.items():
    values = torch.cat(values)
    peak = values.abs().max().item()
    # The inputs of an SE excitation can be all zero.
    maxima[name] = (
        min(
            (peak * percent / 100 for percent in range(5, 101)),
            key=lambda clip: error(values, clip),
        )
        if peak
        else 0.0
    )

# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on 1024 validation images, the
# convolutions of FP32 left in FP32: the scales are rejected when the top-1
# accuracy drops more than 3 points below FP32.
# ===----------------------------------------------------------------------===
tests = list(images("val", 1024))


def accuracy(model):
    with torch.no_grad():
        correct = sum(
            (model(batch).argmax(1) == label).sum().item() for batch, label in tests
        )
    return correct / 1024 * 100


fp32 = accuracy(model)
for name, module in model.named_modules():
    if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) or name in FP32:
        continue
    # All-zero inputs: any positive step encodes them.
    scale = torch.tensor(maxima[name] / 127 if maxima[name] else 1.0)
    # A channel whose folded weights almost vanish takes a coarser step, so
    # that its bias code stays within 2^30.
    axes = list(range(1, module.weight.ndim))
    maximum = module.weight.detach().abs().amax(axes, keepdim=True)
    floor = module.bias.detach().abs().reshape(maximum.shape) / (scale * 2**30)
    step = torch.maximum(maximum / 127, floor)
    step = torch.where(step == 0, 1.0, step)
    module.weight.data = quantize(module.weight.detach(), 1 / step) * step
    bias_step = step.reshape(-1) * scale
    module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, inputs, scale=scale: quantize(inputs[0], 1 / scale) * scale
    )
int8 = accuracy(model)
print(f"FP32 top-1 {fp32:.2f}%, INT8 top-1 {int8:.2f}%")
if fp32 - int8 > 3:
    raise RuntimeError(f"INT8 top-1 {int8:.2f}% is more than 3 below FP32 {fp32:.2f}%")
(output / "calibration.json").write_text(json.dumps(maxima, indent=2) + "\n")
