from importlib import import_module
from pathlib import Path
import json

import torch
from torch.overrides import TorchFunctionMode
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ap_per_class, box_iou

from compiler.frontend.quant import quantize, quantize_bias

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
coco = import_module("models.models.yolo26n.data")


class Products(TorchFunctionMode):
    """Calls `hook(index, operands)` on every attention product, numbered in
    the order the model runs them, and computes the product of what the hook
    returns."""

    def __init__(self, hook):
        super().__init__()
        self.hook = hook
        self.index = 0

    def __torch_function__(self, function, types, args=(), kwargs=None):
        # `a @ b` arrives as the matmul method of the tensor.
        if getattr(function, "__name__", None) == "matmul":
            args = self.hook(self.index, args)
            self.index += 1
        return function(*args, **(kwargs or {}))


# ===----------------------------------------------------------------------===
# Load the checkpoint with every batch normalization folded into the
# convolution before it, as the importer does, and record the largest
# magnitude of every input channel of every convolution over the first 64
# images of COCO128.
# ===----------------------------------------------------------------------===
yolo = import_module("models.models.yolo26n.model")


def load_fused():
    model = yolo.load()
    model.fuse()
    return model


model = load_fused()
convolutions = {
    name: module
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Conv2d)
}
images, _ = coco.load(coco.paths[:64])
channel_maxima = {}


def record_channels(name, x):
    maximum = x.abs().amax([0, 2, 3])
    channel_maxima[name] = torch.maximum(channel_maxima.get(name, maximum), maximum)


hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: record_channels(name, inputs[0])
    )
    for name, module in convolutions.items()
]
product_maxima = {}


def record_product(index, operands):
    for side, operand in enumerate(operands):
        name = f"matmul.{index}.{side}"
        product_maxima[name] = max(
            product_maxima.get(name, 0.0), operand.abs().max().item()
        )
    return operands


with torch.no_grad():
    for image in images.split(1):
        with Products(record_product):
            model(image)
for hook in hooks:
    hook.remove()

# ===----------------------------------------------------------------------===
# Smooth the inputs: the input divides every channel by s while it converts
# to INT8 online, the weight takes s. s = sqrt(a / w) from the largest input
# magnitude a and weight magnitude w of the channel; a depthwise convolution
# reads one input channel per output channel, so it takes all of a. Then
# sample every smoothed input and clip it at the 99.99th percentile of its
# magnitudes: a few outliers would otherwise take most of the 256 codes.
# ===----------------------------------------------------------------------===
smooth = {}
for name, module in convolutions.items():
    a = channel_maxima[name].clamp(min=1e-5)
    if module.groups > 1:
        smooth[name] = a
    else:
        w = module.weight.detach().abs().amax([0, 2, 3]).clamp(min=1e-5)
        smooth[name] = (a / w).sqrt()

torch.manual_seed(0)
samples = {}


def sample(name, x):
    x = (x / smooth[name].reshape(1, -1, 1, 1)).abs().flatten()
    samples.setdefault(name, []).append(x[torch.randperm(x.numel())[:200000]])


hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: sample(name, inputs[0])
    )
    for name, module in convolutions.items()
]
with torch.no_grad():
    for image in images.split(1):
        model(image)
for hook in hooks:
    hook.remove()
clips = {
    name: torch.quantile(torch.cat(values)[: 2**24], 0.9999).item()
    for name, values in samples.items()
}

# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on the other 64 images: the scales
# are rejected when mAP50-95 drops more than 1 point below FP32.
# ===----------------------------------------------------------------------===
tests, labels = coco.load(coco.paths[64:])
validator = DetectionValidator()


def mean_ap(model):
    """mAP50-95 of the end-to-end detections, matched as ultralytics does."""
    correct, scores, classes, targets = [], [], [], []
    with torch.no_grad():
        for image, label in zip(tests.split(1), labels):
            detections = model(image)[0]
            iou = box_iou(label[:, 1:], detections[:, :4])
            correct.append(
                validator.match_predictions(detections[:, 5], label[:, 0], iou)
            )
            scores.append(detections[:, 4])
            classes.append(detections[:, 5])
            targets.append(label[:, 0])
    ap = ap_per_class(
        torch.cat(correct).numpy(),
        torch.cat(scores).numpy(),
        torch.cat(classes).numpy(),
        torch.cat(targets).numpy(),
    )[5]
    return ap.mean() * 100


fp32 = mean_ap(model)
quantized = load_fused()
for name, module in quantized.named_modules():
    if not isinstance(module, torch.nn.Conv2d):
        continue
    scale = torch.tensor(clips[name] / 127)
    shape = (-1, 1, 1, 1) if module.groups > 1 else (1, -1, 1, 1)
    weight = module.weight.detach() * smooth[name].reshape(shape)
    # A channel whose folded weights almost vanish takes a coarser step, so
    # that its bias code stays within 2^30.
    maximum = weight.abs().amax([1, 2, 3], keepdim=True)
    floor = module.bias.detach().abs().reshape(maximum.shape) / (scale * 2**30)
    step = torch.maximum(maximum / 127, floor)
    module.weight.data = quantize(weight, 1 / step) * step
    bias_step = step.reshape(-1) * scale
    module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, inputs, name=name, scale=scale: quantize(
            inputs[0] / smooth[name].reshape(1, -1, 1, 1), 1 / scale
        )
        * scale
    )


def fake_product(index, operands):
    # Every image runs the same products again.
    index %= len(product_maxima) // 2
    return tuple(
        quantize(operand, 127 / product_maxima[f"matmul.{index}.{side}"])
        * (product_maxima[f"matmul.{index}.{side}"] / 127)
        for side, operand in enumerate(operands)
    )


with Products(fake_product):
    quantized_map = mean_ap(quantized)
print(f"FP32 mAP50-95 {fp32:.2f}, INT8 mAP50-95 {quantized_map:.2f} on 64 images")
if fp32 - quantized_map > 1:
    raise RuntimeError(
        f"INT8 mAP50-95 {quantized_map:.2f} is more than 1 below FP32 {fp32:.2f}"
    )
calibration = {
    name: {"smooth": smooth[name].tolist(), "clip": clips[name]} for name in clips
}
calibration.update(product_maxima)
(output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
