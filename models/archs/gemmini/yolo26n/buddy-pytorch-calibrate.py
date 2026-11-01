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


def int8(module):
    """Gemmini computes in INT8 only: every convolution."""
    return isinstance(module, torch.nn.Conv2d)


def depthwise(module):
    return isinstance(module, torch.nn.Conv2d) and module.groups > 1


def channels(module):
    """The broadcast shape of one value per input channel of the weight:
    axis 0 of the [C, 1, H, W] depthwise weights, axis 1 of the others."""
    if depthwise(module):
        return [-1, 1, 1, 1]
    return [1, -1] + [1] * (module.weight.ndim - 2)


def inputs(x):
    """The broadcast shape of one value per channel of an input: axis 1 of a
    convolution input, the last axis of a Linear one."""
    return [1, -1, 1, 1] if x.ndim == 4 else [1] * (x.ndim - 1) + [-1]


# ===----------------------------------------------------------------------===
# Record the largest magnitude of every input channel of every convolution
# of the checkpoint over the first 64 images of COCO128.
# ===----------------------------------------------------------------------===
yolo = import_module("models.models.yolo26n.model")
model = yolo.load()
images, _ = coco.load(coco.paths[:64])
channel_maxima = {}


def record_channels(name, x):
    axes = [
        axis for axis in range(x.ndim) if axis != (1 if x.ndim == 4 else x.ndim - 1)
    ]
    maximum = x.abs().amax(axes)
    channel_maxima[name] = torch.maximum(channel_maxima.get(name, maximum), maximum)


hooks = [
    module.register_forward_pre_hook(
        lambda module, args, name=name: record_channels(name, args[0])
    )
    for name, module in model.named_modules()
    if int8(module)
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
# Smooth the inputs: the CPU divides every input channel by s while it
# converts the input to INT8 online, the weight takes s. s = a^0.9 / w^0.1
# from the largest input magnitude a and weight magnitude w of the channel; a
# depthwise convolution reads one input channel per output channel, so it
# takes all of a. Then record the largest magnitude of every smoothed input.
# ===----------------------------------------------------------------------===
smooth = {}
for name, module in model.named_modules():
    if not int8(module):
        continue
    a = channel_maxima[name].clamp(min=1e-5)
    if depthwise(module):
        smooth[name] = a
    else:
        weight = module.weight.detach().abs()
        w = weight.amax([axis for axis in range(weight.ndim) if axis != 1])
        smooth[name] = a**0.9 / w.clamp(min=1e-5) ** 0.1
maxima = {}


def record_smoothed(name, x):
    x = x / smooth[name].reshape(inputs(x))
    maxima[name] = max(maxima.get(name, 0.0), x.abs().max().item())


hooks = [
    module.register_forward_pre_hook(
        lambda module, args, name=name: record_smoothed(name, args[0])
    )
    for name, module in model.named_modules()
    if int8(module)
]
with torch.no_grad():
    for image in images.split(1):
        model(image)
for hook in hooks:
    hook.remove()

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
quantized = yolo.load()
for name, module in quantized.named_modules():
    if not int8(module):
        continue
    weight = module.weight.detach() * smooth[name].reshape(channels(module))
    axes = list(range(1, weight.ndim))
    maximum = weight.abs().amax(axes, keepdim=True)
    # A padding channel is zero; any positive step encodes it.
    step = torch.where(maximum == 0, 1.0, maximum / 127)
    module.weight.data = quantize(weight, 1 / step) * step
    # All-zero inputs (an SE excitation): any positive step encodes them.
    scale = torch.tensor(maxima[name] / 127 if maxima[name] else 1.0)
    if module.bias is not None:
        bias_step = step.reshape(-1) * scale
        module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, args, name=name, scale=scale: quantize(
            args[0] / smooth[name].reshape(inputs(args[0])), 1 / scale
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
    name: {"smooth": smooth[name].tolist(), "maximum": maxima[name]} for name in maxima
}
calibration.update(product_maxima)
(output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
