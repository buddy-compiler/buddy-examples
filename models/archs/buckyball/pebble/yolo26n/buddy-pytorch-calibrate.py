from importlib import import_module
from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch

torch.set_num_threads(8)
from torch.overrides import TorchFunctionMode
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ap_per_class, box_iou

from compiler.frontend.quant import quantize, quantize_bias
from models.archs.buckyball.pebble.fixed import coefficients
from models.archs.buckyball.pebble.fixed_reference import round_div
from frontend.nonlinear_reference import IntegerNonlinear
from frontend.residual_reference import ResidualAdds
from ultralytics.nn.modules.block import Attention

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
coco = import_module("models.models.yolo26n.data")


class Products(TorchFunctionMode):
    """Calls `hook(index, operands)` on every attention product, numbered in
    the order the model runs them, and computes the product of what the hook
    returns."""

    def __init__(self, hook, result=False):
        super().__init__()
        self.hook = hook
        self.index = 0
        self.result = result

    def __torch_function__(self, function, types, args=(), kwargs=None):
        # `a @ b` arrives as the matmul method of the tensor.
        if getattr(function, "__name__", None) == "matmul":
            args = self.hook(self.index, args)
            self.index += 1
            if self.result:
                return args
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
    # Ultralytics shares its stateless default activation across Conv modules.
    # Per-layer calibration hooks need distinct activation instances.
    for parent in list(model.modules()):
        for name, child in parent.named_children():
            if isinstance(child, torch.nn.SiLU):
                setattr(parent, name, torch.nn.SiLU(inplace=child.inplace))
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
output_samples = {}
output_generator = torch.Generator().manual_seed(1)


def sample(name, x):
    x = (x / smooth[name].reshape(1, -1, 1, 1)).abs().flatten()
    samples.setdefault(name, []).append(x[torch.randperm(x.numel())[:200000]])


def sample_output(name, output):
    values = output.permute(0, 2, 3, 1).reshape(-1, output.shape[1])
    rows = torch.randperm(values.shape[0], generator=output_generator)[
        : 200000 // values.shape[1]
    ]
    output_samples.setdefault(name, []).append(values[rows])


hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: sample(name, inputs[0])
    )
    for name, module in convolutions.items()
]
hooks += [
    module.register_forward_hook(
        lambda module, inputs, output, name=name: sample_output(name, output)
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

silu_input_steps, silu_output_steps = {}, {}
for name, samples_ in output_samples.items():
    values = torch.cat(samples_)
    input_clip = torch.quantile(values.abs(), 0.9999, dim=0)
    output_clip = torch.quantile(torch.nn.functional.silu(values).abs(), 0.9999, dim=0)
    # A zero channel is represented exactly by code zero with unit step.
    silu_input_steps[name] = torch.where(input_clip == 0, 1.0, input_clip / 127)
    silu_output_steps[name] = torch.where(output_clip == 0, 1.0, output_clip / 127)

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
for name, module in quantized.named_modules():
    if not isinstance(module, torch.nn.SiLU):
        continue
    convolution = name.rsplit(".", 1)[0] + ".conv"
    input_step = silu_input_steps[convolution].reshape(1, -1, 1, 1)
    output_step = silu_output_steps[convolution].reshape(1, -1, 1, 1)
    module.register_forward_pre_hook(
        lambda module, inputs, step=input_step: quantize(inputs[0], 1 / step).float()
        * step
    )
    module.register_forward_hook(
        lambda module, inputs, output, step=output_step: quantize(
            output, 1 / step
        ).float()
        * step
    )
with Products(fake_product):
    lookup_map = mean_ap(quantized)
print(f"INT8 SiLU lookup mAP50-95 {lookup_map:.2f} on 64 images")
if fp32 - lookup_map > 1:
    raise RuntimeError(
        f"SiLU lookup mAP50-95 {lookup_map:.2f} is more than 1 below FP32 {fp32:.2f}"
    )


def integer_product(index, operands):
    index %= len(product_maxima) // 2
    steps = [
        torch.tensor(
            product_maxima[f"matmul.{index}.{side}"] / 127, dtype=torch.float32
        )
        for side in (0, 1)
    ]
    codes = [
        quantize(operand, 1 / step).float()
        for operand, step in zip(operands, steps, strict=True)
    ]
    # All integer products and partial sums are exactly representable in FP32.
    if codes[0].shape[-1] * 128 * 128 >= 2**24:
        raise RuntimeError(
            "Integer attention reference exceeds exact FP32 accumulation"
        )
    return (codes[0] @ codes[1]) * (steps[0] * steps[1])


with Products(integer_product, result=True):
    integer_map = mean_ap(quantized)
print(f"INT8 integer attention mAP50-95 {integer_map:.2f} on 64 images")
if fp32 - integer_map > 1:
    raise RuntimeError(
        f"Integer attention mAP50-95 {integer_map:.2f} is more than 1 below FP32 {fp32:.2f}"
    )


class IntegerConv(torch.nn.Module):
    """Integer accumulation with the recipe's single SiLU input multiplier."""

    def __init__(self, module, name, silu):
        super().__init__()
        scale = torch.tensor(clips[name] / 127)
        shape = (-1, 1, 1, 1) if module.groups > 1 else (1, -1, 1, 1)
        weight = module.weight.detach() * smooth[name].reshape(shape)
        maximum = weight.abs().amax([1, 2, 3], keepdim=True)
        floor = module.bias.detach().abs().reshape(maximum.shape) / (scale * 2**30)
        step = torch.maximum(maximum / 127, floor)
        self.weight = quantize(weight, 1 / step).float()
        self.step = (step.reshape(-1) * scale).reshape(1, -1, 1, 1)
        self.bias = (
            quantize_bias(module.bias.detach(), self.step.reshape(-1))
            .long()
            .reshape(1, -1, 1, 1)
        )
        self.multiplier = 1 / (scale * smooth[name].reshape(1, -1, 1, 1))
        self.stride, self.padding, self.dilation, self.groups = (
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )
        self.silu_step = silu_input_steps[name].reshape(1, -1, 1, 1) if silu else None
        self.fixed_scores = False

    def forward(self, x):
        codes = quantize(x, self.multiplier).float()
        product = torch.nn.functional.conv2d(
            codes,
            self.weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        assert product.abs().max() < 2**24
        raw = product.long() + self.bias
        if self.fixed_scores:
            multiplier, shift = coefficients(self.step, 6)
            code = round_div(
                raw * multiplier.long().reshape(self.step.shape),
                1 << shift.long().reshape(self.step.shape),
            )
            return code.clamp(-512, 511).float() / 64
        result = raw.float()
        if self.silu_step is not None:
            return (
                quantize(result, self.step * (1 / self.silu_step)).float()
                * self.silu_step
            )
        return result * self.step


integer_model = load_fused()
silu_convolutions = {
    name.rsplit(".", 1)[0] + ".conv"
    for name, module in integer_model.named_modules()
    if isinstance(module, torch.nn.SiLU)
}
for name, module in list(integer_model.named_modules()):
    if isinstance(module, torch.nn.Conv2d):
        integer_model.set_submodule(
            name, IntegerConv(module, name, name in silu_convolutions)
        )
    elif isinstance(module, torch.nn.SiLU):
        convolution = name.rsplit(".", 1)[0] + ".conv"
        step = silu_output_steps[convolution].reshape(1, -1, 1, 1)
        module.register_forward_hook(
            lambda module, inputs, output, step=step: quantize(output, 1 / step).float()
            * step
        )
with Products(integer_product, result=True):
    accumulator_map = mean_ap(integer_model)
print(f"INT8 integer accumulators mAP50-95 {accumulator_map:.2f} on 64 images")
if fp32 - accumulator_map > 1:
    raise RuntimeError(
        f"Integer accumulator mAP50-95 {accumulator_map:.2f} is more than 1 below FP32 {fp32:.2f}"
    )

calibration = {
    name: {
        "smooth": smooth[name].tolist(),
        "clip": clips[name],
        "silu_input_step": silu_input_steps[name].tolist(),
        "silu_output_step": silu_output_steps[name].tolist(),
    }
    for name in clips
}
calibration.update(product_maxima)
attention_scales = [
    module.scale for module in integer_model.modules() if isinstance(module, Attention)
]
nonlinear_reference = IntegerNonlinear(calibration, attention_scales)
for branch in integer_model.model[-1].one2one_cv3:
    branch[-1].fixed_scores = True
with nonlinear_reference:
    table_map = mean_ap(integer_model)
assert nonlinear_reference.softmaxes == len(labels) * len(attention_scales)
assert (
    nonlinear_reference.sigmoids == len(labels)
    and nonlinear_reference.topks == len(labels) * 2
)
print(f"Integer nonlinear mAP50-95 {table_map:.2f} on 64 images")
if fp32 - table_map > 1:
    raise RuntimeError(
        f"Integer nonlinear mAP50-95 {table_map:.2f} is more than 1 below FP32 {fp32:.2f}"
    )
# The two attention residuals keep FP32; convolution residuals use scalar INT8 steps.
torch.manual_seed(7)
residual_samples = {}
with torch.no_grad(), IntegerNonlinear(calibration, attention_scales):
    for image in images.split(1):
        with ResidualAdds(samples=residual_samples) as additions:
            integer_model(image)
        assert additions.index == 20
residual_steps = {}
for key, values in residual_samples.items():
    clip = torch.quantile(torch.cat(values).abs().flatten(), 0.9999)
    residual_steps[key] = torch.where(clip == 0, 1.0, clip / 127)
del residual_samples
residual_reference = ResidualAdds(steps=residual_steps)
hook = integer_model.register_forward_pre_hook(
    lambda module, inputs: setattr(residual_reference, "index", 0)
)
with residual_reference, IntegerNonlinear(calibration, attention_scales):
    residual_map = mean_ap(integer_model)
# The reference of the baremetal check: the detections of this integer model
# on the image the importer letterboxes for the device.
image = Image.open(
    Path(import_module("models.models.yolo26n").__file__).parent
    / "images/bus_32bit.bmp"
).convert("RGB")
scale = min(640 / image.width, 640 / image.height)
width, height = int(image.width * scale + 0.5), int(image.height * scale + 0.5)
prepared = Image.new("RGB", (640, 640), (114, 114, 114))
prepared.paste(
    image.resize((width, height), Image.Resampling.BILINEAR),
    ((640 - width) // 2, (640 - height) // 2),
)
pixels = torch.from_numpy(np.asarray(prepared, dtype=np.float32) / np.float32(255.0))
with torch.no_grad(), residual_reference, IntegerNonlinear(
    calibration, attention_scales
):
    detections = integer_model(pixels.permute(2, 0, 1)[None].contiguous())[0]
detections.numpy().astype(np.float32).tofile(output / "reference.data")
hook.remove()
print(f"INT8 convolution residual mAP50-95 {residual_map:.2f} on 64 images")
if fp32 - residual_map > 1:
    raise RuntimeError(
        f"INT8 residual mAP50-95 {residual_map:.2f} is more than 1 below FP32 {fp32:.2f}"
    )
calibration["residual_steps"] = {
    f"{index}.{side}": step.item() for (index, side), step in residual_steps.items()
}
calibration["validation"] = {
    "images": 64,
    "fp32_map50_95": fp32,
    "int8_map50_95": quantized_map,
    "lookup_map50_95": lookup_map,
    "integer_attention_map50_95": integer_map,
    "integer_accumulator_map50_95": accumulator_map,
    "integer_nonlinear_map50_95": table_map,
    "integer_residual_map50_95": residual_map,
}
(output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
