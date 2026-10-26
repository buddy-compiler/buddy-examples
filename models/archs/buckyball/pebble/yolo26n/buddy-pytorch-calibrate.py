from pathlib import Path
import json

import torch
from PIL import Image
from torchvision.ops import box_iou
from torchvision.transforms.functional import pil_to_tensor, resize
from ultralytics import YOLO

from stack.compiler.frontend.quant import quantize, quantize_bias
from frontend.quant.graph import FP32

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
model_source = model_dir.parents[3] / "models/yolo26n"
coco = model_dir.parents[3] / "datasets/coco128/images/train2017"

# ===----------------------------------------------------------------------===
# Load the checkpoint with every batch normalization folded into the
# convolution before it, and the Detect head returning its 300 best boxes.
# ===----------------------------------------------------------------------===
model = YOLO(model_source / "output/yolo26n/yolo26n.pt").model.eval()
model.fuse()
head = model.model[-1]
head.end2end = True
head.export = True
head.xyxy = True


def letterbox(path):
    """The runner scales the image into 640x640 and pads it with gray."""
    image = pil_to_tensor(Image.open(path).convert("RGB")).float() / 255
    height, width = image.shape[1:]
    ratio = min(640 / height, 640 / width)
    rows, columns = round(height * ratio), round(width * ratio)
    padded = torch.full((3, 640, 640), 114 / 255)
    top, left = (640 - rows) // 2, (640 - columns) // 2
    padded[:, top : top + rows, left : left + columns] = resize(
        image, [rows, columns], antialias=False
    )
    return padded


images = [letterbox(path)[None] for path in sorted(coco.glob("*.jpg"))]

# ===----------------------------------------------------------------------===
# Sample the inputs of every convolution over the first 96 images and clip
# each at the magnitude whose INT8 encoding has the least squared error: a
# few outliers would otherwise take most of the 256 codes.
# ===----------------------------------------------------------------------===
torch.manual_seed(0)
samples = {}
hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: samples.setdefault(name, []).append(
            inputs[0].flatten()[torch.randperm(inputs[0].numel())[:50000]]
        )
    )
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Conv2d)
]
with torch.no_grad():
    for image in images[:96]:
        model(image)
for hook in hooks:
    hook.remove()


def error(values, clip):
    step = clip / 127
    return ((quantize(values, 1 / step) * step - values) ** 2).mean().item()


maxima = {}
for name, values in samples.items():
    values = torch.cat(values)
    peak = values.abs().max().item()
    maxima[name] = min(
        (peak * percent / 100 for percent in range(5, 101)),
        key=lambda clip: error(values, clip),
    )


# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on the other 32 images, the
# convolutions of FP32 left in FP32: every FP32 detection scoring 0.25 or more
# should be found again by INT8, same class and IoU of at least 0.5. The
# scales are rejected when less than 80% of them are.
# ===----------------------------------------------------------------------===
def detections(model):
    with torch.no_grad():
        return [
            boxes[boxes[:, 4] >= 0.25]
            for boxes in (model(image)[0] for image in images[96:])
        ]


fp32 = detections(model)
for name, module in model.named_modules():
    if not isinstance(module, torch.nn.Conv2d) or name in FP32:
        continue
    scale = torch.tensor(maxima[name] / 127)
    # A channel whose folded weights almost vanish takes a coarser step, so
    # that its bias code stays within 2^30.
    maximum = module.weight.detach().abs().amax([1, 2, 3], keepdim=True)
    floor = module.bias.detach().abs().reshape(maximum.shape) / (scale * 2**30)
    step = torch.maximum(maximum / 127, floor)
    step = torch.where(step == 0, 1.0, step)
    module.weight.data = quantize(module.weight.detach(), 1 / step) * step
    bias_step = step.reshape(-1) * scale
    module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, inputs, scale=scale: quantize(inputs[0], 1 / scale) * scale
    )
int8 = detections(model)
found = sum(
    (
        (box_iou(reference[:, :4], boxes[:, :4]) >= 0.5)
        & (reference[:, 5:6] == boxes[:, 5][None])
    )
    .any(1)
    .sum()
    .item()
    for reference, boxes in zip(fp32, int8)
)
total = sum(len(reference) for reference in fp32)
print(
    f"FP32 detections {total}, found by INT8 {found} ({found / total * 100:.2f}%), "
    f"INT8 detections {sum(len(boxes) for boxes in int8)}"
)
if found < 0.8 * total:
    raise RuntimeError(f"INT8 finds only {found} of {total} FP32 detections")
(output / "calibration.json").write_text(json.dumps(maxima, indent=2) + "\n")
