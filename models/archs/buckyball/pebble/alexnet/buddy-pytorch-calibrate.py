from pathlib import Path
import io
import json

import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import alexnet

from stack.compiler.frontend.quant import quantize, quantize_bias

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
model_source = model_dir.parents[3] / "models/alexnet"
imagenet = model_dir.parents[3] / "datasets/imagenet-256/data"
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def load(pattern, every):
    """The first image of every `every`-th row group: the files are sorted by
    class, so this spreads the images over the classes."""
    images, labels = [], []
    for path in sorted(imagenet.glob(pattern)):
        data = pq.ParquetFile(path)
        for group in range(0, data.num_row_groups, every):
            row = data.read_row_group(group).slice(0, 1).to_pylist()[0]
            image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            images.append(preprocess(image))
            labels.append(row["label"])
    return torch.stack(images), torch.tensor(labels)


# ===----------------------------------------------------------------------===
# Record the largest input magnitude of the last Linear, classifier.6, of the checkpoint over
# about 1024 training images.
# ===----------------------------------------------------------------------===
model = alexnet(weights=None).eval()
model.load_state_dict(
    torch.load(model_source / "output/alexnet/model.pth", weights_only=True)
)
images, _ = load("train-*.parquet", 12)
maxima = {}
hook = model.classifier[6].register_forward_pre_hook(
    lambda module, inputs: maxima.update(
        {
            "classifier.6": max(
                maxima.get("classifier.6", 0.0), inputs[0].abs().max().item()
            )
        }
    )
)
with torch.no_grad():
    for batch in images.split(64):
        model(batch)
hook.remove()

# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on 500 validation images: the scale
# is rejected when the accuracy drops more than 0.5 points below FP32.
# ===----------------------------------------------------------------------===
tests, labels = load("val-*.parquet", 1)


def accuracy(model):
    with torch.no_grad():
        correct = sum(
            (model(batch).argmax(1) == label).sum().item()
            for batch, label in zip(tests.split(100), labels.split(100))
        )
    return correct / len(labels) * 100


fp32 = accuracy(model)
quantized = alexnet(weights=None).eval()
quantized.load_state_dict(model.state_dict())
module = quantized.classifier[6]
step = module.weight.detach().abs().amax(1, keepdim=True) / 127
module.weight.data = quantize(module.weight.detach(), 1 / step) * step
scale = torch.tensor(maxima["classifier.6"] / 127, dtype=torch.float32)
bias_step = step.reshape(-1) * scale
module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
module.register_forward_pre_hook(
    lambda module, inputs: quantize(inputs[0], 1 / scale) * scale
)
int8 = accuracy(quantized)
print(f"FP32 accuracy {fp32:.2f}%, INT8 accuracy {int8:.2f}% on {len(labels)} images")
if fp32 - int8 > 0.5:
    raise RuntimeError(
        f"INT8 accuracy {int8:.2f}% is more than 0.5 below FP32 {fp32:.2f}%"
    )
(output / "calibration.json").write_text(json.dumps(maxima, indent=2) + "\n")
