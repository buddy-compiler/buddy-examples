from pathlib import Path
import json

import torch
from torchvision import datasets, transforms
from models.models.lenet.model import LeNet

from stack.compiler.frontend.quant import quantize, quantize_bias

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
model_source = model_dir.parents[3] / "models/lenet"

# ===----------------------------------------------------------------------===
# Record the largest input magnitude of every convolution and Linear of the
# checkpoint over 1024 training images.
# ===----------------------------------------------------------------------===
model = LeNet().eval()
model.load_state_dict(
    torch.load(model_source / "output/lenet/model.pth", weights_only=True)
)
mnist = datasets.MNIST(
    model_source / "output/datasets",
    train=True,
    download=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
)
images = torch.utils.data.DataLoader(
    torch.utils.data.Subset(mnist, range(1024)), batch_size=64
)
maxima = {}
hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: maxima.update(
            {name: max(maxima.get(name, 0.0), inputs[0].abs().max().item())}
        )
    )
    for name, module in model.named_modules()
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
]
with torch.no_grad():
    for batch, _ in images:
        model(batch)
for hook in hooks:
    hook.remove()

# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on the test images: the scales are
# rejected when the accuracy drops more than 0.5 points below FP32.
# ===----------------------------------------------------------------------===
test = datasets.MNIST(
    model_source / "output/datasets",
    train=False,
    download=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
)
tests = torch.utils.data.DataLoader(test, batch_size=1000)


def accuracy(model):
    with torch.no_grad():
        correct = sum(
            (model(batch).argmax(1) == label).sum().item() for batch, label in tests
        )
    return correct / len(test) * 100


fp32 = accuracy(model)
quantized = LeNet().eval()
quantized.load_state_dict(model.state_dict())
for name, module in quantized.named_modules():
    if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        continue
    axes = list(range(1, module.weight.ndim))
    step = module.weight.detach().abs().amax(axes, keepdim=True) / 127
    module.weight.data = quantize(module.weight.detach(), 1 / step) * step
    scale = torch.tensor(maxima[name] / 127, dtype=torch.float32)
    bias_step = step.reshape(-1) * scale
    module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, inputs, scale=scale: quantize(inputs[0], 1 / scale) * scale
    )
int8 = accuracy(quantized)
print(f"FP32 accuracy {fp32:.2f}%, INT8 accuracy {int8:.2f}%")
if fp32 - int8 > 0.5:
    raise RuntimeError(
        f"INT8 accuracy {int8:.2f}% is more than 0.5 below FP32 {fp32:.2f}%"
    )
(output / "calibration.json").write_text(json.dumps(maxima, indent=2) + "\n")
