from importlib import import_module
from pathlib import Path
import json

import torch
from PIL import Image
from compiler.frontend.quant import quantize, quantize_bias

output = Path(__file__).resolve().parent / "output"
alexnet = import_module("models.models.alexnet.model")
data = import_module("models.models.alexnet.data")
reference = alexnet.load()
calibration = json.loads((output / "calibration.json").read_text())
with Image.open(Path(data.__file__).parent / "images/dog.bmp") as image:
    pixels = data.preprocess(image.convert("RGB")).unsqueeze(0)
with torch.no_grad():
    for name, module in reference.named_modules():
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        entry = calibration[name]
        scale = entry["maximum"] / 127 if entry["maximum"] else 1.0
        smooth = torch.tensor(entry["smooth"])
        dims = (1, -1, 1, 1) if module.weight.ndim == 4 else (1, -1)
        weight = module.weight * smooth.reshape(dims)
        maximum = weight.abs().amax(tuple(range(1, weight.ndim)), keepdim=True)
        step = torch.where(maximum == 0, 1.0, maximum / 127)
        acc_step = (step * scale).reshape(-1)
        module.weight.data = quantize(weight, 1 / step).double()
        module.bias.data = quantize_bias(module.bias, acc_step).double()
        factor = 1 / (scale * smooth.reshape(dims))
        module.register_forward_pre_hook(
            lambda layer, args, factor=factor: quantize(args[0], factor).double()
        )
        out_dims = (1, -1, 1, 1) if weight.ndim == 4 else (1, -1)
        module.register_forward_hook(
            lambda layer, args, value, step=acc_step.reshape(out_dims): value.float()
            * step
        )
    logits = reference(pixels)
pixels.contiguous().numpy().tofile(output / "inputs.data")
logits.contiguous().numpy().tofile(output / "reference.data")
