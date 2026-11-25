from importlib import import_module
from pathlib import Path
import json

import torch
from torch.overrides import TorchFunctionMode
from compiler.frontend.quant import quantize, quantize_bias

output = Path(__file__).resolve().parent / "output"
calibration = json.loads((output / "calibration.json").read_text())
coco = import_module("models.models.yolo26n.data")
model = import_module("models.models.yolo26n.model").load()
pixels, _ = coco.load(coco.paths[64:65])


class QuantizedAttention(TorchFunctionMode):
    def __init__(self):
        self.product = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__name__", None) == "matmul":
            scales = [
                calibration[f"matmul.{self.product}.{side}"] / 127 for side in (0, 1)
            ]
            self.product += 1
            x, y = (
                quantize(v, 1 / scale).double()
                for v, scale in zip(args, scales, strict=True)
            )
            return (x @ y).float() * (scales[0] * scales[1])
        if func in (torch.topk, torch.Tensor.topk):
            x, k = args[:2]
            dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
            indices = torch.argsort(x, dim=dim, descending=True, stable=True).narrow(
                dim, 0, k
            )
            return torch.return_types.topk((x.gather(dim, indices), indices))
        return func(*args, **kwargs)


with torch.no_grad():
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.SiLU):
            module.inplace = False
            module.register_forward_hook(
                lambda silu, args, out: args[0]
                / (1 + torch.exp(-args[0].double()).float())
            )
        if isinstance(module, torch.nn.BatchNorm2d):
            module.register_forward_hook(
                lambda bn, args, out: (args[0] - bn.running_mean[None, :, None, None])
                * (1 / torch.sqrt(bn.running_var + bn.eps))[None, :, None, None]
                * bn.weight[None, :, None, None]
                + bn.bias[None, :, None, None]
            )
        if not isinstance(module, torch.nn.Conv2d):
            continue
        entry = calibration[name]
        smooth = torch.tensor(entry["smooth"])
        shape = (-1, 1, 1, 1) if module.groups > 1 else (1, -1, 1, 1)
        weight = module.weight * smooth.reshape(shape)
        maximum = weight.abs().amax((1, 2, 3), keepdim=True)
        step = torch.where(maximum == 0, 1.0, maximum / 127)
        scale = entry["maximum"] / 127 if entry["maximum"] else 1.0
        acc_step = (step * scale).reshape(-1)
        module.weight.data = quantize(weight, 1 / step).double()
        if module.bias is not None:
            module.bias.data = quantize_bias(module.bias, acc_step).double()
        factor = (1 / (scale * smooth)).reshape(1, -1, 1, 1)
        module.register_forward_pre_hook(
            lambda conv, args, factor=factor: quantize(args[0], factor).double()
        )
        module.register_forward_hook(
            lambda conv, args, out, step=acc_step: out.float()
            * step[None, :, None, None]
        )
    with QuantizedAttention():
        reference = model(pixels)
pixels.contiguous().numpy().tofile(output / "inputs.data")
reference.contiguous().numpy().tofile(output / "reference.data")
