import torch
from torch.overrides import TorchFunctionMode
from compiler.frontend.quant import quantize
from models.archs.buckyball.pebble.fixed_reference import round_div


class IntegerNonlinear(TorchFunctionMode):
    def __init__(self, calibration, attention_scales):
        super().__init__()
        self.steps = {
            key: torch.tensor(value / 127, dtype=torch.float32)
            for key, value in calibration.items()
            if key.startswith("matmul.")
        }
        self.products = len(self.steps) // 2
        self.scales = attention_scales
        assert len(self.scales) * 2 == self.products
        self.product = 0
        self.softmaxes = 0
        self.sigmoids = 0
        self.topks = 0
        self.exponent = torch.round(
            torch.exp(-torch.arange(1024, dtype=torch.float64) / 64) * 2**30
        ).long()
        indices = torch.arange(1024)
        signed = torch.where(indices < 512, indices, indices - 1024).double()
        self.sigmoid = torch.round(torch.sigmoid(signed / 64) * 2**24).long()

    def __torch_function__(self, function, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if function in (torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__):
            index = self.product % self.products
            self.product += 1
            steps = [self.steps[f"matmul.{index}.{side}"] for side in (0, 1)]
            codes = [
                quantize(value, 1 / step).float()
                for value, step in zip(args[:2], steps, strict=True)
            ]
            assert codes[0].shape[-1] * 128 * 128 < 2**24
            raw = codes[0] @ codes[1]
            if index % 2 == 0:
                self.pending = (raw.long(), steps[0] * steps[1], index // 2)
            return raw * (steps[0] * steps[1])
        if function in (torch.softmax, torch.Tensor.softmax):
            self.softmaxes += 1
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            assert dim == -1
            raw, scale, head = self.pending
            mantissa, exponent = torch.frexp((scale * self.scales[head]).double() * 64)
            multiplier = torch.round(mantissa * 2**30).long()
            shift = 30 - exponent.long()
            assert 0 <= shift.item() <= 61
            code = round_div(raw * multiplier, 1 << shift).clamp(
                -(1 << 31), (1 << 31) - 1
            )
            distance = (code.max(-1, keepdim=True).values - code).clamp(0, 1023)
            values = self.exponent[distance]
            total = values.sum(-1, keepdim=True)
            step = self.steps[f"matmul.{head * 2 + 1}.1"]
            coefficient = int(torch.round((1 / step).double() * 65536).item())
            reciprocal = torch.div(coefficient << 32, total, rounding_mode="floor")
            result = round_div(values * reciprocal, 1 << 48).clamp(0, 127)
            return result.float() * step
        if function in (torch.sigmoid, torch.Tensor.sigmoid):
            self.sigmoids += 1
            code = torch.round(args[0] * 64).clamp(-512, 511).long()
            return self.sigmoid[code & 1023].float() * (2.0**-24)
        if function in (torch.topk, torch.Tensor.topk):
            self.topks += 1
            x, k = args[:2]
            dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
            assert kwargs.get("largest", True) and kwargs.get("sorted", True)
            indices = torch.argsort(x, dim=dim, descending=True, stable=True).narrow(
                dim, 0, k
            )
            return torch.return_types.topk((x.gather(dim, indices), indices))
        return function(*args, **kwargs)
