import torch
from torch.overrides import TorchFunctionMode
from compiler.frontend.quant import quantize


class ResidualAdds(TorchFunctionMode):
    def __init__(self, samples=None, steps=None):
        super().__init__()
        self.samples = samples
        self.steps = steps
        self.index = 0

    def __torch_function__(self, function, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if (
            function in (torch.add, torch.Tensor.add, torch.Tensor.__add__)
            and len(args) >= 2
            and all(isinstance(x, torch.Tensor) for x in args[:2])
            and args[0].ndim == 4
            and args[0].shape == args[1].shape
            and args[0].dtype == torch.float32
        ):
            index = self.index
            self.index += 1
            if self.steps is not None and index in (7, 17):
                return function(*args, **kwargs)
            a, b = args[:2]
            assert kwargs.get("alpha", 1) == 1
            if self.steps is None:
                output = function(*args, **kwargs)
                for side, value in enumerate((a, b, output)):
                    rows = value.permute(0, 2, 3, 1).reshape(-1, value.shape[1])
                    selected = torch.randperm(len(rows))[
                        : max(1, 200000 // value.shape[1])
                    ]
                    self.samples.setdefault((index, side), []).append(
                        rows[selected].clone()
                    )
                return output
            sa, sb, so = [
                self.steps[index, side].reshape(1, -1, 1, 1) for side in range(3)
            ]
            qa, qb = quantize(a, 1 / sa).float(), quantize(b, 1 / sb).float()
            return torch.round(qa * (sa / so) + qb * (sb / so)).clamp(-128, 127) * so
        return function(*args, **kwargs)
