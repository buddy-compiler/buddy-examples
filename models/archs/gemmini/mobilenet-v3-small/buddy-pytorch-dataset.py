# ===- buddy-pytorch-dataset.py --------------------------------------------
#
# MobileNetV3-Small on Gemmini: the dataset of the accuracy runs (dataset.cpp), the
# 10000 ImageNet validation images (every fifth), in <output>/dataset/:
#   pixels.data     each image resized to 224x224, as 224x224x3 uint8 (HWC);
#                   dataset.cpp normalizes it as the check images are
#   reference.data  the 1000 logits of the check reference of
#                   buddy-pytorch-calibrate.py (the model computed the way
#                   the compiled graph computes it) on each normalized image
#   labels.data     each label, int64
# Images and logits are computed by 32 processes of 2 threads: many threads
# on one process are slower on a loaded host.
#
#   buddy-pytorch-dataset.py <output>
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import io
import json
import multiprocessing
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms

from compiler.frontend.quant import quantize, quantize_bias

model_dir = Path(__file__).resolve().parent
output = Path(sys.argv[1]) / "dataset"
output.mkdir(parents=True, exist_ok=True)
imagenet = Path(import_module("models.models.mobilenet-v3-small.data").imagenet)
files = sorted(imagenet.glob("val-*.parquet"))
rows = [
    (path, group, row)
    for path in files
    for group in range(pq.ParquetFile(path).num_row_groups)
    for row in range(pq.ParquetFile(path).metadata.row_group(group).num_rows)
]
# Every fifth image, 10000 over all the classes (the files are sorted by
# class): the accuracy run of the design stays within two hours.
rows = rows[::5]
samples = len(rows)
crop = transforms.Resize((224, 224))
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
deviation = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def channels(module):
    if isinstance(module, torch.nn.Conv2d) and module.groups > 1:
        return [-1, 1, 1, 1]
    return [1, -1] + [1] * (module.weight.ndim - 2)


def inputs(x):
    return [1, -1, 1, 1] if x.ndim == 4 else [1] * (x.ndim - 1) + [-1]


def average(values, dims):
    total = values
    count = 1
    for d in dims:
        moved = total.movedim(d, 0)
        accumulated = torch.zeros_like(moved[0])
        for index in range(moved.shape[0]):
            accumulated = accumulated + moved[index]
        total = accumulated.unsqueeze(d)
        count *= values.shape[d]
    return total * torch.tensor(1.0 / count, dtype=torch.float32)


def reference():
    """The check reference of buddy-pytorch-calibrate.py, from its
    calibration."""
    calibration = json.loads((model_dir / "output/calibration.json").read_text())
    model = import_module("models.models.mobilenet-v3-small.model").load()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.register_forward_hook(
                lambda module, args, output: (
                    (args[0] - module.running_mean.reshape(1, -1, 1, 1))
                    * (1 / torch.sqrt(module.running_var + module.eps)).reshape(
                        1, -1, 1, 1
                    )
                )
                * module.weight.reshape(1, -1, 1, 1)
                + module.bias.reshape(1, -1, 1, 1)
            )
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            module.register_forward_hook(
                lambda module, args, output: average(args[0], [2, 3])
            )
        if isinstance(module, torch.nn.Hardswish):
            # The hook reads the input after the module ran.
            module.inplace = False
            module.register_forward_hook(
                lambda module, args, output: (
                    args[0] * torch.clamp(torch.clamp(args[0] + 3, min=0), max=6)
                )
                / 6
            )
        if isinstance(module, torch.nn.Hardsigmoid):
            module.inplace = False
            module.register_forward_hook(
                lambda module, args, output: torch.clamp(
                    torch.clamp(args[0] + 3, min=0), max=6
                )
                / 6
            )
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        smooth = torch.tensor(calibration[name]["smooth"])
        maximum_input = calibration[name]["maximum"]
        weight = module.weight.detach() * smooth.reshape(channels(module))
        axes = list(range(1, weight.ndim))
        maximum = weight.abs().amax(axes, keepdim=True)
        step = torch.where(maximum == 0, 1.0, maximum / 127)
        scale = maximum_input / 127 if maximum_input else 1.0
        accumulator_step = (step * scale).reshape(-1)
        if module.bias is not None:
            module.bias.data = quantize_bias(
                module.bias.detach(), accumulator_step
            ).double()
        module.weight.data = quantize(weight, 1 / step).double()
        module.register_forward_pre_hook(
            lambda module, args, smooth=smooth, scale=scale: quantize(
                args[0], 1 / (scale * smooth.reshape(inputs(args[0])))
            ).double()
        )
        module.register_forward_hook(
            lambda module, args, output, accumulator_step=accumulator_step: output.float()
            * accumulator_step.reshape([1, -1, 1, 1] if output.ndim == 4 else [1, -1])
        )
    return model


def work(part):
    """Images first + part, first + part + parts, ...: their pixels, logits
    and labels written in place."""
    torch.set_num_threads(2)
    model = reference()
    pixels = np.memmap(
        output / "pixels.data", np.uint8, "r+", shape=(samples, 224, 224, 3)
    )
    logits = np.memmap(
        output / "reference.data", np.float32, "r+", shape=(samples, 1000)
    )
    labels = np.memmap(output / "labels.data", np.int64, "r+", shape=(samples,))
    tables = {}
    for index in range(part, samples, parts):
        path, group, row = rows[index]
        if (path, group) not in tables:
            tables = {(path, group): pq.ParquetFile(path).read_row_group(group)}
        record = tables[path, group].slice(row, 1).to_pylist()[0]
        image = crop(Image.open(io.BytesIO(record["image"]["bytes"])).convert("RGB"))
        pixels[index] = np.asarray(image, dtype=np.uint8)
        labels[index] = record["label"]
        # The normalization of dataset.cpp, in its order.
        values = pixels[index].astype(np.float32) / np.float32(255)
        values = (values - mean) / deviation
        with torch.no_grad():
            logits[index] = model(
                torch.from_numpy(values.transpose(2, 0, 1).copy())[None]
            )[0].numpy()
    for array in (pixels, logits, labels):
        array.flush()


parts = 32
for name, dtype, shape in (
    ("pixels.data", np.uint8, (samples, 224, 224, 3)),
    ("reference.data", np.float32, (samples, 1000)),
    ("labels.data", np.int64, (samples,)),
):
    np.memmap(output / name, dtype, "w+", shape=shape).flush()
# Forked: the workers take the rows and paths above as they are.
with multiprocessing.get_context("fork").Pool(parts) as pool:
    pool.map(work, range(parts))
print(f"{samples} ImageNet validation images in {output}")
