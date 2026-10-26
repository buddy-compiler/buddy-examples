from datetime import datetime
from pathlib import Path
import ctypes
import os
import sys

import numpy as np
import torch
from torchvision import datasets, transforms

recipe = Path(__file__).resolve().parents[1]
root = recipe.parents[4]
sys.path[:0] = [str(root / "stack"), str(root / "bbdev/api")]
from models.models.lenet.model import LeNet
from utils.path import log_dir
from buddy_mlir.runtime import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    ranked_memref_to_numpy,
)

output = Path(
    log_dir(
        str(root),
        "x86",
        "model",
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        "numerics",
        "lenet",
    )
)
output.mkdir(parents=True)
os.chdir(output)
torch.set_num_threads(4)
model = LeNet().eval()
model.load_state_dict(
    torch.load(
        root / "stack/models/models/lenet/output/lenet/model.pth",
        weights_only=True,
    )
)
samples = datasets.MNIST(
    root / "stack/models/models/lenet/output/datasets",
    train=False,
    download=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
)
for name in ("libmlir_c_runner_utils.so", "libmlir_runner_utils.so"):
    ctypes.CDLL(
        str(root / "stack/thirdparty/buddy-mlir/llvm/build/lib" / name),
        mode=ctypes.RTLD_GLOBAL,
    )
library = ctypes.CDLL(str(recipe / "output/liblenet-x86-model.so"))
forward = library._mlir_ciface_forward
forward.restype = None
parameters = np.fromfile(recipe / "output/arg0.data", dtype=np.float32)
weights = np.fromfile(recipe / "output/arg1.data", dtype=np.int8)
parameter_descriptor = get_ranked_memref_descriptor(parameters)
weight_descriptor = get_ranked_memref_descriptor(weights)
free = ctypes.CDLL(None).free
free.argtypes = [ctypes.c_void_p]


def contraction(value, layer):
    scale = value.abs().amax() / 127
    scale = torch.where(scale == 0, 1.0, scale)
    activation = (value / scale).round().clamp(-127, 127).to(torch.int8)
    shape = [layer.weight.shape[0], *([1] * (layer.weight.ndim - 1))]
    weight_scale = (
        layer.weight.abs().reshape(layer.weight.shape[0], -1).amax(1).reshape(shape)
        / 127
    )
    weight = (layer.weight / weight_scale).round().clamp(-127, 127).to(torch.int8)
    if isinstance(layer, torch.nn.Conv2d):
        accumulator = torch.nn.functional.conv2d(
            activation.double(), weight.double()
        ).to(torch.int32)
        scale_shape = [1, layer.weight.shape[0], 1, 1]
    else:
        accumulator = (activation.double() @ weight.double().T).to(torch.int32)
        scale_shape = [1, layer.weight.shape[0]]
    return accumulator.float() * (
        scale * weight_scale.reshape(scale_shape)
    ) + layer.bias.reshape(scale_shape)


with torch.no_grad():
    for index in range(32):
        value = samples[index][0].unsqueeze(0)
        reference = torch.nn.functional.max_pool2d(
            torch.relu(contraction(value, model.conv1)), 2
        )
        reference = torch.nn.functional.max_pool2d(
            torch.relu(contraction(reference, model.conv2)), 2
        ).flatten(1)
        reference = torch.relu(contraction(reference, model.fc1))
        reference = torch.relu(contraction(reference, model.fc2))
        reference = contraction(reference, model.fc3)
        image = value.numpy()
        input_descriptor = get_ranked_memref_descriptor(image)
        result = make_nd_memref_descriptor(2, ctypes.c_float)()
        forward(
            ctypes.byref(result),
            ctypes.byref(parameter_descriptor),
            ctypes.byref(weight_descriptor),
            ctypes.byref(input_descriptor),
        )
        actual = ranked_memref_to_numpy(ctypes.pointer(result)).copy()
        free(result.allocated)
        np.testing.assert_array_equal(actual, reference.numpy())
print("PASS compiled LeNet matches independent INT8 reference on 32 MNIST images")
