from buddy.compiler.graph.transform.quantization.activation import calibrate_layers
from buddy.compiler.graph.transform.quantization.symmetric import quantize_symmetric
from examples.balls.gemmini.compiler.python.quant import apply
from ..permute.layout import reorder, pool_result


def prepare(model, source):
    import numpy as np
    import torch
    from PIL import Image

    rgb = np.asarray(
        Image.open(source / "images/8.bmp").convert("RGB"), dtype=np.float32
    )
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    inputs = torch.from_numpy((gray * 2.0 - 1.0)[None, None, :, :].astype(np.float32))
    return inputs, calibrate_layers(model, inputs)


def quantize(graph, params, names, output, name, calibration, rax_pack):
    # One INT8 scale for each complete weight tensor. Bias and nonlinear ops stay FP32.
    quantized = {
        key: quantize_symmetric(value.detach().numpy(), [])
        for key, value in zip(names, params)
        if value.ndim >= 2
    }
    apply(
        graph,
        params,
        names,
        output,
        name,
        calibration,
        quantized,
        rax_pack,
        reorder=reorder,
        pool_result=pool_result,
    )
