from buddy.compiler.graph.transform.quantization.activation import calibrate_layers
from buddy.compiler.graph.transform.quantization.symmetric import quantize_symmetric
from stack.compiler.quant.importer import quantize_model_graph
from ..permute.layout import reorder


def quantize(graph, params, names, output, name, calibration, rax_pack):
    # This design uses symmetric INT8 weights with one scale per output channel.
    weight_scales = {}
    for parameter_name, parameter in zip(names, params):
        if parameter_name.endswith(".weight") and parameter.ndim >= 2:
            _, scales = quantize_symmetric(parameter.detach().numpy(), [0])
            weight_scales[parameter_name] = scales
    quantize_model_graph(graph, params, names, output, name, calibration, rax_pack,
                         reorder=reorder, weight_scales=weight_scales)


def prepare(model, source):
    import numpy as np
    import torch
    from PIL import Image
    rgb = np.asarray(Image.open(source / "images/8.bmp").convert("RGB"), dtype=np.float32)
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    inputs = torch.from_numpy((gray * 2.0 - 1.0)[None, None, :, :].astype(np.float32))
    return inputs, calibrate_layers(model, inputs)
