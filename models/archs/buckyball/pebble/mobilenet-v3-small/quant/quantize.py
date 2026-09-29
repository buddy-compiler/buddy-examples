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
    pixels = np.asarray(Image.open(source / "images/dog-32bit_224x224.bmp").convert("RGB"), dtype=np.float32)
    if pixels.shape != (224, 224, 3):
        raise ValueError(f"Expected a 224x224 RGB calibration image, got {pixels.shape}")
    inputs = torch.from_numpy((pixels / np.float32(255.0)).copy()).permute(2, 0, 1)[None]
    return inputs, calibrate_layers(model, inputs)
