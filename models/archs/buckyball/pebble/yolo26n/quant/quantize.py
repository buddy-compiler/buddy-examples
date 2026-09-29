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


def prepare(model, calibration_images, size, model_path, output_dir):
    import hashlib
    import json
    import sys
    from importlib.metadata import version
    import numpy as np
    import torch
    from .calibration import load_calibration_images
    calibration_tensor = torch.from_numpy(
        load_calibration_images(calibration_images, size)
    )
    calibration = calibrate_layers(model, calibration_tensor)
    input_tensor = calibration_tensor[:1].clone()
    (output_dir / "calibration-manifest.json").write_text(json.dumps(
        {
            "images": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in calibration_images
            ],
            "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(calibration_tensor.numpy().tobytes()).hexdigest(),
            "calibration_batch": len(calibration_images),
            "import_batch": 1,
            "input_size": size,
            "preprocessing": "RGB/255, DIP corner interpolation, letterbox 114/255",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": version("pillow"),
            "ultralytics": version("ultralytics"),
            "python": sys.version.split()[0],
        }, indent=2,
    ))
    return input_tensor, calibration
