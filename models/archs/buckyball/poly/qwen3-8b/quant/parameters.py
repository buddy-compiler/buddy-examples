import torch
from buddy.compiler.graph.transform.quantization.mxfp8 import quantize
from examples.balls.mxmm.compiler.python.layout import plan, pack, bank_bytes
from stack.compiler.quant.rax import QuantTensor, RaxQuantPackage, write_rax


def write_parameters(values, output, name, packer, target, rows_hint):
    tensors, weights, floats, matrices = [], [], [], {}
    weight_offset = float_offset = 0
    capacity = bank_bytes(packer.parent.parent, target)
    for parameter_name, parameter in sorted(values.items()):
        array = parameter.detach()
        if array.dtype != torch.float32:
            raise ValueError(f"parameter {parameter_name} requires FP32 storage")
        shape = list(array.shape)
        if array.ndim == 2:
            layout = plan(max(16, rows_hint), *shape, capacity, mxfp8=True)
            packed = pack(*quantize(array), layout)
            raw = packed.cpu().numpy().tobytes()
            tensors.append(
                QuantTensor(
                    parameter_name,
                    shape,
                    [packed.numel()],
                    "mxfp8",
                    [1],
                    weight_offset,
                    len(raw),
                    0,
                    0,
                    layout,
                )
            )
            matrices[parameter_name] = {
                "offset": weight_offset,
                "bytes": len(raw),
                "layout": layout,
            }
            weights.append(raw)
            weight_offset += len(raw)
        else:
            raw = array.cpu().contiguous().numpy().tobytes()
            tensors.append(
                QuantTensor(
                    parameter_name,
                    shape,
                    shape,
                    "f32",
                    [],
                    float_offset,
                    len(raw),
                    0,
                    0,
                )
            )
            floats.append(raw)
            float_offset += len(raw)
    package = RaxQuantPackage(tensors, b"".join(weights), b"".join(floats), b"", {})
    output.mkdir(parents=True, exist_ok=True)
    write_rax(package, output / f"{name}.rax", packer, name)
    (output / "weights.bin").write_bytes(package.weights)
    (output / "params.f32").write_bytes(package.params_f32)
    return {
        "f32_elements": float_offset // 4,
        "weight_bytes": weight_offset,
        "bank_bytes": capacity,
        "matrices": matrices,
    }
