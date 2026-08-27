from __future__ import annotations

from pathlib import Path
import os
import struct

import numpy as np
import torch

from framework.quant.core.quantize import quantize_symmetric
from framework.quant.core.rax import QuantTensor, RaxQuantPackage, write_rax
from framework.quant.core.rax_metadata import (
    bind_rax_quant_metadata,
    load_rax_weight_bindings,
)
from buddy.compiler.graph.operation import AddMMOp, MatmulOp, QuantizedAddMMOp, QuantizedMatmulOp
from buddy.compiler.graph.type import TensorDType


def _rax_pack() -> Path:
    root = os.environ.get("BUDDY_MLIR_BUILD_DIR")
    if root is None:
        raise RuntimeError("BUDDY_MLIR_BUILD_DIR is required")
    build = Path(root)
    return build.parent / "cores" / build.name / "bin" / "rax-pack"


def _replace_linears(graph) -> None:
    for index, node in enumerate(list(graph._body)):
        if isinstance(node, AddMMOp):
            replacement = QuantizedAddMMOp()
            replacement._arguments = [node.args[0], node.args[1], node.args[2], "weight_scale"]
        elif isinstance(node, MatmulOp):
            replacement = QuantizedMatmulOp()
            replacement._arguments = [node.args[0], node.args[1], "weight_scale"]
        else:
            continue
        weight = graph.node_table[node.args[2] if isinstance(node, AddMMOp) else node.args[1]]
        while weight.tensor_meta["dtype"] != TensorDType.Int8:
            if len(weight._parents) != 1:
                break
            weight = graph.node_table[weight._parents[0]]
        if weight.tensor_meta["dtype"] != TensorDType.Int8:
            continue
        replacement._name = node.name
        replacement._parents = list(node._parents)
        replacement._children = list(node._children)
        replacement._tensor_meta = node._tensor_meta.copy()
        replacement._dw_addr = node._dw_addr
        replacement._dw_bytes = node._dw_bytes
        replacement._per_channel = node._per_channel
        replacement.trace_meta = node.trace_meta
        graph._body[index] = replacement
        graph.node_table[node.name] = replacement
        for group in graph.op_groups.values():
            for group_index, group_node in enumerate(group):
                if group_node is node:
                    group[group_index] = replacement


def _propagate_int8(graph) -> None:
    changed = True
    while changed:
        changed = False
        for node in graph._body:
            if len(node._parents) != 1:
                continue
            parent = graph.node_table[node._parents[0]]
            if parent.tensor_meta["dtype"] != TensorDType.Int8:
                continue
            if node.tensor_meta["dtype"] == TensorDType.Int8:
                continue
            node.tensor_meta["dtype"] = TensorDType.Int8
            changed = True


def quantize_model_graph(graph, params, names: list[str], output_dir: Path,
                         model_name: str, per_channel: bool) -> None:
    if len(graph.params) != len(params) or len(names) != len(params):
        raise ValueError("parameter metadata does not match imported graph")
    tensors = []
    weights, fp_params, scales = [], [], []
    weight_off = param_off = scale_off = 0
    parameter_names = {}
    for node, param, name in zip(graph.params, params, names):
        array = param.detach().cpu().numpy().astype(np.float32)
        parameter_names[node.name] = name
        if name.endswith(".weight") and array.ndim >= 2:
            axes = [0] if per_channel else []
            q, dw = quantize_symmetric(array, axes)
            node.tensor_meta["dtype"] = TensorDType.Int8
            params[graph.params.index(node)] = torch.from_numpy(q.copy())
            raw_scales = np.asarray(dw, dtype=np.float32).reshape(-1)
            padded = (np.pad(raw_scales, (0, (-len(raw_scales)) % 16),
                             constant_values=1.0)
                      if per_channel else raw_scales)
            scale_bytes = padded.tobytes()
            tensors.append(QuantTensor(name, list(array.shape), list(array.shape), "i8",
                                       axes, weight_off, q.nbytes, scale_off, len(scale_bytes)))
            weights.append(q.tobytes())
            scales.append(scale_bytes)
            weight_off += q.nbytes
            scale_off += len(scale_bytes)
        else:
            raw = array.tobytes()
            tensors.append(QuantTensor(name, list(array.shape), list(array.shape), "f32",
                                       [], param_off, len(raw), 0, 0))
            fp_params.append(raw)
            param_off += len(raw)
    package = RaxQuantPackage(tensors, b"".join(weights), b"".join(fp_params), b"".join(scales), {})
    output_dir.mkdir(parents=True, exist_ok=True)
    rax = output_dir / f"{model_name}.rax"
    write_rax(package, rax, _rax_pack(), model_name)
    bindings = load_rax_weight_bindings(rax.with_name(f"{rax.stem}.payload") / "quant-index.json")
    bind_rax_quant_metadata(graph, parameter_names, bindings)
    _propagate_int8(graph)
    _replace_linears(graph)
    (output_dir / "weights.i8").write_bytes(package.weights_i8)
    (output_dir / "params.f32").write_bytes(package.params_f32)
    (output_dir / "scales.bin").write_bytes(package.scales_f32)
