from __future__ import annotations

from pathlib import Path
import os
import struct

import numpy as np
import torch

from framework.quant.core.quantize import quantize_symmetric
from framework.quant.core.rax import QuantTensor, RaxQuantPackage, write_rax
from buddy.compiler.graph.operation import (
    AddOp,
    AddMMOp,
    ClampMaxOp,
    ClampMinOp,
    Conv2dOp,
    DivOp,
    MatmulOp,
    MaxPool2dOp,
    MeanOp,
    MegaKernelOp,
    MegaConv2dOp,
    MegaConv2dDepthwiseOp,
    MegaGlobalAvgPoolOp,
    MegaInt8AddOp,
    MegaInt8MulOp,
    MegaMatmulOp,
    MegaMaxPool2dOp,
    HardswishOp,
    MulOp,
    OutputOp,
    PermuteOp,
    ReluOp,
    ReshapeOp,
    TOp,
    ViewOp,
)
from buddy.compiler.graph.type import TensorDType


def fold_batch_norms(model: torch.nn.Module) -> None:
    if model.training:
        raise ValueError("BatchNorm folding requires an eval-mode model")
    for parent in model.modules():
        names = list(parent._modules)
        for index in range(len(names) - 1):
            conv_name, bn_name = names[index : index + 2]
            conv = parent._modules[conv_name]
            bn = parent._modules[bn_name]
            if not isinstance(conv, torch.nn.Conv2d) or not isinstance(
                bn, torch.nn.BatchNorm2d
            ):
                continue
            parent._modules[conv_name] = torch.nn.utils.fusion.fuse_conv_bn_eval(
                conv, bn
            )
            parent._modules[bn_name] = torch.nn.Identity()


def _rax_pack() -> Path:
    root = os.environ.get("BUDDY_MLIR_BUILD_DIR")
    if root is None:
        raise RuntimeError("BUDDY_MLIR_BUILD_DIR is required")
    build = Path(root)
    return build.parent / "cores" / build.name / "bin" / "rax-pack"


def _form_mega_kernels(graph, parameter_names, arrays, weight_scales, calibration):
    param_nodes = list(graph.params)
    input_node_list = list(graph.inputs)
    original_body = list(graph._body)
    calibration_index = {}
    plans = {}
    renamed = {}
    removed = set()

    for index, node in enumerate(original_body):
        if not isinstance(node, (Conv2dOp, AddMMOp, MatmulOp)):
            continue
        if isinstance(node, Conv2dOp):
            if len(node.args) != 9:
                raise ValueError(f"unexpected Conv2d form for {node.name}")
            if node.args[5] != [1, 1] or node.args[6]:
                raise ValueError(f"unsupported Mega Conv2D form for {node.name}")
            if node.args[3][0] != node.args[3][1]:
                raise ValueError(f"asymmetric Mega Conv2D stride for {node.name}")
            padding = node.args[4]
            if len(padding) == 1:
                padding = [padding[0], padding[0]]
            if len(padding) != 2 or padding[0] != padding[1]:
                raise ValueError(f"asymmetric Mega Conv2D padding for {node.name}")
            activation_name, weight_arg, bias_arg = node.args[:3]
            weight_node = graph.node_table[str(weight_arg)]
            weight_name = parameter_names.get(weight_node.name)
            if weight_name is None:
                raise ValueError(f"Mega Conv2D weight is not a parameter for {node.name}")
            weight_shape = list(arrays[weight_name].shape)
            input_channels = int(
                graph.node_table[str(activation_name)].tensor_meta["shape"][1]
            )
            groups = int(node.args[8])
            if groups == 1:
                replacement = MegaConv2dOp()
                reduction_size = weight_shape[1] * weight_shape[2] * weight_shape[3]
            elif (
                groups == input_channels
                and weight_shape[0] == input_channels
                and weight_shape[1] == 1
            ):
                replacement = MegaConv2dDepthwiseOp()
                weight_node._mega_depthwise_weight = True
                reduction_size = weight_shape[2] * weight_shape[3]
            else:
                raise ValueError(f"unsupported grouped Conv2D for {node.name}")
        else:
            if isinstance(node, AddMMOp):
                if len(node.args) < 3:
                    raise ValueError(f"unexpected AddMM form for {node.name}")
                bias_arg, activation_name, weight_arg = node.args[:3]
            else:
                if len(node.args) != 2:
                    raise ValueError(f"unexpected Matmul form for {node.name}")
                activation_name, weight_arg = node.args
                bias_arg = None
            weight_value = graph.node_table[str(weight_arg)]
            valid_transpose = isinstance(weight_value, TOp) or (
                isinstance(weight_value, PermuteOp)
                and len(weight_value.args) == 2
                and list(weight_value.args[1]) == [1, 0]
            )
            if not valid_transpose or len(weight_value._parents) != 1:
                raise ValueError(
                    f"Mega MatMul weight must be one transpose for {node.name}: "
                    f"{weight_value.name}/{type(weight_value).__name__}, "
                    f"parents={weight_value._parents}, args={weight_value.args}"
                )
            weight_node = graph.node_table[weight_value._parents[0]]
            weight_arg = weight_node.name
            removed.add(weight_value.name)
            replacement = MegaMatmulOp()
            weight_name = parameter_names.get(weight_node.name)
            if weight_name is None:
                raise ValueError(f"Mega MatMul weight is not a parameter for {node.name}")
            weight_shape = list(arrays[weight_name].shape)
            if len(weight_shape) != 2:
                raise ValueError(f"Mega MatMul weight must be rank 2 for {node.name}")
            reduction_size = weight_shape[1]

        weight_name = parameter_names.get(weight_node.name)
        if weight_name is None or weight_name not in weight_scales:
            raise ValueError(f"missing quantized weight for {node.name}")
        weight_node._mega_weight = True
        records = calibration.get(weight_name)
        occurrence = calibration_index.get(weight_name, 0)
        if records is None or occurrence >= len(records):
            raise ValueError(f"missing calibration for {weight_name}")
        calibration_index[weight_name] = occurrence + 1
        calibrated_input, output_scale = records[occurrence]
        if (
            not np.isfinite(calibrated_input)
            or calibrated_input <= 0.0
            or not np.isfinite(output_scale)
            or output_scale <= 0.0
        ):
            raise ValueError(f"invalid calibration scale for {node.name}")
        activation_node = graph.node_table[str(activation_name)]

        dw = np.asarray(weight_scales[weight_name], dtype=np.float32).reshape(-1)
        output_channels = weight_shape[0]
        if dw.size != output_channels:
            raise ValueError(f"weight scale count mismatch for {weight_name}")
        if not np.all(np.isfinite(dw)) or np.any(dw <= 0.0):
            raise ValueError(f"invalid weight scale for {weight_name}")
        if bias_arg is None:
            bias = np.zeros(output_channels, dtype=np.float32)
        else:
            bias_node = graph.node_table[str(bias_arg)]
            bias_name = parameter_names.get(bias_node.name)
            if bias_name is None:
                raise ValueError(f"Mega bias is not an offline parameter for {node.name}")
            bias = np.asarray(arrays[bias_name], dtype=np.float32).reshape(-1)
        if bias.size != output_channels:
            raise ValueError(f"bias channel count mismatch for {node.name}")
        if not np.all(np.isfinite(bias)):
            raise ValueError(f"non-finite bias for {node.name}")
        product_bound = 128 * 128 * reduction_size
        bias_headroom = np.iinfo(np.int32).max - product_bound
        if bias_headroom <= 1:
            raise ValueError(f"INT32 accumulator cannot hold {node.name}")
        minimum_input_scale = float(
            np.max(
                np.abs(bias.astype(np.float64))
                / ((bias_headroom - 1) * dw.astype(np.float64))
            )
        )
        required_input_scale = max(float(calibrated_input), minimum_input_scale)

        direct_users = [
            candidate
            for candidate in graph._body
            if candidate is not node
            and any(str(arg) == node.name for arg in candidate.args)
        ]
        relus = [
            candidate
            for candidate in graph._body
            if isinstance(candidate, (ReluOp, HardswishOp))
            and len(candidate.args) == 1
            and str(candidate.args[0]) == node.name
        ]
        if len(relus) > 1:
            raise ValueError(f"multiple activation users for {node.name}")
        activation = relus[0] if relus else None
        activation_nodes = [activation] if activation is not None else []
        activation_kind = (
            2 if isinstance(activation, HardswishOp)
            else 1 if isinstance(activation, ReluOp)
            else 0
        )
        lut_kind = "hardswish" if activation_kind == 2 else None

        hard_swish_adds = [
            candidate
            for candidate in direct_users
            if isinstance(candidate, AddOp)
            and len(candidate.args) == 2
            and str(candidate.args[0]) == node.name
            and candidate.args[1] == 3
        ]
        if hard_swish_adds:
            if activation is not None or len(hard_swish_adds) != 1:
                raise ValueError(f"ambiguous HardSwish after {node.name}")
            add = hard_swish_adds[0]
            add_users = [
                candidate
                for candidate in graph._body
                if any(str(arg) == add.name for arg in candidate.args)
            ]
            if len(add_users) != 1 or not isinstance(add_users[0], ClampMinOp):
                raise ValueError(f"malformed HardSwish clamp-min after {node.name}")
            clamp_min = add_users[0]
            if list(clamp_min.args) != [add.name, 0]:
                raise ValueError(f"malformed HardSwish clamp-min args after {node.name}")
            clamp_min_users = [
                candidate
                for candidate in graph._body
                if any(str(arg) == clamp_min.name for arg in candidate.args)
            ]
            if len(clamp_min_users) != 1 or not isinstance(
                clamp_min_users[0], ClampMaxOp
            ):
                raise ValueError(f"malformed HardSwish clamp-max after {node.name}")
            clamp_max = clamp_min_users[0]
            if list(clamp_max.args) != [clamp_min.name, 6]:
                raise ValueError(f"malformed HardSwish clamp-max args after {node.name}")
            clamp_max_users = [
                candidate
                for candidate in graph._body
                if any(str(arg) == clamp_max.name for arg in candidate.args)
            ]
            if len(clamp_max_users) != 1:
                raise ValueError(f"ambiguous hard activation after {node.name}")
            after_clamp = clamp_max_users[0]
            if isinstance(after_clamp, DivOp):
                if list(after_clamp.args) != [clamp_max.name, 6]:
                    raise ValueError(f"malformed hard-sigmoid after {node.name}")
                if set(candidate.name for candidate in direct_users) != {add.name}:
                    raise ValueError(f"hard-sigmoid input has extra users after {node.name}")
                activation = after_clamp
                activation_nodes = [add, clamp_min, clamp_max, after_clamp]
                activation_kind = 2
                lut_kind = "hardsigmoid"
            elif isinstance(after_clamp, MulOp):
                mul = after_clamp
                if list(map(str, mul.args)) != [node.name, clamp_max.name]:
                    raise ValueError(f"malformed HardSwish multiply args after {node.name}")
                if set(candidate.name for candidate in direct_users) != {add.name, mul.name}:
                    raise ValueError(f"HardSwish input has extra users after {node.name}")
                mul_users = [
                    candidate
                    for candidate in graph._body
                    if any(str(arg) == mul.name for arg in candidate.args)
                ]
                if len(mul_users) != 1 or not isinstance(mul_users[0], DivOp):
                    raise ValueError(f"malformed HardSwish divide after {node.name}")
                div = mul_users[0]
                if list(div.args) != [mul.name, 6]:
                    raise ValueError(f"malformed HardSwish divide args after {node.name}")
                activation = div
                activation_nodes = [add, clamp_min, clamp_max, mul, div]
                activation_kind = 2
                lut_kind = "hardswish"
            else:
                raise ValueError(f"unsupported hard activation after {node.name}")

        result_name = activation.name if activation is not None else node.name
        result_users = [
            candidate
            for candidate in graph._body
            if candidate is not activation
            and candidate is not node
            and any(str(arg) == result_name for arg in candidate.args)
        ]
        plans[node.name] = {
            "index": index,
            "node": node,
            "replacement": replacement,
            "activation_name": str(activation_name),
            "activation_node": activation_node,
            "weight_arg": str(weight_arg),
            "weight_shape": weight_shape,
            "dw": dw,
            "bias": bias,
            "required_input_scale": required_input_scale,
            "calibrated_output_scale": float(output_scale),
            "activation": activation,
            "activation_nodes": activation_nodes,
            "activation_kind": activation_kind,
            "lut_kind": lut_kind,
            "direct_users": direct_users,
            "result_name": result_name,
            "users": result_users,
            "padding": padding if isinstance(node, Conv2dOp) else None,
            "product_bound": product_bound,
        }

    for plan in plans.values():
        if plan["activation"] is not None:
            renamed[plan["result_name"]] = plan["node"].name
            removed.update(item.name for item in plan["activation_nodes"])

    special = {}
    for node in original_body:
        if node.name in removed:
            continue
        if isinstance(node, MaxPool2dOp):
            if (
                len(node.args) not in (3, 4)
                or len(node.args[1]) != 2
                or node.args[1][0] != node.args[1][1]
                or len(node.args[2]) != 2
                or node.args[2][0] != node.args[2][1]
            ):
                raise ValueError(f"unsupported MegaKernel MaxPool2D for {node.name}")
            padding = node.args[3] if len(node.args) == 4 else [0, 0]
            if len(padding) == 1:
                padding = [padding[0], padding[0]]
            if len(padding) != 2 or padding[0] != padding[1]:
                raise ValueError(f"asymmetric MegaKernel MaxPool2D padding for {node.name}")
            replacement = MegaMaxPool2dOp()
        elif isinstance(node, MeanOp):
            if (
                len(node.args) != 3
                or list(node.args[1]) != [-1, -2]
                or not node.args[2]
            ):
                continue
            replacement = MegaGlobalAvgPoolOp()
        elif isinstance(node, MulOp) and len(node.args) == 2 and all(
            isinstance(arg, str) for arg in node.args
        ):
            replacement = MegaInt8MulOp()
        elif isinstance(node, AddOp) and len(node.args) == 2 and all(
            isinstance(arg, str) for arg in node.args
        ):
            replacement = MegaInt8AddOp()
        else:
            continue
        special[node.name] = {"node": node, "replacement": replacement}

    for plan in plans.values():
        consumers = [
            candidate
            for candidate in plans.values()
            if candidate["activation_name"] == plan["result_name"]
        ]
        plan["output_scale"] = (
            max(candidate["required_input_scale"] for candidate in consumers)
            if consumers
            else plan["calibrated_output_scale"]
        )

    for item in special.values():
        node = item["node"]
        consumers = [
            candidate
            for candidate in plans.values()
            if candidate["activation_name"] == node.name
        ]
        if not consumers and isinstance(node, (MeanOp, MaxPool2dOp)):
            users = [
                candidate
                for candidate in original_body
                if any(str(arg) == node.name for arg in candidate.args)
            ]
            if len(users) != 1 or not isinstance(users[0], (ViewOp, ReshapeOp)):
                raise ValueError(f"MegaKernel mean has no unique consumer: {node.name}")
            consumers = [
                candidate
                for candidate in plans.values()
                if candidate["activation_name"] == users[0].name
            ]
        if len(consumers) != 1:
            raise ValueError(f"MegaKernel stage has no unique compute consumer: {node.name}")
        item["output_scale"] = consumers[0]["required_input_scale"]

    for plan in plans.values():
        max_pool_consumers = [
            item
            for item in special.values()
            if isinstance(item["replacement"], MegaMaxPool2dOp)
            and renamed.get(str(item["node"].args[0]), str(item["node"].args[0]))
            == plan["node"].name
        ]
        if len(max_pool_consumers) > 1:
            raise ValueError(f"multiple MaxPool2D consumers for {plan['node'].name}")
        if max_pool_consumers:
            plan["output_scale"] = max_pool_consumers[0]["output_scale"]

    value_scale = {}
    for plan in plans.values():
        value_scale[plan["node"].name] = plan["output_scale"]
        value_scale[plan["result_name"]] = plan["output_scale"]
    for name, item in special.items():
        value_scale[name] = item["output_scale"]

    for plan in plans.values():
        activation_name = renamed.get(plan["activation_name"], plan["activation_name"])
        input_scale = value_scale.get(activation_name, plan["required_input_scale"])
        bias_i64 = np.rint(
            plan["bias"].astype(np.float64)
            / (input_scale * plan["dw"].astype(np.float64))
        ).astype(np.int64)
        overflow = np.flatnonzero(
            np.abs(bias_i64) + plan["product_bound"] > np.iinfo(np.int32).max
        )
        if overflow.size:
            channel = int(overflow[0])
            raise ValueError(
                f"INT32 accumulator overflow for {plan['node'].name} channel {channel}: "
                f"input_scale={input_scale}, weight_scale={float(plan['dw'][channel])}, "
                f"bias_i32={int(bias_i64[channel])}, "
                f"product_bound={plan['product_bound']}"
            )

        node = plan["node"]
        replacement = plan["replacement"]
        activation = plan["activation"]
        replacement._name = node.name
        replacement._arguments = [activation_name, plan["weight_arg"]]
        replacement._parents = list(replacement._arguments)
        replacement._children = [renamed.get(user.name, user.name) for user in plan["users"]]
        replacement._tensor_meta = node._tensor_meta.copy()
        direct_compute_consumers = [
            candidate
            for candidate in plans.values()
            if candidate["activation_name"] == plan["result_name"]
        ]
        replacement._final_output = not isinstance(node, Conv2dOp) and not direct_compute_consumers
        replacement._tensor_meta["dtype"] = (
            TensorDType.Float32 if replacement._final_output else TensorDType.Int8
        )
        replacement._input_scale = input_scale
        replacement._output_scale = plan["output_scale"]
        replacement._bias_i32 = bias_i64.astype(np.int32).tolist()
        replacement._requant_scale = (
            np.float32(input_scale)
            * plan["dw"]
            / np.float32(plan["output_scale"])
        ).tolist()
        replacement._dequant_scale = (
            np.float32(input_scale) * plan["dw"]
        ).tolist()
        replacement._activation = plan["activation_kind"]
        if replacement._activation == 2:
            raw = np.arange(256, dtype=np.int16)
            signed = np.where(raw < 128, raw, raw - 256).astype(np.float32)
            x = signed * np.float32(plan["output_scale"])
            if plan["lut_kind"] == "hardswish":
                y = x * np.clip(x + np.float32(3.0), 0.0, 6.0) / np.float32(6.0)
            elif plan["lut_kind"] == "hardsigmoid":
                y = np.clip(x + np.float32(3.0), 0.0, 6.0) / np.float32(6.0)
            else:
                raise ValueError(f"missing LUT function for {node.name}")
            replacement._lut_i8 = np.clip(
                np.rint(y / np.float32(plan["output_scale"])), -128, 127
            ).astype(np.int8).tolist()
        else:
            replacement._lut_i8 = [0]
        replacement.trace_meta = node.trace_meta
        if isinstance(replacement, (MegaConv2dOp, MegaConv2dDepthwiseOp)):
            replacement._input_shape = list(plan["activation_node"].tensor_meta["shape"])
            replacement._weight_shape = plan["weight_shape"]
            replacement._output_shape = list(node.tensor_meta["shape"])
            replacement._stride = int(node.args[3][0])
            replacement._padding = int(plan["padding"][0])
    for name, item in special.items():
        node = item["node"]
        replacement = item["replacement"]
        args = [renamed.get(str(arg), str(arg)) for arg in node.args if isinstance(arg, str)]
        if isinstance(replacement, MegaMaxPool2dOp):
            if len(args) != 1 or args[0] not in value_scale:
                raise ValueError(f"invalid Mega MaxPool2D input for {name}")
            replacement._input_scale = value_scale[args[0]]
            replacement._output_scale = replacement._input_scale
            replacement._input_shape = list(
                graph.node_table[str(node.args[0])].tensor_meta["shape"]
            )
            replacement._output_shape = list(node.tensor_meta["shape"])
            replacement._kernel = int(node.args[1][0])
            replacement._stride = int(node.args[2][0])
            padding = node.args[3] if len(node.args) == 4 else [0, 0]
            replacement._padding = int(padding[0])
            replacement._final_output = False
        elif isinstance(replacement, MegaGlobalAvgPoolOp):
            if len(args) != 1 or args[0] not in value_scale:
                raise ValueError(f"invalid Mega global-average input for {name}")
            replacement._input_scale = value_scale[args[0]]
        else:
            if len(args) != 2 or any(arg not in value_scale for arg in args):
                raise ValueError(f"invalid Mega INT8 elementwise input for {name}")
            replacement._lhs_scale = value_scale[args[0]]
            replacement._rhs_scale = value_scale[args[1]]
        replacement._name = name
        replacement._arguments = args
        replacement._parents = list(args)
        replacement._children = [renamed.get(child, child) for child in node._children]
        replacement._tensor_meta = node._tensor_meta.copy()
        replacement._tensor_meta["dtype"] = TensorDType.Int8
        replacement._output_scale = item["output_scale"]
        replacement.trace_meta = node.trace_meta

    conv_stage_names = {
        plan["node"].name
        for plan in plans.values()
        if isinstance(plan["node"], Conv2dOp)
    } | set(special)
    conv_components = []
    stage_component = {}
    for node in original_body:
        if node.name in plans and isinstance(node, Conv2dOp):
            stage = plans[node.name]["replacement"]
        elif node.name in special:
            stage = special[node.name]["replacement"]
        else:
            continue

        parents = {
            stage_component[str(argument)]
            for argument in stage.args
            if str(argument) in stage_component
        }
        if len(parents) > 1:
            raise ValueError(f"MegaKernel stage joins disconnected regions: {node.name}")
        if parents:
            component = next(iter(parents))
        else:
            component = len(conv_components)
            conv_components.append([])
        conv_components[component].append(stage)
        stage_component[stage.name] = component

    for stages in conv_components:
        last = stages[-1]
        if isinstance(last, (MegaConv2dOp, MegaConv2dDepthwiseOp)):
            last._final_output = True
            last._tensor_meta["dtype"] = TensorDType.Float32
        elif isinstance(last, MegaMaxPool2dOp):
            last._final_output = True

    matmul_stages = [
        plan["replacement"]
        for plan in plans.values()
        if not isinstance(plan["node"], Conv2dOp)
    ]
    if not conv_components and not matmul_stages:
        raise ValueError("model has no MegaKernel stages")

    kernels = []
    for stages in [*conv_components, matmul_stages]:
        if not stages:
            continue
        produced = {stage.name for stage in stages}
        arguments = []
        for stage in stages:
            for argument in stage.args:
                name = str(argument)
                if name not in produced and name not in arguments:
                    arguments.append(name)
        kernel = MegaKernelOp()
        kernel._name = stages[-1].name
        kernel._arguments = arguments
        kernel._parents = list(arguments)
        kernel._children = list(stages[-1]._children)
        kernel._tensor_meta = stages[-1]._tensor_meta.copy()
        kernel._stages = stages
        kernel.trace_meta = stages[0].trace_meta
        kernels.append(kernel)

    original_stage_names = conv_stage_names | {
        plan["node"].name for plan in plans.values() if not isinstance(plan["node"], Conv2dOp)
    }
    first_to_kernel = {kernel._stages[0].name: kernel for kernel in kernels}
    drop = original_stage_names | removed
    graph._body = [
        first_to_kernel.get(node.name, node)
        for node in original_body
        if node.name not in drop or node.name in first_to_kernel
    ]
    for name in drop:
        graph.node_table.pop(name, None)
    for kernel in kernels:
        graph.node_table[kernel.name] = kernel
    for group in graph.op_groups.values():
        group[:] = [
            first_to_kernel.get(node.name, node)
            for node in group
            if node.name not in drop or node.name in first_to_kernel
        ]

    for node in graph._body:
        node._parents = [renamed.get(parent, parent) for parent in node._parents]
        node._children = [renamed.get(child, child) for child in node._children]
        missing = [parent for parent in node._parents if parent not in graph.node_table]
        if missing:
            raise ValueError(f"dangling graph parents for {node.name}: {missing}")

    body_index = {id(node): i for i, node in enumerate(graph._body)}
    graph._fake_params = [body_index[id(node)] for node in param_nodes]
    graph._inputs = [body_index[id(node)] for node in input_node_list]


def quantize_model_graph(
    graph,
    params,
    names: list[str],
    output_dir: Path,
    model_name: str,
    calibration: dict,
) -> None:
    if len(graph.params) != len(params) or len(names) != len(params):
        raise ValueError("parameter metadata does not match imported graph")
    param_nodes = list(graph.params)
    parameter_names, arrays, weight_scales = {}, {}, {}
    for node, param, name in zip(param_nodes, params, names):
        array = param.detach().cpu().numpy().astype(np.float32)
        parameter_names[node.name] = name
        arrays[name] = array
        if name.endswith(".weight") and array.ndim >= 2:
            _, weight_scales[name] = quantize_symmetric(array, [0])

    _form_mega_kernels(
        graph,
        parameter_names,
        arrays,
        weight_scales,
        calibration,
    )

    tensors = []
    weights, fp_params, scales = [], [], []
    weight_off = param_off = scale_off = 0
    for index, (node, name) in enumerate(zip(param_nodes, names)):
        array = arrays[name]
        if getattr(node, "_mega_weight", False):
            axes = [0]
            q, dw = quantize_symmetric(array, axes)
            if array.ndim == 4:
                if getattr(node, "_mega_depthwise_weight", False):
                    q = np.transpose(q, (2, 3, 0, 1)).copy()
                    storage_axes = [2]
                else:
                    q = np.transpose(q, (2, 3, 1, 0)).copy()
                    storage_axes = [3]
            elif array.ndim == 2:
                q = q.T.copy()
                storage_axes = [1]
            else:
                raise ValueError(f"unsupported Mega weight rank for {name}")
            node.tensor_meta["dtype"] = TensorDType.Int8
            node.tensor_meta["shape"] = list(q.shape)
            params[index] = torch.from_numpy(q.copy())
            raw_scales = np.asarray(dw, dtype=np.float32).reshape(-1)
            padded = np.pad(raw_scales, (0, (-len(raw_scales)) % 16),
                            constant_values=1.0)
            scale_bytes = padded.tobytes()
            tensors.append(QuantTensor(name, list(q.shape), list(q.shape), "i8",
                                       storage_axes, weight_off, q.nbytes, scale_off, len(scale_bytes)))
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
    (output_dir / "weights.i8").write_bytes(package.weights_i8)
    (output_dir / "params.f32").write_bytes(package.params_f32)
    (output_dir / "scales.bin").write_bytes(package.scales_f32)
