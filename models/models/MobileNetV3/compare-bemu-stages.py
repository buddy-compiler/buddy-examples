#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_dense_constants(mlir: str) -> dict[str, np.ndarray]:
    constants: dict[str, np.ndarray] = {}
    pattern = re.compile(
        r"^\s*(%[A-Za-z0-9_]+) = arith\.constant dense<(.*?)> "
        r": tensor<(\d+)x(i8|i32|f32)>$"
    )
    dtypes = {"i8": np.dtype("i1"), "i32": np.dtype("<i4"), "f32": np.dtype("<f4")}
    for line in mlir.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        name, literal, count_text, element_type = match.groups()
        count = int(count_text)
        dtype = dtypes[element_type]
        if literal.startswith('"0x') and literal.endswith('"'):
            value = np.frombuffer(bytes.fromhex(literal[3:-1]), dtype=dtype).copy()
        elif literal.startswith("[") and literal.endswith("]"):
            value = np.asarray(literal[1:-1].split(","), dtype=dtype)
        else:
            value = np.full(count, literal, dtype=dtype)
        if value.size != count:
            raise ValueError(
                f"{name} has {value.size} elements, expected {count}"
            )
        constants[name] = value
    return constants


def parse_stages(mlir: str) -> tuple[str, np.float32, list[dict]]:
    quant_result = None
    input_multiplier = None
    stages: list[dict] = []
    regions: list[dict] = []
    for line in mlir.splitlines():
        if "buckyball.quant_f32_to_i8" in line:
            result = re.match(r"\s*(%\d+) = linalg\.generic", line)
            scale = re.search(r"scale = ([^ ,}]+) : f32", line)
            if not result or not scale or quant_result is not None:
                raise ValueError("expected one well-formed input quantization op")
            quant_result = result.group(1)
            input_multiplier = np.float32(scale.group(1))
        if "buckyball.mega_kernel = true" not in line:
            continue
        region_id = re.search(r'mega_kernel_id = "([^"]+)"', line)
        region_size = re.search(r"mega_kernel_size = (\d+) : i64", line)
        stage_number = re.search(r"mega_kernel_stage = (\d+) : i64", line)
        if not region_id or not region_size or not stage_number:
            raise ValueError(f"missing MegaKernel metadata: {line}")
        name = region_id.group(1)
        size = int(region_size.group(1))
        local_stage = int(stage_number.group(1))
        if not regions or regions[-1]["id"] != name:
            if regions and regions[-1]["count"] != regions[-1]["size"]:
                raise ValueError(f"incomplete MegaKernel region: {regions[-1]}")
            if any(region["id"] == name for region in regions):
                raise ValueError(f"noncontiguous MegaKernel region: {name}")
            regions.append({"id": name, "size": size, "count": 0})
        region = regions[-1]
        if (
            size <= 0
            or size != region["size"]
            or local_stage != region["count"]
            or local_stage >= size
        ):
            raise ValueError(
                f"invalid MegaKernel stage metadata: {name}, "
                f"size={size}, stage={local_stage}"
            )
        region["count"] += 1
        if "buckyball.mega_matmul" in line:
            if len(regions) != 17 or size != 2 or len(stages) != 77:
                raise ValueError(
                    "expected one final two-stage classifier after "
                    "77 convolution stages"
                )
            continue
        if len(regions) > 16:
            raise ValueError("expected only classifier stages in region 17")
        result = re.match(r"\s*(%\d+) = linalg\.generic", line)
        inputs = re.search(r" ins\((.*?) : tensor<", line)
        output = re.search(r" outs\(.*?tensor<([0-9x]+)xi8>", line)
        if not result or not inputs or not output:
            raise ValueError(f"malformed MegaKernel stage: {line}")
        stage = len(stages)
        if "buckyball.mega_conv2d_depthwise" in line:
            kind = "depthwise"
        elif "buckyball.mega_conv2d" in line:
            kind = "conv2d"
        elif "buckyball.mega_global_avg_pool" in line:
            kind = "global_avg"
        elif "buckyball.mega_int8_mul" in line:
            kind = "mul"
        elif "buckyball.mega_int8_add" in line:
            kind = "add"
        else:
            raise ValueError(f"unsupported stage {stage}: {line}")
        attrs = {}
        for name in (
            "activation",
            "kernel",
            "stride",
            "pad_low",
            "input_scale",
            "lhs_scale",
            "rhs_scale",
            "output_scale",
        ):
            match = re.search(rf"(?:[ {{,]){name} = ([^ ,}}]+)", line)
            if match:
                attrs[name] = float(match.group(1))
        stages.append(
            {
                "number": stage,
                "kind": kind,
                "result": result.group(1),
                "inputs": re.findall(r"%[A-Za-z0-9_]+", inputs.group(1)),
                "shape": tuple(int(x) for x in output.group(1).split("x")),
                "attrs": attrs,
            }
        )
    if quant_result is None or input_multiplier is None:
        raise ValueError("missing input quantization op")
    if len(regions) != 17 or regions[-1]["count"] != regions[-1]["size"]:
        raise ValueError("expected 17 complete MegaKernel regions")
    if len(stages) != 77:
        raise ValueError(f"expected 77 convolution stages, found {len(stages)}")
    return quant_result, input_multiplier, stages


def load_weight(entry: dict, payload_dir: Path, depthwise: bool) -> np.ndarray:
    if entry["payload"] != "weights_i8" or entry["storage"] != "i8":
        raise ValueError(f"{entry['name']} is not an INT8 payload weight")
    payload = np.fromfile(
        payload_dir / "weights.i8",
        dtype=np.int8,
        count=entry["payload_bytes"],
        offset=entry["payload_offset"],
    )
    packed_shape = tuple(entry["payload_shape"])
    if payload.size != int(np.prod(packed_shape)):
        raise ValueError(f"short payload for {entry['name']}")
    packed = payload.reshape(packed_shape)
    cout, cin, kh, kw = entry["shape"]
    if depthwise:
        if packed_shape != (kh, kw, cout, 1) or cin != 1:
            raise ValueError(f"invalid depthwise layout for {entry['name']}")
        return packed.transpose(2, 3, 0, 1).copy()
    padded_k = ((kh * kw + 15) // 16) * 16
    expected = ((cout + 15) // 16, cin, padded_k, 16)
    if packed_shape != expected:
        raise ValueError(f"invalid packed layout for {entry['name']}: {packed_shape}")
    return (
        packed.transpose(0, 3, 1, 2)
        .reshape(-1, cin, padded_k)[:cout, :, : kh * kw]
        .reshape(cout, cin, kh, kw)
        .copy()
    )


def parse_classifier_stage(mlir: str) -> dict:
    matches = [
        line
        for line in mlir.splitlines()
        if "mega_kernel_size = 2 : i64" in line
        and "mega_kernel_stage = 0 : i64" in line
        and "buckyball.mega_matmul" in line
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one classifier stage 0, found {len(matches)}")
    line = matches[0]
    inputs = re.search(r" ins\((.*?) : tensor<", line)
    output = re.search(r" outs\(.*?tensor<(\d+)x(\d+)xi8>", line)
    activation = re.search(r"activation = (\d+) : i64", line)
    if not inputs or not output or not activation:
        raise ValueError(f"malformed classifier stage: {line}")
    return {
        "inputs": re.findall(r"%[A-Za-z0-9_]+", inputs.group(1)),
        "shape": (int(output.group(1)), int(output.group(2))),
        "activation": int(activation.group(1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BEMU stage dumps with the exact INT8 PyTorch pipeline"
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("mlir", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("payload_dir", type=Path)
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--classifier-trace-id", type=int)
    parser.add_argument("--classifier-output-trace", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--reference-output-dir", type=Path)
    args = parser.parse_args()
    if args.reference_output_dir is not None:
        if not args.reference_only:
            raise ValueError("--reference-output-dir requires --reference-only")
        args.reference_output_dir.mkdir(parents=True, exist_ok=True)

    mlir = args.mlir.read_text()
    constants = parse_dense_constants(mlir)
    quant_result, input_multiplier, stages = parse_stages(mlir)
    if not 0 <= args.start_stage < len(stages):
        raise ValueError(f"start stage out of range: {args.start_stage}")
    if args.classifier_trace_id is not None and args.classifier_trace_id < len(stages):
        raise ValueError("classifier trace ID must not overlap convolution stages")
    index = json.loads((args.payload_dir / "quant-index.json").read_text())
    weights = [item for item in index["tensors"] if item["storage"] == "i8"]
    if len(weights) != 54:
        raise ValueError(f"expected 54 INT8 weights, found {len(weights)}")

    parts: dict[int, dict[int, Path]] = {}
    for path in args.trace_dir.glob("trace-*-part-*.i8"):
        match = re.fullmatch(r"trace-(\d+)-part-(\d+)\.i8", path.name)
        if not match:
            raise ValueError(f"malformed stage trace name: {path.name}")
        stage, part = (int(value) for value in match.groups())
        if stage == args.classifier_trace_id:
            continue
        if not 0 <= stage < len(stages):
            raise ValueError(f"stage trace ID out of range: {stage}")
        if stage < args.start_stage:
            continue
        if part in parts.setdefault(stage, {}):
            raise ValueError(f"duplicate trace part for stage {stage}: {part}")
        parts[stage][part] = path
    if not parts and not args.reference_only:
        raise ValueError(
            f"no sharded stage traces at or after {args.start_stage} in "
            f"{args.trace_dir}"
        )
    trace_numbers = list(range(77)) if args.reference_only else sorted(parts)
    traces: dict[int, np.ndarray] = {}
    if not args.reference_only:
        for stage in trace_numbers:
            part_numbers = sorted(parts[stage])
            if part_numbers != list(range(part_numbers[-1] + 1)):
                raise ValueError(
                    f"trace parts for stage {stage} are not contiguous: {part_numbers}"
                )
            traces[stage] = np.concatenate(
                [
                    np.fromfile(parts[stage][part], dtype=np.int8)
                    for part in part_numbers
                ]
            )
            expected_size = int(np.prod(stages[stage]["shape"]))
            if traces[stage].size != expected_size:
                raise ValueError(
                    f"stage {stage} size mismatch: BEMU={traces[stage].size}, "
                    f"expected={expected_size}"
                )

    pixels = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32)
    if pixels.shape != (224, 224, 3):
        raise ValueError(f"expected a 224x224 RGB image, got {pixels.shape}")
    input_i8 = np.clip(
        np.rint((pixels / np.float32(255.0)) * input_multiplier), -128, 127
    ).astype(np.int8)[None, ...]
    values: dict[str, np.ndarray] = {quant_result: input_i8}

    print("stage  op          shape              corr       exact       mae   max")
    first_mismatch = None
    last_reference_stage = (
        len(stages) - 1
        if args.classifier_trace_id is not None or args.classifier_output_trace is not None
        else trace_numbers[-1]
    )
    for stage in stages[: last_reference_stage + 1]:
        number = stage["number"]
        kind = stage["kind"]
        operands = [values[name] for name in stage["inputs"] if name in values]
        attrs = stage["attrs"]
        if kind in ("conv2d", "depthwise"):
            if len(operands) != 1 or len(stage["inputs"]) != 5:
                raise ValueError(f"stage {number} has invalid Conv operands")
            weight_arg = re.fullmatch(r"%arg(\d+)", stage["inputs"][1])
            if not weight_arg:
                raise ValueError(f"stage {number} has no payload weight")
            entry = weights[int(weight_arg.group(1)) - 1]
            weight = load_weight(entry, args.payload_dir, kind == "depthwise")
            bias = constants[stage["inputs"][2]].astype(np.int32)
            scale = constants[stage["inputs"][3]].astype(np.float32)
            lut = constants[stage["inputs"][4]].astype(np.int8)
            source = torch.from_numpy(
                operands[0].transpose(0, 3, 1, 2).astype(np.int32)
            )
            kernel = torch.from_numpy(weight.astype(np.int32))
            groups = source.shape[1] if kind == "depthwise" else 1
            accumulator = F.conv2d(
                source,
                kernel,
                torch.from_numpy(bias),
                stride=int(attrs["stride"]),
                padding=int(attrs["pad_low"]),
                groups=groups,
            ).numpy().transpose(0, 2, 3, 1)
            reference = np.clip(
                np.rint(accumulator.astype(np.float32) * scale), -128, 127
            ).astype(np.int8)
            activation = int(attrs["activation"])
            if activation == 1:
                reference = np.maximum(reference, np.int8(0))
            elif activation == 2:
                indices = reference.view(np.uint8)
                if lut.size == 256:
                    reference = lut[indices]
                elif lut.size == 4096 and reference.shape[-1] == 16:
                    reference = lut.reshape(16, 256)[np.arange(16), indices]
                else:
                    raise ValueError(
                        f"stage {number} has invalid LUT size {lut.size} for "
                        f"{reference.shape[-1]} channels"
                    )
            elif activation != 0:
                raise ValueError(f"stage {number} has invalid activation {activation}")
        elif kind == "global_avg":
            if len(operands) != 1:
                raise ValueError(f"stage {number} has invalid GlobalAvg operands")
            source = operands[0]
            ratio = np.float32(
                attrs["input_scale"]
                / (source.shape[1] * source.shape[2] * attrs["output_scale"])
            )
            total = source.astype(np.int32).sum(axis=(1, 2), dtype=np.int32)
            reference = np.clip(
                np.rint(total.astype(np.float32) * ratio), -128, 127
            ).astype(np.int8)[:, None, None, :]
        elif kind == "mul":
            if len(operands) != 2:
                raise ValueError(f"stage {number} has invalid Mul operands")
            ratio = np.float32(
                attrs["lhs_scale"] * attrs["rhs_scale"] / attrs["output_scale"]
            )
            reference = np.clip(
                np.rint(
                    operands[0].astype(np.float32)
                    * operands[1].astype(np.float32)
                    * ratio
                ),
                -128,
                127,
            ).astype(np.int8)
        elif kind == "add":
            if len(operands) != 2:
                raise ValueError(f"stage {number} has invalid Add operands")
            lhs_ratio = np.float32(attrs["lhs_scale"] / attrs["output_scale"])
            rhs_ratio = np.float32(attrs["rhs_scale"] / attrs["output_scale"])
            reference = np.clip(
                np.rint(
                    operands[0].astype(np.float32) * lhs_ratio
                    + operands[1].astype(np.float32) * rhs_ratio
                ),
                -128,
                127,
            ).astype(np.int8)
            if int(attrs["activation"]) == 1:
                reference = np.maximum(reference, np.int8(0))
        else:
            raise AssertionError(kind)

        if reference.shape != stage["shape"]:
            raise ValueError(
                f"stage {number} reference shape {reference.shape} != {stage['shape']}"
            )
        values[stage["result"]] = reference
        if args.reference_only:
            if args.reference_output_dir is not None:
                reference.tofile(
                    args.reference_output_dir / f"trace-{number}-part-0.i8"
                )
            continue
        if number not in traces:
            continue
        actual = traces[number]
        if actual.size != reference.size:
            raise ValueError(
                f"stage {number} size mismatch: BEMU={actual.size}, "
                f"PyTorch={reference.size}"
            )
        actual = actual.reshape(reference.shape)
        actual_f64 = actual.reshape(-1).astype(np.float64)
        reference_f64 = reference.reshape(-1).astype(np.float64)
        if np.std(actual_f64) == 0.0 or np.std(reference_f64) == 0.0:
            corr = 1.0 if np.array_equal(actual, reference) else 0.0
        else:
            corr = float(np.corrcoef(actual_f64, reference_f64)[0, 1])
        difference = np.abs(actual_f64 - reference_f64)
        exact = float(np.mean(actual == reference))
        shape = "x".join(str(dimension) for dimension in reference.shape)
        print(
            f"{number:5d}  {kind:10s}  {shape:17s}  {corr: .6f}  "
            f"{exact: .6f}  {np.mean(difference):5.3f}  {np.max(difference):3.0f}"
        )
        if exact != 1.0 and first_mismatch is None:
            first_mismatch = number

    if first_mismatch is not None:
        raise SystemExit(f"first mismatching stage: {first_mismatch}")
    if args.reference_only:
        print("computed all 77 integer reference stages")
    else:
        print(f"all {len(traces)} traced stages match exactly")

    if args.classifier_trace_id is not None or args.reference_only:
        classifier = parse_classifier_stage(mlir)
        if classifier["activation"] != 2 or len(classifier["inputs"]) != 5:
            raise ValueError("classifier stage 0 must be a LUT-activated MatMul")
        weight_arg = re.fullmatch(r"%arg(\d+)", classifier["inputs"][1])
        if not weight_arg:
            raise ValueError("classifier stage 0 has no payload weight")
        entry = weights[int(weight_arg.group(1)) - 1]
        payload = np.fromfile(
            args.payload_dir / "weights.i8",
            dtype=np.int8,
            count=entry["payload_bytes"],
            offset=entry["payload_offset"],
        )
        packed_shape = tuple(entry["payload_shape"])
        if payload.size != int(np.prod(packed_shape)):
            raise ValueError("short classifier stage 0 weight payload")
        source = values[stages[-1]["result"]].reshape(1, -1)
        if packed_shape != (source.shape[1], classifier["shape"][1]):
            raise ValueError(
                f"invalid classifier weight layout: {packed_shape}"
            )
        weight = payload.reshape(packed_shape)
        bias = constants[classifier["inputs"][2]].astype(np.int32)
        scale = constants[classifier["inputs"][3]].astype(np.float32)
        lut = constants[classifier["inputs"][4]].astype(np.int8)
        accumulator = (
            torch.from_numpy(source.astype(np.int32))
            @ torch.from_numpy(weight.astype(np.int32))
        ).numpy() + bias
        reference = np.clip(
            np.rint(accumulator.astype(np.float32) * scale), -128, 127
        ).astype(np.int8)
        if lut.size != 256:
            raise ValueError("classifier stage 0 requires a 256-entry LUT")
        reference = lut[reference.view(np.uint8)]

        if args.reference_only:
            classifier_reference = reference
            if args.reference_output_dir is not None:
                classifier_reference.tofile(
                    args.reference_output_dir / "trace-1000-part-0.i8"
                )
        else:
            classifier_parts = sorted(
            args.trace_dir.glob(
                f"trace-{args.classifier_trace_id}-part-*.i8"
            ),
            key=lambda path: int(path.stem.split("-")[-1]),
            )
            part_numbers = [
                int(path.stem.split("-")[-1]) for path in classifier_parts
            ]
            if not classifier_parts or part_numbers != list(
                range(len(classifier_parts))
            ):
                raise ValueError(
                    f"classifier trace {args.classifier_trace_id} parts are missing"
                )
            actual = np.concatenate(
                [np.fromfile(path, dtype=np.int8) for path in classifier_parts]
            )
            if actual.size != reference.size:
                raise ValueError(
                    f"classifier stage 0 size mismatch: BEMU={actual.size}, "
                    f"PyTorch={reference.size}"
                )
            actual = actual.reshape(reference.shape)
            actual_f64 = actual.reshape(-1).astype(np.float64)
            reference_f64 = reference.reshape(-1).astype(np.float64)
            corr = float(np.corrcoef(actual_f64, reference_f64)[0, 1])
            difference = np.abs(actual_f64 - reference_f64)
            exact = float(np.mean(actual == reference))
            print(
                f"classifier 0: corr={corr:.6f} exact={exact:.6f} "
                f"mae={np.mean(difference):.3f} max={np.max(difference):.0f}"
            )
            if exact != 1.0:
                raise SystemExit("first mismatching classifier stage: 0")

    if args.classifier_output_trace is not None or args.reference_only:
        if (
            not args.reference_only
            and args.classifier_output_trace.suffix != ".txt"
        ):
            raise ValueError("classifier output trace must be the FP32 text trace")
        classifier = parse_classifier_stage(mlir)
        final_line = next(
            line
            for line in mlir.splitlines()
            if "mega_kernel_size = 2 : i64" in line
            and "mega_kernel_stage = 1 : i64" in line
            and "buckyball.mega_matmul" in line
        )
        inputs = re.findall(r"%[A-Za-z0-9_]+", re.search(r" ins\((.*?) : tensor<", final_line).group(1))
        weight_arg = re.fullmatch(r"%arg(\d+)", inputs[1])
        if not weight_arg:
            raise ValueError("classifier stage 1 has no payload weight")
        entry = weights[int(weight_arg.group(1)) - 1]
        payload = np.fromfile(
            args.payload_dir / "weights.i8", dtype=np.int8,
            count=entry["payload_bytes"], offset=entry["payload_offset"])
        packed_shape = tuple(entry["payload_shape"])
        if packed_shape != (1024, 1000) or payload.size != 1024000:
            raise ValueError("invalid classifier stage 1 weight payload")
        source = values[stages[-1]["result"]].reshape(1, -1)
        first_entry = weights[int(re.fullmatch(r"%arg(\d+)", classifier["inputs"][1]).group(1)) - 1]
        first_payload = np.fromfile(
            args.payload_dir / "weights.i8", dtype=np.int8,
            count=first_entry["payload_bytes"], offset=first_entry["payload_offset"])
        first_weight = first_payload.reshape((576, 1024))
        first_bias = constants[classifier["inputs"][2]].astype(np.int32)
        first_scale = constants[classifier["inputs"][3]].astype(np.float32)
        first_lut = constants[classifier["inputs"][4]].astype(np.int8)
        first_i8 = np.clip(
            np.rint((source.astype(np.int32) @ first_weight.astype(np.int32) + first_bias) * first_scale),
            -128, 127).astype(np.int8)
        first_i8 = first_lut[first_i8.view(np.uint8)]
        final_bias_name = inputs[2]
        final_scale_name = inputs[3]
        final_weight = payload.reshape(packed_shape)
        final_bias = constants[final_bias_name].astype(np.int32)
        final_scale = constants[final_scale_name].astype(np.float32)
        final_ref = (
            first_i8.astype(np.int32) @ final_weight.astype(np.int32) + final_bias
        ).astype(np.float32) * final_scale
        if args.reference_only:
            if args.reference_output_dir is not None:
                np.savetxt(
                    args.reference_output_dir / "trace-1001.txt",
                    final_ref.reshape(-1),
                )
            print(f"integer reference classification: {int(np.argmax(final_ref))}")
            return
        actual = np.loadtxt(args.classifier_output_trace, dtype=np.float32)
        if actual.size != 1000:
            raise ValueError(f"classifier output has {actual.size} values, expected 1000")
        actual = actual.reshape(final_ref.shape)
        diff = np.abs(actual.astype(np.float64) - final_ref.astype(np.float64))
        corr = float(np.corrcoef(actual.reshape(-1), final_ref.reshape(-1))[0, 1])
        print(
            f"classifier output: corr={corr:.6f} exact={float(np.mean(actual == final_ref)):.6f} "
            f"mae={np.mean(diff):.6g} max={np.max(diff):.6g}"
        )
        if not np.array_equal(actual, final_ref):
            raise SystemExit("first mismatching classifier stage: 1")


if __name__ == "__main__":
    main()
