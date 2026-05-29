#!/usr/bin/env python3

import argparse
from pathlib import Path


TRACE_DECL = (
    "  func.func private @buddyTraceTensorF32(i64, memref<?xf32>) attributes {llvm.emit_c_interface}\n"
)


CHECKPOINTS = [
    ("input_nchw", 0, "%arg0", "tensor<1x1x28x28xf32>", [1, 1, 28, 28]),
    ("conv1_out_nchw", 1, "%5", "tensor<1x6x24x24xf32>", [1, 6, 24, 24]),
    ("relu1_out_nchw", 2, "%7", "tensor<1x6x24x24xf32>", [1, 6, 24, 24]),
    ("pool1_out_nchw", 3, "%9", "tensor<1x6x12x12xf32>", [1, 6, 12, 12]),
    ("conv2_out_nchw", 4, "%17", "tensor<1x16x8x8xf32>", [1, 16, 8, 8]),
    ("relu2_out_nchw", 5, "%19", "tensor<1x16x8x8xf32>", [1, 16, 8, 8]),
    ("pool2_out_nchw", 6, "%21", "tensor<1x16x4x4xf32>", [1, 16, 4, 4]),
    ("flatten_out", 7, "%25", "tensor<1x256xf32>", [1, 256]),
    ("fc1_out", 8, "%30", "tensor<1x120xf32>", [1, 120]),
    ("relu3_out", 9, "%32", "tensor<1x120xf32>", [1, 120]),
    ("fc2_out", 10, "%37", "tensor<1x84xf32>", [1, 84]),
    ("relu4_out", 11, "%39", "tensor<1x84xf32>", [1, 84]),
    ("fc3_out", 12, "%44", "tensor<1x10xf32>", [1, 10]),
]


def tensor_type(shape: list[int]) -> str:
    return "x".join(str(dim) for dim in shape) + "xf32"


def trace_block(base: str, tag_id: int, value: str, value_type: str, shape: list[int], indent: str) -> str:
    flat = 1
    for dim in shape:
        flat *= dim
    return "\n".join(
        [
            f"{indent}%{base}_tag = arith.constant {tag_id} : i64",
            f"{indent}%{base}_shape = tosa.const_shape  {{values = dense<[{', '.join(str(d) for d in shape)}]> : tensor<{len(shape)}xindex>}} : () -> !tosa.shape<{len(shape)}>",
            f"{indent}%{base}_flat = tosa.reshape {value}, %{base}_shape : ({value_type}, !tosa.shape<{len(shape)}>) -> tensor<{tensor_type(shape)}>",
            f"{indent}%{base}_flat_shape = tosa.const_shape  {{values = dense<[{flat}]> : tensor<1xindex>}} : () -> !tosa.shape<1>",
            f"{indent}%{base}_linear = tosa.reshape %{base}_flat, %{base}_flat_shape : (tensor<{tensor_type(shape)}>, !tosa.shape<1>) -> tensor<{flat}xf32>",
            f"{indent}%{base}_buf = bufferization.to_buffer %{base}_linear : tensor<{flat}xf32> to memref<{flat}xf32>",
            f"{indent}%{base}_cast = memref.cast %{base}_buf : memref<{flat}xf32> to memref<?xf32>",
            f"{indent}func.call @buddyTraceTensorF32(%{base}_tag, %{base}_cast) : (i64, memref<?xf32>) -> ()",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lines = Path(args.input).read_text().splitlines()
    result = []
    inserted_decl = False
    checkpoint_iter = iter(
        [
            ("    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x28x28xf32>) -> tensor<1x28x28x1xf32>", CHECKPOINTS[0], False),
            ("    %5 = tosa.transpose %4 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x24x24x6xf32>) -> tensor<1x6x24x24xf32>", CHECKPOINTS[1], False),
            ("    %7 = tosa.maximum %5, %6 : (tensor<1x6x24x24xf32>, tensor<1x6x24x24xf32>) -> tensor<1x6x24x24xf32>", CHECKPOINTS[2], False),
            ("    %9 = bufferization.to_tensor %alloc restrict : memref<1x6x12x12xf32> to tensor<1x6x12x12xf32>", CHECKPOINTS[3], False),
            ("    %17 = tosa.transpose %16 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x8x8x16xf32>) -> tensor<1x16x8x8xf32>", CHECKPOINTS[4], False),
            ("    %19 = tosa.maximum %17, %18 : (tensor<1x16x8x8xf32>, tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>", CHECKPOINTS[5], False),
            ("    %21 = bufferization.to_tensor %alloc_24 restrict : memref<1x16x4x4xf32> to tensor<1x16x4x4xf32>", CHECKPOINTS[6], False),
            ("    %25 = tosa.reshape %21, %24 : (tensor<1x16x4x4xf32>, !tosa.shape<2>) -> tensor<1x256xf32>", CHECKPOINTS[7], False),
            ("    %30 = tosa.add %29, %27 : (tensor<1x120xf32>, tensor<1x120xf32>) -> tensor<1x120xf32>", CHECKPOINTS[8], False),
            ("    %32 = tosa.maximum %30, %31 : (tensor<1x120xf32>, tensor<1x120xf32>) -> tensor<1x120xf32>", CHECKPOINTS[9], False),
            ("    %37 = tosa.add %36, %34 : (tensor<1x84xf32>, tensor<1x84xf32>) -> tensor<1x84xf32>", CHECKPOINTS[10], False),
            ("    %39 = tosa.maximum %37, %38 : (tensor<1x84xf32>, tensor<1x84xf32>) -> tensor<1x84xf32>", CHECKPOINTS[11], False),
            ("    return %44 : tensor<1x10xf32>", CHECKPOINTS[12], True),
        ]
    )
    current = next(checkpoint_iter, None)

    for line in lines:
        result.append(line)
        if line == "module {" and not inserted_decl:
            result.append(TRACE_DECL.rstrip("\n"))
            inserted_decl = True
        if current and line == current[0]:
            _, checkpoint, before_return = current
            base, tag_id, value, value_type, shape = checkpoint
            block = trace_block(base, tag_id, value, value_type, shape, "    ")
            if before_return:
                result.pop()
                result.extend(block.splitlines())
                result.append(line)
            else:
                result.extend(block.splitlines())
            current = next(checkpoint_iter, None)

    Path(args.output).write_text("\n".join(result) + "\n")


if __name__ == "__main__":
    main()
