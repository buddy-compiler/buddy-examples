import ctypes
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from buddy_mlir import ir
from buddy_mlir.dialects import func, tensor
from buddy_mlir.execution_engine import ExecutionEngine
from buddy_mlir.runtime import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    ranked_memref_to_numpy,
)
from stack.compiler.frontend.ops import int8

torch.set_num_threads(4)
generator = torch.Generator().manual_seed(7)
for groups in (1, 2, 4):
    x = torch.randint(-4, 5, (1, 4, 6, 8), dtype=torch.int8, generator=generator)
    w = torch.randint(
        -4, 5, (4, 4 // groups, 2, 3), dtype=torch.int8, generator=generator
    )
    expected_nchw = torch.nn.functional.conv2d(
        x.double(),
        w.double(),
        stride=(2, 1),
        padding=(1, 1),
        dilation=(1, 2),
        groups=groups,
    ).float()
    for nhwc in (False, True):
        permutation = [2, 3, 1, 0] if groups < 4 else [0, 2, 3, 1]
        arguments = {
            "x": x.permute(0, 2, 3, 1).contiguous() if nhwc else x,
            "w": w.permute(permutation).contiguous(),
            "scale": torch.ones(1),
            "ws": torch.ones(4),
        }
        arrays = [value.numpy() for value in arguments.values()]
        expected = (
            expected_nchw.permute(0, 2, 3, 1).contiguous() if nhwc else expected_nchw
        )
        node = SimpleNamespace(
            args=[
                "x",
                "w",
                None,
                [2, 1],
                [1, 1],
                [1, 2],
                False,
                [0, 0],
                groups,
                "scale",
                "ws",
            ],
            tensor_meta={
                "shape": expected_nchw.shape,
                "activation_layout": "nhwc" if nhwc else "nchw",
                "weight_permutation": permutation,
            },
        )
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.create()
            types = [
                ir.RankedTensorType.get(
                    array.shape,
                    (
                        ir.IntegerType.get_signless(8)
                        if array.dtype == np.int8
                        else ir.F32Type.get()
                    ),
                )
                for array in arrays
            ]
            with ir.InsertionPoint(module.body):

                @func.FuncOp.from_py_func(*types, name="forward")
                def forward(*values):
                    symbols = {
                        (name, 0): value
                        for name, value in zip(arguments, values, strict=True)
                    }
                    if nhwc:
                        value = tensor.CastOp(values[0].type, values[0]).result
                        value.owner.attributes["buddy.layout"] = ir.StringAttr.get(
                            "nhwc"
                        )
                        symbols[("x", 0)] = value
                    return int8.conv2d(node, symbols)

            assert module.operation.verify()
            original = str(module)
        first = subprocess.run(
            [
                sys.argv[1],
                "-pass-pipeline=builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))",
            ],
            input=original,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        passes = [
            "--load-pass-plugin=" + sys.argv[2],
            "-one-shot-bufferize=bufferize-function-boundaries",
            "-prepare-gemmini-int8",
            "-convert-linalg-to-loops",
            "-expand-strided-metadata",
            "-lower-affine",
            "-convert-scf-to-cf",
            "-llvm-request-c-wrappers",
            "-convert-math-to-llvm",
            "-convert-arith-to-llvm",
            "-convert-cf-to-llvm",
            "-convert-func-to-llvm",
            "-finalize-memref-to-llvm",
            "-reconcile-unrealized-casts",
        ]
        lowered = subprocess.run(
            [sys.argv[1], *passes],
            input=first,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        assert "buddy.int8_conv" not in lowered
        with ir.Context():
            engine = ExecutionEngine(
                ir.Module.parse(lowered), opt_level=2, shared_libs=[sys.argv[3]]
            )
        result = make_nd_memref_descriptor(4, ctypes.c_float)()
        descriptors = [get_ranked_memref_descriptor(array) for array in arrays]
        engine.invoke(
            "forward",
            ctypes.pointer(ctypes.pointer(result)),
            *[ctypes.pointer(ctypes.pointer(value)) for value in descriptors]
        )
        actual = ranked_memref_to_numpy(ctypes.pointer(result)).copy()
        np.testing.assert_array_equal(actual, expected.numpy())
        free = ctypes.CDLL(None).free
        free.argtypes = [ctypes.c_void_p]
        free(result.allocated)
print(
    "PASS Gemmini INT8 preparation matches independent convolutions for grouped/depthwise and NCHW/NHWC layouts"
)
