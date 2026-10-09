import hashlib
import json
import subprocess
import re
from pathlib import Path

import torch


class Parameters:
    def __init__(self, phase, graph, original, packed, directory):
        records = json.loads(
            (directory / f"{phase}.payload/quant-index.json").read_text()
        )["tensors"]
        by_name = {record["name"]: record for record in records}
        self.entries = []
        original_by_name = original
        for node, value in zip(graph.params, packed):
            record = by_name[node.name]
            name = node.name.removesuffix("_mxfp8")
            source = original_by_name[name]
            resource = (
                "weights.bin"
                if record["payload"] == "weights"
                else f"{phase}/params.f32"
            )
            self.entries.append((source.data_ptr(), value, record, resource))

    def weight(self, tensor, *, matrix):
        matches = [
            entry
            for entry in self.entries
            if entry[0] == tensor.data_ptr()
            and ("tile_k" in entry[2].get("layout", {})) == matrix
            and entry[2]["storage"] == "mxfp8"
        ]
        if len(matches) != 1:
            raise ValueError(
                "Gemma task weight must match one original packed parameter"
            )
        _, packed, record, _ = matches[0]
        if record["shape"] != list(tensor.shape):
            raise ValueError(
                "Gemma task weight shape differs from its original RAX parameter"
            )
        return packed, record["layout"]

    def binding(self, value):
        matches = []
        for _, tensor, record, resource in self.entries:
            if (
                tensor.untyped_storage().data_ptr()
                != value.untyped_storage().data_ptr()
            ):
                continue
            delta = (
                value.storage_offset() - tensor.storage_offset()
            ) * value.element_size()
            size = value.numel() * value.element_size()
            if (
                delta < 0
                or delta + size > record["payload_bytes"]
                or not value.is_contiguous()
            ):
                continue
            matches.append(
                {
                    "resource": resource,
                    "byte_offset": record["payload_offset"] + delta,
                    "shape": list(value.shape),
                    "strides": list(value.stride()),
                    "dtype": str(value.dtype).removeprefix("torch."),
                }
            )
        if len(matches) != 1:
            raise ValueError(
                "Gemma captured task parameter has no unique original RAX byte view"
            )
        return matches[0]


class Exporter:
    def __init__(self, directory, compiler_build, passes, design, isa_directory):
        self.directory = Path(directory) / "tasks"
        self.directory.mkdir(exist_ok=True)
        self.build, self.passes, self.design = Path(compiler_build), passes, design
        self.kernels, self.calls = {}, []
        self.isa_directory = Path(isa_directory)

    def add(
        self,
        phase,
        layer,
        rank,
        stage,
        task,
        inputs,
        slots,
        parameters,
        *,
        target,
        outputs,
        panels=None,
    ):
        from buddy.compiler.frontend import DynamoCompiler
        from buddy.compiler.ops import tosa
        from torch._inductor.decomposition import decompositions

        compiler = DynamoCompiler(
            primary_registry=dict(tosa.ops_registry),
            aot_autograd_decomposition=decompositions,
            func_name="gemma_task",
            verbose=False,
        )
        self.design.entry.register(compiler)
        graph, module = self.design.tasks.capture_task(
            compiler, task, inputs, symbol="gemma_task", prefill=phase == "prefill"
        )
        by_pointer = {value.data_ptr(): slot for value, slot in zip(inputs, slots)}
        if len(by_pointer) != len(inputs):
            raise ValueError("Gemma task capture requires distinct runtime input views")
        runtime = {
            node.name: by_pointer[value.data_ptr()]
            for node, value in zip(graph.inputs, graph._runtime_inputs_ref)
        }
        values = {
            node.name: value
            for node, value in zip(graph.params, compiler.imported_params[graph])
        }
        bindings = [
            dict(runtime[name]) if name in runtime else parameters.binding(values[name])
            for name in graph.task_arguments
        ]
        text = str(module)
        digest = hashlib.sha256((target + text).encode()).hexdigest()[:16]
        symbol = f"gemma_{digest}"
        if symbol not in self.kernels:
            text = text.replace("@gemma_task", "@" + symbol)
            source = self.directory / f"{symbol}.mlir"
            source.write_text(text)
            opt = str(self.build / "bin/buddy-opt")
            log_path = self.directory / f"{symbol}.log"
            with log_path.open("w") as log:
                subprocess.run(
                    [
                        opt,
                        str(source),
                        "-pass-pipeline=builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))",
                        "-o",
                        str(self.directory / f"{symbol}.linalg.mlir"),
                    ],
                    check=True,
                    stderr=log,
                )
                split = self.passes.index("-expand-strided-metadata")
                typed = self.directory / f"{symbol}.bank.mlir"
                subprocess.run(
                    [
                        opt,
                        str(self.directory / f"{symbol}.linalg.mlir"),
                        f"--target={target}",
                        *self.passes[:split],
                        "-o",
                        str(typed),
                    ],
                    check=True,
                    stderr=log,
                )
                final = self.directory / f"{symbol}.llvm.mlir"
                subprocess.run(
                    [
                        opt,
                        str(typed),
                        f"--target={target}",
                        *self.passes[split:],
                        "-o",
                        str(final),
                    ],
                    check=True,
                    stderr=log,
                )
            from .task_abi import export_wrapper

            abi = export_wrapper(symbol, typed, final, self.directory)
            params = (self.isa_directory / target / "params.h").read_text()
            signature = re.search(r"^#define CORE_SIGNATURE ([0-9]+)ULL$", params, re.M)
            if signature is None:
                raise ValueError(
                    f"Gemma task target {target} has no generated core signature"
                )
            self.kernels[symbol] = {
                "symbol": symbol,
                "target": target,
                "core_signature": int(signature[1]),
                **abi,
            }
        if len(outputs) != len(self.kernels[symbol]["outputs"]):
            raise ValueError("Gemma task output slots differ from its real CIF results")
        self.calls.append(
            {
                "kind": "compute",
                "phase": phase,
                "layer": layer,
                "rank": rank,
                "stage": stage,
                "kernel": symbol,
                "bindings": bindings,
                "outputs": list(outputs),
                "output_storage": "workspace-ddr",
                "panels": panels,
            }
        )

    def gather(self, phase, layer, stage, inputs, output, columns):
        if len(inputs) == 1 and len(columns) == 1 and columns[0]["column_start"] == 0:
            source = next(
                event
                for event in reversed(self.calls)
                if event["kind"] == "compute" and inputs[0] in event["outputs"]
            )
            shape = self.kernels[source["kernel"]]["outputs"][
                source["outputs"].index(inputs[0])
            ]["shape"]
            if len(shape) != 3 or shape[2] != columns[0]["columns"]:
                raise ValueError(
                    "Gemma single-source gather must cover every source column"
                )
            self.calls.append(
                {
                    "kind": "view",
                    "phase": phase,
                    "layer": layer,
                    "stage": stage,
                    "input": inputs[0],
                    "output": output,
                    "axis": 2,
                    "start": 0,
                    "count": shape[2],
                }
            )
            return
        self.calls.append(
            {
                "kind": "gather",
                "phase": phase,
                "layer": layer,
                "stage": stage,
                "inputs": inputs,
                "output": output,
                "columns": columns,
            }
        )

    def finish(self, tiles):
        from .task_abi import write_index

        manifest = {
            "tiles": list(tiles),
            "kernels": list(self.kernels.values()),
            "calls": self.calls,
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        symbols = list(self.kernels)
        write_index(self.directory, list(self.kernels.values()))
        (self.directory / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\nproject(GemmaTasks LANGUAGES CXX)\n"
            "set(GEMMA_TASK_OBJECTS)\n"
            + "".join(
                f'add_custom_command(OUTPUT "${{CMAKE_CURRENT_BINARY_DIR}}/{symbol}.o"\n'
                f'  COMMAND bash -o pipefail -c "\\"${{COMPILER_BUILD}}/bin/buddy-translate\\" \\"${{CMAKE_CURRENT_SOURCE_DIR}}/{symbol}.llvm.mlir\\" --buddy-to-llvmir | \\"${{COMPILER_BUILD}}/bin/buddy-llc\\" -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m,+D -float-abi=hard -o \\"${{CMAKE_CURRENT_BINARY_DIR}}/{symbol}.original.o\\""\n'
                f'  COMMAND "${{CMAKE_OBJCOPY}}" --redefine-syms=${{GEMMA_RENAMES}}/{self.kernels[symbol]["target"]}.rename "${{CMAKE_CURRENT_BINARY_DIR}}/{symbol}.original.o" "${{CMAKE_CURRENT_BINARY_DIR}}/{symbol}.o"\n'
                f'  DEPENDS "${{CMAKE_CURRENT_SOURCE_DIR}}/{symbol}.llvm.mlir" "${{GEMMA_RENAMES}}/{self.kernels[symbol]["target"]}.rename" VERBATIM)\n'
                f'list(APPEND GEMMA_TASK_OBJECTS "${{CMAKE_CURRENT_BINARY_DIR}}/{symbol}.o")\n'
                for symbol in symbols
            )
            + "add_library(gemma4_kernels STATIC ${GEMMA_TASK_OBJECTS} task_index.cpp\n"
            + "".join(f"  {symbol}.cpp\n" for symbol in symbols)
            + ")\n"
            + "target_compile_features(gemma4_kernels PRIVATE cxx_std_17)\n"
            + "set_target_properties(gemma4_kernels PROPERTIES LINKER_LANGUAGE CXX)\n"
            + 'target_include_directories(gemma4_kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${DESIGN_DIR}" "${REPO_ROOT}/stack/runtime/include" "${LLVM_MLIR_EXECUTION_ENGINE_DIR}")\n'
        )
        return manifest
