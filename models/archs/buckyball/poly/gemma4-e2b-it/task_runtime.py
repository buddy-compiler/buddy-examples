import argparse
import subprocess
from pathlib import Path


def build_instances(repo, output, isa, images, compiler, ar, nm, objcopy):
    source = repo / "bb-tests/workloads/lib/bbsw/kernels/rvv"
    sources = (
        "dispatch.cpp",
        "matmul_dispatch.cpp",
        "cache_dispatch.cpp",
        "normalization_dispatch.cpp",
        "pointwise_dispatch.cpp",
        "activation_dispatch.cpp",
        "flash_attention_dispatch.cpp",
        "flash_attention_mxfp8_dispatch.cpp",
    )
    output.mkdir(parents=True, exist_ok=True)
    for target in ("ffn", "attention"):
        directory = output / target
        directory.mkdir(exist_ok=True)
        objects, symbols = [], set()
        for filename in sources:
            obj = directory / (filename + ".o")
            subprocess.run(
                [
                    str(compiler),
                    "-std=c++20",
                    "-O2",
                    "-ffp-contract=off",
                    "-DBUCKYBALL_ANT_HOST=1",
                    "-c",
                    str(source / filename),
                    "-I" + str(images),
                    "-I" + str(isa / target),
                    "-I" + str(repo / "bb-tests/workloads/lib"),
                    "-I" + str(repo / "stack/runtime/include"),
                    "-I"
                    + str(
                        repo
                        / "stack/compiler/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine"
                    ),
                    "-o",
                    str(obj),
                ],
                check=True,
            )
            text = subprocess.check_output(
                [str(nm), "--defined-only", str(obj)], text=True
            )
            symbols.update(
                line.split()[-1]
                for line in text.splitlines()
                if line.split() and line.split()[-1].startswith("_mlir_ciface_rvv_")
            )
            objects.append(obj)
        rename = output / (target + ".rename")
        rename.write_text(
            "".join(f"{name} {name}__{target}\n" for name in sorted(symbols))
        )
        renamed = []
        for obj in objects:
            destination = obj.with_suffix(".target.o")
            subprocess.run(
                [
                    str(objcopy),
                    "--redefine-syms=" + str(rename),
                    str(obj),
                    str(destination),
                ],
                check=True,
            )
            renamed.append(destination)
        subprocess.run(
            [
                str(ar),
                "rcs",
                str(output / f"libgemma_rvv_{target}.a"),
                *[str(obj) for obj in renamed],
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("repo", "output", "isa", "images", "compiler", "ar", "nm", "objcopy"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build_instances(**vars(args))
