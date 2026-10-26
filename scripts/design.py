"""Write what a CMake build needs from design.pb: ISA headers and design.cmake.

usage: design.py --design-pb <design.pb> --out <dir>
  <dir>/isa/<target>/params.h   bank geometry and THREAD_SIGNATURE of one compiler target
  <dir>/isa/<target>/ballISA.h  funct7 of every Ball instruction of that target
  <dir>/isa/<target>/target.inc the target table the Buckyball compiler is built with
  <dir>/isa/design/instances.h    cores of the design
  <dir>/isa/control/params.h    the hart that controls tests
  <dir>/design.cmake              lists of targets, threads and cores for CMake
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bbdev/api/steps/config/scripts"))
import design_pb2  # noqa: E402


def load(design_pb) -> "design_pb2.Design":
    design = design_pb2.Design()
    design.ParseFromString(Path(design_pb).read_bytes())
    return design


def target(design, name: str):
    """The compiler target (profile) called `name`; its `config` is the thread it compiles for."""
    return next(profile for profile in design.profiles if profile.name == name)


def stack_build(design, selection) -> Path:
    """Where `bbdev stack --build` builds the stacks of `design`, one directory per level."""
    return REPO / "examples/designs" / design.name / "output/sims" / selection / "stack"


def core_component(design, core) -> str:
    """Cores are named by their component; one component has one configuration per design."""
    same = {c.signature for c in design.device.cores if c.component == core.component}
    if len(same) != 1:
        raise ValueError(
            f"{design.name}: core {core.component} has {len(same)} configurations"
        )
    return core.component


def opt(design, name: str, selection: str) -> list:
    """buddy-opt with the plugin of compiler target `name`, and the engine of the
    cores that run it on hardware threads."""
    engines = sorted(
        {
            core_component(design, core)
            for core in design.device.cores
            for index in core.thread_indices
            if design.threads[index].profile == name
        }
    )
    return [
        str(REPO / "stack/thirdparty/buddy-mlir/build/bin/buddy-opt"),
        f"--load-dialect-plugin={plugin(design, name, selection)}",
        f"--load-pass-plugin={plugin(design, name, selection)}",
        *(
            f"--load-pass-plugin={stack_build(design, selection) / 'core' / engine / 'libengine-compiler.so'}"
            for engine in engines
        ),
    ]


def core_stack(design, core, selection) -> Path:
    if core.component == design.name:
        return REPO / design.simulations[selection].stack / "core" / core.component
    return REPO / "examples/components/cores" / core.component / "stack"


def stacks(design, selection) -> list:
    """(level, name, source, build) of every stack of `design`, in build order:
    threads and cores first, the device last."""
    result = []
    for name in dict.fromkeys(thread.profile for thread in design.threads):
        thread = next(thread for thread in design.threads if thread.profile == name)
        result.append(
            (
                "thread",
                name,
                REPO / thread.source_path / "stack",
                stack_build(design, selection) / "thread" / name,
            )
        )
    for core in {
        core_component(design, core): core for core in design.device.cores
    }.values():
        name = core_component(design, core)
        result.append(
            (
                "core",
                name,
                core_stack(design, core, selection),
                stack_build(design, selection) / "core" / name,
            )
        )
    result.append(
        (
            "device",
            design.name,
            REPO / design.simulations[selection].stack / "runtime",
            stack_build(design, selection) / "device",
        )
    )
    return result


def plugin(design, name: str, selection: str) -> Path:
    """The buddy-opt plugin of compiler target `name`, built by the stack that owns it."""
    if any(thread.profile == name for thread in design.threads):
        return (
            stack_build(design, selection) / "thread" / name / f"lib{name}-compiler.so"
        )
    core = next(core for core in design.device.cores if core.profile == name)
    return (
        stack_build(design, selection)
        / "core"
        / core_component(design, core)
        / f"lib{name}-compiler.so"
    )


def bank_bytes(design_pb, name: str) -> int:
    bank = target(load(design_pb), name).config.mem.bank
    return bank.width * bank.entries // 8


def write(path: Path, text: str) -> None:
    # Unchanged files keep their timestamps so dependent objects are not rebuilt.
    if path.is_file() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def isqrt(n: int) -> int:
    x = n
    while (x + n // x) // 2 < x:
        x = (x + n // x) // 2
    return x


def owners(design, target: str) -> list:
    return [
        core
        for core in design.device.cores
        if core.profile == target
        or any(design.threads[i].profile == target for i in core.thread_indices)
    ]


def params(design, profile) -> str:
    thread = profile.config
    bank, mmio = thread.mem.bank, thread.mem.mmio
    cores = owners(design, profile.name)
    first = cores[0]
    shared = first.shared_mem
    test_hart = (
        profile.test_core
        if profile.HasField("test_core")
        else first.hart_id if len(cores) == 1 else None
    )
    lines = [
        "/* Generated from design.pb. Do not edit. */",
        "#ifndef BBHW_PARAMS_H",
        "#define BBHW_PARAMS_H",
        f"#define THREAD_SIGNATURE {thread.signature}ULL",
        *([f"#define BB_TEST_HART {test_hart}"] if test_hart is not None else []),
        f"#define BANK_NUM {bank.num}",
        f"#define SHARED_BANK_NUM {shared.entries // shared.bank_entries if shared.enable else 0}",
        f"#define BB_SHARED_BANK_LINES {shared.bank_entries}",
        f"#define VIRTUAL_BANK_NUM {first.virtual_bank_count}",
        f"#define BB_PRIVATE_VBANK_MAX {thread.frontend.vbank_id_upper_bound}",
        f"#define BB_SHARED_BANK_BASE {thread.frontend.shared_bank_id_base}",
        f"#define BANK_WIDTH {bank.width}",
        f"#define BANK_LINES {bank.entries}",
        f"#define BANK_ISQRT {isqrt(bank.entries)}",
        f"#define MMIO_BANK_NUM {mmio.bank_num}",
        f"#define MMIO_BANK_ENTRIES {mmio.bank_entries}",
        f"#define MMIO_BANK_WIDTH_BITS {mmio.bank_width}",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def ball_isa(profile) -> str:
    balls = profile.config.balldomain.balls
    counts = {}
    for ball in balls:
        for entry in ball.instructions:
            counts[entry.mnemonic] = counts.get(entry.mnemonic, 0) + 1
    lines = ["#ifndef BALL_ISA_H", "#define BALL_ISA_H"]
    for ball in balls:
        for entry in ball.instructions:
            lines.append(
                f"#define BB_FUNC7_BALL_{ball.ball_id}_{entry.mnemonic} {entry.funct7}"
            )
            if counts[entry.mnemonic] == 1:
                lines.append(f"#define BB_FUNC7_{entry.mnemonic} {entry.funct7}")
    lines += ["#endif", ""]
    return "\n".join(lines)


def target_table(thread) -> str:
    """BuckyballTargetConfig of one compiler target: banks, Balls and their ISA."""
    from ball_parameters import unpack_parameters

    def text(value: str) -> str:
        if not value or '"' in value or "\\" in value:
            raise ValueError(f"unsupported C++ string in design.pb: {value!r}")
        return f'"{value}"'

    lines = ["// Generated from design.pb. Do not edit.", ""]

    def array(type_name: str, name: str, items: list) -> str:
        if not items:
            return f"llvm::ArrayRef<{type_name}>()"
        lines.append(f"static const {type_name} {name}[] = {{")
        lines.extend(f"  {item}," for item in items)
        lines.append("};")
        return f"llvm::ArrayRef({name})"

    rows = []
    for index, ball in enumerate(thread.balldomain.balls):
        values = sorted(
            (field.name, value)
            for field, value in unpack_parameters(ball).ListFields()
            if type(value) is int
        )
        params = array(
            "buckyball_target::BuckyballBallParam",
            f"kBall{index}Params",
            [f"{{{text(name)}, {value}}}" for name, value in values],
        )
        rows.append(
            f"{{{text(ball.ball_name)}, {ball.in_bw}, {ball.out_bw}, {params}}}"
        )
    mappings = array("buckyball_target::BuckyballBallMapping", "kBallMappings", rows)
    isa = array(
        "buckyball_target::BuckyballIsaEntry",
        "kIsa",
        [
            f"{{{text(entry.mnemonic)}, {entry.funct7}, {index}}}"
            for index, ball in enumerate(thread.balldomain.balls)
            for entry in ball.instructions
        ],
    )
    bank = thread.mem.bank
    rvv = "true" if thread.gpdomain.enable else "false"
    lines.append(
        "static const buckyball_target::BuckyballTargetConfig kBuckyballTarget = {"
        f"{text(thread.pkg)}, {bank.num}, {bank.width}, {bank.entries}, "
        f"{thread.frontend.vbank_id_upper_bound}, {rvv}, {mappings}, {isa}}};"
    )
    return "\n".join(lines) + "\n"


def instances(design) -> str:
    targets = {profile.name: index for index, profile in enumerate(design.profiles)}
    rows = ", ".join(
        f"{{{core.hart_id}u, {targets[core.profile] if core.profile else 0xFFFFFFFF}u, {core.signature}ull}}"
        for core in design.device.cores
    )
    thread_cores = ", ".join(
        f"{index}u"
        for index, core in enumerate(design.device.cores)
        if core.thread_indices
    )
    lines = [
        "#ifndef BBHW_INSTANCES_H",
        "#define BBHW_INSTANCES_H",
        f"#define BB_REPORT_HART {design.workload.report_hart}",
        f"#define BB_CORE_NUM {len(design.device.cores)}",
        f"#define BB_HART_NUM {len(design.device.cores)}",
        f"#define BB_THREAD_CORE_COUNT {sum(bool(core.thread_indices) for core in design.device.cores)}",
        "#ifndef __ASSEMBLER__",
        "#include <stdint.h>",
        "typedef struct { uint32_t hart, profile; uint64_t signature; } bb_core_instance;",
        f"static const bb_core_instance bb_cores[BB_CORE_NUM] = {{{rows}}};",
        "/* Where the model image sits in the DDR. */",
        f"#define BB_MODEL_BASE 0x{design.device.ddr.base + design.device.ddr.bytes - design.device.ddr.model:x}ull",
        "/* The cores with hardware threads, in core order. */",
        f"static const uint32_t bb_thread_cores[] = {{{thread_cores}}};",
        "static inline uint32_t bb_hart_slot(uint32_t hart) {",
        "  for (uint32_t i = 0; i < BB_CORE_NUM; ++i) if (bb_cores[i].hart == hart) return i;",
        "  __builtin_trap();",
        "}",
        "static inline uint32_t bb_profile_core_count(uint32_t profile) {",
        "  uint32_t count = 0;",
        "  for (uint32_t i = 0; i < BB_CORE_NUM; ++i) count += bb_cores[i].profile == profile;",
        "  return count;",
        "}",
        "static inline uint32_t bb_hart_profile(uint32_t hart) {",
        "  for (uint32_t i = 0; i < BB_CORE_NUM; ++i) if (bb_cores[i].hart == hart) return bb_cores[i].profile;",
        "  return UINT32_MAX;",
        "}",
        "#endif",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def cmake(design, isa: Path, selection: str) -> str:
    lines = ["# Generated from design.pb. Do not edit."]

    def put(name, *values):
        lines.append(f"set({name} {' '.join(f'\"{value}\"' for value in values)})")

    put("DESIGN_ISA_DIR", isa)
    put("DESIGN_TARGETS", *(profile.name for profile in design.profiles))
    put(
        "DESIGN_BANK_TARGETS",
        *(profile.name for profile in design.profiles if profile.bank_num > 0),
    )
    put(
        "DESIGN_CPU_TARGETS",
        *(
            p.name
            for p in design.profiles
            if any(core.profile == p.name for core in design.device.cores)
        ),
    )
    put(
        "DESIGN_RUNTIME_TARGETS",
        *(
            p.name
            for p in sorted(design.profiles, key=lambda p: p.bank_entries)
            if p.config.balldomain.balls
        ),
    )
    for index, profile in enumerate(design.profiles):
        thread = profile.config
        balls = [ball.ball_dir for ball in thread.balldomain.balls]
        kernels = [REPO / thread.source_path / "stack/kernels"]
        kernels += [
            REPO / "examples/components/balls" / ball / "stack/kernels"
            for ball in balls
        ]
        shared = owners(design, profile.name)[0].shared_mem
        put(f"DESIGN_TARGET_ID_{profile.name}", index)
        put(
            f"DESIGN_TARGET_SHARED_BANKS_{profile.name}",
            shared.entries // shared.bank_entries if shared.enable else 0,
        )
        put(f"DESIGN_TARGET_BALLS_{profile.name}", *balls)
        put(
            f"DESIGN_TARGET_KERNELS_{profile.name}",
            *(k for k in kernels if (k / "CMakeLists.txt").is_file()),
        )
        put(
            f"DESIGN_TARGET_WORKLOADS_{profile.name}",
            REPO / thread.source_path / "workloads",
        )
        put(
            f"DESIGN_TARGET_BALL_STACKS_{profile.name}",
            *dict.fromkeys(
                REPO / "examples/components/balls" / ball / "stack/src"
                for ball in balls
            ),
        )
        put(
            f"DESIGN_TARGET_PLUGIN_{profile.name}",
            plugin(design, profile.name, selection),
        )
        put(f"DESIGN_TARGET_OPT_{profile.name}", *opt(design, profile.name, selection))

    put("DESIGN_REPO", REPO)
    put("DESIGN_COMPILER", REPO / "stack/thirdparty/buddy-mlir/build")
    put("DESIGN_MLIR", REPO / "stack/thirdparty/buddy-mlir/llvm/build/lib/cmake/mlir")
    levels = stacks(design, selection)
    device = levels[-1]
    put("DESIGN_DEVICE_STACK", device[2])
    put("DESIGN_DEVICE_BUILD", device[3])

    put("DESIGN_THREADS", *(name for level, name, _, _ in levels if level == "thread"))
    for level, name, source, build in levels:
        if level != "thread":
            continue
        thread = next(thread for thread in design.threads if thread.profile == name)
        put(f"DESIGN_THREAD_STACK_{name}", source)
        put(f"DESIGN_THREAD_BUILD_{name}", build)
        put(
            f"DESIGN_THREAD_BALLS_{name}",
            *(
                REPO / "examples/components/balls" / ball.ball_dir / "workloads"
                for ball in thread.balldomain.balls
            ),
        )

    put("DESIGN_CORES", *(name for level, name, _, _ in levels if level == "core"))
    for level, name, source, build in levels:
        if level != "core":
            continue
        core = next(core for core in design.device.cores if core.component == name)
        put(f"DESIGN_CORE_SIGNATURE_{name}", core.signature)
        put(f"DESIGN_CORE_STACK_{name}", source)
        put(f"DESIGN_CORE_BUILD_{name}", build)
        put(f"DESIGN_CORE_TARGET_{name}", *([core.profile] if core.profile else []))
        put(
            f"DESIGN_CORE_THREADS_{name}",
            *dict.fromkeys(
                design.threads[index].profile for index in core.thread_indices
            ),
        )

    put("DESIGN_CONTROL_HART", design.workload.control_hart)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--design-pb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    generate(load(args.design_pb), args.out, args.target)


def generate(design, out: Path, selection: str) -> None:
    isa = out / "isa"
    write(isa / "design/instances.h", instances(design))
    write(
        isa / "control/params.h",
        f"#pragma once\n#define BB_TEST_HART {design.workload.control_hart}\n",
    )
    for profile in design.profiles:
        write(isa / profile.name / "params.h", params(design, profile))
        write(isa / profile.name / "ballISA.h", ball_isa(profile))
        write(isa / profile.name / "target.inc", target_table(profile.config))
    write(out / "design.cmake", cmake(design, isa.resolve(), selection))


if __name__ == "__main__":
    main()
