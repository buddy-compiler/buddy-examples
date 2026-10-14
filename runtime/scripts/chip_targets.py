"""Chip-level views of Chip.pb for the runtime: ISA headers, per-target compilers, thread signatures."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _die(message: str) -> None:
    raise ValueError(message)


def signature(thread) -> int:
    data = bytearray(thread.pkg.encode() + b"\0")
    for value in (thread.mem.bank.num, thread.mem.bank.width, thread.mem.bank.entries):
        data.extend(value.to_bytes(8, "little"))
    for entry in sorted(thread.balldomain.isa, key=lambda entry: entry.funct7):
        data.extend(entry.mnemonic.encode() + b"\0")
        data.extend(entry.funct7.to_bytes(4, "little"))
    for entry in sorted(
        thread.balldomain.mappings,
        key=lambda entry: entry.ball_class.rsplit(".", 1)[-1],
    ):
        data.extend(entry.ball_class.rsplit(".", 1)[-1].encode() + b"\0")
        for value in (entry.in_bw, entry.out_bw):
            data.extend(value.to_bytes(8, "little"))
        for name, value in sorted(entry.ball_params.items()):
            data.extend(name.encode() + b"\0" + value.encode() + b"\0")
    if thread.cpu.kind == "ant":
        ant = thread.cpu.ant
        data.extend(b"ant\0")
        for value in (
            ant.code_bytes,
            ant.tls.base,
            ant.tls.bytes,
            ant.tls.data_bits,
            ant.task_bits,
        ):
            data.extend(value.to_bytes(8, "little"))
    result = 0xCBF29CE484222325
    for value in data:
        result = ((result ^ value) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return result


def load_chip(chip_pb: str | Path):
    chip = _load_proto(REPO).Chip()
    chip.ParseFromString(Path(chip_pb).read_bytes())
    return chip


def target_thread(chip, target: str):
    profiles = [profile for profile in chip.profiles if profile.name == target]
    if len(profiles) != 1:
        _die(f"Chip.pb has no compiler target {target!r}")
    return _profile_thread(chip, profiles[0])


def compiler_dir(thread) -> Path:
    config = Path(thread.config_path)
    return (
        REPO / "stack/compiler/thirdparty/buddy-mlir/build" / thread.pkg / config.stem
    )


def bank_bytes(chip_pb: str | Path, target: str) -> int:
    bank = target_thread(load_chip(chip_pb), target).mem.bank
    return bank.width * bank.entries // 8


def _write(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_proto(repo: Path):
    sys.path.insert(0, str(repo / "bbdev" / "api" / "steps" / "config" / "scripts"))
    import chip_pb2

    return chip_pb2


def _validate_ball(entry) -> None:
    if (
        not entry.ball_dir
        or Path(entry.ball_dir).is_absolute()
        or len(Path(entry.ball_dir).parts) != 1
    ):
        _die(f"external Ball {entry.ball_name}: invalid ball_dir {entry.ball_dir!r}")
    if entry.in_bw <= 0 or entry.out_bw <= 0:
        _die(f"external Ball {entry.ball_name}: bandwidth must be positive")


def _target_name(thread) -> str:
    return thread.role or thread.pkg


def _profile_thread(chip, profile):
    matches = [
        thread for thread in chip.threads if _target_name(thread) == profile.name
    ]
    if not matches:
        _die(f"profile {profile.name}: no matching ThreadInstance")
    first = matches[0]
    expected_isa = [
        (entry.mnemonic, entry.funct7, entry.bid) for entry in first.balldomain.isa
    ]
    expected_balls = [entry.ball_name for entry in first.balldomain.mappings]
    expected_bank = (first.mem.bank.num, first.mem.bank.width, first.mem.bank.entries)
    profile_bank = (profile.bank_num, profile.bank_width, profile.bank_entries)
    if expected_bank != profile_bank:
        _die(
            f"profile {profile.name}: profile bank={profile_bank} disagrees with ThreadInstance bank={expected_bank}"
        )
    for thread in matches[1:]:
        isa = [
            (entry.mnemonic, entry.funct7, entry.bid) for entry in thread.balldomain.isa
        ]
        balls = [entry.ball_name for entry in thread.balldomain.mappings]
        bank = (thread.mem.bank.num, thread.mem.bank.width, thread.mem.bank.entries)
        if isa != expected_isa or balls != expected_balls or bank != expected_bank:
            _die(f"profile {profile.name}: ThreadInstance configuration mismatch")
    return first


def _validate_profile(profile, thread) -> None:
    if not profile.name.isidentifier():
        _die(f"profile name is not a C++ identifier: {profile.name!r}")
    has_balls = len(thread.balldomain.mappings) > 0
    if has_balls and (
        profile.bank_num == 0 or profile.bank_width == 0 or profile.bank_entries == 0
    ):
        _die(f"profile {profile.name}: bank geometry must be non-zero")
    for entry in thread.balldomain.mappings:
        _validate_ball(entry)
    balls = [entry.ball_name for entry in thread.balldomain.mappings]
    if len(set(balls)) != len(balls):
        _die(f"profile {profile.name}: duplicate Ball mapping")
    mnemonics: set[str] = set()
    funct7s: set[int] = set()
    for entry in thread.balldomain.isa:
        if entry.funct7 == 15 or entry.mnemonic == "RUN_KERNEL":
            _die("RUN_KERNEL is a global control instruction, not a Ball instruction")
        if not entry.mnemonic.isidentifier():
            _die(f"profile {profile.name}: invalid ISA mnemonic {entry.mnemonic!r}")
        if entry.mnemonic in mnemonics:
            _die(f"profile {profile.name}: duplicate ISA mnemonic {entry.mnemonic}")
        if entry.funct7 in funct7s:
            _die(f"profile {profile.name}: duplicate funct7 {entry.funct7}")
        mnemonics.add(entry.mnemonic)
        funct7s.add(entry.funct7)


def _isqrt(n: int) -> int:
    if n <= 0:
        _die(f"isqrt of non-positive {n}")
    x = n
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y


def _emit_params_header(
    profile, thread, virtual_bank_num: int, isa_dir: Path, chip
) -> None:
    bank = thread.mem.bank
    mmio = thread.mem.mmio
    if bank.num == 0 or bank.width == 0 or bank.entries == 0:
        _die(f"profile {profile.name}: bank geometry must be non-zero")
    if bank.width % 8 != 0:
        _die(f"profile {profile.name}: bank.width must be a multiple of 8")
    if (
        virtual_bank_num < bank.num
        or virtual_bank_num <= thread.frontend.vbank_id_upper_bound
    ):
        _die(f"profile {profile.name}: invalid virtual bank count {virtual_bank_num}")
    if mmio.bank_num == 0 or mmio.bank_entries == 0 or mmio.bank_width == 0:
        _die(f"profile {profile.name}: mmio geometry must be non-zero")
    if mmio.bank_width % 8 != 0:
        _die(f"profile {profile.name}: mmio.bank_width must be a multiple of 8")
    row_bytes = bank.width // 8
    mmio_bytes = mmio.bank_num * mmio.bank_entries * (mmio.bank_width // 8)
    if mmio_bytes % row_bytes != 0:
        _die(
            f"profile {profile.name}: MMIO bytes {mmio_bytes} is not a multiple of SRAM row bytes {row_bytes}"
        )
    core = next(core for core in chip.cores if thread.index in core.thread_indices)
    shared = core.shared_mem
    shared_banks = shared.entries // shared.bank_entries if shared.enable else 0
    lines = [
        "/* Generated from Chip.pb. Do not edit. */",
        "#ifndef BBHW_PARAMS_H",
        "#define BBHW_PARAMS_H",
        "",
        f"#define THREAD_SIGNATURE {signature(thread)}ULL",
        *(
            [f"#define BB_TEST_HART {thread.hart_id}"]
            if thread.HasField("hart_id")
            else [f"#define BB_ANT_CONTEXT {thread.ant_context_id}"]
        ),
        f"#define BANK_NUM {bank.num}",
        f"#define SHARED_BANK_NUM {shared_banks}",
        f"#define VIRTUAL_BANK_NUM {virtual_bank_num}",
        f"#define BB_PRIVATE_VBANK_MAX {thread.frontend.vbank_id_upper_bound}",
        f"#define BB_SHARED_BANK_BASE {thread.frontend.shared_bank_id_base}",
        f"#define BANK_WIDTH {bank.width}",
        f"#define BANK_LINES {bank.entries}",
        f"#define BANK_ISQRT {_isqrt(bank.entries)}",
        f"#define MMIO_BANK_NUM {mmio.bank_num}",
        f"#define MMIO_BANK_ENTRIES {mmio.bank_entries}",
        f"#define MMIO_BANK_WIDTH_BITS {mmio.bank_width}",
        "",
        "#endif",
        "",
    ]
    _write(isa_dir / profile.name / "params.h", "\n".join(lines))


def _emit_isa_headers(chip, isa_dir: Path) -> None:
    """Emit one C ISA header per compiler target.

    A header describes one thread ISA only.  Deliberately do not create a merged
    chip header: funct7 is target-local and may overlap across profiles.
    """
    if not chip.cores or not chip.cores[0].thread_indices:
        _die("chip topology has no core threads")
    if chip.n_cores != len(chip.cores):
        _die(
            f"chip n_cores={chip.n_cores} disagrees with core placements={len(chip.cores)}"
        )
    # Core 0 is the main core; compute cores follow and are homogeneous among themselves.
    # Bank macros describe the compute cores when present, otherwise the main core. A chip whose
    # main core also gains Buckyball banks with a different geometry needs per-kind macros.
    work_core = 1 if len(chip.cores) > 1 else 0
    bank_core = chip.cores[work_core]
    virtual_bank_num = bank_core.virtual_bank_count

    def threads(core):
        for index in core.thread_indices:
            if index >= len(chip.threads):
                _die(f"core Thread index {index} out of range (n={len(chip.threads)})")
        return [
            (chip.threads[index].role, chip.threads[index].pkg)
            for index in core.thread_indices
        ]

    profile_ids = {profile.name: index for index, profile in enumerate(chip.profiles)}
    if len(profile_ids) != len(chip.profiles):
        _die("chip has duplicate compiler profile names")
    compute = list(chip.cores)[1:]
    for core_id, core in enumerate(compute, start=1):
        if threads(core) != threads(compute[0]):
            _die(f"core {core_id}: compute cores are not homogeneous")
        if core.virtual_bank_count != compute[0].virtual_bank_count:
            _die(f"core {core_id}: virtual bank count differs from core 1")
        if core.shared_mem != compute[0].shared_mem:
            _die(f"core {core_id}: shared memory config differs from core 1")

    # Ant contexts are execution resources, not system harts.
    hart_num = sum(thread.HasField("hart_id") for thread in chip.threads)
    threads_per_core = max(len(core.thread_indices) for core in chip.cores)
    hart_core = [None] * hart_num
    hart_thread = [None] * hart_num
    slot_profiles = []
    for core_id, core in enumerate(chip.cores):
        row = []
        for thread_id, thread_index in enumerate(core.thread_indices):
            thread = chip.threads[thread_index]
            if thread.cpu.kind == "ant":
                if thread.HasField("hart_id") or not thread.HasField("ant_context_id"):
                    _die(
                        f"thread {thread_index}: Ant requires context ID and forbids hart ID"
                    )
            else:
                if not thread.HasField("hart_id") or thread.hart_id >= hart_num:
                    _die(f"thread {thread_index}: missing or out-of-range hart_id")
                hart_core[thread.hart_id] = core_id
                hart_thread[thread.hart_id] = thread_id
            target = _target_name(thread)
            if target not in profile_ids:
                _die(
                    f"core {core_id} thread {thread_id} target {target!r} has no compiler profile"
                )
            row.append(profile_ids[target])
        slot_profiles.append(row + [0xFFFFFFFF] * (threads_per_core - len(row)))
    if None in hart_core:
        _die("chip CPU hart ids are not exactly 0..harts-1")
    report_hart = chip.threads[bank_core.thread_indices[0]].hart_id

    shared = bank_core.shared_mem
    if shared.enable:
        endpoints = [
            chip.threads[index]
            for index in bank_core.thread_indices
            if chip.threads[index].balldomain.mappings
        ]
        if not endpoints:
            _die("shared memory requires a compute endpoint")
        shared_bank_entries = shared.bank_entries
        if (
            shared_bank_entries == 0
            or shared.entries == 0
            or shared.entries % shared_bank_entries
        ):
            _die(
                "bank core: shared entries must be a non-zero multiple of compute bank depth"
            )
        shared_physical_bank_num = shared.entries // shared_bank_entries
    else:
        shared_physical_bank_num = 0
    profile_rows = ", ".join(
        "{" + ", ".join(f"{p}u" for p in row) + "}" for row in slot_profiles
    )
    lines = [
        "#ifndef BBHW_TOPOLOGY_H",
        "#define BBHW_TOPOLOGY_H",
        f"#define BB_CORE_NUM {len(chip.cores)}",
        f"#define BB_HART_NUM {hart_num}",
        "// Hart layout: main core threads are harts 0..BB_MAIN_THREADS-1; compute core t (1-based)",
        "// has its controller at BB_MAIN_THREADS+t-1. Ant contexts have no hart ID.",
        f"#define BB_MAIN_THREADS {len(chip.cores[0].thread_indices)}",
        f"#define BB_COMPUTE_CORES {len(chip.cores) - 1}",
        f"#define BB_COMPUTE_THREADS {len(chip.cores[1].thread_indices) if len(chip.cores) > 1 else 0}",
        "// Largest core; core 0 is the main core and compute cores follow. Bank counts below",
        "// describe the compute cores when present, otherwise the main core.",
        f"#define BB_THREADS_PER_CORE {threads_per_core}",
        "// Bare-metal tests run on the work core, the core the bank counts describe; its thread 0",
        "// reports the result. Ant workloads use the local task management interface.",
        f"#define BB_WORK_CORE {work_core}",
        f"#define BB_REPORT_HART {report_hart}",
        f"#define BB_VIRTUAL_BANK_NUM {virtual_bank_num}",
        f"#define BB_SHARED_PHYSICAL_BANK_NUM {shared_physical_bank_num}",
        f"#define BB_SHARED_BANK_LINES {shared.bank_entries}",
        "#ifndef __ASSEMBLER__",
        "#include <stdint.h>",
        "typedef struct { uint32_t core; uint32_t thread; } thread_id_t;",
        "static inline thread_id_t bb_topology_thread_id(uint32_t hart) {",
        f"  static const uint32_t cores[{hart_num}] = {{{', '.join(map(str, hart_core))}}};",
        f"  static const uint32_t threads[{hart_num}] = {{{', '.join(map(str, hart_thread))}}};",
        f"  if (hart >= {hart_num}u) __builtin_trap();",
        "  return (thread_id_t){cores[hart], threads[hart]};",
        "}",
        "static inline uint32_t bb_topology_thread_profile(thread_id_t id) {",
        f"  static const uint32_t profiles[{len(chip.cores)}][{threads_per_core}] = {{{profile_rows}}};",
        f"  if (id.core >= {len(chip.cores)}u || id.thread >= {threads_per_core}u ||",
        "      profiles[id.core][id.thread] == 0xFFFFFFFFu) __builtin_trap();",
        "  return profiles[id.core][id.thread];",
        "}",
        "static inline uint32_t bb_topology_profile_threads_per_core(uint32_t profile) {",
        "  uint32_t count = 0;",
        f"  for (uint32_t thread = 0; thread < {len(bank_core.thread_indices)}u; ++thread)",
        "    count += bb_topology_thread_profile((thread_id_t){BB_WORK_CORE, thread}) == profile;",
        "  return count;",
        "}",
        "#endif",
        "#endif",
        "",
    ]
    _write(isa_dir / "chip" / "topology.h", "\n".join(lines))
    for profile in chip.profiles:
        thread = _profile_thread(chip, profile)
        _validate_profile(profile, thread)
        (isa_dir / profile.name / "topology.h").unlink(missing_ok=True)
        lines = ["#ifndef BALL_ISA_H", "#define BALL_ISA_H", ""]
        lines.extend(
            (
                f"#define BB_FUNC7_{entry.mnemonic} {entry.funct7}"
                for entry in thread.balldomain.isa
            )
        )
        lines.extend(["", "#endif", ""])
        header = isa_dir / profile.name / "ballISA.h"
        _write(header, "\n".join(lines))
        core = next(core for core in chip.cores if thread.index in core.thread_indices)
        _emit_params_header(profile, thread, core.virtual_bank_count, isa_dir, chip)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--chip-pb", type=Path, required=True)
    parser.add_argument("--isa-dir", type=Path)
    parser.add_argument("--print-targets", action="store_true")
    parser.add_argument("--print-compiler-dirs", action="store_true")
    parser.add_argument("--print-runtime-targets", action="store_true")
    parser.add_argument("--print-thread-runtime-dirs", action="store_true")
    parser.add_argument("--print-cpu-targets", action="store_true")
    parser.add_argument("--print-bank-targets", action="store_true")
    parser.add_argument("--print-target-balls", action="store_true")
    parser.add_argument("--print-thread-workload-targets", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    chip = load_chip(args.chip_pb.resolve())
    if args.isa_dir:
        _emit_isa_headers(chip, args.isa_dir)
    if args.print_targets:
        for profile in chip.profiles:
            print(profile.name)
    if args.print_compiler_dirs:
        for profile in chip.profiles:
            print(f"{profile.name}|{compiler_dir(_profile_thread(chip, profile))}")
    if args.print_cpu_targets:
        for profile in chip.profiles:
            if _profile_thread(chip, profile).HasField("hart_id"):
                print(profile.name)
    if args.print_runtime_targets:
        for profile in sorted(chip.profiles, key=lambda profile: profile.bank_entries):
            if _profile_thread(chip, profile).balldomain.mappings:
                print(profile.name)
    if args.print_thread_runtime_dirs:
        for profile in chip.profiles:
            thread = _profile_thread(chip, profile)
            _validate_profile(profile, thread)
            directory = (
                repo / "examples" / "threads" / thread.pkg / "compiler" / "runtime"
            )
            if (directory / "CMakeLists.txt").is_file():
                print(f"{profile.name}|{directory}")
            for entry in thread.balldomain.mappings:
                directory = (
                    repo
                    / "examples"
                    / "balls"
                    / entry.ball_dir
                    / "compiler"
                    / "runtime"
                )
                if (directory / "CMakeLists.txt").is_file():
                    print(f"{profile.name}|{directory}")
    if args.print_bank_targets:
        for profile in chip.profiles:
            if profile.bank_num > 0:
                print(profile.name)
    if args.print_target_balls:
        for profile in chip.profiles:
            thread = _profile_thread(chip, profile)
            _validate_profile(profile, thread)
            dirs = [entry.ball_dir for entry in thread.balldomain.mappings]
            if len(set(dirs)) != len(dirs):
                _die(f"profile {profile.name}: duplicate ball_dir")
            print(f"{profile.name}:{','.join(dirs)}")
    if args.print_thread_workload_targets:
        seen = set()
        for profile_id, profile in enumerate(chip.profiles):
            thread = _profile_thread(chip, profile)
            entry = (profile.name, thread.pkg)
            if entry in seen:
                continue
            seen.add(entry)
            print(f"{profile.name}:{thread.pkg}:{profile_id}")


if __name__ == "__main__":
    main()
