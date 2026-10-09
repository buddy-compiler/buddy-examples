"""Generate the static Buckyball compiler target registry from Chip.pb."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path


def _die(message: str) -> None:
    raise ValueError(message)


def _write(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_proto(repo: Path):
    scripts = repo / "bbdev" / "api" / "steps" / "config" / "scripts"
    if not scripts.is_dir():
        _die(f"missing protobuf bindings directory: {scripts}")
    sys.path.insert(0, str(scripts))
    import chip_pb2

    return chip_pb2


def _builtin_kernel(entry) -> bool:
    if not entry.builtin:
        if not entry.ball_dir or Path(entry.ball_dir).is_absolute() or len(Path(entry.ball_dir).parts) != 1:
            _die(f"external Ball {entry.ball_name}: invalid ball_dir {entry.ball_dir!r}")
        if entry.in_bw <= 0 or entry.out_bw <= 0:
            _die(f"external Ball {entry.ball_name}: bandwidth must be positive")
        return False
    if entry.builtin != "kernel":
        _die(f"unknown builtin Ball {entry.builtin!r}")
    if (entry.ball_name != "kernel" or entry.ball_class != "framework.balldomain.kernel.KernelBall"
            or entry.ball_dir or entry.config_path or entry.in_bw != 0 or entry.out_bw != 0):
        _die("builtin kernel has an invalid mapping")
    return True


def _target_name(core) -> str:
    return core.role or core.pkg


def _cxx_string(value: str) -> str:
    if not value:
        _die("empty string is not a valid compiler target field")
    if '"' in value or "\\" in value:
        _die(f"unsupported C++ string in chip.pb: {value!r}")
    return f'"{value}"'


def _profile_core(chip, profile):
    matches = [core for core in chip.cores if _target_name(core) == profile.name]
    if not matches:
        _die(f"profile {profile.name}: no matching CoreInstance")
    first = matches[0]
    expected_isa = [
        (entry.mnemonic, entry.funct7, entry.bid) for entry in first.balldomain.isa
    ]
    expected_balls = [entry.ball_name for entry in first.balldomain.mappings]
    expected_bank = (first.mem.bank.num, first.mem.bank.width, first.mem.bank.entries)
    profile_bank = (profile.bank_num, profile.bank_width, profile.bank_entries)
    if expected_bank != profile_bank:
        _die(
            f"profile {profile.name}: profile bank={profile_bank} disagrees with CoreInstance bank={expected_bank}"
        )
    for core in matches[1:]:
        isa = [
            (entry.mnemonic, entry.funct7, entry.bid) for entry in core.balldomain.isa
        ]
        balls = [entry.ball_name for entry in core.balldomain.mappings]
        bank = (core.mem.bank.num, core.mem.bank.width, core.mem.bank.entries)
        if isa != expected_isa or balls != expected_balls or bank != expected_bank:
            _die(f"profile {profile.name}: CoreInstance configuration mismatch")
    return first


def _validate_profile(profile, core) -> None:
    if not profile.name.isidentifier():
        _die(f"profile name is not a C++ identifier: {profile.name!r}")
    has_balls = len(core.balldomain.mappings) > 0
    if has_balls and (
        profile.bank_num == 0 or profile.bank_width == 0 or profile.bank_entries == 0
    ):
        _die(f"profile {profile.name}: bank geometry must be non-zero")
    for entry in core.balldomain.mappings:
        _builtin_kernel(entry)
    balls = [entry.ball_name for entry in core.balldomain.mappings]
    if len(set(balls)) != len(balls):
        _die(f"profile {profile.name}: duplicate Ball mapping")
    mnemonics: set[str] = set()
    funct7s: set[int] = set()
    for entry in core.balldomain.isa:
        if not entry.mnemonic.isidentifier():
            _die(f"profile {profile.name}: invalid ISA mnemonic {entry.mnemonic!r}")
        if entry.mnemonic in mnemonics:
            _die(f"profile {profile.name}: duplicate ISA mnemonic {entry.mnemonic}")
        if entry.funct7 in funct7s:
            _die(f"profile {profile.name}: duplicate funct7 {entry.funct7}")
        mnemonics.add(entry.mnemonic)
        funct7s.add(entry.funct7)


def _core_compiler_dirs(chip, repo: Path) -> list[Path]:
    """Resolve the Core compiler packages selected by Chip.pb."""
    result: list[Path] = []
    seen: set[str] = set()
    for profile in chip.profiles:
        core = _profile_core(chip, profile)
        core_name = core.pkg
        if not core_name.isidentifier():
            _die(f"profile {profile.name}: invalid core package {core_name!r}")
        if core_name in seen:
            continue
        compiler_dir = repo / "examples" / "cores" / core_name / "compiler"
        if not (compiler_dir / "CMakeLists.txt").is_file():
            _die(f"Core {core_name}: missing compiler package {compiler_dir}")
        result.append(compiler_dir)
        seen.add(core_name)
    return result


def _core_signature(core) -> int:
    signature_bytes = bytearray(core.pkg.encode() + b"\0")
    for value in (core.mem.bank.num, core.mem.bank.width, core.mem.bank.entries):
        signature_bytes.extend(value.to_bytes(8, "little"))
    for entry in sorted(core.balldomain.isa, key=lambda entry: entry.funct7):
        signature_bytes.extend(entry.mnemonic.encode() + b"\0")
        signature_bytes.extend(entry.funct7.to_bytes(4, "little"))
    for entry in sorted(
        core.balldomain.mappings, key=lambda entry: entry.ball_class.rsplit(".", 1)[-1]
    ):
        signature_bytes.extend(entry.ball_class.rsplit(".", 1)[-1].encode() + b"\0")
        for value in (entry.in_bw, entry.out_bw):
            signature_bytes.extend(value.to_bytes(8, "little"))
        for name, value in sorted(entry.ball_params.items()):
            signature_bytes.extend(name.encode() + b"\0" + value.encode() + b"\0")
    signature = 0xCBF29CE484222325
    for value in signature_bytes:
        signature = ((signature ^ value) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return signature


def _emit(chip, target: str | None = None) -> str:
    profiles = [
        profile for profile in chip.profiles if target is None or profile.name == target
    ]
    if not profiles:
        if target is None:
            _die("Chip.pb has no compiler profiles")
        _die(f"Chip.pb has no compiler profile {target!r}")
    names: set[str] = set()
    chunks = ["// Generated from Chip.pb. Do not edit.\n"]
    targets: list[str] = []
    for profile in profiles:
        if profile.name in names:
            _die(f"duplicate compiler profile {profile.name}")
        names.add(profile.name)
        core = _profile_core(chip, profile)
        _validate_profile(profile, core)
        stem = profile.name
        mappings = list(core.balldomain.mappings)
        isa = list(core.balldomain.isa)
        if mappings:
            chunks.append(f"static const llvm::StringRef k{stem}Balls[] = {{")
            chunks.extend((f"  {_cxx_string(entry.ball_name)}," for entry in mappings))
            chunks.append("};")
            param_refs: list[str] = []
            for index, entry in enumerate(mappings):
                params = sorted(
                    (
                        (name, int(value))
                        for name, value in entry.ball_params.items()
                        if value.lstrip("-").isdigit()
                    )
                )
                if params:
                    chunks.append(
                        f"static const buckyball_target::BuckyballBallParam k{stem}Ball{index}Params[] = {{"
                    )
                    chunks.extend(
                        (
                            f"  {{{_cxx_string(name)}, {value}}},"
                            for name, value in params
                        )
                    )
                    chunks.append("};")
                    param_refs.append(f"llvm::ArrayRef(k{stem}Ball{index}Params)")
                else:
                    param_refs.append(
                        "llvm::ArrayRef<buckyball_target::BuckyballBallParam>()"
                    )
            chunks.append(
                f"static const buckyball_target::BuckyballBallMapping k{stem}BallMappings[] = {{"
            )
            for index, entry in enumerate(mappings):
                chunks.append(
                    f"  {{{_cxx_string(entry.ball_name)}, {entry.in_bw}, {entry.out_bw}, {param_refs[index]}}},"
                )
            chunks.append("};")
            balls_ref = f"llvm::ArrayRef(k{stem}Balls)"
            mappings_ref = f"llvm::ArrayRef(k{stem}BallMappings)"
        else:
            balls_ref = "llvm::ArrayRef<llvm::StringRef>()"
            mappings_ref = "llvm::ArrayRef<buckyball_target::BuckyballBallMapping>()"
        if isa:
            chunks.append(
                f"static const buckyball_target::BuckyballIsaEntry k{stem}Isa[] = {{"
            )
            chunks.extend(
                (
                    f"  {{{_cxx_string(entry.mnemonic)}, {entry.funct7}}},"
                    for entry in isa
                )
            )
            chunks.append("};")
            isa_ref = f"llvm::ArrayRef(k{stem}Isa)"
        else:
            isa_ref = "llvm::ArrayRef<buckyball_target::BuckyballIsaEntry>()"
        chunks.append("")
        signature = _core_signature(core)
        targets.append(
            f"  {{{_cxx_string(profile.name)}, {_cxx_string(core.pkg)}, "
            f"{profile.bank_num}, {profile.bank_width}, {profile.bank_entries}, "
            f"{balls_ref}, {mappings_ref}, {isa_ref}, {signature}ULL}},"
        )
    chunks.append(
        "static const buckyball_target::BuckyballTargetConfig kBuckyballTargets[] = {"
    )
    chunks.extend(targets)
    chunks.append("};\n")
    return "\n".join(chunks)


def _isqrt(n: int) -> int:
    if n <= 0:
        _die(f"isqrt of non-positive {n}")
    x = n
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y


def _emit_params_header(profile, core, virtual_bank_num: int, isa_dir: Path, test_hart: int) -> None:
    bank = core.mem.bank
    mmio = core.mem.mmio
    if bank.num == 0 or bank.width == 0 or bank.entries == 0:
        _die(f"profile {profile.name}: bank geometry must be non-zero")
    if bank.width % 8 != 0:
        _die(f"profile {profile.name}: bank.width must be a multiple of 8")
    if (
        virtual_bank_num < bank.num
        or virtual_bank_num <= core.frontend.vbank_id_upper_bound
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
    lines = [
        "/* Generated from Chip.pb. Do not edit. */",
        "#ifndef BBHW_PARAMS_H",
        "#define BBHW_PARAMS_H",
        "",
        f"#define CORE_SIGNATURE {_core_signature(core)}ULL",
        f"#define BB_TEST_HART {test_hart}",
        f"#define BANK_NUM {bank.num}",
        f"#define VIRTUAL_BANK_NUM {virtual_bank_num}",
        f"#define BB_PRIVATE_VBANK_MAX {core.frontend.vbank_id_upper_bound}",
        f"#define BB_SHARED_BANK_BASE {core.frontend.shared_bank_id_base}",
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

    A header describes one core ISA only.  Deliberately do not create a merged
    chip header: funct7 is target-local and may overlap across profiles.
    """
    if not chip.tiles or not chip.tiles[0].core_indices:
        _die("chip topology has no tile cores")
    if chip.n_tiles != len(chip.tiles):
        _die(
            f"chip n_tiles={chip.n_tiles} disagrees with tile placements={len(chip.tiles)}"
        )
    # Tile 0 is the main tile; compute tiles follow and are homogeneous among themselves.
    # Bank macros describe the compute tiles when present, otherwise the main tile. A chip whose
    # main tile also gains Buckyball banks with a different geometry needs per-kind macros.
    work_tile = 1 if len(chip.tiles) > 1 else 0
    bank_tile = chip.tiles[work_tile]
    virtual_bank_num = bank_tile.virtual_bank_count

    def signature(tile):
        for index in tile.core_indices:
            if index >= len(chip.cores):
                _die(f"tile Core index {index} out of range (n={len(chip.cores)})")
        return [
            (chip.cores[index].role, chip.cores[index].pkg)
            for index in tile.core_indices
        ]

    profile_ids = {profile.name: index for index, profile in enumerate(chip.profiles)}
    if len(profile_ids) != len(chip.profiles):
        _die("chip has duplicate compiler profile names")
    compute = list(chip.tiles)[1:]
    for tile_id, tile in enumerate(compute, start=1):
        if signature(tile) != signature(compute[0]):
            _die(f"tile {tile_id}: compute tiles are not homogeneous")
        if tile.virtual_bank_count != compute[0].virtual_bank_count:
            _die(f"tile {tile_id}: virtual bank count differs from tile 1")
        if tile.shared_mem != compute[0].shared_mem:
            _die(f"tile {tile_id}: shared memory config differs from tile 1")

    # Every core carries its explicit hart id; the header only tabulates them.
    hart_num = len(chip.cores)
    cores_per_tile = max(len(tile.core_indices) for tile in chip.tiles)
    hart_tile = [None] * hart_num
    hart_core = [None] * hart_num
    slot_profiles = []
    for tile_id, tile in enumerate(chip.tiles):
        row = []
        for core_id, core_index in enumerate(tile.core_indices):
            core = chip.cores[core_index]
            if not core.HasField("hart_id") or core.hart_id >= hart_num:
                _die(f"core {core_index}: missing or out-of-range hart_id")
            hart_tile[core.hart_id] = tile_id
            hart_core[core.hart_id] = core_id
            target = _target_name(core)
            if target not in profile_ids:
                _die(f"tile {tile_id} core {core_id} target {target!r} has no compiler profile")
            row.append(profile_ids[target])
        slot_profiles.append(row + [0xFFFFFFFF] * (cores_per_tile - len(row)))
    if None in hart_tile:
        _die("chip hart ids are not exactly 0..cores-1")
    report_hart = chip.cores[bank_tile.core_indices[0]].hart_id

    shared = bank_tile.shared_mem
    if shared.enable:
        endpoints = [chip.cores[index] for index in bank_tile.core_indices
                     if chip.cores[index].balldomain.mappings]
        if not endpoints:
            _die("shared memory requires a compute endpoint")
        shared_bank_entries = endpoints[0].mem.bank.entries
        if shared.entries == 0 or shared.entries % shared_bank_entries:
            _die(
                "bank tile: shared entries must be a non-zero multiple of compute bank depth"
            )
        shared_physical_bank_num = shared.entries // shared_bank_entries
    else:
        shared_physical_bank_num = 0
    profile_rows = ", ".join("{" + ", ".join(f"{p}u" for p in row) + "}" for row in slot_profiles)
    lines = [
        "#ifndef BBHW_TOPOLOGY_H",
        "#define BBHW_TOPOLOGY_H",
        f"#define BB_TILE_NUM {len(chip.tiles)}",
        f"#define BB_HART_NUM {hart_num}",
        "// Hart layout: main tile cores are harts 0..BB_MAIN_CORES-1; compute tile t (1-based)",
        "// has its controller at BB_MAIN_CORES+t-1 and its workers from BB_MAIN_CORES+BB_COMPUTE_TILES.",
        f"#define BB_MAIN_CORES {len(chip.tiles[0].core_indices)}",
        f"#define BB_COMPUTE_TILES {len(chip.tiles) - 1}",
        f"#define BB_COMPUTE_CORES {len(chip.tiles[1].core_indices) if len(chip.tiles) > 1 else 0}",
        "// Largest tile; tile 0 is the main tile and compute tiles follow. Bank counts below",
        "// describe the compute tiles when present, otherwise the main tile.",
        f"#define BB_CORES_PER_TILE {cores_per_tile}",
        "// Bare-metal tests run on the work tile, the tile the bank counts describe; its core 0",
        "// reports the result, since tiles share no coherent cache state.",
        f"#define BB_WORK_TILE {work_tile}",
        f"#define BB_REPORT_HART {report_hart}",
        f"#define BB_VIRTUAL_BANK_NUM {virtual_bank_num}",
        f"#define BB_SHARED_PHYSICAL_BANK_NUM {shared_physical_bank_num}",
        "#ifndef __ASSEMBLER__",
        "#include <stdint.h>",
        "typedef struct { uint32_t tile; uint32_t core; } core_id_t;",
        "static inline core_id_t bb_topology_core_id(uint32_t hart) {",
        f"  static const uint32_t tiles[{hart_num}] = {{{', '.join(map(str, hart_tile))}}};",
        f"  static const uint32_t cores[{hart_num}] = {{{', '.join(map(str, hart_core))}}};",
        f"  if (hart >= {hart_num}u) __builtin_trap();",
        "  return (core_id_t){tiles[hart], cores[hart]};",
        "}",
        "static inline uint32_t bb_topology_core_profile(core_id_t id) {",
        f"  static const uint32_t profiles[{len(chip.tiles)}][{cores_per_tile}] = {{{profile_rows}}};",
        f"  if (id.tile >= {len(chip.tiles)}u || id.core >= {cores_per_tile}u ||",
        "      profiles[id.tile][id.core] == 0xFFFFFFFFu) __builtin_trap();",
        "  return profiles[id.tile][id.core];",
        "}",
        "static inline uint32_t bb_topology_profile_cores_per_tile(uint32_t profile) {",
        "  uint32_t count = 0;",
        f"  for (uint32_t core = 0; core < {len(bank_tile.core_indices)}u; ++core)",
        "    count += bb_topology_core_profile((core_id_t){BB_WORK_TILE, core}) == profile;",
        "  return count;",
        "}",
        "#endif",
        "#endif",
        "",
    ]
    _write(isa_dir / "chip" / "topology.h", "\n".join(lines))
    for profile in chip.profiles:
        core = _profile_core(chip, profile)
        _validate_profile(profile, core)
        (isa_dir / profile.name / "topology.h").unlink(missing_ok=True)
        lines = ["#ifndef BALL_ISA_H", "#define BALL_ISA_H", ""]
        lines.extend(
            (
                f"#define BB_FUNC7_{entry.mnemonic} {entry.funct7}"
                for entry in core.balldomain.isa
            )
        )
        lines.extend(["", "#endif", ""])
        header = isa_dir / profile.name / "ballISA.h"
        _write(header, "\n".join(lines))
        _emit_params_header(
            profile, core, chip.tiles[hart_tile[core.hart_id]].virtual_bank_count,
            isa_dir, core.hart_id)


def _emit_dialect_td(chip, repo: Path) -> str:
    """Emit the union of Ball operation definitions selected by this Chip."""
    ball_dirs: list[str] = []
    seen: set[str] = set()
    for profile in chip.profiles:
        core = _profile_core(chip, profile)
        for entry in core.balldomain.mappings:
            if _builtin_kernel(entry):
                continue
            ball_dir = entry.ball_dir
            if not ball_dir:
                _die(f"profile {profile.name}: {entry.ball_name} has no ball_dir")
            if ball_dir in seen:
                continue
            td_dir = (
                repo
                / "examples"
                / "balls"
                / ball_dir
                / "compiler"
                / "src"
                / "Dialect"
                / "Buckyball"
            )
            td_files = sorted(td_dir.glob("*.td"))
            if len(td_files) != 1:
                _die(f"Ball {ball_dir}: expected one dialect TD file in {td_dir}")
            seen.add(ball_dir)
            ball_dirs.append(td_files[0].name)
    lines = ["// Generated from Chip.pb. Do not edit.", 'include "Buckyball.td"']
    lines.extend((f'include "{td_file}"' for td_file in ball_dirs))
    lines.append("")
    return "\n".join(lines)


def _ball_dialect_dirs(chip, repo: Path) -> list[Path]:
    return [ball["dialect_dir"] for ball in _ball_compilers(chip, repo)]


def _ball_compilers(chip, repo: Path) -> list[dict[str, object]]:
    """Resolve convention-based compiler contributions from selected Balls."""
    result: list[dict[str, object]] = []
    seen: dict[str, str] = {}
    for profile in chip.profiles:
        core = _profile_core(chip, profile)
        _validate_profile(profile, core)
        for entry in core.balldomain.mappings:
            if _builtin_kernel(entry):
                continue
            ball_name = entry.ball_name
            ball_dir = entry.ball_dir
            if not ball_name.isidentifier():
                _die(f"profile {profile.name}: invalid Ball name {ball_name!r}")
            if not ball_dir:
                _die(f"profile {profile.name}: {ball_name} has no ball_dir")
            if Path(ball_dir).is_absolute() or len(Path(ball_dir).parts) != 1:
                _die(f"profile {profile.name}: invalid ball_dir {ball_dir!r}")
            if ball_name in seen:
                if seen[ball_name] != ball_dir:
                    _die(f"Ball {ball_name} maps to multiple directories")
                continue
            source_dir = repo / "examples" / "balls" / ball_dir / "compiler" / "src"
            dialect_dir = source_dir / "Dialect" / "Buckyball"
            td_files = sorted(dialect_dir.glob("*.td"))
            if len(td_files) != 1:
                _die(f"Ball {ball_name}: expected one dialect TD in {dialect_dir}")
            legalize = dialect_dir / "Transforms" / "LegalizeForLLVMExport.cpp"
            if not legalize.is_file():
                _die(f"Ball {ball_name}: missing LLVM export lowering {legalize}")
            lower = source_dir / "Conversion" / "LowerBuckyball"
            tile_dir = source_dir / "Conversion" / "LowerTileToBuckyball"
            result.append(
                {
                    "name": ball_name,
                    "dialect_dir": dialect_dir,
                    "legalize": legalize,
                    "assign": lower / "AssignPhysicalBankPatterns.cpp",
                    "bank_ssa": sorted(lower.glob("*ToBankSSAPatterns.cpp")),
                    "tile": sorted(tile_dir.glob("*.cpp")) if tile_dir.is_dir() else [],
                    "passes": sorted((source_dir / "Transforms").glob("*.cpp")),
                }
            )
            seen[ball_name] = ball_dir
    return result


def _emit_lowering_hooks(chip, repo: Path) -> str:
    balls = _ball_compilers(chip, repo)
    lines = ["// Generated from Chip.pb. Do not edit.", ""]
    stages = (
        ("BUCKYBALL_LEGALIZE_HOOK", "legalize"),
        ("BUCKYBALL_ASSIGN_HOOK", "assign"),
        ("BUCKYBALL_TILE_HOOK", "tile"),
        ("BUCKYBALL_BANK_SSA_HOOK", "bank_ssa"),
    )
    for macro, field in stages:
        lines.append(f"#ifdef {macro}")
        for ball in balls:
            value = ball[field]
            if (
                field == "legalize"
                or (isinstance(value, list) and value)
                or (isinstance(value, Path) and value.is_file())
            ):
                lines.append(f"{macro}({ball['name']})")
        lines.extend(["#endif", ""])
    return "\n".join(lines)


def _print_ball_compiler_paths(chip, repo: Path, kind: str) -> None:
    for ball in _ball_compilers(chip, repo):
        if kind == "dialect-dir":
            print(ball["dialect_dir"])
        elif kind == "legalize":
            print(ball["legalize"])
        elif kind in ("tile", "bank_ssa", "passes"):
            for source in ball[kind]:
                print(source)
        else:
            source = ball[kind]
            if isinstance(source, Path) and source.is_file():
                print(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--chip-pb", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-dialect-td", type=Path)
    parser.add_argument("--lowering-hooks-out", type=Path)
    parser.add_argument("--pass-decls-out", type=Path)
    parser.add_argument("--pass-registration-out", type=Path)
    parser.add_argument("--isa-dir", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--print-core-compiler-dirs", action="store_true")
    parser.add_argument("--print-targets", action="store_true")
    parser.add_argument("--print-runtime-targets", action="store_true")
    parser.add_argument("--print-bank-targets", action="store_true")
    parser.add_argument("--print-target-balls", action="store_true")
    parser.add_argument("--print-core-targets", action="store_true")
    parser.add_argument("--print-core-workload-targets", action="store_true")
    parser.add_argument("--print-ball-dialect-dirs", action="store_true")
    parser.add_argument(
        "--print-ball-compiler-paths",
        choices=("dialect-dir", "legalize", "assign", "bank_ssa", "tile", "passes"),
    )
    args = parser.parse_args()
    if (
        args.out is None
        and args.out_dialect_td is None
        and (args.lowering_hooks_out is None)
        and (args.pass_decls_out is None)
        and (args.pass_registration_out is None)
        and (args.isa_dir is None)
        and (not args.print_core_compiler_dirs)
        and (not args.print_targets)
        and (not args.print_runtime_targets)
        and (not args.print_bank_targets)
        and (not args.print_target_balls)
        and (not args.print_core_targets)
        and (not args.print_core_workload_targets)
        and (not args.print_ball_dialect_dirs)
        and (args.print_ball_compiler_paths is None)
    ):
        _die("one output mode is required")
    repo = args.repo.resolve()
    if not repo.is_dir():
        _die(f"repository does not exist: {repo}")
    pb_path = args.chip_pb.resolve()
    if not pb_path.is_file():
        _die(f"missing Chip.pb: {pb_path}")
    pb = _load_proto(repo)
    chip = pb.Chip()
    chip.ParseFromString(pb_path.read_bytes())
    if args.out:
        _write(args.out, _emit(chip, args.target))
    if args.out_dialect_td:
        _write(args.out_dialect_td, _emit_dialect_td(chip, repo))
    if args.lowering_hooks_out:
        _write(args.lowering_hooks_out, _emit_lowering_hooks(chip, repo))
    if args.pass_decls_out or args.pass_registration_out:
        passes = [ball for ball in _ball_compilers(chip, repo) if ball["passes"]]
        for ball in passes:
            if not any(source.name == "Passes.cpp" for source in ball["passes"]):
                _die(f"Ball {ball['name']}: Transforms contributions need Passes.cpp")
        if args.pass_decls_out:
            _write(args.pass_decls_out, "\n".join(
                f"namespace mlir::buddy {{ void register{ball['name']}Passes(); }}"
                for ball in passes) + "\n")
        if args.pass_registration_out:
            _write(args.pass_registration_out, "\n".join(
                f"::mlir::buddy::register{ball['name']}Passes();"
                for ball in passes) + "\n")
    if args.isa_dir:
        _emit_isa_headers(chip, args.isa_dir)
    if args.print_targets:
        for profile in chip.profiles:
            print(profile.name)
    if args.print_runtime_targets:
        for profile in sorted(chip.profiles, key=lambda profile: profile.bank_entries):
            if _profile_core(chip, profile).balldomain.mappings:
                print(profile.name)
    if args.print_bank_targets:
        for profile in chip.profiles:
            if profile.bank_num > 0:
                print(profile.name)
    if args.print_target_balls:
        for profile in chip.profiles:
            core = _profile_core(chip, profile)
            _validate_profile(profile, core)
            dirs = []
            seen = set()
            for entry in core.balldomain.mappings:
                if _builtin_kernel(entry):
                    continue
                ball_dir = entry.ball_dir
                if not ball_dir:
                    _die(f"profile {profile.name}: {entry.ball_name} has no ball_dir")
                if ball_dir in seen:
                    _die(f"profile {profile.name}: duplicate ball_dir {ball_dir!r}")
                seen.add(ball_dir)
                dirs.append(ball_dir)
            print(f"{profile.name}:{','.join(dirs)}")
    if args.print_core_targets:
        targets = {profile.name for profile in chip.profiles}
        for core in chip.cores:
            target = _target_name(core)
            if target not in targets:
                _die(f"CoreInstance {core.index}: no compiler profile {target}")
            print(f"{core.index}:{target}")
    if args.print_core_workload_targets:
        seen = set()
        for profile_id, profile in enumerate(chip.profiles):
            core = _profile_core(chip, profile)
            if not core.pkg.isidentifier():
                _die(f"profile {profile.name}: invalid Core package {core.pkg!r}")
            entry = (profile.name, core.pkg)
            if entry in seen:
                continue
            seen.add(entry)
            print(f"{profile.name}:{core.pkg}:{profile_id}")
    if args.print_ball_dialect_dirs:
        for dialect_dir in _ball_dialect_dirs(chip, repo):
            print(dialect_dir)
    if args.print_ball_compiler_paths:
        _print_ball_compiler_paths(chip, repo, args.print_ball_compiler_paths)
    if args.print_core_compiler_dirs:
        for compiler_dir in _core_compiler_dirs(chip, repo):
            print(compiler_dir)


if __name__ == "__main__":
    main()
