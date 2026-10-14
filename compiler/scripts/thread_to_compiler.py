"""Generate the Buckyball compiler inputs of one Thread from its thread.pb."""

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


def _load_thread(repo: Path, pb_path: Path):
    sys.path.insert(0, str(repo / "bbdev" / "api" / "steps" / "config" / "scripts"))
    import chip_pb2

    thread = chip_pb2.ThreadInstance()
    thread.ParseFromString(pb_path.read_bytes())
    if not thread.pkg.isidentifier():
        _die(f"{pb_path}: invalid thread package {thread.pkg!r}")
    return thread


def _cxx_string(value: str) -> str:
    if not value or '"' in value or "\\" in value:
        _die(f"unsupported C++ string in thread.pb: {value!r}")
    return f'"{value}"'


def _emit_registry(thread) -> str:
    bank = thread.mem.bank
    mappings = list(thread.balldomain.mappings)
    if mappings and (bank.num == 0 or bank.width == 0 or bank.entries == 0):
        _die(f"thread {thread.pkg}: bank geometry must be non-zero")
    chunks = ["// Generated from thread.pb. Do not edit.", ""]

    def array(type_name: str, name: str, items: list[str]) -> str:
        if not items:
            return f"llvm::ArrayRef<{type_name}>()"
        chunks.append(f"static const {type_name} {name}[] = {{")
        chunks.extend(f"  {item}," for item in items)
        chunks.append("};")
        return f"llvm::ArrayRef({name})"

    balls = array(
        "llvm::StringRef",
        "kBalls",
        [_cxx_string(entry.ball_name) for entry in mappings],
    )
    rows = []
    for index, entry in enumerate(mappings):
        params = sorted(
            (name, int(value))
            for name, value in entry.ball_params.items()
            if value.lstrip("-").isdigit()
        )
        ref = array(
            "buckyball_target::BuckyballBallParam",
            f"kBall{index}Params",
            [f"{{{_cxx_string(name)}, {value}}}" for name, value in params],
        )
        rows.append(
            f"{{{_cxx_string(entry.ball_name)}, {entry.in_bw}, {entry.out_bw}, {ref}}}"
        )
    ball_mappings = array(
        "buckyball_target::BuckyballBallMapping", "kBallMappings", rows
    )
    isa = array(
        "buckyball_target::BuckyballIsaEntry",
        "kIsa",
        [
            f"{{{_cxx_string(entry.mnemonic)}, {entry.funct7}}}"
            for entry in thread.balldomain.isa
        ],
    )
    rvv = "true" if thread.gpdomain.enable else "false"
    chunks.append(
        "static const buckyball_target::BuckyballTargetConfig kBuckyballTarget = {"
        f"{_cxx_string(thread.pkg)}, {bank.num}, {bank.width}, {bank.entries}, "
        f"{thread.frontend.vbank_id_upper_bound}, {rvv}, {balls}, {ball_mappings}, {isa}}};"
    )
    return "\n".join(chunks) + "\n"


def _balls(thread, repo: Path) -> list[dict[str, object]]:
    result = []
    for entry in thread.balldomain.mappings:
        if not entry.ball_name.isidentifier():
            _die(f"thread {thread.pkg}: invalid Ball name {entry.ball_name!r}")
        if not entry.ball_dir or len(Path(entry.ball_dir).parts) != 1:
            _die(f"thread {thread.pkg}: invalid ball_dir {entry.ball_dir!r}")
        source = repo / "examples" / "balls" / entry.ball_dir / "compiler" / "src"
        dialect = source / "Dialect" / "Buckyball"
        td = sorted(dialect.glob("*.td"))
        if len(td) != 1:
            _die(f"Ball {entry.ball_name}: expected one dialect TD in {dialect}")
        legalize = dialect / "Transforms" / "LegalizeForLLVMExport.cpp"
        if not legalize.is_file():
            _die(f"Ball {entry.ball_name}: missing {legalize}")
        lower = source / "Conversion" / "LowerBuckyball"
        tile = source / "Conversion" / "LowerTileToBuckyball"
        assign = lower / "AssignPhysicalBankPatterns.cpp"
        result.append(
            {
                "name": entry.ball_name,
                "td": td[0],
                "dialect": dialect,
                "legalize": [legalize],
                "assign": [assign] if assign.is_file() else [],
                "bank_ssa": sorted(lower.glob("*ToBankSSAPatterns.cpp")),
                "tile": sorted(tile.glob("*.cpp")),
                "passes": sorted((source / "Transforms").glob("*.cpp")),
            }
        )
    return result


def _thread_sources(thread, repo: Path) -> list[Path]:
    source = repo / "examples" / "threads" / thread.pkg / "compiler" / "src"
    return sorted(source.rglob("*.cpp"))


def _thread_rel(thread, repo: Path, path: Path) -> Path:
    return path.relative_to(
        repo / "examples" / "threads" / thread.pkg / "compiler" / "src"
    )


def _emit_hooks(thread, repo: Path) -> str:
    balls = _balls(thread, repo)
    lines = ["// Generated from thread.pb. Do not edit.", ""]
    for macro, field in (
        ("BUCKYBALL_LEGALIZE_HOOK", "legalize"),
        ("BUCKYBALL_ASSIGN_HOOK", "assign"),
        ("BUCKYBALL_TILE_HOOK", "tile"),
        ("BUCKYBALL_BANK_SSA_HOOK", "bank_ssa"),
    ):
        lines.append(f"#ifdef {macro}")
        lines.extend(f"{macro}({ball['name']})" for ball in balls if ball[field])
        lines.extend(["#endif", ""])
    return "\n".join(lines)


def _pass_owners(thread, repo: Path) -> list[str]:
    owners = []
    for ball in _balls(thread, repo):
        if ball["passes"]:
            if not any(path.name == "Passes.cpp" for path in ball["passes"]):
                _die(f"Ball {ball['name']}: Transforms contributions need Passes.cpp")
            owners.append(ball["name"])
    rels = {_thread_rel(thread, repo, path) for path in _thread_sources(thread, repo)}
    if any(rel.parts[0] == "Transforms" for rel in rels):
        if Path("Transforms/Passes.cpp") not in rels:
            _die(f"Thread {thread.pkg}: Transforms contributions need Passes.cpp")
        owners.append("Thread")
    return owners


def _print_sources(thread, repo: Path, kind: str) -> None:
    if kind == "thread":
        for path in _thread_sources(thread, repo):
            print(path)
        return
    if kind == "dialect-dir":
        for ball in _balls(thread, repo):
            print(ball["dialect"])
        return
    for ball in _balls(thread, repo):
        for path in ball[kind]:
            print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--thread-pb", type=Path, required=True)
    parser.add_argument("--registry-out", type=Path)
    parser.add_argument("--dialect-td-out", type=Path)
    parser.add_argument("--hooks-out", type=Path)
    parser.add_argument("--pass-decls-out", type=Path)
    parser.add_argument("--pass-registration-out", type=Path)
    parser.add_argument(
        "--print-sources",
        choices=(
            "thread",
            "dialect-dir",
            "legalize",
            "assign",
            "bank_ssa",
            "tile",
            "passes",
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    thread = _load_thread(repo, args.thread_pb.resolve())
    if args.registry_out:
        _write(args.registry_out, _emit_registry(thread))
    if args.dialect_td_out:
        lines = ["// Generated from thread.pb. Do not edit.", 'include "Buckyball.td"']
        lines.extend(f'include "{ball["td"].name}"' for ball in _balls(thread, repo))
        _write(args.dialect_td_out, "\n".join(lines) + "\n")
    if args.hooks_out:
        _write(args.hooks_out, _emit_hooks(thread, repo))
    if args.pass_decls_out:
        _write(
            args.pass_decls_out,
            "".join(
                f"namespace mlir::buddy {{ void register{owner}Passes(); }}\n"
                for owner in _pass_owners(thread, repo)
            ),
        )
    if args.pass_registration_out:
        _write(
            args.pass_registration_out,
            "".join(
                f"::mlir::buddy::register{owner}Passes();\n"
                for owner in _pass_owners(thread, repo)
            ),
        )
    if args.print_sources:
        _print_sources(thread, repo, args.print_sources)


if __name__ == "__main__":
    main()
