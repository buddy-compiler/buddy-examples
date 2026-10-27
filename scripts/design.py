"""Write design.cmake, everything a CMake build needs from design.pb.

usage: design.py --design-pb <design.pb> --out <dir>

Every value of design.pb reaches CMake and C by one rule, by protobuf reflection:
  - a scalar field is named by its path, field names upper-cased and joined by
    `_`: device.ddr.base is DESIGN_DEVICE_DDR_BASE; a repeated scalar is a list;
  - a repeated message (or map) is the list of its keys, and each element
    continues the path with its key: DESIGN_PROFILES_pebble_THREAD_MEM_BANK_WIDTH.
    The key is the element's field marked [(key) = true] in design.proto
    (profiles by name, cores by hart id, balls by component, instructions by
    mnemonic), else its position. google.protobuf.Any is opaque and skipped.
C sees the same paths without DESIGN_: numbers (64-bit ones with a ULL/LL
suffix), strings as bare tokens, a repeated scalar as a comma list and a
repeated message as the X-list X(key)X(key)...
  DESIGN_TARGET_DEFS_<t>  a program of compiler target t: the sub-messages of
                          its profile by their path from the profile
                          (thread.mem.bank.width is THREAD_MEM_BANK_WIDTH,
                          core.hart_id is CORE_HART_ID)
  DESIGN_DEFS             every program: WORKLOAD_<PATH>, and the profiles and
                          cores without their sub-messages (PROFILES, DEVICE_CORES)
The rest of design.cmake is the build layout: where the stacks of each level
are and are built, the compiler plugins and command of each target.
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


def stack_build(design) -> Path:
    """Where `bbdev stack --build` builds the stacks of the simulation, one directory per level."""
    return REPO / design.simulation.output / "stack"


def core_component(design, core) -> str:
    """Cores are named by their component; one component has one configuration per design."""
    same = {c.signature for c in design.device.cores if c.component == core.component}
    if len(same) != 1:
        raise ValueError(
            f"{design.name}: core {core.component} has {len(same)} configurations"
        )
    return core.component


def opt(design, design_pb: Path, name: str) -> list:
    """buddy-opt with the plugin of compiler target `name`, and the engine of the
    cores that run it on hardware threads. The plugin reads its target from design.pb.
    """
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
        f"--load-dialect-plugin={plugin(design, name)}",
        f"--load-pass-plugin={plugin(design, name)}",
        *(
            f"--load-pass-plugin={stack_build(design) / 'core' / engine / 'libengine-compiler.so'}"
            for engine in engines
        ),
        f"--buckyball-design-pb={design_pb}",
        f"--buckyball-target={name}",
    ]


def design_stack(design) -> Path:
    """The design's own software stack, laid out like its arch: stack/device,
    with the design's own core in stack/device/core."""
    return REPO / Path(design.path).parent.parent / "stack"


def core_stack(design, core) -> Path:
    if core.component == design.name:
        return design_stack(design) / "device/core"
    return REPO / "examples/prefab/cores" / core.component / "stack"


def device_stack(design) -> Path:
    """The device level: the design's SoC, or for a core entry the single-core
    system of the framework, which sees no SoC."""
    if design.simulation.selection.startswith("core/"):
        return REPO / "stack/sims/core"
    return design_stack(design) / "device"


def stacks(design) -> list:
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
                stack_build(design) / "thread" / name,
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
                core_stack(design, core),
                stack_build(design) / "core" / name,
            )
        )
    result.append(
        (
            "device",
            design.name,
            device_stack(design),
            stack_build(design) / "device",
        )
    )
    return result


def plugin(design, name: str) -> Path:
    """The buddy-opt plugin of compiler target `name`, built by the stack that owns it."""
    if any(thread.profile == name for thread in design.threads):
        return stack_build(design) / "thread" / name / f"lib{name}-compiler.so"
    core = next(core for core in design.device.cores if core.profile == name)
    return (
        stack_build(design)
        / "core"
        / core_component(design, core)
        / f"lib{name}-compiler.so"
    )


def write(path: Path, text: str) -> None:
    # Unchanged files keep their timestamps so dependent objects are not rebuilt.
    if path.is_file() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def elements(field, value) -> dict:
    """The elements of a repeated message or map field, by key."""
    if field.message_type.GetOptions().map_entry:
        return dict(value)
    marked = [
        f
        for f in field.message_type.fields
        if f.GetOptions().Extensions[design_pb2.key]
    ]
    items = {}
    for index, element in enumerate(value):
        key = str(getattr(element, marked[0].name)) if marked else str(index)
        if key in items:
            raise ValueError(f"{field.full_name}: two elements keyed {key}")
        items[key] = element
    return items


def listing(field, value, path: str, deep: bool = True):
    """A repeated message or map: (path, None, its keys), then its elements."""
    items = elements(field, value)
    yield path, None, list(items)
    entry = field.message_type.fields_by_name.get("value")
    for key, item in items.items():
        if field.message_type.GetOptions().map_entry and entry.message_type is None:
            yield f"{path}_{key}", entry, item
        else:
            yield from flatten(item, f"{path}_{key}", deep)


def flatten(message, path: str, deep: bool = True):
    """(name, field, value) of every value of `message`, named below `path`.
    Not `deep`: the scalars of `message` itself, none of its sub-messages."""
    for field in message.DESCRIPTOR.fields:
        name = f"{path}_{field.name.upper()}" if path else field.name.upper()
        value = getattr(message, field.name)
        if field.message_type is None:
            if field.is_repeated:
                yield name, field, list(value)
            elif not field.has_presence or message.HasField(field.name):
                yield name, field, value
        elif field.message_type.full_name == "google.protobuf.Any" or not deep:
            continue
        elif field.is_repeated:
            yield from listing(field, value, name)
        elif message.HasField(field.name):
            yield from flatten(value, name)


def text(value) -> str:
    return str(int(value)) if isinstance(value, bool) else str(value)


def c_defs(values) -> list:
    """Values of `flatten` as C compile definitions."""
    defs = []
    for name, field, value in values:
        if field is None:
            value = "".join(f"X({key})" for key in value)
        elif isinstance(value, list):
            value = ",".join(text(v) for v in value)
        elif field.type in (field.TYPE_UINT64, field.TYPE_FIXED64):
            value = f"{value}ULL"
        elif field.type in (field.TYPE_INT64, field.TYPE_SINT64, field.TYPE_SFIXED64):
            value = f"{value}LL"
        defs.append(f"{name}={text(value)}")
    return defs


def cmake(design, design_pb: Path) -> str:
    values = {}

    def put(name, *items):
        if name in values:
            raise ValueError(f"{name} is both design.pb data and build layout")
        values[name] = items

    for name, field, value in flatten(design, "DESIGN"):
        put(name, *(value if isinstance(value, list) else [value]))

    profiles = design.DESCRIPTOR.fields_by_name["profiles"]
    cores = design.device.DESCRIPTOR.fields_by_name["cores"]
    put(
        "DESIGN_DEFS",
        *c_defs(flatten(design.workload, "WORKLOAD")),
        *c_defs(listing(profiles, design.profiles, "PROFILES", deep=False)),
        *c_defs(listing(cores, design.device.cores, "DEVICE_CORES", deep=False)),
    )
    for profile in design.profiles:
        put(
            f"DESIGN_TARGET_DEFS_{profile.name}",
            *(
                define
                for field in profile.DESCRIPTOR.fields
                if field.message_type is not None
                for define in c_defs(
                    flatten(getattr(profile, field.name), field.name.upper())
                )
            ),
        )

    # The build layout.
    put("DESIGN_PB", design_pb)
    put("DESIGN_REPO", REPO)
    put("DESIGN_OUTPUT", REPO / design.simulation.output)
    put("DESIGN_WORKLOADS", REPO / design.simulation.workloads)
    put("DESIGN_COMPILER", REPO / "stack/thirdparty/buddy-mlir/build")
    put("DESIGN_MLIR", REPO / "stack/thirdparty/buddy-mlir/llvm/build/lib/cmake/mlir")
    put(
        "DESIGN_CPU_TARGETS",
        *dict.fromkeys(core.profile for core in design.device.cores if core.profile),
    )
    put(
        "DESIGN_RUNTIME_TARGETS",
        *(
            p.name
            for p in sorted(design.profiles, key=lambda p: p.thread.mem.bank.entries)
            if p.thread.balldomain.balls
        ),
    )
    for profile in design.profiles:
        thread = profile.thread
        balls = [ball.component for ball in thread.balldomain.balls]
        kernels = [REPO / thread.source_path / "stack/kernels"]
        kernels += [
            REPO / "examples/prefab/balls" / ball / "stack/kernels" for ball in balls
        ]
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
                src
                for ball in balls
                if (src := REPO / "examples/prefab/balls" / ball / "stack/src").is_dir()
            ),
        )
        put(f"DESIGN_TARGET_PLUGIN_{profile.name}", plugin(design, profile.name))
        put(f"DESIGN_TARGET_OPT_{profile.name}", *opt(design, design_pb, profile.name))

    levels = stacks(design)
    put("DESIGN_DEVICE_STACK", levels[-1][2])
    put("DESIGN_DEVICE_BUILD", levels[-1][3])
    put(
        "DESIGN_THREAD_STACKS",
        *(name for level, name, _, _ in levels if level == "thread"),
    )
    for level, name, source, build in levels:
        if level != "thread":
            continue
        thread = next(thread for thread in design.threads if thread.profile == name)
        put(f"DESIGN_THREAD_STACK_{name}", source)
        put(f"DESIGN_THREAD_BUILD_{name}", build)
        put(
            f"DESIGN_THREAD_BALLS_{name}",
            *(
                REPO / "examples/prefab/balls" / ball.component / "workloads"
                for ball in thread.balldomain.balls
            ),
        )
    put("DESIGN_CORES", *(name for level, name, _, _ in levels if level == "core"))
    for level, name, source, build in levels:
        if level != "core":
            continue
        core = next(core for core in design.device.cores if core.component == name)
        put(f"DESIGN_CORE_HART_{name}", core.hart_id)
        put(f"DESIGN_CORE_STACK_{name}", source)
        put(f"DESIGN_CORE_BUILD_{name}", build)

    lines = ["# Generated from design.pb. Do not edit."]
    for name, items in values.items():
        quoted = " ".join(
            '"' + text(item).replace("\\", "\\\\").replace('"', '\\"') + '"'
            for item in items
        )
        lines.append(f"set({name} {quoted})")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--design-pb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.design_pb, args.out)


def generate(design_pb: Path, out: Path) -> None:
    design_pb = Path(design_pb).resolve()
    write(out / "design.cmake", cmake(load(design_pb), design_pb))


if __name__ == "__main__":
    main()
