import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-pb", type=Path, required=True)
    parser.add_argument("--proto-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--hart-count", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.proto_dir))
    import chip_pb2

    chip = chip_pb2.Chip()
    chip.ParseFromString(args.chip_pb.read_bytes())
    # Core 0 is the main core; host workers map one-to-one onto compute cores.
    compute = [core for core in chip.cores if core.kind == chip_pb2.CORE_KIND_COMPUTE]
    if len(compute) != args.workers or chip.n_cores != len(chip.cores):
        raise ValueError(
            f"workers={args.workers} differs from chip compute Core count={len(compute)}"
        )
    cpus = []
    for index, core in enumerate(compute, start=1):
        if not core.HasField("controller_thread_index"):
            raise ValueError(f"Core {index} has no explicit controller_thread_index")
        controller = core.controller_thread_index
        if controller not in core.thread_indices or controller >= len(chip.threads):
            raise ValueError(f"Core {index} controller is not a thread in this Core")
        thread = chip.threads[controller]
        if not thread.HasField("hart_id"):
            raise ValueError(f"Core {index} controller has no explicit hart_id")
        cpu = thread.hart_id
        if cpu >= args.hart_count:
            raise ValueError(
                f"Core {index} controller hart {cpu} is outside Linux CPUs [0,{args.hart_count}): "
                "compute controllers must be Linux-visible to host workers"
            )
        if cpu in cpus:
            raise ValueError(f"Core {index} controller hart {cpu} is duplicated")
        cpus.append(cpu)
    print(";".join(map(str, cpus)))


if __name__ == "__main__":
    main()
