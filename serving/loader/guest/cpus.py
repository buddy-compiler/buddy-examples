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
    # Tile 0 is the main tile; host workers map one-to-one onto compute tiles.
    compute = [tile for tile in chip.tiles if tile.kind == chip_pb2.TILE_KIND_COMPUTE]
    if len(compute) != args.workers or chip.n_tiles != len(chip.tiles):
        raise ValueError(f"workers={args.workers} differs from chip compute Tile count={len(compute)}")
    cpus = []
    for index, tile in enumerate(compute, start=1):
        if not tile.HasField("controller_core_index"):
            raise ValueError(f"Tile {index} has no explicit controller_core_index")
        controller = tile.controller_core_index
        if controller not in tile.core_indices or controller >= len(chip.cores):
            raise ValueError(f"Tile {index} controller is not a core in this Tile")
        core = chip.cores[controller]
        if not core.HasField("hart_id"):
            raise ValueError(f"Tile {index} controller has no explicit hart_id")
        cpu = core.hart_id
        if cpu >= args.hart_count:
            raise ValueError(
                f"Tile {index} controller hart {cpu} is outside Linux CPUs [0,{args.hart_count}): "
                "compute controllers must be Linux-visible to host workers"
            )
        if cpu in cpus:
            raise ValueError(f"Tile {index} controller hart {cpu} is duplicated")
        cpus.append(cpu)
    print(";".join(map(str, cpus)))


if __name__ == "__main__":
    main()
