#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # e2e/
sys.path.insert(0, str(ROOT))

from framework.quant.core.pack import pack

HERE = Path(__file__).resolve().parent
QUANT = HERE / "quant"
MODES = [
    ("i8-per-channel", QUANT / "modes" / "i8-per-channel.toml"),
    ("i8-per-tensor", QUANT / "modes" / "i8-per-tensor.toml"),
]


def main() -> None:
    arg0 = HERE / "arg0.data"
    shapes = QUANT / "shapes.toml"
    if not arg0.is_file():
        raise SystemExit(f"missing {arg0}")
    if not shapes.is_file():
        raise SystemExit(f"missing {shapes}")

    for name, mode in MODES:
        if not mode.is_file():
            raise SystemExit(f"missing {mode}")
        out_dir = QUANT / "out" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_bwq = out_dir / "lenet.bwq"
        out_recon = out_dir / "arg0_recon.data"
        nmse_val = pack(arg0, shapes, mode, out_bwq, out_recon)
        print(f"{name}: nmse={nmse_val:.6e}")
        print(f"  wrote {out_bwq}")
        print(f"  wrote {out_recon}")


if __name__ == "__main__":
    main()
