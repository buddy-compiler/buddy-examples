#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # e2e/
sys.path.insert(0, str(ROOT))

from framework.quant.core.pack import pack

HERE = Path(__file__).resolve().parent
QUANT = HERE / "quant"
MODES = {
    "mxfp4": QUANT / "modes" / "mxfp4.toml",
    "mxfp8": QUANT / "modes" / "mxfp8.toml",
}


def out_base() -> Path:
    env = os.environ.get("QWEN3_QUANT_OUT")
    if env:
        return Path(env)
    link = QUANT / "out"
    if link.is_symlink():
        return link.resolve()
    return link


def run_mode(name: str, mode: Path) -> float:
    arg0 = HERE / "arg0_0_6b.data"
    shapes = QUANT / "shapes.toml"
    if not arg0.is_file():
        raise SystemExit(f"missing {arg0}")
    if not shapes.is_file():
        raise SystemExit(f"missing {shapes}")
    if not mode.is_file():
        raise SystemExit(f"missing {mode}")

    out_dir = out_base() / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_bwq = out_dir / "qwen3.bwq"
    out_recon = out_dir / "arg0_recon.data"
    nmse_val = pack(arg0, shapes, mode, out_bwq, out_recon, dtype="bf16")
    print(f"{name}: nmse={nmse_val:.6e}")
    print(f"  wrote {out_bwq}")
    print(f"  wrote {out_recon}")
    return nmse_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modes",
        nargs="*",
        choices=sorted(MODES),
        help="modes to pack (default: all, one at a time)",
    )
    args = parser.parse_args()
    names = args.modes or sorted(MODES)
    for name in names:
        run_mode(name, MODES[name])


if __name__ == "__main__":
    main()
