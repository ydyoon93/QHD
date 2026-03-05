#!/usr/bin/env python3
"""Read rank-local HDF5 outputs and reconstruct a global 2D field."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Iterable

import h5py
import numpy as np

STEP_RE = re.compile(r"step_(\d{6})_rank_\d{4}\.h5$")


def discover_steps(output_dir: pathlib.Path) -> list[int]:
    steps: set[int] = set()
    for path in output_dir.glob("step_*_rank_*.h5"):
        m = STEP_RE.match(path.name)
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)


def files_for_step(output_dir: pathlib.Path, step: int) -> list[pathlib.Path]:
    files = sorted(output_dir.glob(f"step_{step:06d}_rank_*.h5"))
    return files


def read_global_field(files: Iterable[pathlib.Path], field: str) -> tuple[np.ndarray, dict]:
    files = list(files)
    if not files:
        raise RuntimeError("No files provided")

    global_nx = None
    global_ny = None
    time = None

    out = None
    meta = {}

    for p in files:
        with h5py.File(p, "r") as h5:
            if field not in h5:
                raise RuntimeError(f"Field '{field}' not found in {p}")

            gx = int(h5.attrs["global_nx"])
            gy = int(h5.attrs["global_ny"])
            lx = int(h5.attrs["local_nx"])
            ly = int(h5.attrs["local_ny"])
            ox = int(h5.attrs["offset_x"])
            oy = int(h5.attrs["offset_y"])
            t = float(h5.attrs["time"])

            if global_nx is None:
                global_nx = gx
                global_ny = gy
                time = t
                out = np.zeros((global_ny, global_nx), dtype=np.float64)
                meta = {
                    "global_nx": gx,
                    "global_ny": gy,
                    "dx": float(h5.attrs["dx"]),
                    "dy": float(h5.attrs["dy"]),
                    "time": t,
                }
            else:
                if gx != global_nx or gy != global_ny:
                    raise RuntimeError("Inconsistent global grid among rank files")

            data = h5[field][...]
            if data.shape != (ly, lx):
                raise RuntimeError(
                    f"Unexpected data shape in {p}: {data.shape}, expected {(ly, lx)}"
                )

            out[oy : oy + ly, ox : ox + lx] = data

    assert out is not None
    return out, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output", help="Directory with step_XXXXXX_rank_YYYY.h5 files")
    parser.add_argument("--step", type=int, default=None, help="Step to read (default: latest)")
    parser.add_argument(
        "--field",
        default="Bz",
        choices=["Bx", "By", "Bz", "Ux", "Uy", "Uz", "Qx", "Qy", "Qz"],
        help="Field component to reconstruct",
    )
    parser.add_argument("--save-npy", default=None, help="Optional output .npy path")
    parser.add_argument("--list-steps", action="store_true", help="List available steps and exit")

    args = parser.parse_args()
    output_dir = pathlib.Path(args.output_dir)

    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}", file=sys.stderr)
        return 1

    steps = discover_steps(output_dir)
    if not steps:
        print(f"No step files found in {output_dir}", file=sys.stderr)
        return 1

    if args.list_steps:
        print("Available steps:", " ".join(str(s) for s in steps))
        return 0

    step = args.step if args.step is not None else steps[-1]
    if step not in steps:
        print(f"Step {step} not found. Available: {steps}", file=sys.stderr)
        return 1

    files = files_for_step(output_dir, step)
    arr, meta = read_global_field(files, args.field)

    print(f"Read field={args.field} at step={step} from {len(files)} rank files")
    print(
        f"Grid={meta['global_nx']}x{meta['global_ny']}  "
        f"dx={meta['dx']:.6g} dy={meta['dy']:.6g}  time={meta['time']:.6g}"
    )
    print(
        f"Stats: min={arr.min():.6e} max={arr.max():.6e} "
        f"mean={arr.mean():.6e} rms={np.sqrt(np.mean(arr * arr)):.6e}"
    )

    if args.save_npy:
        out_path = pathlib.Path(args.save_npy)
        np.save(out_path, arr)
        print(f"Saved array to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
